use crate::convert::*;
use crate::translate_api;
use crate::types::*;
use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse},
    Json,
};
use futures::StreamExt;
use reqwest::{Client, RequestBuilder};
use serde_json::json;
use std::{env, error::Error, sync::Arc, time::SystemTime};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Max retries for transient network errors.
const MAX_RETRIES: u32 = 5;
/// Max mid-stream retries (stream died without finish, dead connection, etc.)
const MID_STREAM_MAX_RETRIES: u32 = 5;
/// Timeout waiting for next SSE chunk from CC before considering stream dead.
const CHUNK_TIMEOUT_SECS: u64 = 20;

/// Check if an error is a transient network issue worth retrying.
fn is_retryable_error(err: &reqwest::Error) -> bool {
    if err.is_timeout() || err.is_connect() {
        return true;
    }
    // Walk the error chain for io::ErrorKind::ConnectionReset
    let mut source = err.source();
    while let Some(s) = source {
        if let Some(io_err) = s.downcast_ref::<std::io::Error>() {
            return matches!(
                io_err.kind(),
                std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
                    | std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::UnexpectedEof
            );
        }
        source = s.source();
    }
    false
}

/// Map a reqwest error to the appropriate HTTP status code.
fn map_upstream_error(err: &reqwest::Error) -> StatusCode {
    if err.is_timeout() {
        return StatusCode::GATEWAY_TIMEOUT;
    }
    let mut source = err.source();
    while let Some(s) = source {
        if let Some(io_err) = s.downcast_ref::<std::io::Error>() {
            match io_err.kind() {
                std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::UnexpectedEof => {
                    return StatusCode::BAD_GATEWAY;
                }
                _ => {}
            }
        }
        source = s.source();
    }
    StatusCode::BAD_GATEWAY
}

fn log_error_body_if_network_lost(text: &str) {
    let lower = text.to_lowercase();
    if lower.contains("network connection lost") || lower.contains("connection lost") {
        eprintln!("[NETWORK] CC API response indicates network connection lost: {}", text);
    }
}

struct CcResponseDumper {
    path: std::path::PathBuf,
    buf: String,
    created: i64,
    model: String,
    stream: bool,
    // Accumulated fields
    text: String,
    reasoning: String,
    tool_calls: Vec<serde_json::Value>,
    usage: Option<serde_json::Value>,
    finish_reason: Option<String>,
    event_count: usize,
}

impl CcResponseDumper {
    fn new(label: &str, stream: bool, model: &str) -> Option<Self> {
        let now = chrono::Local::now();
        let date_dir = now.format("%d-%m-%Y").to_string();
        let dump_dir = format!("data/{}/cc_responses", date_dir);
        let _ = std::fs::create_dir_all(&dump_dir);
        let ts = now.format("%d-%m-%Y_%H-%M-%S%.3f").to_string();
        let path = std::path::PathBuf::from(format!("{}/cc_resp_{}_{}.json", dump_dir, label, ts));
        let created = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        Some(Self {
            path,
            buf: String::new(),
            created,
            model: model.to_string(),
            stream,
            text: String::new(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: None,
            event_count: 0,
        })
    }

    fn feed_bytes(&mut self, raw_bytes: &[u8]) {
        self.buf.push_str(&String::from_utf8_lossy(raw_bytes));
        while let Some(pos) = self.buf.find('\n') {
            let line = self.buf[..pos].trim_end_matches('\r').to_string();
            self.buf = self.buf[pos + 1..].to_string();
            if line.is_empty() { continue; }
            if let Some(event) = parse_cc_event_line(&line) {
                self.process_event(&event);
            }
        }
    }

    fn feed_full_body(&mut self, body: &str) {
        for line in body.lines() {
            let line = line.trim().to_string();
            if line.is_empty() { continue; }
            if let Some(event) = parse_cc_event_line(&line) {
                self.process_event(&event);
            }
        }
    }

    fn process_event(&mut self, event: &CcStreamEvent) {
        self.event_count += 1;
        match event.event_type.as_str() {
            "text-delta" => {
                if let Some(ref t) = event.text { self.text.push_str(t); }
            }
            "reasoning-delta" => {
                if let Some(ref t) = event.text { self.reasoning.push_str(t); }
            }
            "tool-call" => {
                let id = event.tool_call_id.clone().unwrap_or_default();
                let name = event.tool_name.clone().unwrap_or_default();
                let args = event.input.as_ref()
                    .or(event.args.as_ref())
                    .or(event.arguments.as_ref())
                    .map(|v| serde_json::to_string(v).unwrap_or_default())
                    .unwrap_or_default();
                self.tool_calls.push(serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args,
                    }
                }));
            }
            "finish" => {
                self.finish_reason = event.finish_reason.clone();
                if let Some(ref u) = event.total_usage {
                    self.usage = Some(serde_json::json!({
                        "prompt_tokens": u.input_tokens.unwrap_or(0),
                        "completion_tokens": u.output_tokens.unwrap_or(0),
                    }));
                }
            }
            _ => {}
        }
    }

    fn finalize(&mut self) {
        let content = if self.text.is_empty() { None } else { Some(&self.text) };
        let reasoning_content = if self.reasoning.is_empty() { None } else { Some(&self.reasoning) };
        let tool_calls = if self.tool_calls.is_empty() { None } else { Some(&self.tool_calls) };

        let message = {
            let mut m = serde_json::json!({ "role": "assistant" });
            if let Some(c) = content { m["content"] = serde_json::Value::from(c.as_str()); }
            if let Some(r) = reasoning_content { m["reasoning_content"] = serde_json::Value::from(r.as_str()); }
            if let Some(tc) = tool_calls { m["tool_calls"] = serde_json::Value::from(tc.as_slice()); }
            m
        };

        let usage = &self.usage;
        let finish_reason = &self.finish_reason;

        let report = serde_json::json!({
            "ts": chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.3f").to_string(),
            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "stream": self.stream,
            "total_events": self.event_count,
            "choices": [{
                "index": 0,
                "finish_reason": finish_reason,
                "message": message,
            }],
            "usage": usage,
        });
        let _ = std::fs::write(&self.path, serde_json::to_string_pretty(&report).unwrap_or_default());
    }
}

impl Drop for CcResponseDumper {
    fn drop(&mut self) {
        self.finalize();
    }
}

/// Execute a POST request with exponential-backoff retry for transient errors.
async fn retry_post_request(
    build: impl Fn() -> RequestBuilder,
    max_retries: u32,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut last_err = None;
    for attempt in 0..=max_retries {
        let builder = build();
        match builder.send().await {
            Ok(resp) => return Ok(resp),
            Err(e) => {
                if !is_retryable_error(&e) || attempt == max_retries {
                    eprintln!(
                        "[ERROR] Upstream POST failed (attempt {}/{}): {}",
                        attempt + 1,
                        max_retries + 1,
                        e
                    );
                    return Err(e);
                }
                last_err = Some(e);
                let delay = std::time::Duration::from_millis(500 * (1 << attempt));
                eprintln!(
                    "[WARN] Upstream request failed (attempt {}/{}), retrying in {:?}: {}",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    last_err.as_ref().unwrap()
                );
                tokio::time::sleep(delay).await;
            }
        }
    }
    Err(last_err.unwrap())
}

const MODELS_JSON_URL: &str =
    "https://raw.githubusercontent.com/ninehills/pi-commandcode-provider/main/models.json";

#[derive(Clone)]
pub struct AppState {
    pub client: Client,
    pub api_base: String,
    pub default_api_key: Option<String>,
    pub models_json: ModelsJson,
}

/// Try to fetch the latest models.json from upstream, fall back to local file.
pub async fn load_models_json(client: &Client) -> ModelsJson {
    // 1. Try remote
    match client.get(MODELS_JSON_URL).send().await {
        Ok(resp) if resp.status().is_success() => {
            match resp.text().await {
                Ok(text) => match serde_json::from_str::<ModelsJson>(&text) {
                    Ok(models) => {
                        // Cache to local file for offline fallback
                        if let Err(e) = tokio::fs::write("models.json", &text).await {
                            tracing::warn!("Failed to cache models.json: {}", e);
                        } else {
                            tracing::info!("models.json updated from remote and cached");
                        }
                        return models;
                    }
                    Err(e) => tracing::warn!("Remote models.json parse failed: {}", e),
                },
                Err(e) => tracing::warn!("Failed to read remote models.json body: {}", e),
            }
        }
        Ok(resp) => tracing::warn!("Remote models.json returned status {}", resp.status()),
        Err(e) => tracing::warn!("Failed to fetch remote models.json: {}", e),
    }

    // 2. Fall back to local file
    match tokio::fs::read_to_string("models.json").await {
        Ok(text) => match serde_json::from_str::<ModelsJson>(&text) {
            Ok(models) => {
                tracing::info!("Loaded models.json from local cache");
                return models;
            }
            Err(e) => tracing::warn!("Local models.json parse failed: {}", e),
        },
        Err(e) => tracing::warn!("No local models.json found: {}", e),
    }

    // 3. Ultimate fallback: empty list so the server can still start
    tracing::error!("Could not load models.json from remote or local cache; model list will be empty");
    ModelsJson {
        providers: Default::default(),
        provider_groups: vec![],
        models: vec![],
        pricing: vec![],
    }
}

/// Build the CommandCode request from an OpenAI request.
pub(crate) fn build_cc_request(
    req: &ChatCompletionRequest,
    working_dir: &str,
    environment: &str,
) -> CcRequest {
    let system = req
        .system
        .clone()
        .unwrap_or_else(|| extract_system(&req.messages));
    let max_tokens = req.max_tokens.unwrap_or(64000).min(200_000);
    // CommandCode API always requires stream=true; we buffer for non-streaming clients
    let _client_stream = req.stream.unwrap_or(false);
    let stream = true;

    let cc_req = CcRequest {
        config: CcConfig {
            working_dir: working_dir.to_string(),
            date: chrono::Local::now().format("%Y-%m-%d").to_string(),
            environment: environment.to_string(),
            structure: vec![],
            is_git_repo: false,
            current_branch: String::new(),
            main_branch: String::new(),
            git_status: String::new(),
            recent_commits: vec![],
        },
        memory: String::new(),
        taste: String::new(),
        skills: None,
        permission_mode: "standard".to_string(),
        params: CcParams {
            model: req.model.clone(),
            reasoning_effort: req.reasoning_effort.clone(),
            thinking: req.thinking.clone(),
            messages: messages_to_cc(&req.messages),
            tools: tools_to_cc(req.tools.as_deref()),
            system,
            max_tokens,
            stream,
        },
    };

    // Dump full JSON to data/{date}/cc_requests/ for debugging
    let now = chrono::Local::now();
    let date_dir = now.format("%d-%m-%Y").to_string();
    let dump_dir = format!("data/{}/cc_requests", date_dir);
    let _ = std::fs::create_dir_all(&dump_dir);
    let timestamp = now.format("%d-%m-%Y_%H-%M-%S").to_string();
    let dump_path = format!("{}/cc_req_{}.json", dump_dir, timestamp);
    if let Ok(json) = serde_json::to_string_pretty(&cc_req) {
        let _ = std::fs::write(&dump_path, &json);
    }
    cc_req
}

/// Helper to extract Authorization header.
pub(crate) fn get_api_key(headers: &HeaderMap, default: &Option<String>) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|s| s.to_string())
        .or_else(|| default.clone())
}

fn proxy_error_response(status: StatusCode, message: &'static str) -> Response {
    (
        status,
        Json(json!({ "error": { "message": message, "type": "proxy_error" } })),
    )
        .into_response()
}

fn usage_from_cc_usage(cc_usage: CcUsage) -> Usage {
    let mut usage = Usage {
        prompt_tokens: cc_usage.input_tokens.unwrap_or(0),
        completion_tokens: cc_usage.output_tokens.unwrap_or(0),
        ..Usage::default()
    };

    if let Some(details) = cc_usage.input_token_details {
        usage.prompt_tokens +=
            details.cache_read_tokens.unwrap_or(0) + details.cache_write_tokens.unwrap_or(0);
    }
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
    usage
}

/// Dangling-intent detection: checks if assistant text/thought ends mid-sentence.
fn ends_with_dangling_intent(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() { return false; }
    if t.ends_with(':') { return true; }
    let lower = t.to_lowercase();
    let phrases = [
        "let me", "let's", "i'll", "i will",
        "i'm going to", "i am going to", "we'll", "we will",
    ];
    for phrase in &phrases {
        if lower.ends_with(phrase) || lower.ends_with(&format!("{} ", phrase)) {
            return true;
        }
    }
    false
}

/// Non-streaming completion handler with recovery loop.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::extract::Json(req): axum::extract::Json<ChatCompletionRequest>,
) -> Result<Response, StatusCode> {
    let api_key = get_api_key(&headers, &state.default_api_key).ok_or(StatusCode::UNAUTHORIZED)?;

    let working_dir = env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let environment = format!(
        "{}-{}, Rust {}",
        env::consts::OS,
        env::consts::ARCH,
        env!("CARGO_PKG_VERSION")
    );

    let model_id = req.model.clone();
    let mut cc_req = build_cc_request(&req, &working_dir, &environment);
    cc_req.params.stream = true;

    let mut nudge_count = 0u32;
    let mut length_count = 0u32;
    let mut intent_count = 0u32;
    const MAX_EMPTY_TURNS: u32 = 8;

    'recovery_loop: loop {
        let body = serde_json::to_string(&cc_req).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let url = format!("{}/alpha/generate", state.api_base);
        let session_id = uuid::Uuid::new_v4().to_string();

        let cc_response = retry_post_request(
            || {
                state
                    .client
                    .post(&url)
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("x-command-code-version", "0.37.2")
                    .header("x-cli-environment", "production")
                    .header("x-project-slug", "commandcode2api")
                    .header("x-taste-learning", "false")
                    .header("x-co-flag", "false")
                    .header("x-session-id", &session_id)
                    .body(body.clone())
            },
            MAX_RETRIES,
        )
        .await
        .map_err(|e| {
            eprintln!("[ERROR] Non-stream POST failed after all retries: {}", e);
            map_upstream_error(&e)
        })?;

        if !cc_response.status().is_success() {
            let status = cc_response.status();
            let text = cc_response
                .text()
                .await
                .unwrap_or_default()
                .chars()
                .take(500)
                .collect::<String>();
            log_error_body_if_network_lost(&text);
            eprintln!("[ERROR] CC API returned HTTP {}: {}", status, text);
            return Ok((
                StatusCode::BAD_GATEWAY,
                format!("CommandCode API error {}: {}", status, text),
            )
                .into_response());
        }

        let bytes = cc_response
            .bytes()
            .await
            .map_err(|_| StatusCode::BAD_GATEWAY)?;
        let text = String::from_utf8_lossy(&bytes);

        if let Some(mut dumper) = CcResponseDumper::new("openai_nonstream", false, &model_id) {
            dumper.feed_full_body(&text);
        }

        let mut full_text = String::new();
        let mut reasoning_text = String::new();
        let mut tool_calls: Vec<ToolCall> = vec![];
        let mut usage = Usage::default();
        let mut finish_reason: Option<String> = None;
        let mut raw_finish_reason: Option<String> = None;
        let mut finished = false;
        let mut has_error = false;
        let mut error_msg: Option<String> = None;

        for line in text.lines() {
            if let Some(event) = parse_cc_event_line(line) {
                match event.event_type.as_str() {
                    "text-delta" => {
                        if let Some(delta) = event.text { full_text.push_str(&delta); }
                    }
                    "reasoning-delta" => {
                        if let Some(delta) = event.text { reasoning_text.push_str(&delta); }
                    }
                    "reasoning-end" => {}
                    "tool-call" => {
                        let id = event.tool_call_id.unwrap_or_default();
                        let name = event.tool_name.unwrap_or_default();
                        let args = event
                            .input
                            .or(event.args)
                            .or(event.arguments)
                            .and_then(|v| serde_json::to_string(&v).ok())
                            .unwrap_or_default();
                        tool_calls.push(ToolCall {
                            index: None,
                            id: Some(id),
                            tool_type: Some("function".to_string()),
                            function: Some(ToolCallFunction {
                                name: Some(name),
                                arguments: Some(args),
                            }),
                        });
                    }
                    "finish" => {
                        raw_finish_reason = event.finish_reason.clone();
                        finish_reason = map_finish_reason(event.finish_reason.as_deref());
                        finished = true;
                        if let Some(u) = event.total_usage {
                            usage = usage_from_cc_usage(u);
                        }
                    }
                    "error" => {
                        has_error = true;
                        error_msg = event
                            .error
                            .as_ref()
                            .and_then(|e| e.get("message").and_then(|m| m.as_str()))
                            .or_else(|| event.error.as_ref().and_then(|e| e.as_str()))
                            .map(|s| s.to_string());
                    }
                    _ => {}
                }
                if finished || has_error { break; }
            }
        }

        if has_error {
            let msg = error_msg.unwrap_or_else(|| "Stream error".to_string());
            return Ok((
                StatusCode::BAD_GATEWAY,
                format!("CommandCode stream error: {}", msg),
            ).into_response());
        }

        let has_text = !full_text.trim().is_empty();
        let has_tool_calls = !tool_calls.is_empty();
        let has_reasoning = !reasoning_text.is_empty();

        // Success: has visible content or tool calls
        if has_text || has_tool_calls {
            let message = ChatCompletionMessage {
                role: "assistant".to_string(),
                content: if full_text.is_empty() { None } else { Some(full_text) },
                reasoning_content: if reasoning_text.is_empty() { None } else { Some(reasoning_text) },
                tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
            };

            let response = ChatCompletionResponse {
                id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                object: "chat.completion".to_string(),
                created: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64,
                model: model_id,
                choices: vec![Choice { index: 0, message, finish_reason }],
                usage: if usage.total_tokens > 0 { Some(usage) } else { None },
            };
            return Ok(axum::Json(response).into_response());
        }

        // Length recovery: max_tokens truncated — auto-continue up to 3 times
        if matches!(raw_finish_reason.as_deref(), Some("length") | Some("max_tokens") | Some("max-tokens") | Some("max_output_tokens")) {
            if length_count < 3 {
                length_count += 1;
                eprintln!("[WARN] Output truncated by token cap, auto-continuing ({}/3)", length_count);
                cc_req.params.messages.push(serde_json::json!({
                    "role": "user", "content": "Please continue from where you left off."
                }));
                continue 'recovery_loop;
            }
        }

        // Dangling-intent detection: stop reason but sentence seems incomplete
        if raw_finish_reason.as_deref() == Some("stop")
            && intent_count < 2
            && ends_with_dangling_intent(&reasoning_text)
        {
            intent_count += 1;
            eprintln!("[WARN] Dangling-intent turn, auto-continuing ({}/2)", intent_count);
            cc_req.params.messages.push(serde_json::json!({
                "role": "user", "content": "Please continue from where you left off."
            }));
            continue 'recovery_loop;
        }

        // Empty-turn recovery: nudge (inject "Please continue.") up to 2 times
        if nudge_count < 2 {
            nudge_count += 1;
            eprintln!("[WARN] Empty assistant response, nudging ({}/2)", nudge_count);
            cc_req.params.messages.push(serde_json::json!({
                "role": "user", "content": "Please continue."
            }));
            continue 'recovery_loop;
        }

        // After nudges, pause up to MAX_EMPTY_TURNS total
        if nudge_count < MAX_EMPTY_TURNS {
            nudge_count += 1;
            eprintln!("[WARN] Empty assistant response, pausing (attempt {})", nudge_count);
            continue 'recovery_loop;
        }

        // Exhausted: return reasoning as content or empty
        eprintln!("[WARN] Exhausted empty-turn recovery, returning empty");
        let message = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: if has_reasoning { Some(reasoning_text) } else { Some(String::new()) },
            reasoning_content: None,
            tool_calls: None,
        };
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_secs() as i64,
            model: model_id,
            choices: vec![Choice { index: 0, message, finish_reason: Some("stop".to_string()) }],
            usage: if usage.total_tokens > 0 { Some(usage) } else { None },
        };
        return Ok(axum::Json(response).into_response());
    }
}

/// Streaming completion handler using SSE.
pub async fn chat_completions_stream(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::extract::Json(req): axum::extract::Json<ChatCompletionRequest>,
) -> Result<
    Sse<
        impl tokio_stream::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>,
    >,
    StatusCode,
> {
    let api_key = get_api_key(&headers, &state.default_api_key).ok_or(StatusCode::UNAUTHORIZED)?;

    let working_dir = env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let environment = format!(
        "{}-{}, Rust {}",
        env::consts::OS,
        env::consts::ARCH,
        env!("CARGO_PKG_VERSION")
    );

    let mut cc_req = build_cc_request(&req, &working_dir, &environment);
    cc_req.params.stream = true;
    let body = serde_json::to_string(&cc_req).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let url = format!("{}/alpha/generate", state.api_base);
    let model_id = req.model.clone();

    let (tx, rx) =
        mpsc::channel::<Result<axum::response::sse::Event, std::convert::Infallible>>(128);
    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    tokio::spawn(async move {
        let mut mid_stream_retries = MID_STREAM_MAX_RETRIES;
        let mut emitted_visible = false;

        'stream_retry: loop {
            // NEW session_id on every retry to route to different backend
            let retry_session_id = uuid::Uuid::new_v4().to_string();
            let cc_response = match retry_post_request(
                || {
                    state
                        .client
                        .post(&url)
                        .header(header::CONTENT_TYPE, "application/json")
                        .header("Authorization", format!("Bearer {}", api_key))
                        .header("x-command-code-version", "0.37.2")
                        .header("x-cli-environment", "production")
                        .header("x-project-slug", "commandcode2api")
                        .header("x-taste-learning", "false")
                        .header("x-co-flag", "false")
                        .header("x-session-id", &retry_session_id)
                        .body(body.clone())
                },
                MAX_RETRIES,
            )
            .await
            {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[ERROR] OpenAI stream POST failed (retries left: {}): {}", mid_stream_retries, e);
                    if emitted_visible || mid_stream_retries == 0 {
                        let _ = tx
                            .send(Ok(error_event(
                                &completion_id, created, &model_id,
                                &format!("Upstream request failed: {}", e),
                            )))
                            .await;
                        break 'stream_retry;
                    }
                    mid_stream_retries -= 1;
                    let delay = std::time::Duration::from_millis(500 * (1 << (MID_STREAM_MAX_RETRIES - mid_stream_retries)));
                    let retry_n = MID_STREAM_MAX_RETRIES - mid_stream_retries;
                    eprintln!("[WARN] OpenAI stream POST failed, retrying #{retry_n} in {:?}...", delay);
                    tokio::time::sleep(delay).await;
                    continue 'stream_retry;
                }
            };

            if !cc_response.status().is_success() {
                let status = cc_response.status();
                let text = cc_response
                    .text()
                    .await
                    .unwrap_or_default()
                    .chars()
                    .take(500)
                    .collect::<String>();
                log_error_body_if_network_lost(&text);
                eprintln!(
                    "[ERROR] CC API returned HTTP {} (OpenAI stream, retries left: {}): {}",
                    status, mid_stream_retries, text
                );
                if emitted_visible || mid_stream_retries == 0 {
                    let _ = tx
                        .send(Ok(error_event(
                            &completion_id, created, &model_id,
                            &format!("CC API error {}", status),
                        )))
                        .await;
                    break 'stream_retry;
                }
                mid_stream_retries -= 1;
                let delay = std::time::Duration::from_millis(500 * (1 << (MID_STREAM_MAX_RETRIES - mid_stream_retries)));
                let retry_n = MID_STREAM_MAX_RETRIES - mid_stream_retries;
                eprintln!("[WARN] Retrying OpenAI stream POST after HTTP {} #{retry_n} in {:?}...", status, delay);
                tokio::time::sleep(delay).await;
                continue 'stream_retry;
            }

            if mid_stream_retries < MID_STREAM_MAX_RETRIES {
                eprintln!("[INFO] OpenAI stream retry succeeded, reading started");
            }

            let mut buf = String::new();
            let mut byte_stream = cc_response.bytes_stream();
            let mut tool_call_idx = 0usize;
            let mut sent_role = false;
            let mut reasoning_buf = String::new();
            let mut stream_success = false;
            let mut has_text_content = false;
            let mut has_tool_calls = false;
            let mut chunk_dumper = CcResponseDumper::new("openai_stream", true, &model_id);
            let chunk_timeout = std::time::Duration::from_secs(CHUNK_TIMEOUT_SECS);
            'read_stream: loop {
                let next_chunk = tokio::time::timeout(chunk_timeout, byte_stream.next()).await;
                match next_chunk {
                    Ok(Some(Ok(chunk))) => {
                        if let Some(ref mut d) = chunk_dumper {
                            d.feed_bytes(&chunk);
                        }
                        buf.push_str(&String::from_utf8_lossy(&chunk));

                        while let Some(pos) = buf.find('\n') {
                            let line = buf[..pos].trim_end_matches('\r').to_string();
                            buf = buf[pos + 1..].to_string();

                            if let Some(event) = parse_cc_event_line(&line) {
                                match event.event_type.as_str() {
                                    "text-delta" => {
                                        emitted_visible = true;
                                        has_text_content = true;
                                        if !sent_role {
                                            let _ = tx
                                                .send(Ok(chunk_event(
                                                    &completion_id,
                                                    created,
                                                    &model_id,
                                                    0,
                                                    ChunkDelta {
                                                        role: Some("assistant".to_string()),
                                                        content: None,
                                                        reasoning_content: None,
                                                        tool_calls: None,
                                                    },
                                                    None,
                                                    None,
                                                )))
                                                .await;
                                            sent_role = true;
                                        }
                                        if let Some(delta) = event.text {
                                            let _ = tx
                                                .send(Ok(chunk_event(
                                                    &completion_id,
                                                    created,
                                                    &model_id,
                                                    0,
                                                    ChunkDelta {
                                                        role: None,
                                                        content: Some(delta),
                                                        reasoning_content: None,
                                                        tool_calls: None,
                                                    },
                                                    None,
                                                    None,
                                                )))
                                                .await;
                                        }
                                    }
                                    "reasoning-delta" => {
                                        emitted_visible = true;
                                        if let Some(delta) = event.text {
                                            reasoning_buf.push_str(&delta);
                                            let _ = tx
                                                .send(Ok(chunk_event(
                                                    &completion_id,
                                                    created,
                                                    &model_id,
                                                    0,
                                                    ChunkDelta {
                                                        role: None,
                                                        content: None,
                                                        reasoning_content: Some(delta),
                                                        tool_calls: None,
                                                    },
                                                    None,
                                                    None,
                                                )))
                                                .await;
                                        }
                                    }
                                    "reasoning-end" => {}
                                    "tool-call" => {
                                        emitted_visible = true;
                                        has_tool_calls = true;
                                        let id = event.tool_call_id.unwrap_or_default();
                                        let name = event.tool_name.unwrap_or_default();
                                        let args = event
                                            .input
                                            .or(event.args)
                                            .or(event.arguments)
                                            .and_then(|v| serde_json::to_string(&v).ok())
                                            .unwrap_or_default();
                                        let _ = tx
                                            .send(Ok(chunk_event(
                                                &completion_id,
                                                created,
                                                &model_id,
                                                0,
                                                ChunkDelta {
                                                    role: None,
                                                    content: None,
                                                    reasoning_content: None,
                                                    tool_calls: Some(vec![ToolCall {
                                                        index: Some(tool_call_idx),
                                                        id: Some(id),
                                                        tool_type: Some("function".to_string()),
                                                        function: Some(ToolCallFunction {
                                                            name: Some(name),
                                                            arguments: Some(args),
                                                        }),
                                                    }]),
                                                },
                                                None,
                                                None,
                                            )))
                                            .await;
                                        tool_call_idx += 1;
                                    }
                                    "finish" => {
                                        if !has_text_content && !has_tool_calls && !reasoning_buf.is_empty() {
                                            let _ = tx
                                                .send(Ok(chunk_event(
                                                    &completion_id,
                                                    created,
                                                    &model_id,
                                                    0,
                                                    ChunkDelta {
                                                        role: None,
                                                        content: Some(reasoning_buf.clone()),
                                                        reasoning_content: None,
                                                        tool_calls: None,
                                                    },
                                                    None,
                                                    None,
                                                )))
                                                .await;
                                        }
                                        let finish_reason = map_finish_reason(event.finish_reason.as_deref());
                                        let usage = event
                                            .total_usage
                                            .map(usage_from_cc_usage)
                                            .filter(|usage| usage.total_tokens > 0);
                                        stream_success = true;
                                        let _ = tx
                                            .send(Ok(chunk_event(
                                                &completion_id,
                                                created,
                                                &model_id,
                                                0,
                                                ChunkDelta::default(),
                                                finish_reason,
                                                usage,
                                            )))
                                            .await;
                                        break 'read_stream;
                                    }
                                    "error" => {
                                        let msg = event
                                            .error
                                            .as_ref()
                                            .and_then(|e| e.get("message").and_then(|m| m.as_str()))
                                            .or_else(|| event.error.as_ref().and_then(|e| e.as_str()))
                                            .unwrap_or("Stream error");
                                        let _ = tx
                                            .send(Ok(error_event(&completion_id, created, &model_id, msg)))
                                            .await;
                                        break 'read_stream;
                                    }
                                    _ => {}
                                }
                      
                            }
                        }
                    }
                    Ok(Some(Err(e))) => {
                        eprintln!(
                            "[ERROR] OpenAI stream read error (retries left: {}, emitted_visible: {}): {}",
                            mid_stream_retries, emitted_visible, e
                        );
                        if emitted_visible {
                            let _ = tx
                                .send(Ok(chunk_event(
                                    &completion_id, created, &model_id, 0,
                                    ChunkDelta::default(),
                                    Some("stop".to_string()),
                                    None,
                                )))
                                .await;
                            break 'stream_retry;
                        }
                        if mid_stream_retries > 0 {
                            mid_stream_retries -= 1;
                            let delay = std::time::Duration::from_millis(500 * (1 << (MID_STREAM_MAX_RETRIES - mid_stream_retries)));
                            let retry_n = MID_STREAM_MAX_RETRIES - mid_stream_retries;
                            eprintln!("[WARN] Retrying OpenAI stream read #{retry_n} in {:?}...", delay);
                            tokio::time::sleep(delay).await;
                            continue 'stream_retry;
                        }
                        let _ = tx
                            .send(Ok(error_event(
                                &completion_id,
                                created,
                                &model_id,
                                &format!("Stream read error after retries: {}", e),
                            )))
                            .await;
                        break 'stream_retry;
                    }
                    Ok(None) => {
                        // Process remaining data in buffer before finishing
                        if !buf.is_empty() {
                            if let Some(event) = parse_cc_event_line(&buf) {
                                if event.event_type == "finish" {
                                    if !has_text_content && !has_tool_calls && !reasoning_buf.is_empty() {
                                        let _ = tx
                                            .send(Ok(chunk_event(
                                                &completion_id,
                                                created,
                                                &model_id,
                                                0,
                                                ChunkDelta {
                                                    role: None,
                                                    content: Some(reasoning_buf.clone()),
                                                    reasoning_content: None,
                                                    tool_calls: None,
                                                },
                                                None,
                                                None,
                                            )))
                                            .await;
                                    }
                                    let finish_reason = map_finish_reason(event.finish_reason.as_deref());
                                    let usage = event
                                        .total_usage
                                        .map(usage_from_cc_usage)
                                        .filter(|usage| usage.total_tokens > 0);
                                    let _ = tx
                                        .send(Ok(chunk_event(
                                            &completion_id,
                                            created,
                                            &model_id,
                                            0,
                                            ChunkDelta::default(),
                                            finish_reason,
                                            usage,
                                        )))
                                        .await;
                                    stream_success = true;
                                }
                            }
                        }
                        break 'read_stream;
                    }
                    Err(_elapsed) => {
                        eprintln!(
                            "[ERROR] OpenAI stream chunk timeout after {}s (retries left: {}, emitted_visible: {}, buf_remaining: {})",
                            CHUNK_TIMEOUT_SECS, mid_stream_retries, emitted_visible, buf.len()
                        );
                        if emitted_visible {
                            let _ = tx
                                .send(Ok(chunk_event(
                                    &completion_id, created, &model_id, 0,
                                    ChunkDelta::default(),
                                    Some("stop".to_string()),
                                    None,
                                )))
                                .await;
                            break 'stream_retry;
                        }
                        if !buf.is_empty() {
                            eprintln!("[WARN] OpenAI stream partial buffer on timeout: {}", buf.chars().take(200).collect::<String>());
                        }
                        if mid_stream_retries > 0 {
                            mid_stream_retries -= 1;
                            let delay = std::time::Duration::from_millis(500 * (1 << (MID_STREAM_MAX_RETRIES - mid_stream_retries)));
                            let retry_n = MID_STREAM_MAX_RETRIES - mid_stream_retries;
                            eprintln!("[WARN] Retrying OpenAI stream after chunk timeout #{retry_n} in {:?}...", delay);
                            tokio::time::sleep(delay).await;
                            continue 'stream_retry;
                        }
                        let _ = tx
                            .send(Ok(error_event(
                                &completion_id, created, &model_id,
                                &format!("Stream timed out after {}s", CHUNK_TIMEOUT_SECS),
                            )))
                            .await;
                        break 'stream_retry;
                    }
                }

            }

            if stream_success {
                break 'stream_retry;
            }

            if mid_stream_retries > 0 {
                mid_stream_retries -= 1;
                let delay = std::time::Duration::from_millis(500 * (1 << (MID_STREAM_MAX_RETRIES - mid_stream_retries)));
                let retry_n = MID_STREAM_MAX_RETRIES - mid_stream_retries;
                eprintln!("[WARN] OpenAI stream ended without finish, retrying #{retry_n} in {:?}...", delay);
                tokio::time::sleep(delay).await;
                continue 'stream_retry;
            }

            // All retries exhausted: graceful end if we emitted content
            if emitted_visible {
                let _ = tx
                    .send(Ok(chunk_event(
                        &completion_id, created, &model_id, 0,
                        ChunkDelta::default(),
                        Some("stop".to_string()),
                        None,
                    )))
                    .await;
            } else {
                let _ = tx
                    .send(Ok(error_event(
                        &completion_id, created, &model_id,
                        "Stream ended unexpectedly after all retries",
                    )))
                    .await;
            }
            break 'stream_retry;
        }

        let _ = tx
            .send(Ok(axum::response::sse::Event::default().data("[DONE]")))
            .await;
    });

    let stream = ReceiverStream::new(rx);

    Ok(Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::new().text("keep-alive")))
}

fn chunk_event(
    id: &str,
    created: i64,
    model: &str,
    choice_index: usize,
    delta: ChunkDelta,
    finish_reason: Option<String>,
    usage: Option<Usage>,
) -> axum::response::sse::Event {
    let chunk = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: choice_index,
            delta,
            finish_reason,
        }],
        usage,
    };
    axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
}

fn error_event(id: &str, created: i64, model: &str, message: &str) -> axum::response::sse::Event {
    let chunk = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: Some("assistant".to_string()),
                content: Some(message.to_string()),
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: None,
    };
    axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
}

// We need a unified handler that picks streaming vs non-streaming.
pub async fn chat_completions_handler(
    state: State<Arc<AppState>>,
    headers: HeaderMap,
    axum::extract::Json(req): axum::extract::Json<ChatCompletionRequest>,
) -> Response {
    let stream = req.stream.as_ref().copied().unwrap_or(false);
    if stream {
        match chat_completions_stream(state, headers, axum::extract::Json(req)).await {
            Ok(sse) => sse.into_response(),
            Err(status) => proxy_error_response(status, "Failed to start stream"),
        }
    } else {
        match chat_completions(state, headers, axum::extract::Json(req)).await {
            Ok(resp) => resp,
            Err(status) => proxy_error_response(status, "Failed to complete request"),
        }
    }
}

// ============================================================================
// Anthropic /v1/messages endpoint
// ============================================================================

fn anthropic_error_response(status: StatusCode, message: &str) -> Response {
    let error_type = match status {
        StatusCode::BAD_REQUEST => "invalid_request_error",
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        _ => "api_error",
    };
    (
        status,
        Json(json!({"type":"error","error":{"type": error_type, "message": message}})),
    )
        .into_response()
}

fn anthropic_sse_event(ev: &translate_api::MessageEvent) -> axum::response::sse::Event {
    let data = serde_json::to_string(ev).unwrap_or_default();
    axum::response::sse::Event::default()
        .event(ev.event_type_str())
        .data(data)
}

pub async fn messages_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<translate_api::MessageRequest>,
) -> Response {
    let is_stream = req.stream.unwrap_or(false);
    if is_stream {
        match messages_stream(&state, &headers, req).await {
            Ok(sse) => sse.into_response(),
            Err(status) => anthropic_error_response(status, "stream failed"),
        }
    } else {
        match messages_non_streaming(&state, &headers, req).await {
            Ok(response) => Json(response).into_response(),
            Err(status) => anthropic_error_response(status, "request failed"),
        }
    }
}

async fn messages_non_streaming(
    state: &Arc<AppState>,
    headers: &HeaderMap,
    anthropic_req: translate_api::MessageRequest,
) -> Result<translate_api::MessageResponse, StatusCode> {
    let api_key = get_api_key(headers, &state.default_api_key).ok_or(StatusCode::UNAUTHORIZED)?;
    let original_model = anthropic_req.model.clone();

    let openai_req = translate_api::anthropic_request_to_openai(&anthropic_req);

    let working_dir = env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let environment = format!(
        "{}-{}, Rust {}",
        env::consts::OS,
        env::consts::ARCH,
        env!("CARGO_PKG_VERSION")
    );

    let cc_req = build_cc_request(&openai_req, &working_dir, &environment);
    let body = serde_json::to_string(&cc_req).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let url = format!("{}/alpha/generate", state.api_base);
    let session_id = uuid::Uuid::new_v4().to_string();

    let cc_response = retry_post_request(
        || {
            state
                .client
                .post(&url)
                .header(header::CONTENT_TYPE, "application/json")
                .header("Authorization", format!("Bearer {}", api_key))
                .header("x-command-code-version", "0.24.1")
                .header("x-cli-environment", "production")
                .header("x-project-slug", "commandcode2api")
                .header("x-taste-learning", "false")
                .header("x-co-flag", "false")
                .header("x-session-id", &session_id)
                .body(body.clone())
        },
        MAX_RETRIES,
    )
    .await
    .map_err(|e| {
        eprintln!("[ERROR] Non-stream POST failed after all retries: {}", e);
        map_upstream_error(&e)
    })?;

    if !cc_response.status().is_success() {
        let status = cc_response.status();
        let text = cc_response
            .text()
            .await
            .unwrap_or_default()
            .chars()
            .take(500)
            .collect::<String>();
        log_error_body_if_network_lost(&text);
        eprintln!("[ERROR] CC API returned HTTP {}: {}", status, text);
        return Err(StatusCode::BAD_GATEWAY);
    }

    // Buffer SSE stream, accumulate into OpenAI response
    let bytes = cc_response
        .bytes()
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?;
    let text = String::from_utf8_lossy(&bytes);

    // Dump raw CC response
    if let Some(mut dumper) = CcResponseDumper::new("anthropic_nonstream", false, &original_model) {
        dumper.feed_full_body(&text);
    }

    let mut full_text = String::new();
    let mut reasoning_text = String::new();
    let mut tool_calls: Vec<ToolCall> = vec![];
    let mut usage = Usage::default();
    let mut finish_reason: Option<String> = None;
    let mut finished = false;

    for line in text.lines() {
        if let Some(event) = parse_cc_event_line(line) {
            match event.event_type.as_str() {
                "text-delta" => {
                    if let Some(delta) = event.text {
                        full_text.push_str(&delta);
                    }
                }
                "reasoning-delta" => {
                    if let Some(delta) = event.text {
                        reasoning_text.push_str(&delta);
                    }
                }
                "tool-call" => {
                    let id = event.tool_call_id.unwrap_or_default();
                    let name = event.tool_name.unwrap_or_default();
                    let args = event
                        .input
                        .or(event.args)
                        .or(event.arguments)
                        .and_then(|v| serde_json::to_string(&v).ok())
                        .unwrap_or_default();
                    tool_calls.push(ToolCall {
                        index: None,
                        id: Some(id),
                        tool_type: Some("function".to_string()),
                        function: Some(ToolCallFunction {
                            name: Some(name),
                            arguments: Some(args),
                        }),
                    });
                }
                "finish" => {
                    finish_reason = map_finish_reason(event.finish_reason.as_deref());
                    finished = true;
                    if let Some(u) = event.total_usage {
                        usage = usage_from_cc_usage(u);
                    }
                }
                "error" => {
                    let _msg = event
                        .error
                        .as_ref()
                        .and_then(|e| e.get("message").and_then(|m| m.as_str()))
                        .or_else(|| event.error.as_ref().and_then(|e| e.as_str()))
                        .unwrap_or("Stream error");
                    return Err(StatusCode::BAD_GATEWAY);
                }
                _ => {}
            }
            if finished {
                break;
            }
        }
    }

    let mut message = ChatCompletionMessage {
        role: "assistant".to_string(),
        content: if full_text.is_empty() {
            None
        } else {
            Some(full_text)
        },
        reasoning_content: if reasoning_text.is_empty() {
            None
        } else {
            Some(reasoning_text)
        },
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
    };

    if message.content.is_none() && message.tool_calls.is_none() {
        message.content = Some(String::new());
    }
    if message.tool_calls.is_none()
        && message.content.as_deref().map_or(true, |c| c.is_empty())
        && message.reasoning_content.as_deref().map_or(false, |r| !r.is_empty())
    {
        message.content = message.reasoning_content.take();
    }

    let openai_resp = ChatCompletionResponse {
        id: format!("msg_{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        model: anthropic_req.model.clone(),
        choices: vec![Choice {
            index: 0,
            message,
            finish_reason,
        }],
        usage: if usage.total_tokens > 0 {
            Some(usage)
        } else {
            None
        },
    };

    Ok(translate_api::openai_response_to_anthropic(
        &openai_resp,
        &original_model,
    ))
}

async fn messages_stream(
    state: &Arc<AppState>,
    headers: &HeaderMap,
    anthropic_req: translate_api::MessageRequest,
) -> Result<
    Sse<impl tokio_stream::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>>,
    StatusCode,
> {
    let api_key = get_api_key(headers, &state.default_api_key).ok_or(StatusCode::UNAUTHORIZED)?;
    let original_model = anthropic_req.model.clone();
    let msg_id = format!("msg_{}", uuid::Uuid::new_v4());

    let mut openai_req = translate_api::anthropic_request_to_openai(&anthropic_req);
    openai_req.stream = Some(true);

    let working_dir = env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let environment = format!(
        "{}-{}, Rust {}",
        env::consts::OS,
        env::consts::ARCH,
        env!("CARGO_PKG_VERSION")
    );

    let cc_req = build_cc_request(&openai_req, &working_dir, &environment);
    let body = serde_json::to_string(&cc_req).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let url = format!("{}/alpha/generate", state.api_base);

    let (tx, rx) =
        mpsc::channel::<Result<axum::response::sse::Event, std::convert::Infallible>>(128);
    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let state = state.clone();

    tokio::spawn(async move {
        let mut mid_stream_retries = MID_STREAM_MAX_RETRIES;
        let mut emitted_visible = false;

        'stream_retry: loop {
            // NEW session_id on every retry to route to different backend
            let retry_session_id = uuid::Uuid::new_v4().to_string();
            let cc_response = match retry_post_request(
                || {
                    state
                        .client
                        .post(&url)
                        .header(header::CONTENT_TYPE, "application/json")
                        .header("Authorization", format!("Bearer {}", api_key))
                        .header("x-command-code-version", "0.24.1")
                        .header("x-cli-environment", "production")
                        .header("x-project-slug", "commandcode2api")
                        .header("x-taste-learning", "false")
                        .header("x-co-flag", "false")
                        .header("x-session-id", &retry_session_id)
                        .body(body.clone())
                },
                MAX_RETRIES,
            )
            .await
            {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[ERROR] Anthropic stream POST failed (retries left: {}): {}", mid_stream_retries, e);
                    if emitted_visible || mid_stream_retries == 0 {
                        let ev = translate_api::MessageEvent::MessageDelta {
                            delta: translate_api::MessageDeltaInfo {
                                stop_reason: "end_turn".to_string(),
                                stop_sequence: None,
                            },
                            usage: None,
                        };
                        let _ = tx.send(Ok(anthropic_sse_event(&ev))).await;
                        let _ = tx.send(Ok(anthropic_sse_event(&translate_api::MessageEvent::MessageStop))).await;
                        break 'stream_retry;
                    }
                    mid_stream_retries -= 1;
                    let delay = std::time::Duration::from_millis(500 * (1 << (MID_STREAM_MAX_RETRIES - mid_stream_retries)));
                    let retry_n = MID_STREAM_MAX_RETRIES - mid_stream_retries;
                    eprintln!("[WARN] Retrying Anthropic stream POST #{retry_n} in {:?}...", delay);
                    tokio::time::sleep(delay).await;
                    continue 'stream_retry;
                }
            };

            if !cc_response.status().is_success() {
                let status = cc_response.status();
                let text = cc_response
                    .text()
                    .await
                    .unwrap_or_default()
                    .chars()
                    .take(500)
                    .collect::<String>();
                log_error_body_if_network_lost(&text);
                eprintln!("[ERROR] CC API returned HTTP {} (Anthropic stream, retries left: {}): {}", status, mid_stream_retries, text);
                if emitted_visible || mid_stream_retries == 0 {
                    let ev = translate_api::MessageEvent::MessageDelta {
                        delta: translate_api::MessageDeltaInfo {
                            stop_reason: "end_turn".to_string(),
                            stop_sequence: None,
                        },
                        usage: None,
                    };
                    let _ = tx.send(Ok(anthropic_sse_event(&ev))).await;
                    let _ = tx.send(Ok(anthropic_sse_event(&translate_api::MessageEvent::MessageStop))).await;
                    break 'stream_retry;
                }
                mid_stream_retries -= 1;
                let delay = std::time::Duration::from_millis(500 * (1 << (MID_STREAM_MAX_RETRIES - mid_stream_retries)));
                let retry_n = MID_STREAM_MAX_RETRIES - mid_stream_retries;
                eprintln!("[WARN] Retrying Anthropic stream POST after HTTP {} #{retry_n} in {:?}...", status, delay);
                tokio::time::sleep(delay).await;
                continue 'stream_retry;
            }

            if mid_stream_retries < MID_STREAM_MAX_RETRIES {
                eprintln!("[INFO] Anthropic stream retry succeeded, reading started");
            }

            let mut buf = String::new();
            let mut byte_stream = cc_response.bytes_stream();
            let mut translate_state = translate_api::TranslateState::new(
                msg_id.clone(),
                original_model.clone(),
            );
            let mut tool_call_idx = 0usize;
            let mut sent_role = false;
            let mut reasoning_buf = String::new();
            let mut stream_success = false;
            let mut chunk_dumper = CcResponseDumper::new("anthropic_stream", true, &original_model);
            let chunk_timeout = std::time::Duration::from_secs(CHUNK_TIMEOUT_SECS);

            'read_stream: loop {
                let next_chunk = tokio::time::timeout(chunk_timeout, byte_stream.next()).await;
                match next_chunk {
                    Ok(Some(Ok(chunk))) => {
                        if let Some(ref mut d) = chunk_dumper {
                            d.feed_bytes(&chunk);
                        }
                        buf.push_str(&String::from_utf8_lossy(&chunk));

                        while let Some(pos) = buf.find('\n') {
                            let line = buf[..pos].trim_end_matches('\r').to_string();
                            buf = buf[pos + 1..].to_string();

                            if let Some(cc_event) = parse_cc_event_line(&line) {
                                let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
                                let oai_model = anthropic_req.model.clone();

                                match cc_event.event_type.as_str() {
                                    "text-delta" => {
                                        emitted_visible = true;
                                        if let Some(delta_text) = cc_event.text {
                                            if !delta_text.is_empty() {
                                                let delta = ChunkDelta {
                                                    role: if !sent_role {
                                                        sent_role = true;
                                                        Some("assistant".to_string())
                                                    } else {
                                                        None
                                                    },
                                                    content: Some(delta_text),
                                                    reasoning_content: None,
                                                    tool_calls: None,
                                                };
                                                let chunk = ChatCompletionChunk {
                                                    id: chunk_id,
                                                    object: "chat.completion.chunk".to_string(),
                                                    created,
                                                    model: oai_model,
                                                    choices: vec![ChunkChoice {
                                                        index: 0,
                                                        delta,
                                                        finish_reason: None,
                                                    }],
                                                    usage: None,
                                                };
                                                for ev in translate_api::openai_chunk_to_anthropic_events(&chunk, &mut translate_state) {
                                                    let _ = tx.send(Ok(anthropic_sse_event(&ev))).await;
                                                }
                                            }
                                        }
                                    }
                                    "reasoning-delta" => {
                                        emitted_visible = true;
                                        if let Some(delta_text) = cc_event.text {
                                            reasoning_buf.push_str(&delta_text);
                                            let delta = ChunkDelta {
                                                role: None,
                                                content: None,
                                                reasoning_content: Some(delta_text),
                                                tool_calls: None,
                                            };
                                            let chunk = ChatCompletionChunk {
                                                id: chunk_id,
                                                object: "chat.completion.chunk".to_string(),
                                                created,
                                                model: oai_model,
                                                choices: vec![ChunkChoice {
                                                    index: 0,
                                                    delta,
                                                    finish_reason: None,
                                                }],
                                                usage: None,
                                            };
                                            for ev in translate_api::openai_chunk_to_anthropic_events(&chunk, &mut translate_state) {
                                                let _ = tx.send(Ok(anthropic_sse_event(&ev))).await;
                                            }
                                        }
                                    }
                                    "reasoning-end" => {}
                                    "tool-call" => {
                                        emitted_visible = true;
                                        let id = cc_event.tool_call_id.unwrap_or_default();
                                        let name = cc_event.tool_name.unwrap_or_default();
                                        let args = cc_event
                                            .input
                                            .or(cc_event.args)
                                            .or(cc_event.arguments)
                                            .and_then(|v| serde_json::to_string(&v).ok())
                                            .unwrap_or_default();
                                        let delta = ChunkDelta {
                                            role: None,
                                            content: None,
                                            reasoning_content: None,
                                            tool_calls: Some(vec![ToolCall {
                                                index: Some(tool_call_idx),
                                                id: Some(id),
                                                tool_type: Some("function".to_string()),
                                                function: Some(ToolCallFunction {
                                                    name: Some(name),
                                                    arguments: Some(args),
                                                }),
                                            }]),
                                        };
                                        tool_call_idx += 1;
                                        let chunk = ChatCompletionChunk {
                                            id: chunk_id,
                                            object: "chat.completion.chunk".to_string(),
                                            created,
                                            model: oai_model,
                                            choices: vec![ChunkChoice {
                                                index: 0,
                                                delta,
                                                finish_reason: None,
                                            }],
                                            usage: None,
                                        };
                                        for ev in translate_api::openai_chunk_to_anthropic_events(&chunk, &mut translate_state) {
                                            let _ = tx.send(Ok(anthropic_sse_event(&ev))).await;
                                        }
                                    }
                                    "finish" => {
                                        let finish = map_finish_reason(cc_event.finish_reason.as_deref());
                                        let oai_usage = cc_event
                                            .total_usage
                                            .map(usage_from_cc_usage)
                                            .filter(|u| u.total_tokens > 0);
                                        let delta = ChunkDelta::default();
                                        let chunk = ChatCompletionChunk {
                                            id: chunk_id,
                                            object: "chat.completion.chunk".to_string(),
                                            created,
                                            model: oai_model,
                                            choices: vec![ChunkChoice {
                                                index: 0,
                                                delta,
                                                finish_reason: finish,
                                            }],
                                            usage: oai_usage,
                                        };
                                        for ev in translate_api::openai_chunk_to_anthropic_events(&chunk, &mut translate_state) {
                                            let _ = tx.send(Ok(anthropic_sse_event(&ev))).await;
                                        }
                                        stream_success = true;
                                        break 'read_stream;
                                    }
                                    "error" => {
                                        let _msg = cc_event
                                            .error
                                            .as_ref()
                                            .and_then(|e| e.get("message").and_then(|m| m.as_str()))
                                            .or_else(|| cc_event.error.as_ref().and_then(|e| e.as_str()))
                                            .unwrap_or("Stream error");
                                        let ev = translate_api::MessageEvent::MessageDelta {
                                            delta: translate_api::MessageDeltaInfo {
                                                stop_reason: "end_turn".to_string(),
                                                stop_sequence: None,
                                            },
                                            usage: None,
                                        };
                                        let _ = tx.send(Ok(anthropic_sse_event(&ev))).await;
                                        break 'read_stream;
                                    }
                                    _ => {}
                                }
                     
                            }
                        }
                    }
                    Ok(Some(Err(e))) => {
                        eprintln!(
                            "[ERROR] Anthropic stream read error (retries left: {}, emitted_visible: {}): {}",
                            mid_stream_retries, emitted_visible, e
                        );
                        if !emitted_visible && mid_stream_retries > 0 {
                            mid_stream_retries -= 1;
                            let delay = std::time::Duration::from_millis(500 * (1 << (MID_STREAM_MAX_RETRIES - mid_stream_retries)));
                            let retry_n = MID_STREAM_MAX_RETRIES - mid_stream_retries;
                            eprintln!("[WARN] Retrying Anthropic stream read #{retry_n} in {:?}...", delay);
                            tokio::time::sleep(delay).await;
                            continue 'stream_retry;
                        }
                        let ev = translate_api::MessageEvent::MessageDelta {
                            delta: translate_api::MessageDeltaInfo {
                                stop_reason: "end_turn".to_string(),
                                stop_sequence: None,
                            },
                            usage: None,
                        };
                        let _ = tx.send(Ok(anthropic_sse_event(&ev))).await;
                        break 'stream_retry;
                    }
                    Ok(None) => {
                        if !buf.is_empty() {
                            if let Some(event) = parse_cc_event_line(&buf) {
                                if event.event_type == "finish" {
                                    let finish = map_finish_reason(event.finish_reason.as_deref());
                                    let oai_usage = event
                                        .total_usage
                                        .map(usage_from_cc_usage)
                                        .filter(|u| u.total_tokens > 0);
                                    let chunk = ChatCompletionChunk {
                                        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                                        object: "chat.completion.chunk".to_string(),
                                        created,
                                        model: anthropic_req.model.clone(),
                                        choices: vec![ChunkChoice {
                                            index: 0,
                                            delta: ChunkDelta::default(),
                                            finish_reason: finish,
                                        }],
                                        usage: oai_usage,
                                    };
                                    for ev in translate_api::openai_chunk_to_anthropic_events(&chunk, &mut translate_state) {
                                        let _ = tx.send(Ok(anthropic_sse_event(&ev))).await;
                                    }
                                    stream_success = true;
                                }
                            }
                        }
                        break 'read_stream;
                    }
                    Err(_elapsed) => {
                        eprintln!(
                            "[ERROR] Anthropic stream chunk timeout after {}s (retries left: {}, emitted_visible: {}, buf_remaining: {})",
                            CHUNK_TIMEOUT_SECS, mid_stream_retries, emitted_visible, buf.len()
                        );
                        if !buf.is_empty() {
                            eprintln!("[WARN] Anthropic stream partial buffer on timeout: {}", buf.chars().take(200).collect::<String>());
                        }
                        if !emitted_visible && mid_stream_retries > 0 {
                            mid_stream_retries -= 1;
                            let delay = std::time::Duration::from_millis(500 * (1 << (MID_STREAM_MAX_RETRIES - mid_stream_retries)));
                            let retry_n = MID_STREAM_MAX_RETRIES - mid_stream_retries;
                            eprintln!("[WARN] Retrying Anthropic stream after chunk timeout #{retry_n} in {:?}...", delay);
                            tokio::time::sleep(delay).await;
                            continue 'stream_retry;
                        }
                        let ev = translate_api::MessageEvent::MessageDelta {
                            delta: translate_api::MessageDeltaInfo {
                                stop_reason: "end_turn".to_string(),
                                stop_sequence: None,
                            },
                            usage: None,
                        };
                        let _ = tx.send(Ok(anthropic_sse_event(&ev))).await;
                        break 'stream_retry;
                    }
                }
            }

            if stream_success {
                // Close any open content blocks
                if let Some(idx) = translate_state.text_block_idx.take() {
                    let ev = translate_api::MessageEvent::ContentBlockStop { index: idx };
                    let _ = tx.send(Ok(anthropic_sse_event(&ev))).await;
                }
                if let Some(idx) = translate_state.thinking_block_idx.take() {
                    let ev = translate_api::MessageEvent::ContentBlockStop { index: idx };
                    let _ = tx.send(Ok(anthropic_sse_event(&ev))).await;
                }
                for block_idx in translate_state.tool_block_indices.values() {
                    let ev = translate_api::MessageEvent::ContentBlockStop { index: *block_idx };
                    let _ = tx.send(Ok(anthropic_sse_event(&ev))).await;
                }

                if !translate_state.stop_sent {
                    let stop_reason = if !translate_state.tool_block_indices.is_empty() {
                        "tool_use"
                    } else {
                        "end_turn"
                    };
                    let ev = translate_api::MessageEvent::MessageDelta {
                        delta: translate_api::MessageDeltaInfo {
                            stop_reason: stop_reason.to_string(),
                            stop_sequence: None,
                        },
                        usage: None,
                    };
                    let _ = tx.send(Ok(anthropic_sse_event(&ev))).await;
                }

                let _ = tx.send(Ok(anthropic_sse_event(&translate_api::MessageEvent::MessageStop))).await;
                break 'stream_retry;
            }

            if !emitted_visible && mid_stream_retries > 0 {
                mid_stream_retries -= 1;
                let delay = std::time::Duration::from_millis(500 * (1 << (MID_STREAM_MAX_RETRIES - mid_stream_retries)));
                let retry_n = MID_STREAM_MAX_RETRIES - mid_stream_retries;
                eprintln!("[WARN] Anthropic stream ended without finish, retrying #{retry_n} in {:?}...", delay);
                tokio::time::sleep(delay).await;
                continue 'stream_retry;
            }

            let ev = translate_api::MessageEvent::MessageDelta {
                delta: translate_api::MessageDeltaInfo {
                    stop_reason: "end_turn".to_string(),
                    stop_sequence: None,
                },
                usage: None,
            };
            let _ = tx.send(Ok(anthropic_sse_event(&ev))).await;
            let _ = tx.send(Ok(anthropic_sse_event(&translate_api::MessageEvent::MessageStop))).await;
            break 'stream_retry;
        }
    });

    let stream = ReceiverStream::new(rx);
    Ok(Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::new().text("keep-alive")))
}
