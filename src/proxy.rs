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
const MAX_RETRIES: u32 = 3;

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

    // Dump full JSON to data/ for debugging
    let dump_dir = "data";
    let _ = std::fs::create_dir_all(dump_dir);
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S_%3f").to_string();
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

/// Non-streaming completion handler.
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

    let cc_req = build_cc_request(&req, &working_dir, &environment);
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
        return Ok((
            StatusCode::BAD_GATEWAY,
            format!("CommandCode API error {}: {}", status, text),
        )
            .into_response());
    }

    // For non-streaming, we still receive an SSE stream from CC and buffer it.
    let bytes = cc_response
        .bytes()
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?;
    let text = String::from_utf8_lossy(&bytes);

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
                "reasoning-end" => {
                    // reasoning already accumulated
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
                    let msg = event
                        .error
                        .as_ref()
                        .and_then(|e| e.get("message").and_then(|m| m.as_str()))
                        .or_else(|| event.error.as_ref().and_then(|e| e.as_str()))
                        .unwrap_or("Stream error");
                    return Ok((
                        StatusCode::BAD_GATEWAY,
                        format!("CommandCode stream error: {}", msg),
                    )
                        .into_response());
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

    // If no content and no tool_calls, return empty content to be valid
    if message.content.is_none() && message.tool_calls.is_none() {
        message.content = Some(String::new());
    }

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        model: req.model.clone(),
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

    Ok(axum::Json(response).into_response())
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
    let session_id = uuid::Uuid::new_v4().to_string();
    let model_id = req.model.clone();

    let (tx, rx) =
        mpsc::channel::<Result<axum::response::sse::Event, std::convert::Infallible>>(128);
    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    tokio::spawn(async move {
        let mut mid_stream_retries = MAX_RETRIES;
        let mut emitted_visible = false;

        'stream_retry: loop {
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
                        .header("x-session-id", &session_id)
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
                    let delay = std::time::Duration::from_millis(500 * (1 << (MAX_RETRIES - mid_stream_retries)));
                    eprintln!("[WARN] OpenAI stream POST failed, retrying in {:?}...", delay);
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
                let delay = std::time::Duration::from_millis(500 * (1 << (MAX_RETRIES - mid_stream_retries)));
                eprintln!("[WARN] Retrying OpenAI stream POST after HTTP {} in {:?}...", status, delay);
                tokio::time::sleep(delay).await;
                continue 'stream_retry;
            }

            let mut buf = String::new();
            let mut byte_stream = cc_response.bytes_stream();
            let mut tool_call_idx = 0usize;
            let mut sent_role = false;
            let mut reasoning_buf = String::new();
            let mut stream_success = false;
            'read_stream: loop {
                match byte_stream.next().await {
                    Some(Ok(chunk)) => {
                        buf.push_str(&String::from_utf8_lossy(&chunk));

                        while let Some(pos) = buf.find('\n') {
                            let line = buf[..pos].to_string();
                            buf = buf[pos + 1..].to_string();

                            if let Some(event) = parse_cc_event_line(&line) {
                                match event.event_type.as_str() {
                                    "text-delta" => {
                                        emitted_visible = true;
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
                    Some(Err(e)) => {
                        eprintln!(
                            "[ERROR] OpenAI stream read error (retries left: {}, emitted_visible: {}): {}",
                            mid_stream_retries, emitted_visible, e
                        );
                        if !emitted_visible && mid_stream_retries > 0 {
                            mid_stream_retries -= 1;
                            let delay = std::time::Duration::from_millis(500 * (1 << (MAX_RETRIES - mid_stream_retries)));
                            eprintln!("[WARN] Retrying OpenAI stream in {:?}...", delay);
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
                    None => {
                        if !buf.is_empty() {
                            if let Some(event) = parse_cc_event_line(&buf) {
                                if event.event_type == "finish" {
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
                }

            }

            if stream_success {
                break 'stream_retry;
            }

            if mid_stream_retries > 0 {
                mid_stream_retries -= 1;
                let delay = std::time::Duration::from_millis(500 * (1 << (MAX_RETRIES - mid_stream_retries)));
                eprintln!("[WARN] OpenAI stream ended without finish, retrying in {:?}...", delay);
                tokio::time::sleep(delay).await;
                continue 'stream_retry;
            }

            let _ = tx
                .send(Ok(error_event(
                    &completion_id, created, &model_id,
                    "Stream ended unexpectedly after all retries",
                )))
                .await;
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
    let session_id = uuid::Uuid::new_v4().to_string();

    let (tx, rx) =
        mpsc::channel::<Result<axum::response::sse::Event, std::convert::Infallible>>(128);
    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let state = state.clone();

    tokio::spawn(async move {
        let mut mid_stream_retries = MAX_RETRIES;
        let mut emitted_visible = false;

        'stream_retry: loop {
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
                        .header("x-session-id", &session_id)
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
                    let delay = std::time::Duration::from_millis(500 * (1 << (MAX_RETRIES - mid_stream_retries)));
                    eprintln!("[WARN] Retrying Anthropic stream POST in {:?}...", delay);
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
                let delay = std::time::Duration::from_millis(500 * (1 << (MAX_RETRIES - mid_stream_retries)));
                eprintln!("[WARN] Retrying Anthropic stream POST after HTTP {} in {:?}...", status, delay);
                tokio::time::sleep(delay).await;
                continue 'stream_retry;
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

            'read_stream: loop {
                match byte_stream.next().await {
                    Some(Ok(chunk)) => {
                        buf.push_str(&String::from_utf8_lossy(&chunk));

                        while let Some(pos) = buf.find('\n') {
                            let line = buf[..pos].to_string();
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
                    Some(Err(e)) => {
                        eprintln!(
                            "[ERROR] Anthropic stream read error (retries left: {}, emitted_visible: {}): {}",
                            mid_stream_retries, emitted_visible, e
                        );
                        if !emitted_visible && mid_stream_retries > 0 {
                            mid_stream_retries -= 1;
                            let delay = std::time::Duration::from_millis(500 * (1 << (MAX_RETRIES - mid_stream_retries)));
                            eprintln!("[WARN] Retrying Anthropic stream in {:?}...", delay);
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
                    None => {
                        break 'read_stream;
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
                let delay = std::time::Duration::from_millis(500 * (1 << (MAX_RETRIES - mid_stream_retries)));
                eprintln!("[WARN] Anthropic stream ended without finish, retrying in {:?}...", delay);
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
