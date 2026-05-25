use crate::convert::*;
use crate::types::*;
use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse},
    Json,
};
use futures::StreamExt;
use reqwest::Client;
use serde_json::json;
use std::{env, sync::Arc, time::SystemTime};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

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
fn build_cc_request(
    req: &ChatCompletionRequest,
    working_dir: &str,
    environment: &str,
) -> CcRequest {
    let system = req
        .system
        .as_ref()
        .map(|s| s.clone())
        .unwrap_or_else(|| extract_system(&req.messages));
    let max_tokens = req.max_tokens.unwrap_or(8192).min(200_000);
    // CommandCode API always requires stream=true; we buffer for non-streaming clients
    let _client_stream = req.stream.unwrap_or(false);
    let stream = true;

    CcRequest {
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
            messages: messages_to_cc(&req.messages),
            tools: tools_to_cc(req.tools.as_deref()),
            system,
            max_tokens,
            stream,
        },
    }
}

/// Helper to extract Authorization header.
fn get_api_key(headers: &HeaderMap, default: &Option<String>) -> Option<String> {
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
        total_tokens: 0,
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

    let cc_response = state
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
        .body(body)
        .send()
        .await
        .map_err(|e| {
            tracing::error!("CC upstream request failed: {}", e);
            StatusCode::BAD_GATEWAY
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
        tracing::error!("CC API error {}: {}", status, text);
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

    let cc_response = state
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
        .body(body)
        .send()
        .await
        .map_err(|e| {
            tracing::error!("CC upstream request failed: {}", e);
            StatusCode::BAD_GATEWAY
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
        tracing::error!("CC API error {}: {}", status, text);
        return Err(StatusCode::BAD_GATEWAY);
    }

    let (tx, rx) =
        mpsc::channel::<Result<axum::response::sse::Event, std::convert::Infallible>>(128);
    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // Manual SSE parsing from the upstream stream.
    tokio::spawn(async move {
        let mut buf = String::new();
        let mut stream = cc_response.bytes_stream();
        let mut tool_call_idx = 0usize;
        let mut sent_role = false;
        let mut finished = false;
        let mut reasoning_buf = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx
                        .send(Ok(error_event(
                            &completion_id,
                            created,
                            &model_id,
                            &format!("Stream read error: {}", e),
                        )))
                        .await;
                    break;
                }
            };

            buf.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].to_string();
                buf = buf[pos + 1..].to_string();

                if let Some(event) = parse_cc_event_line(&line) {
                    match event.event_type.as_str() {
                        "text-delta" => {
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
                        "reasoning-end" => {
                            // delta already sent, nothing more
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
                            finished = true;
                            // final chunk with finish_reason and usage
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
                            finished = true;
                        }
                        _ => {}
                    }
                    if finished {
                        break;
                    }
                }
            }

            if finished {
                break;
            }
        }

        // Handle remaining buffer
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
                }
            }
        }

        // OpenAI-compatible end marker
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
