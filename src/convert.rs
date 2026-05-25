use crate::types::*;
use serde_json::json;

/// Convert OpenAI chat messages to CommandCode message format.
pub fn messages_to_cc(messages: &[ChatMessage]) -> Vec<serde_json::Value> {
    let mut out = Vec::new();

    // Collect paired tool call ids (assistant toolCall + tool result)
    let mut call_ids = std::collections::HashSet::new();
    let mut result_ids = std::collections::HashSet::new();
    for msg in messages {
        if msg.role == "assistant" {
            if let Some(calls) = &msg.tool_calls {
                for call in calls {
                    if let Some(id) = &call.id {
                        call_ids.insert(id.clone());
                    }
                }
            }
            // Also handle content-based tool calls
            if let Some(content) = &msg.content {
                if let Some(arr) = content.as_array() {
                    for item in arr {
                        if let Some(obj) = item.as_object() {
                            if obj.get("type").and_then(|t| t.as_str()) == Some("toolCall") {
                                if let Some(id) = obj.get("id").and_then(|v| v.as_str()) {
                                    call_ids.insert(id.to_string());
                                }
                            }
                        }
                    }
                }
            }
        } else if msg.role == "tool" {
            if let Some(id) = &msg.tool_call_id {
                result_ids.insert(id.clone());
            }
        }
    }
    let paired: std::collections::HashSet<String> =
        call_ids.intersection(&result_ids).cloned().collect();

    for msg in messages {
        match msg.role.as_str() {
            "user" => {
                let content = match &msg.content {
                    Some(v) if v.is_string() => v.as_str().unwrap_or("").to_string(),
                    Some(v) => v.to_string(),
                    None => String::new(),
                };
                out.push(json!({ "role": "user", "content": content }));
            }
            "assistant" => {
                let mut parts = Vec::new();

                // Content may be string or array of parts
                if let Some(content) = &msg.content {
                    if let Some(text) = content.as_str() {
                        if !text.is_empty() {
                            parts.push(json!({ "type": "text", "text": text }));
                        }
                    } else if let Some(arr) = content.as_array() {
                        for item in arr {
                            if let Some(obj) = item.as_object() {
                                match obj.get("type").and_then(|t| t.as_str()) {
                                    Some("text") => {
                                        let text =
                                            obj.get("text").and_then(|t| t.as_str()).unwrap_or("");
                                        parts.push(json!({ "type": "text", "text": text }));
                                    }
                                    Some("thinking") => {
                                        let text = obj
                                            .get("thinking")
                                            .and_then(|t| t.as_str())
                                            .unwrap_or("");
                                        parts.push(json!({ "type": "reasoning", "text": text }));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                // Tool calls from OpenAI format
                if let Some(calls) = &msg.tool_calls {
                    for call in calls {
                        if let Some(id) = &call.id {
                            if !paired.contains(id) {
                                continue;
                            }
                            let name = call
                                .function
                                .as_ref()
                                .and_then(|f| f.name.clone())
                                .unwrap_or_default();
                            let args = call
                                .function
                                .as_ref()
                                .and_then(|f| f.arguments.clone())
                                .unwrap_or_default();
                            let input = if let Ok(parsed) =
                                serde_json::from_str::<serde_json::Value>(&args)
                            {
                                parsed
                            } else {
                                serde_json::Value::Object(Default::default())
                            };
                            parts.push(json!({
                                "type": "tool-call",
                                "toolCallId": id,
                                "toolName": name,
                                "input": input,
                            }));
                        }
                    }
                }

                if !parts.is_empty() {
                    out.push(json!({ "role": "assistant", "content": parts }));
                }
            }
            "tool" => {
                if let Some(id) = &msg.tool_call_id {
                    if !paired.contains(id) {
                        continue;
                    }
                    let tool_name = msg.name.clone().unwrap_or_default();
                    let value = msg.content.as_ref().and_then(|c| c.as_str()).unwrap_or("");
                    let is_error = msg
                        .extra
                        .get("isError")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let output = if is_error {
                        json!({ "type": "error-text", "value": value })
                    } else {
                        json!({ "type": "text", "value": value })
                    };
                    out.push(json!({
                        "role": "tool",
                        "content": [{
                            "type": "tool-result",
                            "toolCallId": id,
                            "toolName": tool_name,
                            "output": output,
                        }]
                    }));
                }
            }
            "system" => {
                // System messages go into params.system, skip here
            }
            _ => {}
        }
    }

    out
}

/// Convert OpenAI tools to CommandCode tool format.
/// CommandCode expects standard JSON Schema for input_schema, so we pass it through.
pub fn tools_to_cc(tools: Option<&[ToolDefinition]>) -> Vec<serde_json::Value> {
    let Some(tools) = tools else { return vec![] };
    tools
        .iter()
        .map(|t| {
            let schema = t.function.parameters.clone().unwrap_or_else(|| json!({}));
            json!({
                "type": "function",
                "name": t.function.name,
                "description": t.function.description,
                "input_schema": schema,
            })
        })
        .collect()
}

/// Extract system prompt from messages.
pub fn extract_system(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .filter(|m| m.role == "system")
        .filter_map(|m| m.content.as_ref().and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Map CommandCode finish reason to OpenAI finish_reason.
pub fn map_finish_reason(reason: Option<&str>) -> Option<String> {
    match reason {
        Some("tool-calls") => Some("tool_calls".to_string()),
        Some("length" | "max_tokens" | "max-tokens" | "max_output_tokens") => {
            Some("length".to_string())
        }
        Some("stop") | Some(_) => Some("stop".to_string()),
        None => None,
    }
}

/// Parse a SSE line from CommandCode into an event.
pub fn parse_cc_event_line(line: &str) -> Option<CcStreamEvent> {
    let mut trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with(':') {
        return None;
    }
    if trimmed.starts_with("event:") {
        return None;
    }
    if trimmed.starts_with("data:") {
        trimmed = trimmed[5..].trim();
    }
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return None;
    }
    serde_json::from_str(trimmed).ok()
}
