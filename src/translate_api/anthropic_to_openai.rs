use super::*;
use serde::Deserialize;
use std::collections::HashMap;

// ============================================================================
// Anthropic request types
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct MessageRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    pub system: Option<SystemPrompt>,
    pub stream: Option<bool>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub tools: Option<Vec<AnthropicTool>>,
    pub tool_choice: Option<AnthropicToolChoice>,
    pub metadata: Option<AnthropicMetadata>,
    pub thinking: Option<ThinkingConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: String,
    #[serde(default)]
    pub budget_tokens: Option<u32>,
}

impl Default for MessageRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            messages: Vec::new(),
            max_tokens: 1024,
            system: None,
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            thinking: None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    String(String),
    Blocks(Vec<SystemTextBlock>),
}

#[derive(Debug, Deserialize)]
pub struct SystemTextBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
    #[serde(default)]
    pub cache_control: Option<CacheControl>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: String,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMetadata {
    pub user_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: serde_json::Value,
}

impl AnthropicMessage {
    pub fn content_blocks(&self) -> Vec<ContentBlock> {
        if let Some(s) = self.content.as_str() {
            return vec![ContentBlock::Text {
                text: s.to_string(),
            }];
        }
        serde_json::from_value(self.content.clone()).unwrap_or_default()
    }
}

#[derive(Debug, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicToolChoice {
    #[serde(rename = "type")]
    pub choice_type: String,
    #[serde(default)]
    pub name: Option<String>,
}

// ============================================================================
// Conversion: Anthropic MessageRequest → OpenAI ChatCompletionRequest
// ============================================================================

fn tool_result_text(content: &serde_json::Value) -> String {
    if let Some(s) = content.as_str() {
        return s.to_string();
    }
    if let Some(arr) = content.as_array() {
        return arr
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("");
    }
    content.to_string()
}

pub fn anthropic_request_to_openai(req: &MessageRequest) -> ChatCompletionRequest {
    let mut messages: Vec<ChatMessage> = Vec::new();

    // ── System ──
    let system: Option<String> = match &req.system {
        Some(SystemPrompt::String(s)) => {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(serde_json::Value::String(s.clone())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                extra: HashMap::new(),
            });
            Some(s.clone())
        }
        Some(SystemPrompt::Blocks(blocks)) => {
            let text: String = blocks
                .iter()
                .filter(|b| b.block_type == "text")
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join("");
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(serde_json::Value::String(text.clone())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                extra: HashMap::new(),
            });
            Some(text)
        }
        None => None,
    };

    // ── Messages ──
    for am in &req.messages {
        let blocks = am.content_blocks();
        match am.role.as_str() {
            "user" => {
                let mut tool_results: Vec<ChatMessage> = Vec::new();
                let mut text_parts: Vec<String> = Vec::new();
                let mut has_image = false;

                for block in &blocks {
                    match block {
                        ContentBlock::Text { text } => text_parts.push(text.clone()),
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            let text = tool_result_text(content);
                            let mut extra = HashMap::new();
                            if *is_error == Some(true) {
                                extra.insert("isError".to_string(), serde_json::Value::Bool(true));
                            }
                            tool_results.push(ChatMessage {
                                role: "tool".to_string(),
                                content: Some(serde_json::Value::String(text)),
                                name: None,
                                tool_calls: None,
                                tool_call_id: Some(tool_use_id.clone()),
                                extra,
                            });
                        }
                        ContentBlock::Image { .. } => has_image = true,
                        _ => {}
                    }
                }

                messages.extend(tool_results);

                let mut user_text = text_parts.join("");
                if has_image {
                    if !user_text.is_empty() {
                        user_text.push_str("\n\n");
                    }
                    user_text.push_str("[Image]");
                }
                if !user_text.is_empty() || has_image {
                    messages.push(ChatMessage {
                        role: "user".to_string(),
                        content: Some(serde_json::Value::String(user_text)),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                        extra: HashMap::new(),
                    });
                }
            }
            "assistant" => {
                let mut text_parts: Vec<String> = Vec::new();
                let mut thinking_parts: Vec<String> = Vec::new();
                let mut tool_calls: Vec<ToolCall> = Vec::new();

                for block in &blocks {
                    match block {
                        ContentBlock::Text { text } => text_parts.push(text.clone()),
                        ContentBlock::Thinking { thinking, .. } => {
                            thinking_parts.push(thinking.clone())
                        }
                        ContentBlock::ToolUse {
                            id,
                            name,
                            input,
                            thinking,
                        } => {
                            if let Some(t) = thinking {
                                thinking_parts.push(t.clone());
                            }
                            let arguments =
                                serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
                            tool_calls.push(ToolCall {
                                index: None,
                                id: Some(id.clone()),
                                tool_type: Some("function".to_string()),
                                function: Some(ToolCallFunction {
                                    name: Some(name.clone()),
                                    arguments: Some(arguments),
                                }),
                            });
                        }
                        _ => {}
                    }
                }

                let mut content_arr: Vec<serde_json::Value> = Vec::new();
                let text_all = text_parts.join("");
                if !text_all.is_empty() {
                    content_arr.push(serde_json::json!({"type": "text", "text": text_all}));
                }
                for t in &thinking_parts {
                    if !t.is_empty() {
                        content_arr
                            .push(serde_json::json!({"type": "thinking", "thinking": t}));
                    }
                }

                let content = if content_arr.is_empty() {
                    None
                } else {
                    Some(serde_json::Value::Array(content_arr))
                };
                let tc = if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                };

                messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content,
                    name: None,
                    tool_calls: tc,
                    tool_call_id: None,
                    extra: HashMap::new(),
                });
            }
            _ => {}
        }
    }

    // ── Tools ──
    let tools: Option<Vec<ToolDefinition>> = req.tools.as_ref().map(|ants| {
        ants.iter()
            .map(|t| ToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: Some(t.input_schema.clone()),
                },
            })
            .collect()
    });

    // ── Tool choice ──
    let tool_choice: Option<ToolChoice> = req.tool_choice.as_ref().and_then(|tc| match tc.choice_type.as_str() {
        "auto" => Some(ToolChoice::String("auto".to_string())),
        "any" | "required" => Some(ToolChoice::String("required".to_string())),
        "tool" => tc.name.as_ref().map(|name| ToolChoice::Object {
            tool_type: "function".to_string(),
            function: ToolChoiceFunction {
                name: name.clone(),
            },
        }),
        _ => None,
    });

    // ── Extra fields ──
    let mut extra = HashMap::new();
    if let Some(ref stops) = req.stop_sequences {
        extra.insert(
            "stop".to_string(),
            serde_json::Value::Array(stops.iter().map(|s| serde_json::Value::String(s.clone())).collect()),
        );
    }

    // ── Thinking config → reasoning_effort ──
    let reasoning_effort = req.thinking.as_ref().and_then(|tc| {
        match tc.thinking_type.as_str() {
            "disabled" => None,
            "enabled" | "auto" => {
                let budget = tc.budget_tokens.unwrap_or(16384);
                if budget >= 32768 {
                    Some("max".to_string())
                } else if budget >= 16384 {
                    Some("xhigh".to_string())
                } else if budget >= 4096 {
                    Some("high".to_string())
                } else if budget >= 1024 {
                    Some("medium".to_string())
                } else {
                    Some("low".to_string())
                }
            }
            _ => None,
        }
    });

    ChatCompletionRequest {
        model: req.model.clone(),
        messages,
        tools,
        tool_choice,
        max_tokens: Some(req.max_tokens as u64),
        temperature: req.temperature,
        top_p: req.top_p,
        stream: req.stream,
        user: req.metadata.as_ref().and_then(|m| m.user_id.clone()),
        system,
        reasoning_effort,
        thinking: None,
        extra,
    }
}
