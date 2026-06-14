use super::*;
use serde::Serialize;
use std::collections::HashMap;

// ============================================================================
// Anthropic response types
// ============================================================================

#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

// ============================================================================
// Anthropic streaming event types
// ============================================================================

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum MessageEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartInfo },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: StreamDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaInfo,
        usage: Option<AnthropicUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
}

impl MessageEvent {
    pub fn event_type_str(&self) -> &'static str {
        match self {
            MessageEvent::MessageStart { .. } => "message_start",
            MessageEvent::ContentBlockStart { .. } => "content_block_start",
            MessageEvent::ContentBlockDelta { .. } => "content_block_delta",
            MessageEvent::ContentBlockStop { .. } => "content_block_stop",
            MessageEvent::MessageDelta { .. } => "message_delta",
            MessageEvent::MessageStop => "message_stop",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct MessageStartInfo {
    pub id: String,
    #[serde(rename = "type")]
    pub t: String,
    pub role: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum StreamDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Serialize)]
pub struct MessageDeltaInfo {
    pub stop_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

// ============================================================================
// Streaming state machine
// ============================================================================

pub struct TranslateState {
    pub message_id: String,
    pub model: String,
    pub text_block_idx: Option<usize>,
    pub thinking_block_idx: Option<usize>,
    pub tool_block_indices: HashMap<usize, usize>,
    pub started: bool,
    pub stop_sent: bool,
    next_index: usize,
}

impl TranslateState {
    pub fn new(message_id: String, model: String) -> Self {
        Self {
            message_id,
            model,
            text_block_idx: None,
            thinking_block_idx: None,
            tool_block_indices: HashMap::new(),
            started: false,
            stop_sent: false,
            next_index: 0,
        }
    }

    fn assign_blk_idx(&mut self) -> usize {
        self.next_index += 1;
        self.next_index
    }
}

// ============================================================================
// Conversion: OpenAI ChatCompletionChunk → Vec<Anthropic MessageEvent>
// ============================================================================

pub fn openai_chunk_to_anthropic_events(
    chunk: &ChatCompletionChunk,
    state: &mut TranslateState,
) -> Vec<MessageEvent> {
    let mut events = Vec::new();

    if !state.started {
        events.push(MessageEvent::MessageStart {
            message: MessageStartInfo {
                id: state.message_id.clone(),
                t: "message".to_string(),
                role: "assistant".to_string(),
                model: state.model.clone(),
                content: vec![],
            },
        });
        state.started = true;
    }

    if let Some(choice) = chunk.choices.first() {
        // ── Reasoning content delta ──
        if let Some(ref reasoning) = choice.delta.reasoning_content {
            if !reasoning.is_empty() {
                if let Some(idx) = state.text_block_idx.take() {
                    events.push(MessageEvent::ContentBlockStop { index: idx });
                }
                if state.thinking_block_idx.is_none() {
                    let idx = state.assign_blk_idx();
                    state.thinking_block_idx = Some(idx);
                    events.push(MessageEvent::ContentBlockStart {
                        index: idx,
                        content_block: ContentBlock::Thinking {
                            thinking: String::new(),
                            signature: None,
                        },
                    });
                }
                if let Some(idx) = state.thinking_block_idx {
                    events.push(MessageEvent::ContentBlockDelta {
                        index: idx,
                        delta: StreamDelta::ThinkingDelta {
                            thinking: reasoning.clone(),
                        },
                    });
                }
            }
        }

        // ── Text content delta ──
        if let Some(ref content) = choice.delta.content {
            if !content.is_empty() {
                if let Some(idx) = state.thinking_block_idx.take() {
                    events.push(MessageEvent::ContentBlockStop { index: idx });
                }
                if state.text_block_idx.is_none() {
                    let idx = state.assign_blk_idx();
                    state.text_block_idx = Some(idx);
                    events.push(MessageEvent::ContentBlockStart {
                        index: idx,
                        content_block: ContentBlock::Text {
                            text: String::new(),
                        },
                    });
                }
                if let Some(idx) = state.text_block_idx {
                    events.push(MessageEvent::ContentBlockDelta {
                        index: idx,
                        delta: StreamDelta::TextDelta {
                            text: content.clone(),
                        },
                    });
                }
            }
        }

        // ── Tool call deltas ──
        if let Some(ref calls) = choice.delta.tool_calls {
            for tc in calls {
                let oai_idx = tc.index.unwrap_or(0);
                let name = tc
                    .function
                    .as_ref()
                    .and_then(|f| f.name.clone())
                    .unwrap_or_default();

                if !state.tool_block_indices.contains_key(&oai_idx) {
                    if name.is_empty() && tc.id.is_none() {
                        continue;
                    }
                    if let Some(idx) = state.text_block_idx.take() {
                        events.push(MessageEvent::ContentBlockStop { index: idx });
                    }
                    if let Some(idx) = state.thinking_block_idx.take() {
                        events.push(MessageEvent::ContentBlockStop { index: idx });
                    }
                    let idx = state.assign_blk_idx();
                    state.tool_block_indices.insert(oai_idx, idx);
                    events.push(MessageEvent::ContentBlockStart {
                        index: idx,
                        content_block: ContentBlock::ToolUse {
                            id: tc
                                .id
                                .clone()
                                .unwrap_or_else(|| format!("toolu_{}", uuid::Uuid::new_v4())),
                            name,
                            input: serde_json::Value::Object(Default::default()),
                            thinking: None,
                        },
                    });
                }

                let args = tc
                    .function
                    .as_ref()
                    .and_then(|f| f.arguments.clone())
                    .unwrap_or_default();
                if !args.is_empty() {
                    if let Some(&idx) = state.tool_block_indices.get(&oai_idx) {
                        events.push(MessageEvent::ContentBlockDelta {
                            index: idx,
                            delta: StreamDelta::InputJsonDelta {
                                partial_json: args,
                            },
                        });
                    }
                }
            }
        }

        // ── Finish reason ──
        if let Some(ref reason) = choice.finish_reason {
            if !reason.is_empty() && !state.stop_sent {
                let mut closings: Vec<(usize, MessageEvent)> = Vec::new();

                if let Some(idx) = state.text_block_idx.take() {
                    closings.push((idx, MessageEvent::ContentBlockStop { index: idx }));
                }
                if let Some(idx) = state.thinking_block_idx.take() {
                    closings.push((idx, MessageEvent::ContentBlockStop { index: idx }));
                }
                for block_idx in state.tool_block_indices.values() {
                    closings.push((
                        *block_idx,
                        MessageEvent::ContentBlockStop { index: *block_idx },
                    ));
                }
                state.tool_block_indices.clear();

                closings.sort_by_key(|(idx, _)| *idx);
                for (_, ev) in closings {
                    events.push(ev);
                }

                let stop_reason = map_anthropic_stop_reason(Some(reason))
                    .unwrap_or_else(|| "end_turn".to_string());
                let usage = chunk.usage.as_ref().map(map_anthropic_usage);
                events.push(MessageEvent::MessageDelta {
                    delta: MessageDeltaInfo {
                        stop_reason,
                        stop_sequence: None,
                    },
                    usage,
                });
                state.stop_sent = true;
            }
        }
    } else if let Some(usage) = &chunk.usage {
        if !state.stop_sent {
            events.push(MessageEvent::MessageDelta {
                delta: MessageDeltaInfo {
                    stop_reason: "end_turn".to_string(),
                    stop_sequence: None,
                },
                usage: Some(map_anthropic_usage(usage)),
            });
            state.stop_sent = true;
        }
    }

    events
}

// ============================================================================
// Conversion: OpenAI ChatCompletionResponse → Anthropic MessageResponse
// ============================================================================

pub fn openai_response_to_anthropic(
    resp: &ChatCompletionResponse,
    model: &str,
) -> MessageResponse {
    let mut content_blocks: Vec<ContentBlock> = Vec::new();

    if let Some(choice) = resp.choices.first() {
        // reasoning → thinking block (first, per Anthropic convention)
        if let Some(ref reasoning) = choice.message.reasoning_content {
            if !reasoning.is_empty() {
                content_blocks.push(ContentBlock::Thinking {
                    thinking: reasoning.clone(),
                    signature: None,
                });
            }
        }

        // tool_calls → tool_use blocks
        if let Some(ref calls) = choice.message.tool_calls {
            for tc in calls {
                let id = tc.id.clone().unwrap_or_default();
                let name = tc
                    .function
                    .as_ref()
                    .and_then(|f| f.name.clone())
                    .unwrap_or_default();
                let input: serde_json::Value = tc
                    .function
                    .as_ref()
                    .and_then(|f| f.arguments.clone())
                    .and_then(|args| serde_json::from_str(&args).ok())
                    .unwrap_or_else(|| serde_json::Value::Object(Default::default()));

                content_blocks.push(ContentBlock::ToolUse {
                    id,
                    name,
                    input,
                    thinking: None,
                });
            }
        }

        // text content
        if let Some(ref text) = choice.message.content {
            if !text.is_empty() {
                content_blocks.push(ContentBlock::Text {
                    text: text.clone(),
                });
            }
        }
    }

    if content_blocks.is_empty() {
        content_blocks.push(ContentBlock::Text {
            text: String::new(),
        });
    }

    let stop_reason = resp
        .choices
        .first()
        .and_then(|c| c.finish_reason.as_deref())
        .and_then(|r| map_anthropic_stop_reason(Some(r)));

    let usage = resp
        .usage
        .as_ref()
        .map(map_anthropic_usage)
        .unwrap_or(AnthropicUsage {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        });

    MessageResponse {
        id: resp.id.clone(),
        msg_type: "message".to_string(),
        role: "assistant".to_string(),
        content: content_blocks,
        model: model.to_string(),
        stop_reason,
        stop_sequence: None,
        usage,
    }
}
