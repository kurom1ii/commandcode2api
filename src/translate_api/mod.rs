use crate::types::*;
use serde::{Deserialize, Serialize};

// Sub-modules
pub mod anthropic_to_openai;
pub mod openai_to_anthropic;

// ============================================================================
// Shared Anthropic types (used by both conversion directions)
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(default)]
        thinking: Option<String>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: serde_json::Value,
        #[serde(default)]
        is_error: Option<bool>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Map OpenAI finish_reason → Anthropic stop_reason.
pub fn map_anthropic_stop_reason(reason: Option<&str>) -> Option<String> {
    match reason {
        Some("stop") => Some("end_turn".to_string()),
        Some("length") => Some("max_tokens".to_string()),
        Some("tool_calls") | Some("tool_use") => Some("tool_use".to_string()),
        Some("content_filter") => Some("end_turn".to_string()),
        Some(other) => {
            tracing::debug!("Unknown finish reason, mapping to end_turn: {}", other);
            Some("end_turn".to_string())
        }
        None => None,
    }
}

/// Map OpenAI Usage → Anthropic Usage.
/// Anthropic's `input_tokens` excludes cache-read and cache-write tokens
/// (billed at 10%). OpenAI's `prompt_tokens` is the total including cache.
pub fn map_anthropic_usage(usage: &Usage) -> AnthropicUsage {
    let cache_hit = usage.prompt_cache_hit_tokens.unwrap_or(0);
    let cache_miss = usage.prompt_cache_miss_tokens.unwrap_or(0);
    let input_tokens = usage
        .prompt_tokens
        .saturating_sub(cache_hit + cache_miss);

    AnthropicUsage {
        input_tokens: input_tokens as u32,
        output_tokens: usage.completion_tokens as u32,
        cache_creation_input_tokens: if cache_miss > 0 {
            Some(cache_miss as u32)
        } else {
            None
        },
        cache_read_input_tokens: if cache_hit > 0 {
            Some(cache_hit as u32)
        } else {
            None
        },
    }
}

// ============================================================================
// Re-exports
// ============================================================================

pub use anthropic_to_openai::*;
pub use openai_to_anthropic::*;
