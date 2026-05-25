use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// OpenAI-compatible types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    // extra fields we ignore but accept for compatibility
    #[serde(flatten)]
    #[allow(dead_code)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    String(String),
    Object {
        tool_type: String,
        function: ToolChoiceFunction,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub index: Option<usize>,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    pub function: Option<ToolCallFunction>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCallFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

// Non-streaming response
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: ChatCompletionMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Serialize, Default)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

// Streaming response (SSE)
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

// Model list
#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

// ---------------------------------------------------------------------------
// CommandCode types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CcRequest {
    pub config: CcConfig,
    pub memory: String,
    pub taste: String,
    pub skills: Option<serde_json::Value>,
    pub permission_mode: String,
    pub params: CcParams,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CcConfig {
    pub working_dir: String,
    pub date: String,
    pub environment: String,
    pub structure: Vec<serde_json::Value>,
    pub is_git_repo: bool,
    pub current_branch: String,
    pub main_branch: String,
    pub git_status: String,
    pub recent_commits: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct CcParams {
    pub model: String,
    pub messages: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub system: String,
    pub max_tokens: u64,
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct CcStreamEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    #[serde(rename = "toolCallId")]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    #[serde(rename = "toolName")]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub input: Option<serde_json::Value>,
    #[serde(default)]
    pub args: Option<serde_json::Value>,
    #[serde(default)]
    pub arguments: Option<serde_json::Value>,
    #[serde(default)]
    #[serde(rename = "finishReason")]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub error: Option<serde_json::Value>,
    #[serde(default)]
    #[serde(rename = "totalUsage")]
    pub total_usage: Option<CcUsage>,
}

#[derive(Debug, Deserialize)]
pub struct CcUsage {
    #[serde(default)]
    #[serde(rename = "inputTokens")]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    #[serde(rename = "outputTokens")]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    #[serde(rename = "inputTokenDetails")]
    pub input_token_details: Option<CcTokenDetails>,
}

#[derive(Debug, Deserialize)]
pub struct CcTokenDetails {
    #[serde(default)]
    #[serde(rename = "cacheReadTokens")]
    pub cache_read_tokens: Option<u64>,
    #[serde(default)]
    #[serde(rename = "cacheWriteTokens")]
    pub cache_write_tokens: Option<u64>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub struct CcTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "input_schema")]
    pub input_schema: serde_json::Value,
}

// ---------------------------------------------------------------------------
// models.json types (from command-code dist extraction)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Clone)]
pub struct ModelsJson {
    pub providers: HashMap<String, String>,
    #[serde(default)]
    pub provider_groups: Vec<ProviderGroup>,
    pub models: Vec<ModelDef>,
    #[serde(default)]
    pub pricing: Vec<PricingEntry>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ProviderGroup {
    pub id: String,
    pub label: String,
    pub short_label: String,
    pub description: String,
    pub providers: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ModelDef {
    pub key: String,
    pub id: String,
    pub provider: String,
    pub spec: String,
    pub label: String,
    pub name: String,
    pub description: String,
    pub reasoning: bool,
    #[serde(default)]
    pub reasoning_efforts: Option<Vec<String>>,
    pub context_window: u64,
    pub max_output_tokens: u64,
    #[serde(default)]
    pub vendor_label: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PricingEntry {
    pub provider: String,
    pub id: String,
    pub category: String,
    pub prompt_cost: f64,
    pub completion_cost: f64,
    #[serde(default)]
    pub cache_write5m_cost: f64,
    #[serde(default)]
    pub cache_write1h_cost: f64,
    #[serde(default)]
    pub cache_hit_cost: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cc_request_serializes_expected_commandcode_keys() {
        let request = CcRequest {
            config: CcConfig {
                working_dir: "/repo".to_string(),
                date: "2026-05-25".to_string(),
                environment: "test".to_string(),
                structure: vec![],
                is_git_repo: true,
                current_branch: "feature".to_string(),
                main_branch: "main".to_string(),
                git_status: "clean".to_string(),
                recent_commits: vec!["abc123".to_string()],
            },
            memory: String::new(),
            taste: String::new(),
            skills: None,
            permission_mode: "standard".to_string(),
            params: CcParams {
                model: "test-model".to_string(),
                messages: vec![],
                tools: vec![],
                system: String::new(),
                max_tokens: 500,
                stream: true,
            },
        };

        let value = serde_json::to_value(request).expect("CcRequest serializes");

        assert_eq!(value["permissionMode"], "standard");
        assert_eq!(value["config"]["workingDir"], "/repo");
        assert_eq!(value["config"]["isGitRepo"], true);
        assert_eq!(value["config"]["currentBranch"], "feature");
        assert_eq!(value["config"]["mainBranch"], "main");
        assert_eq!(value["config"]["gitStatus"], "clean");
        assert_eq!(value["config"]["recentCommits"], json!(["abc123"]));
        assert!(value.get("permission_mode").is_none());
        assert!(value["config"].get("working_dir").is_none());
        assert_eq!(value["params"]["max_tokens"], 500);
    }
}
