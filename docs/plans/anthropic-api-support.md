# Plan: Anthropic Messages API (/v1/messages) Support for commandcode2api

## Summary

Port Anthropic↔OpenAI translation logic from `oc-go-cc` (Go) into the existing Rust `commandcode2api` proxy. Serve Anthropic's `/v1/messages` on the same port alongside the existing OpenAI `/v1/chat/completions`. The flow: Anthropic request → translate to OpenAI → existing CommandCode upstream → parse CC SSE → translate OpenAI chunks back to Anthropic SSE events.

---

## Files to Create/Modify

| # | Action | File | Purpose |
|---|--------|------|---------|
| 1 | CREATE | `src/translate_api.rs` | Anthropic types + pure conversion functions |
| 2 | MODIFY | `src/types.rs` | Add `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` to `Usage` |
| 3 | MODIFY | `src/proxy.rs` | Add `messages_handler`, make `build_cc_request` + `get_api_key` pub(crate) |
| 4 | MODIFY | `src/main.rs` | Register `translate_api` module + `POST /v1/messages` route |
| 5 | CREATE | `tests/translate_api_test.rs` | Unit tests for all conversion functions |
| 6 | CREATE | `docs/ANTHROPIC_API.md` | API documentation |
| 7 | CREATE | `task_progress.json` | Progress tracking |

---

## Step 1: `src/translate_api.rs` — Anthropic Types + Conversions

### Types (serde-annotated)

```rust
// === Request types ===

#[derive(Debug, Deserialize)]
pub struct MessageRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    pub max_tokens: u32,
    pub stream: Option<bool>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub tools: Option<Vec<AnthropicTool>>,
    pub tool_choice: Option<AnthropicToolChoice>,
    pub metadata: Option<AnthropicMetadata>,
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
    pub cache_control: Option<CacheControl>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: String,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: serde_json::Value,  // string or Vec<ContentBlock>
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String, input: serde_json::Value, #[serde(default)] thinking: Option<String> },
    #[serde(rename = "tool_result")]
    ToolResult { tool_use_id: String, content: serde_json::Value, #[serde(default)] is_error: Option<bool> },
    #[serde(rename = "thinking")]
    Thinking { thinking: String, #[serde(default)] signature: Option<String> },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageSource { pub r#type: String, pub media_type: String, pub data: String }

#[derive(Debug, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicToolChoice {
    Auto { r#type: String },
    Any { r#type: String },
    Tool { r#type: String, name: String },
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMetadata {
    pub user_id: Option<String>,
}

// === Response types ===

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

// === Streaming Event types ===

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum MessageEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartInfo },
    #[serde(rename = "content_block_start")]
    ContentBlockStart { index: usize, content_block: ContentBlock },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: StreamDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta { delta: MessageDeltaInfo, usage: Option<AnthropicUsage> },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
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

// === Streaming state machine ===

pub struct TranslateState {
    pub message_id: String,
    pub model: String,
    pub text_block_idx: Option<usize>,
    pub thinking_block_idx: Option<usize>,
    pub tool_block_indices: std::collections::HashMap<usize, usize>,
    pub started: bool,
    pub stop_sent: bool,
    next_index: usize,
}

impl TranslateState {
    pub fn new(message_id: String, model: String) -> Self { ... }
    fn next_block_index(&mut self) -> usize { self.next_index += 1; self.next_index }
}
```

### Functions

```rust
/// Anthropic MessageRequest → OpenAI ChatCompletionRequest
pub fn anthropic_request_to_openai(req: &MessageRequest) -> ChatCompletionRequest

/// OpenAI CC chunk → 0-N Anthropic MessageEvents (per chunk call)
pub fn openai_chunk_to_anthropic_events(
    chunk: &ChatCompletionChunk, state: &mut TranslateState) -> Vec<MessageEvent>

/// Non-streaming: OpenAI ChatCompletionResponse → Anthropic MessageResponse
pub fn openai_response_to_anthropic(resp: &ChatCompletionResponse, model: &str) -> MessageResponse

/// OpenAI finish_reason → Anthropic stop_reason
pub fn map_anthropic_stop_reason(reason: Option<&str>) -> Option<String>
// "stop" → "end_turn", "length" → "max_tokens", "tool_calls" → "tool_use", other → "end_turn"

/// OpenAI UsageInfo → Anthropic Usage
pub fn map_anthropic_usage(usage: &Usage) -> AnthropicUsage
// input_tokens = prompt_tokens - cache_hit_tokens - cache_miss_tokens
```

### Conversion Logic Detail

**`anthropic_request_to_openai()`:**

1. **System**: `String("text")` → system message. `Blocks([...])` → concatenate text parts → system message with optional cache_control.

2. **Messages** — iterate and flatten (one Anthropic message can produce 1+ OpenAI messages):
   - **user**: Extract text blocks → content string. Extract tool_result blocks → separate `role: "tool"` messages (tool_call_id = tool_use_id). Remaining text → `role: "user"` message emitted AFTER tool results.
   - **assistant**: text → content string. thinking blocks (and tool_use.thinking) → concatenate → `reasoning_content` field. tool_use blocks → `tool_calls[]` array (id, function {name, arguments: input}).

3. **Tools**: AnthropicTool `{name, description, input_schema}` → OpenAI ToolDefinition `{type: "function", function: {name, description, parameters: input_schema}}`

4. **Params**: max_tokens, temperature, top_p map directly. stop_sequences → `Vec<String>` stop.

5. **tool_choice**: `{type: "auto"}` → `"auto"`. `{type: "any"}` → `"required"`. `{type: "tool", name: "x"}` → `{type: "function", function: {name: "x"}}`.

**`openai_chunk_to_anthropic_events()` per call:**

State machine:
- **First call**: emit `message_start` with id/role/model; set started=true.
- **`delta.content` present**: close thinking block if active → start text block if needed → emit `content_block_delta(text_delta)`.
- **`delta.reasoning_content` present**: close text block if active → start thinking block if needed → emit `content_block_delta(thinking_delta)`.
- **`delta.tool_calls[]`**: For each ToolCall with index i. If tool_block_indices[i] doesn't exist → close text/thinking blocks → emit `content_block_start(tool_use, idx=N, id=..., name=...)` → track. If arguments non-empty → emit `content_block_delta(input_json_delta)`.
- **`finish_reason` present**: close all active blocks sorted by index → emit `message_delta` with mapped stop_reason + usage → set stop_sent=true.
- **`usage` without finish_reason**: emit `message_delta` with usage only (stop_reason empty).

**`openai_response_to_anthropic()`:**

1. reasoning_content → thinking ContentBlock (first in array)
2. tool_calls[] → tool_use ContentBlocks (id, name, input: parsed arguments)
3. text content → text ContentBlock
4. Ensure at least one empty text block if nothing else
5. Map finish_reason → stop_reason
6. Map usage with map_anthropic_usage()

---

## Step 2: `src/types.rs` — Add Cache Token Fields

Add to the `Usage` struct:
```rust
#[serde(default, skip_serializing_if = "Option::is_none")]
pub prompt_cache_hit_tokens: Option<u32>,
#[serde(default, skip_serializing_if = "Option::is_none")]
pub prompt_cache_miss_tokens: Option<u32>,
```

---

## Step 3: `src/proxy.rs` — Add Anthropic Handler

### Visibility changes (pub → pub(crate)):
- `fn build_cc_request()` at ~line 154
- `fn get_api_key()` at ~line 84

### New public handler:

```rust
pub async fn messages_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<translate_api::MessageRequest>,
) -> Response
```

Dispatches to `messages_stream()` or `messages_non_streaming()` based on `req.stream`.

### Streaming path — `async fn messages_stream()`:

1. `anthropic_request_to_openai(&req)` → `ChatCompletionRequest` (with stream=true)
2. `build_cc_request(&openai_req)` → `CcRequest`
3. POST to `{api_base}/alpha/generate` with retry
4. Spawn tokio task reading CC byte stream
5. Parse SSE lines via `parse_cc_event_line()` → deserialize to `ChatCompletionChunk`
6. For each chunk → `translate_api::openai_chunk_to_anthropic_events()` → emit named SSE events:
   ```
   event: content_block_start
   data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

   event: content_block_delta
   data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

   event: message_stop
   data: {"type":"message_stop"}
   ```
7. Use `mpsc::channel(128)` → `tokio_stream::wrappers::ReceiverStream` → `Sse::new().keep_alive()`
8. Write helper: `fn write_anthropic_events(events: &[MessageEvent]) -> String`

### Non-streaming path — `async fn messages_non_streaming()`:

1. Same conversion flow
2. POST to CC, buffer entire SSE
3. Accumulate into `ChatCompletionResponse` (reuse pattern from `chat_completions()`)
4. `translate_api::openai_response_to_anthropic(&openai_resp, &req.model)` → return `Json<MessageResponse>`

---

## Step 4: `src/main.rs` — Register Route

```rust
mod translate_api;

// Add to imports:
use proxy::messages_handler;

// In Router::new():
.route("/v1/messages", post(messages_handler))
```

---

## Step 5: `tests/translate_api_test.rs`

```rust
#[cfg(test)]
mod tests {
    use commandcode2api::translate_api::*;
    use commandcode2api::types::*;

    // Request conversion tests
    #[test] fn test_basic_message_conversion() { ... }
    #[test] fn test_system_string_to_message() { ... }
    #[test] fn test_system_blocks_to_message() { ... }
    #[test] fn test_tool_conversion() { ... }
    #[test] fn test_thinking_to_reasoning() { ... }
    #[test] fn test_tool_use_to_tool_calls() { ... }
    #[test] fn test_tool_result_to_tool_message() { ... }
    #[test] fn test_tool_use_inline_thinking() { ... }
    #[test] fn test_user_with_tool_results_ordering() { ... }

    // Streaming chunk conversion tests
    #[test] fn test_text_delta_events() { ... }
    #[test] fn test_reasoning_delta_events() { ... }
    #[test] fn test_tool_call_delta_events() { ... }
    #[test] fn test_finish_reason_closes_blocks() { ... }
    #[test] fn test_multiple_content_blocks_interleaved() { ... }
    #[test] fn test_usage_in_message_delta() { ... }

    // Response conversion tests
    #[test] fn test_full_response_conversion() { ... }
    #[test] fn test_response_with_tool_calls() { ... }
    #[test] fn test_response_with_reasoning() { ... }

    // Stop reason mapping tests
    #[test] fn test_stop_to_end_turn() { ... }
    #[test] fn test_length_to_max_tokens() { ... }
    #[test] fn test_tool_calls_to_tool_use() { ... }
    #[test] fn test_unknown_to_end_turn() { ... }

    // Usage mapping tests
    #[test] fn test_basic_token_mapping() { ... }
    #[test] fn test_cache_token_subtraction() { ... }
}
```

---

## Step 6: `docs/ANTHROPIC_API.md`

Document sections:
- **Endpoint**: `POST /v1/messages` — Anthropic Messages API compatible
- **Supported Features**: text, thinking, tool_use, tool_result; system as string/array; top_p, temperature, stop_sequences
- **Streaming Format**: SSE with named events (message_start, content_block_start/delta/stop, message_delta, message_stop)
- **Field Mappings**: Table showing Anthropic → OpenAI → CommandCode translations
- **Limitations**: no image support, no thinking budget_tokens, no deepseek-specific reasoning placeholders, signature field dropped

---

## Step 7: `task_progress.json`

```json
{
  "version": "1.0",
  "project": "commandcode2api",
  "feature": "Anthropic Messages API (/v1/messages)",
  "tasks": [
    {"id":"1","module":"src/translate_api.rs","action":"create","desc":"Anthropic types and conversion functions","reason":"Core translation layer ported from oc-go-cc","status":"pending"},
    {"id":"2","module":"src/types.rs","action":"modify","desc":"Add cache token fields to Usage struct","reason":"Accurate Anthropic input_tokens requires subtracting cache reads","status":"pending"},
    {"id":"3","module":"src/proxy.rs","action":"modify","desc":"Add messages_handler; make 2 fns pub(crate)","reason":"Reuse existing CommandCode proxying; minimize visibility expansion","status":"pending"},
    {"id":"4","module":"src/main.rs","action":"modify","desc":"Register translate_api module + POST /v1/messages","reason":"New endpoint on same port as /v1/chat/completions","status":"pending"},
    {"id":"5","module":"tests/translate_api_test.rs","action":"create","desc":"Unit tests for all conversion functions","reason":"Verify Anthropic↔OpenAI translation correctness","status":"pending"},
    {"id":"6","module":"docs/ANTHROPIC_API.md","action":"create","desc":"API documentation","reason":"Document supported features, mappings, and limitations","status":"pending"},
    {"id":"7","module":"task_progress.json","action":"create","desc":"Progress tracking manifest","reason":"Harness engineering — trace what changed and rationale","status":"completed"}
  ]
}
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Handler in `proxy.rs` not new file | Reuse 200+ lines of proxying; only 2 functions need visibility bump |
| Types in separate `translate_api.rs` | Clean separation of Anthropic concern per user's specification |
| `#[serde(tag="type")]` for ContentBlock/MessageEvent | Directly matches Anthropic JSON schema; zero custom serialization needed |
| `TranslateState` struct | Clean streaming state machine; testable without I/O dependencies |
| Skip images, budget_tokens, deepseek hacks | Keeps v1 scope manageable; port only core translation logic |
| Drop `signature` on thinking blocks | Matches oc-go-cc behavior; add later if needed |
| No `[DONE]` sentinel in Anthropic streaming | `message_stop` is the canonical terminator; Anthropic protocol doesn't use [DONE] |
| `pub(crate)` not `pub` for internal fns | Minimum necessary visibility; prevents module fragmentation |

---

## Verification

1. `cargo build` — compiles cleanly
2. `cargo test` — all translate_api tests pass
3. `cargo clippy` — no warnings
4. Manual test: `curl -X POST localhost:3000/v1/messages -H 'Content-Type: application/json' -d '{"model":"claude-sonnet","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}'` → valid Anthropic `MessageResponse`
5. Streaming test: `curl ... -d '{"stream":true,...}'` → SSE with `message_start`, `content_block_*`, `message_delta`, `message_stop`
