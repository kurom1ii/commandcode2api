# Anthropic Messages API (/v1/messages)

commandcode2api serves an Anthropic-compatible **Messages API** endpoint alongside the existing OpenAI Chat Completions endpoint on the same port.

**Endpoint**: `POST /v1/messages`

---

## Supported Features

### Request Format

Full Anthropic Messages API request schema:

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model ID from models.json |
| `messages` | array | Conversation messages |
| `max_tokens` | integer | Maximum output tokens (required) |
| `system` | string or array | System prompt |
| `stream` | boolean | Enable SSE streaming |
| `temperature` | number | Sampling temperature (0-1) |
| `top_p` | number | Nucleus sampling |
| `stop_sequences` | array of strings | Stop sequences |
| `tools` | array | Tool definitions |
| `tool_choice` | object | Tool selection: `{type:"auto"}`, `{type:"any"}`, `{type:"tool", name:"x"}` |
| `metadata` | object | User metadata (`user_id`) |

### Content Block Types Supported

| Block Type | Input (Request) | Output (Response) |
|-----------|----------------|-------------------|
| `text` | Yes | Yes |
| `thinking` | Yes (preserved as reasoning) | Yes (from reasoning_content) |
| `tool_use` | Yes | Yes |
| `tool_result` | Yes | N/A (request-only) |
| `image` | Partially¹ | No |

¹ Images are replaced with `[Image]` text placeholder.

### Streaming Format

SSE with named Anthropic events:

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","type":"message","role":"assistant","model":"...","content":[]}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}
```

### Non-Streaming Response

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {"type": "thinking", "thinking": "Let me think..."},
    {"type": "text", "text": "The answer is 42"}
  ],
  "model": "claude-sonnet",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 100,
    "output_tokens": 20
  }
}
```

---

## Field Mappings (Anthropic → OpenAI → CommandCode)

| Anthropic | OpenAI | CommandCode |
|-----------|--------|-------------|
| `messages[].content` (string or blocks) | `messages[].content` (string or array) | `params.messages[].content` |
| `thinking` block | `reasoning_content` field | (bundled in content array as `{type:"reasoning"}`) |
| `tool_use` block | `tool_calls[]` array | `{type:"tool-call"}` in content |
| `tool_result` message | `role: "tool"` message | `{role:"tool"}` with `{type:"tool-result"}` |
| `system` (string/array) | `system` message + `system` field | `params.system` |
| `tools[]` with `input_schema` | `tools[]` with `function.parameters` | `params.tools[]` with `input_schema` |
| `stop_sequences` | `stop` (extra field) | (not forwarded to CC) |
| `max_tokens` | `max_tokens` | `params.max_tokens` |
| `temperature` | `temperature` | (not forwarded) |
| `top_p` | `top_p` | (not forwarded) |

**Stop reason mapping (response direction):**

| OpenAI `finish_reason` | Anthropic `stop_reason` |
|------------------------|------------------------|
| `stop` | `end_turn` |
| `length` | `max_tokens` |
| `tool_calls` | `tool_use` |
| `tool_use` | `tool_use` |
| `content_filter` | `end_turn` |
| (anything else) | `end_turn` |

**Usage mapping (response direction):**

Anthropic `input_tokens` = OpenAI `prompt_tokens` − `prompt_cache_hit_tokens` − `prompt_cache_miss_tokens`  
(because Anthropic bills cache reads at 10% and excludes them from `input_tokens`)

---

## Architecture

```
Anthropic Client (Claude Code, etc.)
    │
    ▼
POST /v1/messages (MessageRequest)
    │
    ├── anthropic_request_to_openai()  ── Anthropic → OpenAI ChatCompletionRequest
    │
    ▼
build_cc_request()                     ── OpenAI → CommandCode CcRequest
    │
    ▼
POST /alpha/generate (to CommandCode)  ── Receives SSE stream
    │
    ▼
parse_cc_event_line()                  ── SSE → CcStreamEvent → ChatCompletionChunk
    │
    ├── openai_chunk_to_anthropic_events()  ── OpenAI chunks → Anthropic MessageEvents (streaming)
    ├── openai_response_to_anthropic()      ── OpenAI response → Anthropic MessageResponse (non-streaming)
    │
    ▼
Anthropic Client receives SSE events or JSON response
```

---

## Limitations

- **No image content block support**: Images are replaced with `[Image]` text placeholder. Vision model support is a future enhancement.
- **No `thinking.budget_tokens` mapping**: The Anthropic `thinking` block's `budget_tokens` field is not translated to any downstream parameter. Thinking mode is always enabled by default.
- **No deepseek-specific reasoning placeholders**: Unlike `oc-go-cc`, this proxy does not inject placeholder `reasoning_content` for DeepSeek models in thinking mode. Use the `/v1/chat/completions` endpoint directly for DeepSeek models requiring this behavior.
- **`signature` field dropped**: Anthropic `thinking` blocks have a cryptographic `signature` that is dropped during translation (matching `oc-go-cc` behavior).
- **`cache_control` dropped**: System block `cache_control` is parsed but not forwarded.
- **No `[DONE]` sentinel**: Anthropic streaming protocol uses `message_stop` as the terminator. The `[DONE]` sentinel (OpenAI convention) is not emitted on the Anthropic endpoint.
