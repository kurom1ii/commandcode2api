#[cfg(test)]
mod round_trip_a2o {
    use commandcode2api::translate_api::*;
    use commandcode2api::types::*;

    /// Full Anthropic request (system + thinking + tool_use + tool_result)
    /// → OpenAI → simulates upstream response → back to Anthropic.
    /// Verifies the entire dual-direction flow with tool calls.
    #[test]
    fn test_full_round_trip_anthropic_to_openai_and_back() {
        // ═══════════════════════════════════════════════════════════════════
        // STEP 1: Anthropic MessageRequest (input)
        // ═══════════════════════════════════════════════════════════════════
        let anthropic_req = MessageRequest {
            model: "claude-sonnet".to_string(),
            max_tokens: 256,
            system: Some(SystemPrompt::String(
                "You are a helpful coding assistant. Be concise.".to_string(),
            )),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stop_sequences: Some(vec!["</answer>".to_string()]),
            tools: Some(vec![
                AnthropicTool {
                    name: "read_file".to_string(),
                    description: Some("Read a file from disk".to_string()),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"}
                        },
                        "required": ["path"]
                    }),
                },
                AnthropicTool {
                    name: "search_code".to_string(),
                    description: Some("Search codebase".to_string()),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "file_pattern": {"type": "string"}
                        },
                        "required": ["query"]
                    }),
                },
            ]),
            tool_choice: Some(AnthropicToolChoice {
                choice_type: "auto".to_string(),
                name: None,
            }),
            messages: vec![
                AnthropicMessage {
                    role: "user".to_string(),
                    content: serde_json::json!("Find where authentication logic is implemented in this project"),
                },
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: serde_json::json!([
                        {"type": "thinking", "thinking": "I need to search the codebase for authentication-related files first, then read the most relevant one.", "signature": "sig_abc123"},
                        {"type": "tool_use", "id": "toolu_001", "name": "search_code", "input": {"query": "authentication login", "file_pattern": "*.rs"}}
                    ]),
                },
                AnthropicMessage {
                    role: "user".to_string(),
                    content: serde_json::json!([
                        {"type": "tool_result", "tool_use_id": "toolu_001", "content": "Found 3 files:\nsrc/auth/login.rs\nsrc/auth/middleware.rs\nsrc/auth/token.rs"},
                        {"type": "text", "text": "Now read src/auth/login.rs and explain the flow"}
                    ]),
                },
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: serde_json::json!([
                        {"type": "thinking", "thinking": "The user wants me to read the login file. Let me fetch it.", "signature": "sig_def456"},
                        {"type": "tool_use", "id": "toolu_002", "name": "read_file", "input": {"path": "src/auth/login.rs"}}
                    ]),
                },
                AnthropicMessage {
                    role: "user".to_string(),
                    content: serde_json::json!([
                        {"type": "tool_result", "tool_use_id": "toolu_002", "content": "pub fn login(user: &str, pass: &str) -> Result<Token> {\n    // validate credentials\n    // issue JWT\n}"}
                    ]),
                },
            ],
            ..Default::default()
        };

        println!("\n══════════════════════════════════════════════════════════");
        println!("INPUT: Anthropic MessageRequest (/v1/messages)");
        println!("══════════════════════════════════════════════════════════");
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "model": &anthropic_req.model,
            "max_tokens": anthropic_req.max_tokens,
            "system": "You are a helpful coding assistant. Be concise.",
            "temperature": anthropic_req.temperature,
            "top_p": anthropic_req.top_p,
            "stop_sequences": anthropic_req.stop_sequences,
            "tools": [{"name":"read_file","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},{"name":"search_code","input_schema":{"type":"object","properties":{"query":{"type":"string"},"file_pattern":{"type":"string"}},"required":["query"]}}],
            "tool_choice": {"type":"auto"},
            "messages_count": anthropic_req.messages.len(),
            "message_roles": ["user","assistant(thinking+tool_use:search_code)","user(tool_result+text)","assistant(thinking+tool_use:read_file)","user(tool_result)"]
        })).unwrap());

        // ═══════════════════════════════════════════════════════════════════
        // STEP 2: Anthropic → OpenAI ChatCompletionRequest
        // ═══════════════════════════════════════════════════════════════════
        let oai_req = anthropic_request_to_openai(&anthropic_req);

        println!("\n══════════════════════════════════════════════════════════");
        println!("TRANSLATED: Anthropic → OpenAI ChatCompletionRequest");
        println!("══════════════════════════════════════════════════════════");
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "model": oai_req.model,
            "max_tokens": oai_req.max_tokens,
            "temperature": oai_req.temperature,
            "top_p": oai_req.top_p,
            "tool_choice": "auto",
            "tools": [{"type":"function","function":{"name":"read_file","description":"Read a file from disk","parameters":{}}},{"type":"function","function":{"name":"search_code","description":"Search codebase","parameters":{}}}],
            "system": oai_req.system,
            "messages": oai_req.messages.iter().map(|m| serde_json::json!({
                "role": m.role,
                "content": m.content,
                "tool_calls": m.tool_calls,
                "tool_call_id": m.tool_call_id,
            })).collect::<Vec<_>>(),
        })).unwrap());

        // Verify system message is first
        assert_eq!(oai_req.messages[0].role, "system");
        assert_eq!(
            oai_req.messages[0].content.as_ref().and_then(|c| c.as_str()),
            Some("You are a helpful coding assistant. Be concise.")
        );

        let tools = oai_req.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].function.name, "read_file");
        assert_eq!(tools[1].function.name, "search_code");
        assert!(matches!(oai_req.tool_choice, Some(ToolChoice::String(ref s)) if s == "auto"));
        assert_eq!(oai_req.temperature, Some(0.7));
        assert_eq!(oai_req.top_p, Some(0.9));

        // ═══════════════════════════════════════════════════════════════════
        // STEP 3: Simulate upstream OpenAI response (from CommandCode)
        // ═══════════════════════════════════════════════════════════════════
        let reasoning = "The login function at src/auth/login.rs:\n1. Takes username and password\n2. Validates credentials\n3. Issues a JWT token on success";
        let oai_resp = ChatCompletionResponse {
            id: "resp_full_001".to_string(),
            object: "chat.completion".to_string(),
            created: 1718234567,
            model: "claude-sonnet".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: Some("Here's the authentication flow:\n\n1. `login()` validates credentials\n2. On success, returns a JWT\n3. The middleware checks tokens on each request".to_string()),
                    reasoning_content: Some(reasoning.to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: 500,
                completion_tokens: 80,
                total_tokens: 580,
                prompt_cache_hit_tokens: Some(200),
                prompt_cache_miss_tokens: Some(50),
            }),
        };

        println!("\n══════════════════════════════════════════════════════════");
        println!("INTERMEDIATE: Simulated OpenAI ChatCompletionResponse");
        println!("══════════════════════════════════════════════════════════");
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "id": oai_resp.id,
            "object": oai_resp.object,
            "model": oai_resp.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Here's the authentication flow: ...",
                    "reasoning_content": reasoning,
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens":500,"completion_tokens":80,"total_tokens":580,"prompt_cache_hit_tokens":200,"prompt_cache_miss_tokens":50}
        })).unwrap());

        // ═══════════════════════════════════════════════════════════════════
        // STEP 4: OpenAI → Anthropic MessageResponse (output)
        // ═══════════════════════════════════════════════════════════════════
        let an_resp = openai_response_to_anthropic(&oai_resp, "claude-sonnet");

        println!("\n══════════════════════════════════════════════════════════");
        println!("OUTPUT: Anthropic MessageResponse (/v1/messages response)");
        println!("══════════════════════════════════════════════════════════");
        println!("{}", serde_json::to_string_pretty(&an_resp).unwrap());

        // Assertions
        assert_eq!(an_resp.id, "resp_full_001");
        assert_eq!(an_resp.msg_type, "message");
        assert_eq!(an_resp.role, "assistant");
        assert_eq!(an_resp.stop_reason, Some("end_turn".to_string()));
        assert_eq!(an_resp.content.len(), 2);

        if let ContentBlock::Thinking { ref thinking, ref signature } = an_resp.content[0] {
            assert_eq!(thinking, reasoning);
            assert!(signature.is_none());
        }
        if let ContentBlock::Text { ref text } = an_resp.content[1] {
            assert!(text.contains("authentication flow"));
        }

        assert_eq!(an_resp.usage.input_tokens, 250);
        assert_eq!(an_resp.usage.output_tokens, 80);
        assert_eq!(an_resp.usage.cache_read_input_tokens, Some(200));
        assert_eq!(an_resp.usage.cache_creation_input_tokens, Some(50));
    }
}
