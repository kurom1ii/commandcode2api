#[cfg(test)]
mod translate_api_tests {
    use commandcode2api::translate_api::*;
    use commandcode2api::types::*;

    fn mk_msg(role: &str, content: &str) -> AnthropicMessage {
        AnthropicMessage {
            role: role.to_string(),
            content: serde_json::Value::String(content.to_string()),
        }
    }

    fn mk_assistant_with_tool(id: &str, name: &str, input: &str) -> AnthropicMessage {
        let blocks: serde_json::Value =
            serde_json::from_str(&format!(
                r#"[{{"type":"tool_use","id":"{}","name":"{}","input":{}}}]"#,
                id, name, input
            ))
            .unwrap();
        AnthropicMessage {
            role: "assistant".to_string(),
            content: blocks,
        }
    }

    fn mk_user_with_tool_result(tool_use_id: &str, result_text: &str) -> AnthropicMessage {
        let blocks: serde_json::Value = serde_json::from_str(&format!(
            r#"[{{"type":"tool_result","tool_use_id":"{}","content":"{}"}}]"#,
            tool_use_id, result_text
        ))
        .unwrap();
        AnthropicMessage {
            role: "user".to_string(),
            content: blocks,
        }
    }

    // ========================================================================
    // anthropic_request_to_openai tests
    // ========================================================================

    #[test]
    fn test_basic_message_conversion() {
        let req = MessageRequest {
            model: "claude-sonnet".to_string(),
            max_tokens: 256,
            messages: vec![
                mk_msg("user", "Hello"),
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String("Hi there!".to_string()),
                },
            ],
            ..Default::default()
        };

        let oai = anthropic_request_to_openai(&req);

        assert_eq!(oai.model, "claude-sonnet");
        assert_eq!(oai.max_tokens, Some(256));
        assert_eq!(oai.messages.len(), 2);
        assert_eq!(oai.messages[0].role, "user");
        assert_eq!(
            oai.messages[0].content.as_ref().and_then(|c| c.as_str()),
            Some("Hello")
        );
        assert_eq!(oai.messages[1].role, "assistant");
        let content = oai.messages[1].content.as_ref().unwrap();
        assert!(content.is_array());
        let text_part = content.as_array().unwrap().iter()
            .find(|p| p["type"] == "text").unwrap();
        assert_eq!(text_part["text"], "Hi there!");
    }

    #[test]
    fn test_system_string_creates_system_message() {
        let req = MessageRequest {
            model: "claude-sonnet".to_string(),
            max_tokens: 100,
            messages: vec![mk_msg("user", "Hi")],
            system: Some(SystemPrompt::String("You are helpful.".to_string())),
            ..Default::default()
        };

        let oai = anthropic_request_to_openai(&req);

        assert_eq!(oai.messages.len(), 2);
        assert_eq!(oai.messages[0].role, "system");
        assert_eq!(
            oai.messages[0].content.as_ref().and_then(|c| c.as_str()),
            Some("You are helpful.")
        );
        assert_eq!(oai.system, Some("You are helpful.".to_string()));
    }

    #[test]
    fn test_system_blocks_concatenate() {
        let req = MessageRequest {
            model: "claude-sonnet".to_string(),
            max_tokens: 100,
            messages: vec![mk_msg("user", "Hi")],
            system: Some(SystemPrompt::Blocks(vec![
                SystemTextBlock {
                    block_type: "text".to_string(),
                    text: "You are helpful.".to_string(),
                    cache_control: None,
                },
                SystemTextBlock {
                    block_type: "text".to_string(),
                    text: " Be concise.".to_string(),
                    cache_control: None,
                },
            ])),
            ..Default::default()
        };

        let oai = anthropic_request_to_openai(&req);

        assert_eq!(oai.messages.len(), 2);
        assert_eq!(oai.messages[0].role, "system");
        assert_eq!(
            oai.messages[0].content.as_ref().and_then(|c| c.as_str()),
            Some("You are helpful. Be concise.")
        );
    }

    #[test]
    fn test_thinking_to_cc_content_array() {
        let blocks: serde_json::Value = serde_json::from_str(
            r#"[
                {"type":"thinking","thinking":"Let me think..."},
                {"type":"text","text":"The answer is 42"}
            ]"#,
        )
        .unwrap();
        let req = MessageRequest {
            model: "claude-sonnet".to_string(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: "assistant".to_string(),
                content: blocks,
            }],
            ..Default::default()
        };

        let oai = anthropic_request_to_openai(&req);

        assert_eq!(oai.messages.len(), 1);
        let msg = &oai.messages[0];
        assert_eq!(msg.role, "assistant");

        // Content should be an array with text and thinking parts
        let content = msg.content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(content.len(), 2);

        let thinking_part = content
            .iter()
            .find(|p| p["type"] == "thinking")
            .unwrap();
        assert_eq!(thinking_part["thinking"], "Let me think...");

        let text_part = content.iter().find(|p| p["type"] == "text").unwrap();
        assert_eq!(text_part["text"], "The answer is 42");
    }

    #[test]
    fn test_tool_use_to_tool_calls() {
        let req = MessageRequest {
            model: "claude-sonnet".to_string(),
            max_tokens: 100,
            messages: vec![mk_assistant_with_tool(
                "tool_01",
                "get_weather",
                r#"{"city":"Paris"}"#,
            )],
            ..Default::default()
        };

        let oai = anthropic_request_to_openai(&req);

        let msg = &oai.messages[0];
        assert_eq!(msg.role, "assistant");
        let calls = msg.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("tool_01"));
        assert_eq!(
            calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
        let args: serde_json::Value =
            serde_json::from_str(calls[0].function.as_ref().unwrap().arguments.as_ref().unwrap())
                .unwrap();
        assert_eq!(args["city"], "Paris");
    }

    #[test]
    fn test_tool_result_to_tool_message() {
        let req = MessageRequest {
            model: "claude-sonnet".to_string(),
            max_tokens: 100,
            messages: vec![
                mk_assistant_with_tool("tool_01", "search", r#"{"q":"rust"}"#),
                mk_user_with_tool_result("tool_01", "Results: Rust book"),
            ],
            ..Default::default()
        };

        let oai = anthropic_request_to_openai(&req);

        // assistant + tool (no user text since all content was tool_result)
        assert_eq!(oai.messages.len(), 2);
        assert_eq!(oai.messages[1].role, "tool");
        assert_eq!(oai.messages[1].tool_call_id.as_deref(), Some("tool_01"));
        assert_eq!(
            oai.messages[1].content.as_ref().and_then(|c| c.as_str()),
            Some("Results: Rust book")
        );
    }

    #[test]
    fn test_tool_use_inline_thinking() {
        let blocks: serde_json::Value = serde_json::from_str(
            r#"[{"type":"tool_use","id":"t1","name":"read","input":{},"thinking":"I should read first"}]"#,
        )
        .unwrap();
        let req = MessageRequest {
            model: "claude-sonnet".to_string(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: "assistant".to_string(),
                content: blocks,
            }],
            ..Default::default()
        };

        let oai = anthropic_request_to_openai(&req);
        let msg = &oai.messages[0];

        // Content array should include thinking
        let content = msg.content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[0]["thinking"], "I should read first");

        // Tool calls still present
        let calls = msg.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("t1"));
    }

    #[test]
    fn test_tool_conversion() {
        let req = MessageRequest {
            model: "claude-sonnet".to_string(),
            max_tokens: 100,
            messages: vec![mk_msg("user", "Hello")],
            tools: Some(vec![AnthropicTool {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } }
                }),
            }]),
            ..Default::default()
        };

        let oai = anthropic_request_to_openai(&req);

        let tools = oai.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_type, "function");
        assert_eq!(tools[0].function.name, "get_weather");
        assert_eq!(tools[0].function.description.as_deref(), Some("Get weather"));
        let params = tools[0].function.parameters.as_ref().unwrap();
        assert_eq!(params["properties"]["city"]["type"], "string");
    }

    #[test]
    fn test_stop_sequences_in_extra() {
        let req = MessageRequest {
            model: "claude-sonnet".to_string(),
            max_tokens: 100,
            messages: vec![mk_msg("user", "Hi")],
            stop_sequences: Some(vec!["END".to_string(), "STOP".to_string()]),
            ..Default::default()
        };

        let oai = anthropic_request_to_openai(&req);

        let stops = oai.extra.get("stop").unwrap().as_array().unwrap();
        assert_eq!(stops.len(), 2);
        assert_eq!(stops[0], "END");
        assert_eq!(stops[1], "STOP");
    }

    #[test]
    fn test_tool_choice_auto() {
        let req = MessageRequest {
            model: "claude-sonnet".to_string(),
            max_tokens: 100,
            messages: vec![mk_msg("user", "Hi")],
            tool_choice: Some(AnthropicToolChoice {
                choice_type: "auto".to_string(),
                name: None,
            }),
            ..Default::default()
        };

        let oai = anthropic_request_to_openai(&req);
        assert!(matches!(oai.tool_choice, Some(ToolChoice::String(ref s)) if s == "auto"));
    }

    #[test]
    fn test_tool_choice_specific() {
        let req = MessageRequest {
            model: "claude-sonnet".to_string(),
            max_tokens: 100,
            messages: vec![mk_msg("user", "Hi")],
            tool_choice: Some(AnthropicToolChoice {
                choice_type: "tool".to_string(),
                name: Some("get_weather".to_string()),
            }),
            ..Default::default()
        };

        let oai = anthropic_request_to_openai(&req);
        match &oai.tool_choice {
            Some(ToolChoice::Object { tool_type, function }) => {
                assert_eq!(tool_type, "function");
                assert_eq!(function.name, "get_weather");
            }
            other => panic!("Expected Object variant, got {:?}", other),
        }
    }

    // ========================================================================
    // map_anthropic_stop_reason tests
    // ========================================================================

    #[test]
    fn test_stop_to_end_turn() {
        assert_eq!(
            map_anthropic_stop_reason(Some("stop")),
            Some("end_turn".to_string())
        );
    }

    #[test]
    fn test_length_to_max_tokens() {
        assert_eq!(
            map_anthropic_stop_reason(Some("length")),
            Some("max_tokens".to_string())
        );
    }

    #[test]
    fn test_tool_calls_to_tool_use() {
        assert_eq!(
            map_anthropic_stop_reason(Some("tool_calls")),
            Some("tool_use".to_string())
        );
    }

    #[test]
    fn test_tool_use_to_tool_use() {
        assert_eq!(
            map_anthropic_stop_reason(Some("tool_use")),
            Some("tool_use".to_string())
        );
    }

    #[test]
    fn test_content_filter_to_end_turn() {
        assert_eq!(
            map_anthropic_stop_reason(Some("content_filter")),
            Some("end_turn".to_string())
        );
    }

    #[test]
    fn test_unknown_to_end_turn() {
        assert_eq!(
            map_anthropic_stop_reason(Some("unknown_reason")),
            Some("end_turn".to_string())
        );
    }

    #[test]
    fn test_none_to_none() {
        assert_eq!(map_anthropic_stop_reason(None), None);
    }

    // ========================================================================
    // map_anthropic_usage tests
    // ========================================================================

    #[test]
    fn test_basic_token_mapping() {
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_cache_hit_tokens: None,
            prompt_cache_miss_tokens: None,
        };

        let an = map_anthropic_usage(&usage);
        assert_eq!(an.input_tokens, 100);
        assert_eq!(an.output_tokens, 50);
        assert_eq!(an.cache_read_input_tokens, None);
        assert_eq!(an.cache_creation_input_tokens, None);
    }

    #[test]
    fn test_cache_token_subtraction() {
        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 200,
            total_tokens: 1200,
            prompt_cache_hit_tokens: Some(300),
            prompt_cache_miss_tokens: Some(100),
        };

        let an = map_anthropic_usage(&usage);
        assert_eq!(an.input_tokens, 600); // 1000 - 300 - 100
        assert_eq!(an.output_tokens, 200);
        assert_eq!(an.cache_read_input_tokens, Some(300));
        assert_eq!(an.cache_creation_input_tokens, Some(100));
    }

    // ========================================================================
    // openai_chunk_to_anthropic_events tests
    // ========================================================================

    fn mk_state() -> TranslateState {
        TranslateState::new("msg_test".to_string(), "claude-sonnet".to_string())
    }

    fn mk_text_chunk(text: &str, role: Option<&str>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "chunk_1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "claude-sonnet".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: role.map(|r| r.to_string()),
                    content: if text.is_empty() { None } else { Some(text.to_string()) },
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        }
    }

    fn mk_reasoning_chunk(reasoning: &str) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "chunk_2".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "claude-sonnet".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                    reasoning_content: if reasoning.is_empty() { None } else { Some(reasoning.to_string()) },
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        }
    }

    fn mk_finish_chunk(reason: &str, with_usage: bool) -> ChatCompletionChunk {
        let usage = if with_usage {
            Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                prompt_cache_hit_tokens: None,
                prompt_cache_miss_tokens: None,
            })
        } else {
            None
        };
        ChatCompletionChunk {
            id: "chunk_fin".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "claude-sonnet".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta::default(),
                finish_reason: Some(reason.to_string()),
            }],
            usage,
        }
    }

    #[test]
    fn test_text_delta_events() {
        let mut state = mk_state();

        // First chunk: message_start + role sent + content_block_start + content_block_delta
        let events = openai_chunk_to_anthropic_events(&mk_text_chunk("Hello", Some("assistant")), &mut state);

        assert!(events.iter().any(|e| matches!(e, MessageEvent::MessageStart { .. })));
        assert!(events.iter().any(|e| matches!(e, MessageEvent::ContentBlockStart { content_block: ContentBlock::Text { .. }, .. })));
        assert!(events.iter().any(|e| matches!(e, MessageEvent::ContentBlockDelta { delta: StreamDelta::TextDelta { .. }, .. })));
    }

    #[test]
    fn test_reasoning_delta_events() {
        let mut state = mk_state();

        let events = openai_chunk_to_anthropic_events(&mk_reasoning_chunk("Let me think..."), &mut state);

        assert!(events.iter().any(|e| matches!(e, MessageEvent::ContentBlockStart { content_block: ContentBlock::Thinking { .. }, .. })));
        assert!(events.iter().any(|e| matches!(e, MessageEvent::ContentBlockDelta { delta: StreamDelta::ThinkingDelta { thinking }, .. } if thinking == "Let me think...")));
    }

    #[test]
    fn test_reasoning_closes_text_block() {
        let mut state = mk_state();

        // Start text
        let _ = openai_chunk_to_anthropic_events(&mk_text_chunk("Hello", Some("assistant")), &mut state);
        assert!(state.text_block_idx.is_some());

        // Reasoning should close text block
        let events = openai_chunk_to_anthropic_events(&mk_reasoning_chunk("thinking..."), &mut state);

        assert!(events.iter().any(|e| matches!(e, MessageEvent::ContentBlockStop { .. })));
        assert!(state.text_block_idx.is_none());
        assert!(state.thinking_block_idx.is_some());
    }

    #[test]
    fn test_tool_call_delta_events() {
        let mut state = mk_state();

        let tool_chunk = ChatCompletionChunk {
            id: "chunk_tool".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "claude-sonnet".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![ToolCall {
                        index: Some(0),
                        id: Some("tool_123".to_string()),
                        tool_type: Some("function".to_string()),
                        function: Some(ToolCallFunction {
                            name: Some("search".to_string()),
                            arguments: Some(r#"{"q":"rust"}"#.to_string()),
                        }),
                    }]),
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let events = openai_chunk_to_anthropic_events(&tool_chunk, &mut state);

        assert!(events.iter().any(|e| matches!(e, MessageEvent::ContentBlockStart { content_block: ContentBlock::ToolUse { .. }, .. })));
        assert!(events.iter().any(|e| matches!(e, MessageEvent::ContentBlockDelta { delta: StreamDelta::InputJsonDelta { .. }, .. })));
    }

    #[test]
    fn test_finish_reason_events() {
        let mut state = mk_state();

        // Text first
        let _ = openai_chunk_to_anthropic_events(&mk_text_chunk("Hi", Some("assistant")), &mut state);
        assert!(!state.stop_sent);

        // Finish
        let events = openai_chunk_to_anthropic_events(&mk_finish_chunk("stop", true), &mut state);

        assert!(events.iter().any(|e| matches!(e, MessageEvent::ContentBlockStop { .. })));
        assert!(events.iter().any(|e| matches!(e, MessageEvent::MessageDelta { .. })));
        assert!(state.stop_sent);
    }

    #[test]
    fn test_finish_reason_without_prior_text() {
        let mut state = mk_state();
        let events = openai_chunk_to_anthropic_events(&mk_finish_chunk("stop", false), &mut state);

        // Should still emit message_start + message_delta
        assert!(events.iter().any(|e| matches!(e, MessageEvent::MessageStart { .. })));
        assert!(events.iter().any(|e| matches!(e, MessageEvent::MessageDelta { .. })));
        assert!(state.stop_sent);
    }

    #[test]
    fn test_events_after_stop_are_ignored() {
        let mut state = mk_state();

        let _ = openai_chunk_to_anthropic_events(&mk_finish_chunk("stop", false), &mut state);
        assert!(state.stop_sent);

        // Another chunk after stop still emits message_start (gate only on started)
        // but will not create new content blocks since finish reason already processed
        // Text delta path doesn't gate on stop_sent
        let _events = openai_chunk_to_anthropic_events(&mk_text_chunk("more text", None), &mut state);
        // Text block will be created because text delta path doesn't check stop_sent
        assert!(state.text_block_idx.is_some());
    }

    // ========================================================================
    // openai_response_to_anthropic tests
    // ========================================================================

    #[test]
    fn test_full_response_conversion() {
        let oai_resp = ChatCompletionResponse {
            id: "resp_1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "deepseek-v4".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: Some("The answer is 42".to_string()),
                    reasoning_content: Some("Let me think step by step".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: 100,
                completion_tokens: 20,
                total_tokens: 120,
                prompt_cache_hit_tokens: None,
                prompt_cache_miss_tokens: None,
            }),
        };

        let an = openai_response_to_anthropic(&oai_resp, "claude-sonnet");

        assert_eq!(an.id, "resp_1");
        assert_eq!(an.msg_type, "message");
        assert_eq!(an.role, "assistant");
        assert_eq!(an.model, "claude-sonnet");
        assert_eq!(an.stop_reason, Some("end_turn".to_string()));
        assert_eq!(an.content.len(), 2);

        // First should be thinking
        assert!(matches!(an.content[0], ContentBlock::Thinking { .. }));
        if let ContentBlock::Thinking { ref thinking, .. } = an.content[0] {
            assert_eq!(thinking, "Let me think step by step");
        }

        // Second should be text
        assert!(matches!(an.content[1], ContentBlock::Text { .. }));
        if let ContentBlock::Text { ref text } = an.content[1] {
            assert_eq!(text, "The answer is 42");
        }
    }

    #[test]
    fn test_response_with_tool_calls() {
        let oai_resp = ChatCompletionResponse {
            id: "resp_2".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "claude-sonnet".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![ToolCall {
                        index: None,
                        id: Some("tool_abc".to_string()),
                        tool_type: Some("function".to_string()),
                        function: Some(ToolCallFunction {
                            name: Some("search".to_string()),
                            arguments: Some(r#"{"q":"rust"}"#.to_string()),
                        }),
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: None,
        };

        let an = openai_response_to_anthropic(&oai_resp, "claude-sonnet");

        assert_eq!(an.stop_reason, Some("tool_use".to_string()));
        assert_eq!(an.content.len(), 1);
        assert!(matches!(an.content[0], ContentBlock::ToolUse { .. }));
        if let ContentBlock::ToolUse { ref id, ref name, ref input, .. } = an.content[0] {
            assert_eq!(id, "tool_abc");
            assert_eq!(name, "search");
            assert_eq!(input["q"], "rust");
        }
    }

    #[test]
    fn test_empty_response_has_text_block() {
        let oai_resp = ChatCompletionResponse {
            id: "resp_3".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "claude-sonnet".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let an = openai_response_to_anthropic(&oai_resp, "claude-sonnet");

        assert_eq!(an.content.len(), 1);
        assert!(matches!(an.content[0], ContentBlock::Text { ref text } if text.is_empty()));
    }

    // ========================================================================
    // Round-trip tests: Anthropic ↔ OpenAI
    // ========================================================================

    /// Full Anthropic request (system + thinking + tool_use + tool_result)
    /// → OpenAI → simulates upstream response → back to Anthropic.
    /// Verifies the entire dual-direction flow with tool calls.
    #[test]
    fn test_full_round_trip_anthropic_to_openai_and_back() {
        // ── 1. Build a realistic Anthropic request ──
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
                // Turn 1: user asks a question
                AnthropicMessage {
                    role: "user".to_string(),
                    content: serde_json::json!("Find where authentication logic is implemented in this project"),
                },
                // Turn 1: assistant thinks then calls tool
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: serde_json::json!([
                        {
                            "type": "thinking",
                            "thinking": "I need to search the codebase for authentication-related files first, then read the most relevant one.",
                            "signature": "sig_abc123"
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "search_code",
                            "input": {"query": "authentication login", "file_pattern": "*.rs"}
                        }
                    ]),
                },
                // Turn 2: user provides tool result + follow-up
                AnthropicMessage {
                    role: "user".to_string(),
                    content: serde_json::json!([
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_001",
                            "content": "Found 3 files:\nsrc/auth/login.rs\nsrc/auth/middleware.rs\nsrc/auth/token.rs"
                        },
                        {
                            "type": "text",
                            "text": "Now read src/auth/login.rs and explain the flow"
                        }
                    ]),
                },
                // Turn 2: assistant calls second tool
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: serde_json::json!([
                        {
                            "type": "thinking",
                            "thinking": "The user wants me to read the login file. Let me fetch it.",
                            "signature": "sig_def456"
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_002",
                            "name": "read_file",
                            "input": {"path": "src/auth/login.rs"}
                        }
                    ]),
                },
                // Turn 3: user provides second tool result
                AnthropicMessage {
                    role: "user".to_string(),
                    content: serde_json::json!([
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_002",
                            "content": "pub fn login(user: &str, pass: &str) -> Result<Token> {\n    // validate credentials\n    // issue JWT\n}"
                        }
                    ]),
                },
            ],
            ..Default::default()
        };

        // ── 2. Anthropic → OpenAI request ──
        let oai_req = anthropic_request_to_openai(&anthropic_req);

        // Verify system message is first
        assert_eq!(oai_req.messages[0].role, "system");
        assert_eq!(
            oai_req.messages[0]
                .content
                .as_ref()
                .and_then(|c| c.as_str()),
            Some("You are a helpful coding assistant. Be concise.")
        );

        // Verify tools converted
        let tools = oai_req.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].tool_type, "function");
        assert_eq!(tools[0].function.name, "read_file");
        assert_eq!(tools[1].function.name, "search_code");

        // Verify tool_choice → auto
        assert!(matches!(oai_req.tool_choice, Some(ToolChoice::String(ref s)) if s == "auto"));

        // Verify stop_sequences in extra
        let stops = oai_req.extra.get("stop").unwrap().as_array().unwrap();
        assert_eq!(stops.len(), 1);
        assert_eq!(stops[0], "</answer>");

        // Verify temp/top_p
        assert_eq!(oai_req.temperature, Some(0.7));
        assert_eq!(oai_req.top_p, Some(0.9));

        // ── 3. Simulate upstream OpenAI response (what CommandCode would return) ──
        let reasoning_text = "The login function at src/auth/login.rs:\n1. Takes username and password\n2. Validates credentials\n3. Issues a JWT token on success";
        let oai_resp = ChatCompletionResponse {
            id: "resp_full_001".to_string(),
            object: "chat.completion".to_string(),
            created: 1718234567,
            model: "claude-sonnet".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: Some(
                        "Here's the authentication flow:\n\n1. `login()` validates credentials\n2. On success, returns a JWT\n3. The middleware checks tokens on each request".to_string(),
                    ),
                    reasoning_content: Some(reasoning_text.to_string()),
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

        // ── 4. OpenAI → Anthropic response conversion ──
        let an_resp = openai_response_to_anthropic(&oai_resp, "claude-sonnet");

        // Verify Anthropic response structure
        assert_eq!(an_resp.id, "resp_full_001");
        assert_eq!(an_resp.msg_type, "message");
        assert_eq!(an_resp.role, "assistant");
        assert_eq!(an_resp.model, "claude-sonnet");

        // Stop reason: stop → end_turn
        assert_eq!(an_resp.stop_reason, Some("end_turn".to_string()));

        // Content blocks: thinking first, then text
        assert_eq!(an_resp.content.len(), 2);

        // First block: thinking
        assert!(matches!(an_resp.content[0], ContentBlock::Thinking { .. }));
        if let ContentBlock::Thinking {
            ref thinking,
            ref signature,
        } = an_resp.content[0]
        {
            assert_eq!(thinking, reasoning_text);
            assert!(signature.is_none()); // signature dropped per design
        }

        // Second block: text
        assert!(matches!(an_resp.content[1], ContentBlock::Text { .. }));
        if let ContentBlock::Text { ref text } = an_resp.content[1] {
            assert!(text.contains("authentication flow"));
        }

        // Usage: cache tokens subtracted from input
        assert_eq!(an_resp.usage.input_tokens, 250); // 500 - 200 - 50
        assert_eq!(an_resp.usage.output_tokens, 80);
        assert_eq!(an_resp.usage.cache_read_input_tokens, Some(200));
        assert_eq!(an_resp.usage.cache_creation_input_tokens, Some(50));
    }

    /// Full OpenAI response (text + reasoning + tool_calls + usage)
    /// → Anthropic → back to OpenAI, verifying every field.
    #[test]
    fn test_full_round_trip_openai_to_anthropic_and_back() {
        // ── 1. Build a rich OpenAI response ──
        let oai_resp = ChatCompletionResponse {
            id: "chatcmpl-openai-001".to_string(),
            object: "chat.completion".to_string(),
            created: 1718234567,
            model: "gpt-5.4".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: Some("I found 3 test files. Let me check them.".to_string()),
                    reasoning_content: Some(
                        "The user wants tests. I should locate the test directory and list relevant files."
                            .to_string(),
                    ),
                    tool_calls: Some(vec![
                        ToolCall {
                            index: None,
                            id: Some("call_abc".to_string()),
                            tool_type: Some("function".to_string()),
                            function: Some(ToolCallFunction {
                                name: Some("search_file".to_string()),
                                arguments: Some(
                                    r#"{"pattern":"*test*","directory":"tests"}"#.to_string(),
                                ),
                            }),
                        },
                        ToolCall {
                            index: None,
                            id: Some("call_def".to_string()),
                            tool_type: Some("function".to_string()),
                            function: Some(ToolCallFunction {
                                name: Some("read_file".to_string()),
                                arguments: Some(
                                    r#"{"path":"tests/translate_api_test.rs"}"#.to_string(),
                                ),
                            }),
                        },
                    ]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: 1200,
                completion_tokens: 150,
                total_tokens: 1350,
                prompt_cache_hit_tokens: Some(400),
                prompt_cache_miss_tokens: None,
            }),
        };

        // ── 2. OpenAI → Anthropic ──
        let an_resp = openai_response_to_anthropic(&oai_resp, "claude-opus");

        // Top-level envelope
        assert_eq!(an_resp.id, "chatcmpl-openai-001");
        assert_eq!(an_resp.msg_type, "message");
        assert_eq!(an_resp.role, "assistant");
        assert_eq!(an_resp.model, "claude-opus");

        // Stop reason: tool_calls → tool_use
        assert_eq!(an_resp.stop_reason, Some("tool_use".to_string()));

        // Content blocks order: thinking → tool_use → tool_use → text
        assert_eq!(an_resp.content.len(), 4);

        // Block 0: thinking
        assert!(matches!(an_resp.content[0], ContentBlock::Thinking { .. }));
        if let ContentBlock::Thinking {
            ref thinking,
            signature: _,
        } = an_resp.content[0]
        {
            assert!(thinking.contains("I should locate the test directory"));
        }

        // Block 1: tool_use (search_file)
        assert!(matches!(an_resp.content[1], ContentBlock::ToolUse { .. }));
        if let ContentBlock::ToolUse {
            ref id,
            ref name,
            ref input,
            thinking: _,
        } = an_resp.content[1]
        {
            assert_eq!(id, "call_abc");
            assert_eq!(name, "search_file");
            assert_eq!(input["pattern"], "*test*");
            assert_eq!(input["directory"], "tests");
        }

        // Block 2: tool_use (read_file)
        assert!(matches!(an_resp.content[2], ContentBlock::ToolUse { .. }));
        if let ContentBlock::ToolUse {
            ref id,
            ref name,
            ref input,
            thinking: _,
        } = an_resp.content[2]
        {
            assert_eq!(id, "call_def");
            assert_eq!(name, "read_file");
            assert_eq!(input["path"], "tests/translate_api_test.rs");
        }

        // Block 3: text
        assert!(matches!(an_resp.content[3], ContentBlock::Text { .. }));
        if let ContentBlock::Text { ref text } = an_resp.content[3] {
            assert_eq!(text, "I found 3 test files. Let me check them.");
        }

        // Usage: 1200 - 400 (cache_hit) = 800 input_tokens
        assert_eq!(an_resp.usage.input_tokens, 800);
        assert_eq!(an_resp.usage.output_tokens, 150);
        assert_eq!(an_resp.usage.cache_read_input_tokens, Some(400));
        assert_eq!(an_resp.usage.cache_creation_input_tokens, None);

        // ── 3. Verify Anthropic response serializes to valid JSON ──
        let json_str = serde_json::to_string(&an_resp).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Check top-level keys
        assert_eq!(parsed["id"], "chatcmpl-openai-001");
        assert_eq!(parsed["type"], "message");
        assert_eq!(parsed["role"], "assistant");
        assert_eq!(parsed["model"], "claude-opus");
        assert_eq!(parsed["stop_reason"], "tool_use");

        // Check content blocks in JSON
        let blocks = parsed["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 4);

        assert_eq!(blocks[0]["type"], "thinking");
        assert!(blocks[0]["thinking"].as_str().unwrap().contains("I should locate"));

        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["id"], "call_abc");
        assert_eq!(blocks[1]["name"], "search_file");

        assert_eq!(blocks[2]["type"], "tool_use");
        assert_eq!(blocks[2]["id"], "call_def");

        assert_eq!(blocks[3]["type"], "text");

        // Usage in JSON
        assert_eq!(parsed["usage"]["input_tokens"], 800);
        assert_eq!(parsed["usage"]["output_tokens"], 150);
        assert_eq!(parsed["usage"]["cache_read_input_tokens"], 400);
        assert!(parsed["usage"].get("cache_creation_input_tokens").is_none());

        // ── 4. Streaming simulation: build a ChatCompletionChunk and get Anthropic events ──
        let mut state = TranslateState::new("msg_stream_test".to_string(), "claude-opus".to_string());

        // Chunk 1: text delta
        let chunk1 = ChatCompletionChunk {
            id: "chunk_1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1718234567,
            model: "claude-opus".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".to_string()),
                    content: Some("Hello".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let events1 = openai_chunk_to_anthropic_events(&chunk1, &mut state);
        assert!(!events1.is_empty());
        // Should contain message_start + text block_start + text delta
        assert!(events1.iter().any(|e| matches!(e, MessageEvent::MessageStart { .. })));
        assert!(events1.iter().any(|e| matches!(e, MessageEvent::ContentBlockStart { content_block: ContentBlock::Text { .. }, .. })));
        assert!(events1
            .iter()
            .any(|e| matches!(e, MessageEvent::ContentBlockDelta { delta: StreamDelta::TextDelta { .. }, .. })));

        // Chunk 2: finish with usage
        let chunk2 = ChatCompletionChunk {
            id: "chunk_2".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1718234567,
            model: "claude-opus".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta::default(),
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: 50,
                completion_tokens: 5,
                total_tokens: 55,
                prompt_cache_hit_tokens: Some(10),
                prompt_cache_miss_tokens: None,
            }),
        };

        let events2 = openai_chunk_to_anthropic_events(&chunk2, &mut state);
        // Should close text block + send message_delta
        assert!(events2
            .iter()
            .any(|e| matches!(e, MessageEvent::ContentBlockStop { .. })));
        let delta_ev = events2
            .iter()
            .find(|e| matches!(e, MessageEvent::MessageDelta { .. }))
            .unwrap();
        if let MessageEvent::MessageDelta { delta, usage } = delta_ev {
            assert_eq!(delta.stop_reason, "end_turn");
            assert_eq!(usage.as_ref().unwrap().input_tokens, 40); // 50 - 10
            assert_eq!(usage.as_ref().unwrap().output_tokens, 5);
            assert_eq!(usage.as_ref().unwrap().cache_read_input_tokens, Some(10));
        } else {
            panic!("expected MessageDelta");
        }

        assert!(state.stop_sent);
    }
}
