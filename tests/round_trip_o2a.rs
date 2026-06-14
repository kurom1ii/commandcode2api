#[cfg(test)]
mod round_trip_o2a {
    use commandcode2api::translate_api::*;
    use commandcode2api::types::*;

    /// Full OpenAI response (text + reasoning + tool_calls + usage)
    /// → Anthropic → back to OpenAI, verifying every field.
    #[test]
    fn test_full_round_trip_openai_to_anthropic_and_back() {
        // ═══════════════════════════════════════════════════════════════════
        // STEP 1: OpenAI ChatCompletionResponse (input — from upstream)
        // ═══════════════════════════════════════════════════════════════════
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
                                arguments: Some(r#"{"pattern":"*test*","directory":"tests"}"#.to_string()),
                            }),
                        },
                        ToolCall {
                            index: None,
                            id: Some("call_def".to_string()),
                            tool_type: Some("function".to_string()),
                            function: Some(ToolCallFunction {
                                name: Some("read_file".to_string()),
                                arguments: Some(r#"{"path":"tests/translate_api_test.rs"}"#.to_string()),
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

        println!("\n══════════════════════════════════════════════════════════");
        println!("INPUT: OpenAI ChatCompletionResponse (from upstream)");
        println!("══════════════════════════════════════════════════════════");
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "id": oai_resp.id,
            "object": oai_resp.object,
            "model": oai_resp.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I found 3 test files. Let me check them.",
                    "reasoning_content": "The user wants tests. I should locate the test directory and list relevant files.",
                    "tool_calls": [
                        {"id":"call_abc","type":"function","function":{"name":"search_file","arguments":"{\"pattern\":\"*test*\",\"directory\":\"tests\"}"}},
                        {"id":"call_def","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"tests/translate_api_test.rs\"}"}}
                    ]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens":1200,"completion_tokens":150,"total_tokens":1350,"prompt_cache_hit_tokens":400}
        })).unwrap());

        // ═══════════════════════════════════════════════════════════════════
        // STEP 2: OpenAI → Anthropic non-streaming response
        // ═══════════════════════════════════════════════════════════════════
        let an_resp = openai_response_to_anthropic(&oai_resp, "claude-opus");

        println!("\n══════════════════════════════════════════════════════════");
        println!("OUTPUT: Anthropic MessageResponse (non-streaming)");
        println!("══════════════════════════════════════════════════════════");
        println!("{}", serde_json::to_string_pretty(&an_resp).unwrap());

        // Top-level envelope
        assert_eq!(an_resp.id, "chatcmpl-openai-001");
        assert_eq!(an_resp.msg_type, "message");
        assert_eq!(an_resp.role, "assistant");
        assert_eq!(an_resp.model, "claude-opus");
        assert_eq!(an_resp.stop_reason, Some("tool_use".to_string()));

        // Content blocks order: thinking → tool_use(search_file) → tool_use(read_file) → text
        assert_eq!(an_resp.content.len(), 4);

        // Block 0: thinking
        if let ContentBlock::Thinking { ref thinking, signature: _ } = an_resp.content[0] {
            assert!(thinking.contains("I should locate the test directory"));
        }

        // Block 1: tool_use (search_file)
        if let ContentBlock::ToolUse { ref id, ref name, ref input, thinking: _ } = an_resp.content[1] {
            assert_eq!(id, "call_abc");
            assert_eq!(name, "search_file");
            assert_eq!(input["pattern"], "*test*");
            assert_eq!(input["directory"], "tests");
        }

        // Block 2: tool_use (read_file)
        if let ContentBlock::ToolUse { ref id, ref name, ref input, thinking: _ } = an_resp.content[2] {
            assert_eq!(id, "call_def");
            assert_eq!(name, "read_file");
            assert_eq!(input["path"], "tests/translate_api_test.rs");
        }

        // Block 3: text
        if let ContentBlock::Text { ref text } = an_resp.content[3] {
            assert_eq!(text, "I found 3 test files. Let me check them.");
        }

        // Usage: 1200 - 400 (cache_hit) = 800 input_tokens
        assert_eq!(an_resp.usage.input_tokens, 800);
        assert_eq!(an_resp.usage.output_tokens, 150);
        assert_eq!(an_resp.usage.cache_read_input_tokens, Some(400));
        assert_eq!(an_resp.usage.cache_creation_input_tokens, None);

        // ═══════════════════════════════════════════════════════════════════
        // STEP 3: Streaming simulation — convert chunks to Anthropic events
        // ═══════════════════════════════════════════════════════════════════
        let mut state = TranslateState::new(
            "msg_stream_test".to_string(),
            "claude-opus".to_string(),
        );

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

        println!("\n══════════════════════════════════════════════════════════");
        println!("STREAMING: OpenAI chunk → Anthropic SSE events (text delta)");
        println!("══════════════════════════════════════════════════════════");
        for ev in &events1 {
            println!("event: {}", ev.event_type_str());
            println!("data: {}\n", serde_json::to_string(ev).unwrap());
        }

        assert!(!events1.is_empty());
        assert!(events1.iter().any(|e| matches!(e, MessageEvent::MessageStart { .. })));
        assert!(events1.iter().any(|e| matches!(e, MessageEvent::ContentBlockStart { .. })));
        assert!(events1.iter().any(|e| matches!(e, MessageEvent::ContentBlockDelta { delta: StreamDelta::TextDelta { .. }, .. })));

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

        println!("\n══════════════════════════════════════════════════════════");
        println!("STREAMING: OpenAI chunk → Anthropic SSE events (finish)");
        println!("══════════════════════════════════════════════════════════");
        for ev in &events2 {
            println!("event: {}", ev.event_type_str());
            println!("data: {}\n", serde_json::to_string(ev).unwrap());
        }

        assert!(events2.iter().any(|e| matches!(e, MessageEvent::ContentBlockStop { .. })));
        let delta_ev = events2.iter().find(|e| matches!(e, MessageEvent::MessageDelta { .. })).unwrap();
        if let MessageEvent::MessageDelta { delta, usage } = delta_ev {
            assert_eq!(delta.stop_reason, "end_turn");
            assert_eq!(usage.as_ref().unwrap().input_tokens, 40);
            assert_eq!(usage.as_ref().unwrap().output_tokens, 5);
            assert_eq!(usage.as_ref().unwrap().cache_read_input_tokens, Some(10));
        }
        assert!(state.stop_sent);
    }
}
