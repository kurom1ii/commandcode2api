#[cfg(test)]
mod stream_usage_test {
    use colored::Colorize;
    use serde_json::Value;

    fn api_base() -> String {
        std::env::var("TEST_API_BASE").unwrap_or_else(|_| "http://localhost:3000".to_string())
    }

    const API_KEY: &str = "user_3J26hW7aZbpjveq2cb83jZ2Rxz5AhSzbKWE6K8CYpQE3JpdZoqEp55GAddY8wg3i71N6oF54ispzRNwvLeTZCUZC";

    fn api_key() -> String {
        std::env::var("COMMANDCODE_API_KEY").unwrap_or_else(|_| API_KEY.to_string())
    }

    async fn stream_and_collect(model: &str) -> (Value, Vec<String>, Option<Value>) {
        let client = reqwest::Client::new();
        let body = serde_json::json!({
            "model": model,
            "max_tokens": 64000,
            "messages": [{"role": "user", "content": "Nói xin chào "}],
            "stream": true
        });

        let resp = client
            .post(format!("{}/v1/messages", api_base()))
            .header("Authorization", format!("Bearer {}", api_key()))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&body)
            .send()
            .await
            .expect("Failed to send request");

        assert!(
            resp.status().is_success(),
            "HTTP {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        );

        let text = resp.text().await.expect("Failed to read response body");
        let mut events: Vec<String> = Vec::new();
        let mut usage: Option<Value> = None;

        // Parse SSE events
        let mut current_event: Option<String> = None;
        for line in text.lines() {
            let line = line.trim().to_string();
            if line.starts_with("event:") {
                current_event = Some(line[6..].trim().to_string());
            } else if line.starts_with("data:") {
                let data = &line[5..].trim();
                if let Ok(parsed) = serde_json::from_str::<Value>(data) {
                    let ev_type = parsed["type"].as_str().unwrap_or("?");
                    let ev_name = current_event.as_deref().unwrap_or("?");
                    events.push(format!("{} {}", format!("[{}]", ev_name).yellow(), ev_type));

                    // Check for usage in message_delta
                    if ev_type == "message_delta" {
                        if let Some(u) = parsed.get("usage") {
                            usage = Some(u.clone());
                        }
                    }
                }
                current_event = None;
            }
        }

        (body, events, usage)
    }

    async fn print_test(model: &str) -> (bool, bool) {
        println!();
        println!(
            "{} {}",
            "══════════════════════════════════════════════════".bright_black(),
            model.cyan().bold()
        );

        println!("\n{}", "📤 REQUEST:".green().bold());
        let (body, events, usage) = stream_and_collect(model).await;
        println!("{}", serde_json::to_string_pretty(&body).unwrap().blue());

        println!("\n{}", "📥 SSE EVENTS:".green().bold());
        for ev in &events {
            println!("  {}", ev);
        }

        println!("\n{}", "📊 USAGE CHECK:".green().bold());
        let has_usage = usage.is_some();
        let has_input_tokens = usage
            .as_ref()
            .and_then(|u| u.get("input_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v > 0)
            .unwrap_or(false);

        if has_usage {
            println!(
                "  {} {}",
                "usage present:".white(),
                "YES".green().bold()
            );
            let pretty = serde_json::to_string_pretty(usage.as_ref().unwrap()).unwrap();
            println!("  {}", pretty.yellow());
        } else {
            println!("  {} {}", "usage present:".white(), "NO".red().bold());
        }

        if has_input_tokens {
            let val = usage
                .as_ref()
                .and_then(|u| u.get("input_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap();
            println!(
                "  {} {} (value: {})",
                "input_tokens > 0:".white(),
                "YES".green().bold(),
                val.to_string().cyan()
            );
        } else {
            println!(
                "  {} {}",
                "input_tokens > 0:".white(),
                "NO".red().bold()
            );
        }

        println!();
        (has_usage, has_input_tokens)
    }

    #[tokio::test]
    async fn test_stream_usage_all_models() {
        let models = vec![
            "deepseek/deepseek-v4-pro",
            "deepseek/deepseek-v4-flash",
        ];

        println!();
        println!(
            "{}  API: {}",
            "🔬 STREAM USAGE TEST".bold().white().on_blue(),
            api_base().yellow()
        );
        println!();

        let mut all_pass = true;
        for model in models {
            let (has_usage, has_input) = print_test(model).await;

            if !has_usage {
                all_pass = false;
            }
            if !has_input {
                all_pass = false;
            }
        }

        println!(
            "{} {}",
            "══════════════════════════════════════════════════".bright_black(),
            if all_pass {
                "✅ ALL PASS".green().bold()
            } else {
                "❌ SOME FAILED".red().bold()
            }
        );
        println!();

        assert!(all_pass, "Một hoặc nhiều model không trả về usage.input_tokens trong stream!");
    }
}
