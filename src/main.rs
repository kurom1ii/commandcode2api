mod convert;
mod proxy;
mod types;

use std::{env, net::SocketAddr, sync::Arc};

use axum::{
    extract::State,
    routing::{get, post},
    Router,
};
use proxy::{chat_completions_handler, load_models_json, AppState};
use reqwest::Client;
use tower_http::cors::CorsLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "commandcode2api=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let api_base = env::var("COMMANDCODE_API_BASE")
        .unwrap_or_else(|_| "https://api.commandcode.ai".to_string());
    let default_api_key = env::var("COMMANDCODE_API_KEY").ok();
    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);
    let host: String = env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());

    let client = Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(300))
        .pool_idle_timeout(std::time::Duration::from_secs(30))
        .pool_max_idle_per_host(10)
        .tcp_keepalive(std::time::Duration::from_secs(60))
        .build()
        .expect("Failed to build reqwest client");

    let models_json = load_models_json(&client).await;

    let state = Arc::new(AppState {
        client,
        api_base,
        default_api_key,
        models_json,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", host, port)
        .parse()
        .expect("Invalid HOST or PORT");
    tracing::info!("Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> &'static str {
    "ok"
}

async fn list_models(State(state): State<Arc<AppState>>) -> axum::Json<types::ModelList> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let data = state
        .models_json
        .models
        .clone()
        .into_iter()
        .map(|m| types::ModelInfo {
            id: m.id,
            object: "model".to_string(),
            created: now,
            owned_by: m.vendor_label.unwrap_or(m.provider),
        })
        .collect();

    axum::Json(types::ModelList {
        object: "list".to_string(),
        data,
    })
}
