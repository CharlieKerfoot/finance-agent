use axum::{
    Json, Router,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;

struct AppState {
    model_loaded: bool,
}

#[derive(Deserialize)]
struct InferenceRequest {
    query: String,
    history: Option<Vec<String>>,
}

#[derive(Serialize)]
struct InferenceResponse {
    answer: String,
    reasoning_time: f64,
    sources: Vec<String>,
}

async fn health_check() -> &'static str {
    "Arb-Agent is working!"
}

async fn generate(Json(payload): Json<InferenceRequest>) -> impl IntoResponse {
    println!("Recieved query: {}", payload.query);

    let mock_response = InferenceResponse {
        answer: "This is a placeholder response from the Rust Engine.".to_string(),
        reasoning_time: 0.05,
        sources: vec!["AAPL 10-K Item 7".to_string()],
    };

    Json(mock_response)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let state = Arc::new(Mutex::new(AppState { model_loaded: true }));

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/generate", post(generate));

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
