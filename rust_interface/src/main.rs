use axum::{Json, Router, extract::State, response::IntoResponse, routing::post};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::task;

mod model;
use model::TextGeneration;

struct AppState {
    engine: Mutex<TextGeneration>,
}

#[derive(Deserialize)]
struct InferenceRequest {
    query: String,
    context: String,
}

#[derive(Serialize)]
struct InferenceResponse {
    answer: String,
    reasoning_time: f64,
    status: String,
}

async fn generate(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<InferenceRequest>,
) -> impl IntoResponse {
    let start_time = std::time::Instant::now();

    let prompt = format!(
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n",
        payload.query, payload.context
    );

    let state_clone = state.clone();

    let prediction = task::spawn_blocking(move || {
        let mut engine = state_clone.engine.lock().unwrap();
        engine.run(&prompt, 256)
    })
    .await
    .unwrap();

    let duration = start_time.elapsed().as_secs_f64();

    match prediction {
        Ok(answer) => Json(InferenceResponse {
            answer,
            reasoning_time: duration,
            status: "success".to_string(),
        }),
        Err(e) => Json(InferenceResponse {
            answer: format!("Error: {}", e),
            reasoning_time: duration,
            status: "error".to_string(),
        }),
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let model_path = "arbagent-q4.gguf";

    println!("Initializing Inference Engine...");

    let engine = match TextGeneration::new(model_path) {
        Ok(e) => e,
        Err(e) => panic!("Failed to load model: {}", e),
    };

    let app = Router::new()
        .route("/generate", post(generate))
        .with_state(Arc::new(AppState {
            engine: Mutex::new(engine),
        }));

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
