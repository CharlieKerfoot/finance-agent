use axum::{Json, Router, extract::State, routing::post};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::task;

mod model;
use model::TextGeneration;

mod qdrant_client;
use qdrant_client::QdrantDb;

struct AppState {
    engine: Mutex<TextGeneration>,
    embedder: Mutex<TextEmbedding>,
    qdrant: QdrantDb,
}

#[derive(Deserialize)]
struct RagRequest {
    query: String,
}

#[derive(Serialize)]
struct RagResponse {
    answer: String,
    context_used: Vec<String>,
    reasoning_time: f64,
}

async fn rag_inference(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RagRequest>,
) -> Json<RagResponse> {
    let start_time = std::time::Instant::now();
    println!("Processing RAG Request: {}", payload.query);

    let embedding = {
        let mut embedder = state.embedder.lock().unwrap();
        match embedder.embed(vec![payload.query.clone()], None) {
            Ok(vecs) => vecs[0].clone(),
            Err(e) => {
                return Json(RagResponse {
                    answer: format!("Embedding Error: {}", e),
                    context_used: vec![],
                    reasoning_time: 0.0,
                });
            }
        }
    };

    let search_results = match state.qdrant.search(embedding, 3).await {
        Ok(results) => results,
        Err(e) => {
            return Json(RagResponse {
                answer: format!("DB Error: {}", e),
                context_used: vec![],
                reasoning_time: 0.0,
            });
        }
    };

    let context_texts: Vec<String> = search_results
        .into_iter()
        .filter_map(|point| {
            point
                .payload
                .get("text")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .collect();

    let full_context = context_texts.join("\n\n---\n\n");

    // Truncate at 8000 characters (for now)
    let safe_context = if full_context.len() > 8000 {
        let end = full_context[..8000].rfind('.').unwrap_or(12000);
        &full_context[..end]
    } else {
        &full_context
    };

    let prompt = format!(
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n",
        payload.query, safe_context
    );

    let state_clone = state.clone();
    let prediction = task::spawn_blocking(move || {
        let mut engine = state_clone.engine.lock().unwrap();
        engine.run(&prompt, 512) // Generate up to 512 tokens
    })
    .await
    .unwrap();

    let duration = start_time.elapsed().as_secs_f64();

    match prediction {
        Ok(answer) => Json(RagResponse {
            answer,
            context_used: context_texts,
            reasoning_time: duration,
        }),
        Err(e) => Json(RagResponse {
            answer: format!("Inference Error: {}", e),
            context_used: vec![],
            reasoning_time: duration,
        }),
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    println!("Initializing Search Engine...");

    let model_path = "arbagent-q4.gguf";
    let engine = match TextGeneration::new(model_path) {
        Ok(e) => e,
        Err(e) => panic!("Failed to load model: {}", e),
    };

    let mut embed_opts = InitOptions::default();
    embed_opts.model_name = EmbeddingModel::BGEBaseENV15;
    embed_opts.show_download_progress = true;
    let embedder = TextEmbedding::try_new(embed_opts).expect("Failed to load Embedder");

    let qdrant = QdrantDb::new_from_env().expect("Failed to connect to Qdrant");

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        embedder: Mutex::new(embedder),
        qdrant,
    });

    let app = Router::new()
        .route("/rag", post(rag_inference))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
