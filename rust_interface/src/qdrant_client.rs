use anyhow::Result;
use qdrant_client::Qdrant; // The new main entry point
use qdrant_client::qdrant::{ScoredPoint, SearchPoints};

pub struct QdrantDb {
    client: Qdrant,
    collection: String,
}

impl QdrantDb {
    pub fn new_from_env() -> Result<Self> {
        let url = std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6334".into());
        let api_key = std::env::var("QDRANT_API_KEY").ok();
        let collection =
            std::env::var("QDRANT_COLLECTION").unwrap_or_else(|_| "finance_chunks".into());

        let mut builder = Qdrant::from_url(&url);

        if let Some(key) = api_key {
            builder = builder.api_key(key);
        }

        let client = builder.build()?;

        Ok(Self { client, collection })
    }

    pub async fn search(&self, embedding: Vec<f32>, top_k: usize) -> Result<Vec<ScoredPoint>> {
        let search_request = SearchPoints {
            collection_name: self.collection.clone(),
            vector: embedding,
            limit: top_k as u64,
            with_payload: Some(true.into()),
            ..Default::default()
        };

        let response = self.client.search_points(search_request).await?;

        Ok(response.result)
    }
}
