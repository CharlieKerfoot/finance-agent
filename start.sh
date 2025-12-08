#!/bin/bash
echo "Starting Arb-Agent..."
echo ""

echo "Running Qdrant Embedding Database..."
if [ "$(docker ps -qa -f name=arbagent-qdrant)" ]; then
  docker rm -f arbagent-qdrant
fi

docker run -d --name arbagent-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

cd data_pipeline
echo "Generating Vector Embeddings..."
uv run embed.py

cd ../rust_interface
echo "Running Rust Inference Engine..."
echo "(Press Ctrl + C to stop)"

echo "Open another terminal tab"
echo "And try a post request to the rag endpoint:
curl -s -X POST http://localhost:3000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the debt levels and interest rate risks for Apple, Microsoft, and Google?"}'"

cargo run --release

