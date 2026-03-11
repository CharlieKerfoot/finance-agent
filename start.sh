#!/bin/bash
set -e

echo "Starting Arb-Agent..."

echo "Starting Qdrant..."
cd /app/qdrant
./qdrant &
QDRANT_PID=$!
trap "kill $QDRANT_PID 2>/dev/null" EXIT
sleep 5

cd /app
echo "Generating vector embeddings..."
uv run data_pipeline/embed.py

echo "Starting Rust inference engine..."
echo "(Press Ctrl+C to stop)"
cd rust_interface
./target/release/rust_interface
