# Finance Agent

RAG-powered financial analyst fine-tuned on SEC 10-K filings. Llama-3-8B + LoRA, served via a Rust inference engine with Qdrant vector search.

## Setup

Requires [UV](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync
```

Add your OpenAI key to `.env`:
```
OPENAI_KEY=sk-...
```

## Data Pipeline

1. Download 10-K filings from SEC EDGAR
```bash
uv run data_pipeline/sec_scraper.py
```
2. Parse HTML into scored financial chunks
```bash
uv run data_pipeline/sec_parser.py
```
3. Generate synthetic QA training pairs
```bash
uv run data_pipeline/generate_dataset.py
```
4. Validate the dataset
```bash
uv run data_pipeline/validate_data.py
```

## Training

Hardware profiles handle batch size, sequence length, and gradient accumulation automatically.

```bash
# RTX 3070 Super (12GB VRAM)
uv run train.py --profile rtx3070

# H100 (80GB VRAM)
uv run train.py --profile h100

# Options
uv run train.py --profile rtx3070 --no-wandb --epochs 3 --lr 1e-4
```

| Parameter | RTX 3070 Super | H100 |
|-----------|---------------|------|
| Sequence length | 4096 | 8192 |
| Batch size | 2 | 8 |
| Gradient accumulation | 8 | 2 |
| Effective batch | 16 | 16 |

## GGUF Export

Merge LoRA weights and quantize for inference:

```bash
uv run export_gguf.py                          # default: q4_k_m
uv run export_gguf.py --quantize q8_0           # higher quality
uv run export_gguf.py --model ckerf/arbagent-llama3-8b-lora
```

## Inference Server

The Rust server provides a RAG endpoint that embeds queries, searches Qdrant, and generates answers.

### Prerequisites

- [Qdrant](https://qdrant.tech/) running on port 6333/6334
- `arbagent-q4.gguf` in `rust_interface/`

### Build & Run

```bash
cd rust_interface
cargo build --release
./target/release/rust_interface
```

Populate the vector store first:

```bash
uv run data_pipeline/embed.py
```

### API

```bash
curl -X POST http://127.0.0.1:3000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What drove Apple revenue growth in 2023?"}'
```

Response:

```json
{
  "answer": "...",
  "context_used": ["chunk1...", "chunk2..."],
  "reasoning_time": 2.4
}
```

## Benchmarking

Evaluates against [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) (150 financial QA samples) using GPT-4o-mini as a judge:

```bash
uv run data_pipeline/benchmark.py
```

Results are saved to `benchmark_results.csv`.

## License

MIT
