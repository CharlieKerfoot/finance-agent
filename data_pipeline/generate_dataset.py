# /// script
# dependencies = [
#  "openai",
#  "tqdm",
#  "python-dotenv"
# ]
# ///

import json
import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = "dataset/finance_chunks.jsonl"
OUTPUT_FILE = "dataset/train_data.jsonl"

MAX_SAMPLES = 50

client = OpenAI(api_key=os.getenv("API_KEY"))

SYSTEM_PROMPT = """
You are an expert financial analyst creating a training dataset for a junior analyst AI.
Your goal is to create ONE high-quality Question-Answer pair based STRICTLY on the provided text chunk.

Rules:
1. The Question must be specific and require reasoning (not just keyword lookup).
2. The Answer must be detailed and cite the logic from the text.
3. If the text contains a Table (marked with | pipes), ask a question that requires comparing two numbers (e.g. "How much did revenue grow 2022 vs 2023?").
4. Output JSON format: {"instruction": "The Question", "output": "The Answer"}
"""

def generate_qa(chunk_text, meta):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"SOURCE ({meta['ticker']} {meta['section']}): \n\n{chunk_text}"}
            ],
            response_format={"type":"json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error generating Q&A: {e}")
        return None

if __name__ == "__main__":
    with open(INPUT_FILE, "r")as f:
        chunks = [json.loads(line) for line in f]

    chunks = [c for c in chunks if c['token_count'] > 200]

    selected_chunks = chunks[:MAX_SAMPLES]

    print(f"Generating synthetic data for {len(selected_chunks)} chunks...")

    dataset = []

    for chunk in tqdm(selected_chunks):
        qa_pair = generate_qa(chunk['text'], chunk['meta'])

        if qa_pair:
            entry = {
                "instruction": qa_pair['instruction'],
                "input": chunk['text'],
                "output": qa_pair['output']
            }
            dataset.append(entry)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Done. Generated {len(dataset)} training examples in {OUTPUT_FILE}")
