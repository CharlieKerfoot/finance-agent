# /// script
# dependencies = [
#  "tiktoken",
#  "tqdm"
# ]
# ///

import json
import os
import tiktoken
from tqdm import tqdm

INPUT_DIR = "sec_cleaned_data"
OUTPUT_FILE = "dataset/finance_chunks.jsonl"

ENC = tiktoken.get_encoding("o200k_base")

TARGET_CHUNK_SIZE = 800
OVERLAP = 100 

def count_tokens(text):
    return len(ENC.encode(text))

def chunk_text(text, source_meta):
    """
    Splits text into semantic chunks based on paragraphs
    """
    paragraphs = text.split('\n\n')

    current_chunk = []
    current_length = 0
    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = count_tokens(para)

        if current_length + para_tokens > TARGET_CHUNK_SIZE and current_chunk:
            chunk_text = "\n\n".join(current_chunk)

            chunks.append({
                "text": chunk_text,
                "meta": source_meta,
                "token_count": current_length
            })

            if count_tokens(current_chunk[-1]) < TARGET_CHUNK_SIZE:
                current_chunk = [current_chunk[-1]]
                current_length = count_tokens(current_chunk[-1])
            else:
                current_chunk = []
                current_length = 0

        current_chunk.append(para)
        current_length += para_tokens

    if current_chunk:
        chunks.append({
            "text": "\n\n".join(current_chunk),
            "meta": source_meta,
            "token_count": current_length
        })

    return chunks

if __name__ == "__main__":
    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    all_chunks = []
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]

    print(f"Chunking {len(files)} files...")

    for filename in tqdm(files):
        with open(f"{INPUT_DIR}/{filename}", "r", encoding="utf-8") as f:
            content = f.read()

        ticker = filename.split("_")[0] # Filename format: AAPL_10K_2023-09-30_cleaned.txt

        sections = content.split("--- ")

        for section in sections[1:]:
            header_end = section.find(" ---")
            section_name = section[:header_end]
            section_body = section[header_end+4:].strip()

            meta = {"ticker": ticker, "section": section_name, "source": filename}

            file_chunks = chunk_text(section_body, meta)
            all_chunks.extend(file_chunks)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"Saved to {OUTPUT_FILE}")
