# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "pandas",
#     "tiktoken"
# ]
# ///

import json
import pandas as pd
import tiktoken
import matplotlib.pyplot as plt

INPUT_FILE = "dataset/finance_chunks.jsonl"
ENC = tiktoken.get_encoding("o200k_base")

def analyze_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} not found.")
        return

    print("Loading dataset...")
    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    print(f"Total Rows: {len(df)}")

    # 1. Check Length Distribution
    df['length'] = df['text'].apply(lambda x: len(ENC.encode(x)))
        
    print(f"Average Token Length: {df['length'].mean():.2f}")
    print(f"Min Length: {df['length'].min()}")
    print(f"Max Length: {df['length'].max()}")
    
    # 2. Check for Duplicates
    dupes = df.duplicated(subset=['text']).sum()
    print(f"Duplicate Rows: {dupes}")
    
    # 3. Check Financial Density (Heuristic)
    df['has_numbers'] = df['text'].apply(lambda x: any(char.isdigit() for char in x))
    print(f"Rows with numbers (Financial Data): {df['has_numbers'].sum()} ({df['has_numbers'].mean()*100:.1f}%)")

    # 4. Plot
    plt.hist(df['length'], bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of Chunk/Answer Lengths")
    plt.xlabel("Tokens")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    import os
    analyze_dataset()
