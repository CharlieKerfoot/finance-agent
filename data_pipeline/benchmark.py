# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "openai",
#     "requests",
#     "tqdm",
#     "pandas",
#     "python-dotenv"
# ]
# ///

import requests
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import os
from dotenv import load_dotenv
import time

load_dotenv()

RUST_ENDPOINT = "http://localhost:3000/rag"
OUTPUT_FILE = "benchmark_results.csv"
SAMPLE_SIZE = 20 # Intital small sample (FinanceBench has 150)

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

JUDGE_PROMPT = """
You are an impartial judge evaluating the quality of an AI financial analyst.
You will be given a QUESTION, a GROUND TRUTH answer, and a PREDICTED answer.

Your job is to rate the PREDICTION on a scale of 0 to 1 (Binary Pass/Fail).

CRITERIA FOR PASS (1):
- The prediction contains the correct numbers/facts found in the ground truth.
- The prediction answers the specific question asked.

CRITERIA FOR FAIL (0):
- The prediction contains wrong numbers.
- The prediction hallucinates info not in the context.
- The prediction is vague ("I don't know") when the ground truth has specific numbers.

Output ONLY a JSON object: {"score": 0 or 1, "reason": "Short explanation"}
"""

def grade_answer(question, truth, prediction):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"QUESTION: {question}\n\nGROUND TRUTH: {truth}\n\nPREDICTION: {prediction}"}
            ],
            response_format={ "type": "json_object" }
        )
        import json
        return json.loads(response.choices[0].message.content)
    except:
        return {"score": 0, "reason": "Grading Error"}

def main():
    print("Loading FinanceBench...")
    dataset = load_dataset("PatronusAI/financebench", split="train")
    dataset = dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))

    results = []
    print(f"Starting Evaluation on {SAMPLE_SIZE} questions...")

    for row in tqdm(dataset):
        question = row['question']
        truth = row['answer']

        try:
            start = time.time()
            res = requests.post(RUST_ENDPOINT, json={"query": question}, timeout=60)
            res.raise_for_status()
            data = res.json()
            prediction = data['answer']
            latency = time.time() - start
        except Exception as e:
            prediction = f"Error: {e}"
            latency = 0
            continue

        grade = grade_answer(question, truth, prediction)

        results.append({
            "question": question,
            "ground_truth": truth,
            "prediction": prediction,
            "score": grade['score'],
            "reason": grade['reason'],
            "latency": latency,
            "context_used": len(data.get('context_used', [])) if 'data' in locals() else 0
        })

    df = pd.DataFrame(results)
    accuracy = df['score'].mean() * 100
    avg_latency = df['latency'].mean()
    
    print("\n" + "-"*40)
    print(f"FINAL SCORECARD")
    print("-"*40)
    print(f"Accuracy:      {accuracy:.1f}%")
    print(f"Avg Latency:   {avg_latency:.2f}s")
    print(f"Pass Count:    {df['score'].sum()} / {len(df)}")
    print("-"*40)
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
