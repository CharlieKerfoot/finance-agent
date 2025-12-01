# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beautifulsoup4",
#     "lxml",
#     "regex",
#     "tqdm",
#     "tiktoken"
# ]
# ///

import os
import re
import json
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from tqdm import tqdm
import tiktoken

INPUT_DIR = "sec_raw_data"
OUTPUT_DIR = "dataset"
OUTPUT_FILE = OUTPUT_DIR + "/finance_chunks.jsonl"

ENC = tiktoken.get_encoding("o200k_base")
CHUNK_SIZE = 800
OVERLAP = 100

# Filtering for financial density (Items 1A, 7, 7A)
POSITIVE_SIGNALS = [
    # MD&A signals (Reasoning)
    "increased by", "decreased by", "primarily due to", "offset by", 
    "gross margin", "operating expenses", "cash flow", "liquidity",
    "capital resources", "net income", "revenue growth", "fiscal year",

    # Risk signals (Uncertainty)
    "adversely affect", "could result in", "significant risk", 
    "subject to", "uncertainty", "competition", "regulation",
    "supply chain", "inflation", "interest rates"
]

NEGATIVE_SIGNALS = [
    # Boilerplate / Navigation
    "table of contents", "index to financial statements", 
    "page", "click here", "exhibit", "signature", "pursuant to",
    "indicate by check mark", "description of securities"
]

def score_chunk(text):
    """
    Determines a financial density score based on used terms
    """
    text_lower = text.lower()
    score = 0

    for word in POSITIVE_SIGNALS:
        if word in text_lower:
            score += 1

    for word in NEGATIVE_SIGNALS:
        if word in text_lower:
            score -= 5

    return score

def convert_tables_to_markdown(soup):
    """
    Finds all HTML tables and replaces them with a Markdown representation.
    """
    for table in soup.find_all("table"):
        if not table.get_text(strip=True): continue
        rows = table.find_all("tr")
        markdown_rows = []
        for tr in rows:
            cells = tr.find_all(["td", "th"])
            row_data = [cell.get_text(separator=" ", strip=True).replace("\n", " ") for cell in cells]

            if any(len(c) > 1 for c in row_data): 
                markdown_rows.append("| " + " | ".join(row_data) + " |")

        if markdown_rows:
            table.replace_with(f"\n\n{chr(10).join(markdown_rows)}\n\n")
    return soup

def parse_html_to_clean_text(filepath):
    """
    Strips HMTL tags and standardizes whitespace
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(raw, 'lxml')

    for tag in soup(["script", "style", "svg", "ix:header", "head"]):
        tag.extract()

    soup = convert_tables_to_markdown(soup)

    text = soup.get_text(separator='\n\n')
    text = text.replace('’', "'").replace('“', '"').replace('”', '"').replace('\xa0', ' ')
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text

def create_chunks(text, meta):
    chunks = []
    paras = text.split('\n\n')

    current_chunk = []
    current_tokens = 0

    for para in paras:
        para = para.strip()
        if not para: continue

        count = len(ENC.encode(para))

        if current_tokens + count > CHUNK_SIZE and current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if score_chunk(chunk_text) >= 2: # Hyperparamter threshold: (greater = stricter)
                chunks.append({
                    "text": chunk_text,
                    "meta": meta,
                    "tokens": current_tokens,
                    "score": score_chunk(chunk_text)
                })

            current_chunk = current_chunk[-3:] # Keep last 3 paragraphs as context
            current_tokens = len(ENC.encode("\n\n".join(current_chunk)))
            
        current_chunk.append(para)
        current_tokens += count
        
    return chunks

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    raw_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".html")]
    all_chunks = []

    print(f"Processing {len(raw_files)} 10-Ks")

    for filename in tqdm(raw_files):
        try:
            clean_text = parse_html_to_clean_text(os.path.join(INPUT_DIR, filename))

            ticker = filename.split('_')[0]
            year = filename.split('_')[2][:4]
            meta = {"ticker": ticker, "year": year, "source": filename}

            file_chunks = create_chunks(clean_text, meta)
            all_chunks.extend(file_chunks)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nFinal Dataset: {len(all_chunks)} high-quality financial chunks.")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")
    print(f"Saved to {OUTPUT_FILE}")
