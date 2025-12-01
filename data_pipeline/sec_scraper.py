# /// script
# dependencies = [
#   "requests",
#   "tqdm"
# ]
# ///

import requests
import json
import time 
import os
import random
from tqdm.auto import tqdm

headers = {
    "User-Agent": "FinanceAgentProject charliekerfoot@gmail.com"
}

# QuantOxide Portfolio (Top ~50 by Weight + Sector Reps)
TARGET_TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "GOOGL",
    # Media
    "META", "NFLX", "DIS", "CMCSA",
    # Financials
    "JPM", "BAC", "V", "MA", "WFC", "MS", "GS", "BLK",
    # Healthcare
    "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "PFE",
    # Consumer Retail
    "AMZN", "WMT", "HD", "PG", "COST", "KO", "PEP", "MCD",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Industrials
    "GE", "CAT", "UNP", "HON", "UPS", "BA",
    # Misc
    "LIN", "NEE", "PLD"
]

OUTPUT_DIR = "sec_raw_data"
NUM_YEARS = 2

def get_cik_map():
    """
    Downloads the official list of all SEC registered companies and creates a ticker -> CIK map.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=headers)
    data = response.json()

    cik_map = {}
    for entry in data.values():
        ticker = entry['ticker']
        cik = str(entry['cik_str']).zfill(10) #CIKs must be 10 digits long (padded with zeros) for URLs to work
        cik_map[ticker] = cik

    return cik_map

def get_10k_metadata(cik, count=NUM_YEARS):
    """
    Fetches the submission history for a specific CIK and finds the latest 10-K
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=headers)
    data = response.json()

    recent = data['filings']['recent']

    filings = []
    found = 0

    for i, form in enumerate(recent['form']):
        if form == '10-K':
            accession_number = recent['accessionNumber'][i]
            primary_document = recent['primaryDocument'][i]
            report_date = recent['reportDate'][i]

            filings.append({
                "accession_number": accession_number,
                "primary_document": primary_document,
                "report_date": report_date,
                "year": report_date[:4]
            })

            found += 1
            if found == count:
                break

    return filings

def download_10k(cik, metadata, ticker):
    """
    Downloads 10-K file
    """
    accession_number = metadata['accession_number'].replace("-", "") # URL requires accession number without dashes
    primary_document = metadata['primary_document']

    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{primary_document}"

    response = requests.get(url, headers=headers)

    filename = f"{OUTPUT_DIR}/{ticker}_10K_{metadata['report_date']}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    cik_map = get_cik_map()

    for ticker in tqdm(TARGET_TICKERS):
        if ticker in cik_map:
            cik = cik_map[ticker]
            tqdm.write(f"Processing {ticker} (CIK: {cik})...")

            filings = get_10k_metadata(cik, NUM_YEARS)

            if not filings:
                print(f"No 10-Ks found for {ticker}")

            for filing in filings:
                download_10k(cik, filing, ticker)
                time.sleep(0.1 + random.random() * 0.1) # Rate limiting. SEC allows 10 requests/sec
        else:
            print(f"Error: Could not find CIK for {ticker}")

        time.sleep(0.1)

    print(f"Downloads done. Check the {OUTPUT_DIR} folder.")
