# /// script
# dependencies = [
#   "requests",
# ]
# ///

import requests
import json
import time 
import os

headers = {
    "User-Agent": "FinanceAgentProject charliekerfoot@gmail.com"
}

TARGET_TICKERS = ["AAPL", "MSFT", "TSLA"]

OUTPUT_DIR = "sec_raw_data"

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

def get_latest_10k_metadata(cik):
    """
    Fetches the submission history for a specific CIK and finds the latest 10-K
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=headers)
    data = response.json()

    recent = data['filings']['recent']

    for i, form in enumerate(recent['form']):
        if form == '10-K':
            accession_number = recent['accessionNumber'][i]
            primary_document = recent['primaryDocument'][i]
            report_date = recent['reportDate'][i]

            return {
                "accession_number": accession_number,
                "primary_document": primary_document,
                "report_date": report_date
            }

    print(f"No 10-K found for CIK {cik}")
    return None

def download_10k(cik, metadata, ticker):
    """
    Downloads 10-K file
    """
    accession_number = metadata['accession_number'].replace("-", "") # URL requires accession number without dashes
    primary_document = metadata['primary_document']

    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{primary_document}"
    print(url)

    response = requests.get(url, headers=headers)
    
    filename = f"{OUTPUT_DIR}/{ticker}_10K_{metadata['report_date']}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    cik_map = get_cik_map()

    for ticker in TARGET_TICKERS:
        if ticker in cik_map:
            cik = cik_map[ticker]
            print(f"\nProcessing {ticker} (CIK: {cik})...")

            metadata = get_latest_10k_metadata(cik)

            if metadata:
                download_10k(cik, metadata, ticker)

                time.sleep(0.2) # Rate limiting. SEC allows 10 requests/sec
        else:
            print(f"Error: Could not find CIK for {ticker}")

    print(f"Downloads done. Check the {OUTPUT_DIR} folder.")
