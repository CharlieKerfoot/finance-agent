# /// script
# dependencies = [
#  "bs4",
#  "lxml"
# ]
# ///

import os
import re
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

INPUT_DIR = "sec_raw_data"
OUTPUT_DIR = "sec_cleaned_data"

SECTIONS_CONFIG = {
    "Item 1A": {
        "start_keywords": ["Item 1A. Risk Factors", "Item 1A Risk Factors"],
        "end_keywords": ["Item 1B. Unresolved Staff Comments", "Item 1B Unresolved Staff Comments", "Item 2. Properties"]
    },
    "Item 7": {
        "start_keywords": ["Item 7. Management's Discussion and Analysis", "Item 7 Management's Discussion and Analysis"],
        "end_keywords": ["Item 7A. Quantitative and Qualitative Disclosures", "Item 7A Quantitative and Qualitative Disclosures", "Item 8. Financial Statements"]
    },
    "Item 7A": {
        "start_keywords": ["Item 7A. Quantitative and Qualitative Disclosures", "Item 7A Quantitative and Qualitative Disclosures"],
        "end_keywords": ["Item 8. Financial Statements", "Item 8 Financial Statements"]
    }
}

def generate_robust_pattern(keyword):
    """
    Creates a regex pattern that accounts for uneven whitespace between characters
    """

    pattern = ""
    for char in keyword:
        if char.isspace():
            pattern += r"\s+"
        elif char.isalnum():
            pattern += char + r"\s*"
        else:
            pattern += re.escape(char) + r"\s*"

    return pattern

def convert_tables_to_markdown(soup):
    """
    Finds all HTML tables and replaces them with a Markdown representation.
    """
    for table in soup.find_all("table"):
        if not table.get_text(strip=True):
            continue

        markdown_rows = []

        rows = table.find_all("tr")
        for tr in rows:
            cells = tr.find_all(["td", "th"])
            row_data = [cell.get_text(separator=" ", strip=True).replace("\n", " ") for cell in cells]

            if row_data:
                markdown_row = "| " + " | ".join(row_data) + " |"
                markdown_rows.append(markdown_row)

        if markdown_rows:
            table_text = "\n".join(markdown_rows)
            replacement_text = f"\n\n[START TABLE]\n{table_text}\n[END TABLE]\n\n"
            table.replace_with(replacement_text)

    return soup

def clean_html_text(html_content):
    """
    Strips HMTL tags and standardizes whitespace
    """
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    soup = BeautifulSoup(html_content, 'lxml')

    for script in soup(["script", "style"]):
        script.extract()

    soup = convert_tables_to_markdown(soup)

    text = soup.get_text(separator='\n\n')
    text = text.replace('’', "'").replace('“', '"').replace('”', '"')
    text = text.replace('\xa0', ' ') # remove NBSPs
    text = re.sub(r'\n{3,}','\n\n',text)

    return text

def find_match_index(text, keywords, is_start=True):
    """
    Tries to find the best match for a list of possible headers.
    Returns the index (start or end) or None.
    """
    for kw in keywords:
        robust_pattern = generate_robust_pattern(kw)
        matches = list(re.finditer(robust_pattern, text, re.IGNORECASE))

        if matches:
            if is_start:
                return matches[-1].end()
            else:
                return matches[0].start()
    return None

def extract_section_content(text, config, section_name):
    start_candidates = []
    for start_kw in config["start_keywords"]:
        pattern = generate_robust_pattern(start_kw)
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start_candidates.append(match.end())

    if not start_candidates:
        print(f"Error: Start pattern not found for {section_name}")
        return None

    start_candidates.sort()

    candidates = []

    for start_idx in start_candidates:
        text_after_start = text[start_idx:]
        end_match = None

        for end_kw in config["end_keywords"]:
            end_pattern = generate_robust_pattern(end_kw)
            match = re.search(end_pattern, text_after_start, re.IGNORECASE)
            if match:
                if end_match is None or match.start() < end_match.start():
                    end_match = match

        if end_match:
            relative_end = end_match.start()
            content = text_after_start[:relative_end].strip()

            candidates.append(content)

    if not candidates:
        print(f"Error: Found start matches for {section_name}, but no end patterns found.")
        return None

    # Heuristic approach - Pick longest content
    best_content = max(candidates, key=len)

    return best_content

def process_file(filename):
    with open(f"{INPUT_DIR}/{filename}", "r", encoding="utf-8") as f:
        raw_html = f.read()

    clean_text = clean_html_text(raw_html)

    extracted_data = {}
    for section, config in SECTIONS_CONFIG.items():
        extracted_data[section] = extract_section_content(clean_text, config, section)

    base_name = filename.replace(".html", "")
    output_path = f"{OUTPUT_DIR}/{base_name}_cleaned.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for sec, text in extracted_data.items():
            f.write(f"--- {sec} ---\n")
            f.write(text)
            f.write("\n" + "-"*50 + "\n\n")

    print(f"Saved clean text to {output_path}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".html")]

    for f in files:
        process_file(f)
