"""
Scraper for daily silk cocoon prices from Kannada Top News
(kannadatopnews.com), covering multiple Karnataka cocoon markets.

STRUCTURE OF THE SITE (confirmed by manual inspection):
  1. A category listing page per market, paginated:
     https://kannadatopnews.com/category/announcement/sericulture/silk-cocoon/{category_slug}/
     https://kannadatopnews.com/category/announcement/sericulture/silk-cocoon/{category_slug}/page/2/
     ... etc.
  2. Each listing page links to individual article pages, one per market
     per day, named like:
     https://kannadatopnews.com/{market_key}-silk-cocoon-market-21-july-2026/
  3. Each article page contains a table with columns:
     Variety | Lots | Qty (kg) | Min | Max | Avg
     and a line "Date: DD/MM/YYYY".

STRATEGY:
  Stage 1 (discover): crawl the paginated listing pages for each market,
    collect every article URL that matches that market's naming pattern.
  Stage 2 (extract): visit each article URL, pull the table + date.

  Both stages checkpoint to disk, so if your connection drops or you need
  to stop halfway through an 8-hour run, re-running the script picks up
  where it left off instead of starting over.

BEFORE RUNNING:
  pip install requests beautifulsoup4
  Check https://kannadatopnews.com/robots.txt yourself in a browser to
  confirm it still allows crawling before you run this at scale.
"""

import csv
import re
import time
import random
import json
from pathlib import Path
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------
# CONFIG & ARCHITECTURE SETUP
# ---------------------------------------------------------------------
MARKETS = {
    "kolar": "kolar-silk-cocoon-market",
    "ramanagara": "ramanagara-silk-cocoon-market",
    "kanakapura": "kanakapura-silk-cocoon-market"
}

BASE_CATEGORY_URL = "https://kannadatopnews.com/category/announcement/sericulture/silk-cocoon"
REQUEST_DELAY_SECONDS = 2.5
MAX_LISTING_PAGES = 250

# Set base directory to where the script is located (Market data/)
BASE_DIR = Path(__file__).parent
CHECKPOINT_FILE = BASE_DIR / "checkpoint.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

session = requests.Session()
session.headers.update(HEADERS)
_retry_strategy = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
_adapter = HTTPAdapter(max_retries=_retry_strategy)
session.mount("https://", _adapter)
session.mount("http://", _adapter)

def polite_get(url: str):
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=20)
            time.sleep(REQUEST_DELAY_SECONDS + random.uniform(0, 1.5))
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return None
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    return None

def extract_date_from_url(url: str) -> datetime.date:
    """Extracts and parses the date directly from the article URL slug."""
    match = re.search(r'-(\d{1,2}-[a-z]+-\d{4})/?$', url)
    if match:
        date_str = match.group(1)
        for fmt in ("%d-%B-%Y", "%d-%b-%Y"):
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
    return None

# ---------------------------------------------------------------------
# STAGE 1 & 2: DISCOVER AND EXTRACT NEW DATA
# ---------------------------------------------------------------------
VARIETY_BLOCK_PATTERN = re.compile(
    r"(Cross-Breed|Bi-Voltine)[^\n]*.*?Lots\s*:\s*(\d+).*?Qty\s*:\s*([\d.]+)\s*Kg"
    r".*?(?:Mx|Max)\s*:\s*₹?\s*([\d.]+).*?(?:Mn|Min)\s*:\s*₹?\s*([\d.]+).*?Avg\s*:\s*₹?\s*([\d.]+)",
    re.DOTALL,
)
VARIETY_CODE = {"Cross-Breed": "CB", "Bi-Voltine": "BV"}

def clean_number(s):
    if s is None: return None
    s = re.sub(r"[^\d.]", "", s)
    return s if s else None

def extract_article_data(url: str, market_key: str) -> list[dict]:
    """Extracts the table/text block data from a single article."""
    resp = polite_get(url)
    if not resp: return []
    
    soup = BeautifulSoup(resp.text, "html.parser")
    page_text = soup.get_text("\n")
    date_match = re.search(r"Date\s*:\s*(\d{1,2}/\d{1,2}/\d{4})", page_text)
    price_date = date_match.group(1) if date_match else None

    rows = []
    # Try table extraction first
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
            if len(cells) < 6 or not re.search(r"\d", cells[1]): continue
            variety_raw = cells[0]
            variety_code = "CB" if "Cross-Breed" in variety_raw or "CB" in variety_raw else ("BV" if "Bi-Voltine" in variety_raw or "BV" in variety_raw else variety_raw)
            rows.append({
                "market": market_key, "date": price_date, "variety": variety_code,
                "lots": clean_number(cells[1]), "qty_kg": clean_number(cells[2]),
                "min_price": clean_number(cells[3]), "max_price": clean_number(cells[4]),
                "avg_price": clean_number(cells[5]), "source_url": url
            })
            
    if rows: return rows

    # Fallback to text block extraction
    for match in VARIETY_BLOCK_PATTERN.finditer(page_text):
        variety_raw, lots, qty, max_p, min_p, avg_p = match.groups()
        rows.append({
            "market": market_key, "date": price_date, "variety": VARIETY_CODE.get(variety_raw, variety_raw),
            "lots": clean_number(lots), "qty_kg": clean_number(qty),
            "min_price": clean_number(min_p), "max_price": clean_number(max_p),
            "avg_price": clean_number(avg_p), "source_url": url
        })
    return rows

def process_market(market_name: str, category_slug: str, last_scraped_date: datetime.date):
    """Discovers and scrapes articles up until the checkpoint date."""
    print(f"\n[{market_name.upper()}] Scraping data newer than {last_scraped_date}...")
    
    market_dir = BASE_DIR / market_name
    market_dir.mkdir(exist_ok=True)
    csv_file = market_dir / "cocoon_prices.csv"
    write_header = not csv_file.exists()
    
    pattern = re.compile(rf"https://kannadatopnews\.com/{re.escape(category_slug)}-[a-z0-9\-]+/")
    new_urls_to_scrape = []
    
    # 1. DISCOVERY
    page_num = 1
    stop_pagination = False
    
    while page_num <= MAX_LISTING_PAGES and not stop_pagination:
        url = f"{BASE_CATEGORY_URL}/{category_slug}/" if page_num == 1 else f"{BASE_CATEGORY_URL}/{category_slug}/page/{page_num}/"
        resp = polite_get(url)
        if not resp: break
            
        page_urls = set(pattern.findall(resp.text))
        if not page_urls: break
            
        for article_url in page_urls:
            url_date = extract_date_from_url(article_url)
            
            # If we successfully parsed the date and it's older/equal to checkpoint, we've reached the past
            if url_date and url_date <= last_scraped_date:
                stop_pagination = True
            elif article_url not in new_urls_to_scrape:
                new_urls_to_scrape.append(article_url)
                
        print(f"  Page {page_num}: Found {len(new_urls_to_scrape)} new articles to process so far...")
        page_num += 1

    if not new_urls_to_scrape:
        print(f"[{market_name.upper()}] No new data found. Up to date.")
        return

    # 2. EXTRACTION & SAVING
    fieldnames = ["market", "date", "variety", "lots", "qty_kg", "min_price", "max_price", "avg_price", "source_url"]
    rows_written = 0
    
    with open(csv_file, "a", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            
        for i, url in enumerate(new_urls_to_scrape, 1):
            rows = extract_article_data(url, market_name)
            for row in rows:
                writer.writerow(row)
            rows_written += len(rows)
            f_csv.flush()
            
            if i % 10 == 0 or i == len(new_urls_to_scrape):
                print(f"  ...Extracted {i}/{len(new_urls_to_scrape)} articles ({rows_written} rows written)")

    print(f"[{market_name.upper()}] Done. Appended {rows_written} new rows to {market_name}/cocoon_prices.csv")

# ---------------------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------------------
def main():
    if not CHECKPOINT_FILE.exists():
        print(f"Error: {CHECKPOINT_FILE} not found. Please create it first.")
        return

    with open(CHECKPOINT_FILE, "r") as f:
        checkpoints = json.load(f)

    today_str = datetime.today().strftime('%Y-%m-%d')
    
    for market_name, category_slug in MARKETS.items():
        last_scraped_str = checkpoints.get(market_name, "2000-01-01")
        last_scraped_date = datetime.strptime(last_scraped_str, "%Y-%m-%d").date()
        
        process_market(market_name, category_slug, last_scraped_date)
        
        # Update checkpoint for this market immediately upon success
        checkpoints[market_name] = today_str
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(checkpoints, f, indent=4)

    print("\nAll markets updated successfully. Checkpoint file updated to today.")

if __name__ == "__main__":
    main()