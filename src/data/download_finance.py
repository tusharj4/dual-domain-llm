"""
Week 1 — Finance Data Downloader
==================================
What this does:
  1. Downloads the Financial PhraseBank sentiment dataset from HuggingFace
  2. Downloads 10-K filings for a sample of large companies via SEC EDGAR API
  3. Downloads recent stock price data via yfinance
  4. Saves everything to data/raw/finance/

Why each piece matters:
  - Financial PhraseBank: 4900 labelled sentences (positive/neutral/negative).
    Used for sentiment classification — one of our finance eval tasks.
  - SEC 10-K filings: Annual reports with Risk Factors, MD&A, financials.
    This is the backbone of our finance RAG knowledge base.
  - Stock prices: Numerical time-series data for any forecasting tasks.
"""

import json
import time
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

RAW_FIN = Path("data/raw/finance")
RAW_FIN.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 1. Financial PhraseBank (HuggingFace)
# ──────────────────────────────────────────────
# This dataset contains 4,840 sentences from financial news, each labelled
# with sentiment: positive, neutral, or negative.
# It's the standard benchmark for financial NLP sentiment tasks.

def download_phrasebank():
    """Download Financial PhraseBank and save as JSONL."""
    print("\n[Financial PhraseBank] Downloading from HuggingFace...")
    print("  What: 4840 labelled financial news sentences")
    print("  Why: Sentiment classification benchmark — we'll use this to")
    print("       evaluate how well the model understands financial tone.\n")

    try:
        from datasets import load_dataset

        # 'sentences_allagree' = only sentences all annotators agreed on (highest quality)
        dataset = load_dataset(
            "financial_phrasebank", "sentences_allagree"
        )

        out_path = RAW_FIN / "financial_phrasebank.jsonl"
        label_map = {0: "negative", 1: "neutral", 2: "positive"}

        count = 0
        with open(out_path, "w") as f:
            for split_name, split_data in dataset.items():
                for row in split_data:
                    record = {
                        "domain": "finance",
                        "task": "sentiment_classification",
                        "instruction": f"What is the sentiment of this financial statement: {row['sentence']}",
                        "output": label_map.get(row["label"], "neutral"),
                        "raw_sentence": row["sentence"],
                        "source": "financial-phrasebank",
                    }
                    f.write(json.dumps(record) + "\n")
                    count += 1

        print(f"  Saved {count} examples -> {out_path}")
        return count

    except Exception as e:
        print(f"  [ERROR] {e}")
        print("  Try: pip install datasets")
        return 0


# ──────────────────────────────────────────────
# 2. SEC EDGAR 10-K Filings
# ──────────────────────────────────────────────
# The SEC EDGAR full-text search API is completely free — no key needed.
# 10-K = annual report. Contains:
#   - Business description
#   - Risk Factors (long, detailed)
#   - Management Discussion & Analysis (MD&A)
#   - Financial statements
# This is our primary RAG knowledge base for the finance domain.

# Sample tickers — well-known companies with clear, readable filings
SAMPLE_TICKERS = ["AAPL", "MSFT", "JPM", "GS", "BAC"]


def get_cik_for_ticker(ticker: str) -> str | None:
    """
    Look up a company's CIK (Central Index Key) from its ticker symbol.
    CIK is the unique identifier SEC uses for each company.
    """
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2023-01-01&enddt=2024-12-31&forms=10-K"
    try:
        # SEC requires a User-Agent header identifying who you are
        headers = {"User-Agent": "research-project tushar@example.com"}
        resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{ticker}.json",
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("cik", None)
    except Exception:
        pass

    # Fallback: use the company search endpoint
    try:
        headers = {"User-Agent": "research-project tushar@example.com"}
        resp = requests.get(
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K",
            headers=headers,
            timeout=10,
        )
        hits = resp.json().get("hits", {}).get("hits", [])
        if hits:
            return hits[0]["_source"].get("entity_id")
    except Exception:
        pass

    return None


def download_edgar_filings(tickers=None, max_per_ticker=1):
    """
    Download recent 10-K filings from SEC EDGAR for a list of tickers.
    Uses sec-edgar-downloader which wraps the EDGAR API cleanly.
    """
    if tickers is None:
        tickers = SAMPLE_TICKERS

    print(f"\n[SEC EDGAR] Downloading 10-K filings for {tickers}...")
    print("  What: Annual reports (10-K) from SEC EDGAR — completely free")
    print("  Why: These are the richest source of financial narrative text.")
    print("       Risk factors, MD&A sections are gold for finance QA tasks.\n")

    filings_dir = RAW_FIN / "edgar_10k"
    filings_dir.mkdir(exist_ok=True)

    try:
        from sec_edgar_downloader import Downloader

        # Downloader saves files to a local directory, organised by ticker
        dl = Downloader("ResearchProject", "research@example.com", filings_dir)

        for ticker in tqdm(tickers, desc="Downloading 10-Ks"):
            try:
                # Download the most recent N 10-K filings for this ticker
                dl.get("10-K", ticker, limit=max_per_ticker)
                print(f"  Downloaded 10-K for {ticker}")
            except Exception as e:
                print(f"  [WARN] Could not download {ticker}: {e}")
            time.sleep(1)  # respect SEC rate limits

        print(f"  Filings saved -> {filings_dir}")

    except ImportError:
        print("  [ERROR] sec-edgar-downloader not installed.")
        print("  Run: pip install sec-edgar-downloader")


# ──────────────────────────────────────────────
# 3. Stock Price Data (yfinance)
# ──────────────────────────────────────────────
# yfinance is a free Python library that fetches Yahoo Finance data.
# We download 2 years of daily closing prices for our sample tickers.
# This gives us numerical financial data for any forecasting tasks.

def download_stock_prices(tickers=None, period="2y"):
    """Download historical stock prices using yfinance."""
    if tickers is None:
        tickers = SAMPLE_TICKERS

    print(f"\n[Stock Prices] Downloading {period} price history via yfinance...")
    print("  What: Daily OHLCV (open/high/low/close/volume) for sample tickers")
    print("  Why: Numerical financial data for any time-series or forecasting tasks.\n")

    try:
        import yfinance as yf

        prices_dir = RAW_FIN / "stock_prices"
        prices_dir.mkdir(exist_ok=True)

        for ticker in tqdm(tickers, desc="Downloading prices"):
            try:
                df = yf.download(ticker, period=period, progress=False)
                if not df.empty:
                    out_path = prices_dir / f"{ticker}.csv"
                    df.to_csv(out_path)
            except Exception as e:
                print(f"  [WARN] {ticker}: {e}")

        print(f"  Prices saved -> {prices_dir}")

    except ImportError:
        print("  [ERROR] yfinance not installed. Run: pip install yfinance")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Week 1 — Finance Data Collection")
    print("=" * 60)

    phrasebank_count = download_phrasebank()
    download_edgar_filings()
    download_stock_prices()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  PhraseBank examples:  {phrasebank_count}")
    print(f"  10-K filings:         data/raw/finance/edgar_10k/")
    print(f"  Stock prices:         data/raw/finance/stock_prices/")
    print("\n  Next step: run src/data/preprocess.py to clean and chunk everything")
