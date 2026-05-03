"""
Week 1 — Preprocessing
========================
WHAT THIS DOES IN PLAIN ENGLISH:
  Think of our raw data like a messy stack of papers fresh off a printer —
  some have staples, some have sticky notes, some are 300 pages long.
  This script:
    1. Removes the mess (HTML tags, legal headers, page numbers)
    2. Cuts long documents into readable chunks (like splitting a book into pages)
    3. Saves everything in one clean, consistent format

WHY THE MODEL NEEDS THIS:
  LLMs have a "context window" — a limit on how much text they can read at once.
  LLaMA-3.1 8B can read about 2048 tokens (~1500 words) at a time.
  A single 10-K filing is ~150,000 words. We must chop it up first.
"""

import json
import re
from pathlib import Path

# Where our raw data lives
RAW_NET = Path("data/raw/networking")
RAW_FIN = Path("data/raw/finance")

# Where we'll save clean, ready-to-use data
PROC_NET = Path("data/processed/networking")
PROC_FIN = Path("data/processed/finance")

PROC_NET.mkdir(parents=True, exist_ok=True)
PROC_FIN.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────
# HELPER: Text Cleaner
# ─────────────────────────────────────────────────────────
# Think of this like a text highlighter that removes everything
# you DON'T want to highlight.

def clean_text(text: str) -> str:
    """Remove HTML tags, extra whitespace, and page-break artifacts."""
    # Remove HTML tags like <div>, <p>, <b>, etc.
    # The pattern <.*?> means: anything between < and >
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove SEC EDGAR header boilerplate (everything before the actual filing)
    # These filings start with machine-readable metadata we don't need
    edgar_start = re.search(r"(ITEM\s+1[\.\s]|PART\s+I[\.\s])", text, re.IGNORECASE)
    if edgar_start:
        text = text[edgar_start.start():]

    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces into one
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove page number artifacts like "- 12 -" or "Page 12 of 150"
    text = re.sub(r"-\s*\d+\s*-", "", text)
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)

    return text.strip()


# ─────────────────────────────────────────────────────────
# HELPER: Text Chunker
# ─────────────────────────────────────────────────────────
# Imagine cutting a long newspaper article into index cards.
# Each card is ~500 words. Cards overlap by 50 words so we
# don't lose context at the edges (like cutting a sentence in half).

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word-level chunks.

    chunk_size = 500 words per chunk
    overlap    = 50 words shared between consecutive chunks
                 (so we don't cut sentences awkwardly at boundaries)
    """
    words = text.split()

    if len(words) <= chunk_size:
        # Short enough to keep as one piece
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # Move forward by (chunk_size - overlap) so next chunk
        # starts 50 words before where this one ended
        start += chunk_size - overlap

    return chunks


# ─────────────────────────────────────────────────────────
# 1. Process Networking Data
# ─────────────────────────────────────────────────────────

def process_nit(input_path: Path, output_path: Path):
    """
    The NIT dataset is already clean JSONL — we just copy it with
    a light normalisation (strip extra whitespace from configs).
    """
    print("  Processing NIT intent->config pairs...")
    count = 0
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            record = json.loads(line)
            # Light clean: normalise whitespace in the config output
            record["output"] = record["output"].strip()
            record["instruction"] = record["instruction"].strip()
            f_out.write(json.dumps(record) + "\n")
            count += 1
    print(f"    {count} NIT examples -> {output_path}")
    return count


def process_rfcs(rfc_dir: Path, output_path: Path):
    """
    RFCs are long plain-text documents (BGP RFC is 104 pages!).
    We clean them and chunk each one into 500-word pieces.
    Each chunk becomes a separate record we'll later index in ChromaDB for RAG.

    Why RAG? Instead of cramming all RFC knowledge into the model's weights
    (which is expensive), we store it in a searchable database and retrieve
    the relevant bit at query time — like a textbook you look things up in.
    """
    print("  Processing IETF RFCs into chunks...")
    count = 0
    rfc_files = list(rfc_dir.glob("*.txt"))

    with open(output_path, "w") as f_out:
        for rfc_file in rfc_files:
            raw_text = rfc_file.read_text(encoding="utf-8", errors="replace")
            cleaned = clean_text(raw_text)
            chunks = chunk_text(cleaned, chunk_size=500, overlap=50)

            for i, chunk in enumerate(chunks):
                if len(chunk.split()) < 30:
                    # Skip tiny fragments — not useful
                    continue
                record = {
                    "domain": "networking",
                    "task": "rag_chunk",          # This will go into the vector DB
                    "source_doc": rfc_file.stem,  # e.g. "RFC4271"
                    "chunk_id": i,
                    "text": chunk,
                }
                f_out.write(json.dumps(record) + "\n")
                count += 1

    print(f"    {count} RFC chunks from {len(rfc_files)} files -> {output_path}")
    return count


# ─────────────────────────────────────────────────────────
# 2. Process Finance Data
# ─────────────────────────────────────────────────────────

def process_phrasebank(input_path: Path, output_path: Path):
    """
    PhraseBank is already clean JSONL — just copy it through.
    This is our labelled sentiment data for evaluation.
    """
    print("  Processing Financial PhraseBank...")
    count = 0
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            record = json.loads(line)
            record["instruction"] = record["instruction"].strip()
            f_out.write(json.dumps(record) + "\n")
            count += 1
    print(f"    {count} sentiment examples -> {output_path}")
    return count


def extract_primary_document(submission_text: str) -> str:
    """
    A full-submission.txt is a bundle of many files (10-K HTML, exhibits,
    XBRL data, etc.) all concatenated together with SEC-specific tags.

    This function pulls out just the primary 10-K document — the one humans
    actually read — by finding the first <DOCUMENT> block with TYPE=10-K.
    """
    # Find all document blocks
    doc_pattern = re.compile(
        r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL | re.IGNORECASE
    )
    type_pattern = re.compile(r"<TYPE>\s*10-K\b", re.IGNORECASE)

    for match in doc_pattern.finditer(submission_text):
        doc_content = match.group(1)
        # Check if this block is the primary 10-K (not an exhibit)
        if type_pattern.search(doc_content[:200]):
            # Extract the actual text content (between <TEXT> tags)
            text_match = re.search(
                r"<TEXT>(.*?)</TEXT>", doc_content, re.DOTALL | re.IGNORECASE
            )
            if text_match:
                return text_match.group(1)
            return doc_content

    # Fallback: return the whole thing if we can't find the primary doc
    return submission_text


def process_edgar(edgar_dir: Path, output_path: Path):
    """
    SEC 10-K filings are massive bundles (9-65MB each).
    The full-submission.txt contains 90+ separate documents (HTML, XBRL,
    exhibits) all concatenated. We:
      1. Extract just the primary 10-K HTML document from the bundle
      2. Strip HTML tags and decode HTML entities (e.g. &#8217; -> ')
      3. Find key narrative sections (Risk Factors, MD&A, Business)
      4. Chunk them into 500-word pieces for RAG

    Why only Risk Factors and MD&A?
      These are the sections with rich narrative text — management explaining
      what went wrong, what they're worried about, what their strategy is.
      The financial tables (numbers) are less useful for language tasks.
    """
    print("  Processing SEC EDGAR 10-K filings...")
    count = 0

    # Walk through: sec-edgar-filings/TICKER/10-K/ACCESSION/full-submission.txt
    filing_files = list(edgar_dir.rglob("full-submission.txt"))
    print(f"    Found {len(filing_files)} filings")

    with open(output_path, "w") as f_out:
        for filing_path in filing_files:
            # Extract ticker from path: .../sec-edgar-filings/AAPL/10-K/...
            parts = filing_path.parts
            ticker = "UNKNOWN"
            for i, part in enumerate(parts):
                if part == "sec-edgar-filings" and i + 1 < len(parts):
                    ticker = parts[i + 1]
                    break

            raw = filing_path.read_text(encoding="utf-8", errors="replace")

            # Step 1: Pull out just the primary 10-K HTML from the bundle
            primary_doc = extract_primary_document(raw)

            # Step 2: Clean HTML and decode entities (&#8217; -> ', &amp; -> &)
            import html
            cleaned = html.unescape(clean_text(primary_doc))

            # Step 3: Find key sections
            sections = extract_10k_sections(cleaned)

            for section_name, section_text in sections.items():
                chunks = chunk_text(section_text, chunk_size=500, overlap=50)
                for i, chunk in enumerate(chunks):
                    if len(chunk.split()) < 30:
                        continue
                    record = {
                        "domain": "finance",
                        "task": "rag_chunk",
                        "ticker": ticker,
                        "section": section_name,
                        "chunk_id": i,
                        "text": chunk,
                    }
                    f_out.write(json.dumps(record) + "\n")
                    count += 1

            print(f"    {ticker}: {sum(1 for s in sections.values() for _ in chunk_text(s))} chunks from {list(sections.keys())}")

    print(f"    {count} 10-K chunks total -> {output_path}")
    return count


def extract_10k_sections(text: str) -> dict[str, str]:
    """
    Find and extract the narrative sections from a 10-K filing.
    Returns a dict like {"risk_factors": "...", "mda": "..."}

    SEC filings use standard item numbers:
      Item 1A = Risk Factors
      Item 7  = MD&A (Management Discussion & Analysis)
      Item 1  = Business description
    """
    sections = {}

    # Patterns to find section starts (case-insensitive)
    section_patterns = {
        "business": r"ITEM\s+1[\.\s]+BUSINESS",
        "risk_factors": r"ITEM\s+1A[\.\s]+RISK\s+FACTORS",
        "mda": r"ITEM\s+7[\.\s]+MANAGEMENT",
    }

    # Find where each section starts.
    # We use re.findall to get ALL occurrences, then pick the last one —
    # the first occurrences are table-of-contents entries (short, close together).
    # The last occurrence is the actual section body (has real content after it).
    positions = {}
    for name, pattern in section_patterns.items():
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            # Pick last match — actual section content, not TOC reference
            positions[name] = matches[-1].start()

    # Extract text from each section start to the next section start
    sorted_sections = sorted(positions.items(), key=lambda x: x[1])
    for i, (name, start_pos) in enumerate(sorted_sections):
        if i + 1 < len(sorted_sections):
            end_pos = sorted_sections[i + 1][1]
        else:
            # Last section: take up to 10000 words max
            end_pos = start_pos + 50000

        section_text = text[start_pos:end_pos]
        # Only keep sections with meaningful content
        if len(section_text.split()) > 100:
            sections[name] = section_text

    # If we couldn't find named sections, just chunk the whole document
    if not sections:
        sections["full_document"] = text[:100000]  # first 100k chars

    return sections


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Preprocessing — cleaning and chunking raw data")
    print("=" * 55)

    totals = {}

    # --- Networking ---
    print("\n[Networking]")

    nit_in  = RAW_NET / "nit_synthetic.jsonl"
    nit_out = PROC_NET / "nit_train.jsonl"
    if nit_in.exists():
        totals["nit"] = process_nit(nit_in, nit_out)

    rfc_in  = RAW_NET / "rfcs"
    rfc_out = PROC_NET / "rfc_chunks.jsonl"
    if rfc_in.exists():
        totals["rfc_chunks"] = process_rfcs(rfc_in, rfc_out)

    # --- Finance ---
    print("\n[Finance]")

    pb_in  = RAW_FIN / "financial_phrasebank.jsonl"
    pb_out = PROC_FIN / "phrasebank_train.jsonl"
    if pb_in.exists():
        totals["phrasebank"] = process_phrasebank(pb_in, pb_out)

    edgar_in  = RAW_FIN / "edgar_10k"
    edgar_out = PROC_FIN / "edgar_chunks.jsonl"
    if edgar_in.exists():
        totals["edgar_chunks"] = process_edgar(edgar_in, edgar_out)

    # --- Summary ---
    print("\n" + "=" * 55)
    print("  Done! Here's what we created:")
    print("=" * 55)
    for name, count in totals.items():
        print(f"  {name:<20} {count:>6} records")

    total = sum(totals.values())
    print(f"  {'TOTAL':<20} {total:>6} records")
    print(f"\n  All saved to data/processed/")
    print("\n  What's next:")
    print("  These processed files are what we'll feed into the model.")
    print("  RAG chunks -> ChromaDB index  (for retrieval)")
    print("  Train examples -> fine-tuning  (for learning)")
