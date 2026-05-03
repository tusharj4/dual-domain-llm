"""
Week 1 — Networking Data Downloader
====================================
What this does:
  1. Downloads the NIT dataset from HuggingFace (intent -> Juniper config pairs)
  2. Downloads a sample of IETF RFCs (plain text) from the IETF datatracker
  3. Saves everything to data/raw/networking/

Why each piece matters:
  - NIT dataset: 1000 real examples of "human intent -> device config" — this is our
    primary training signal for the networking side.
  - RFCs: The authoritative specs for every protocol (BGP, OSPF, VLAN, etc.).
    We'll use these for RAG — so the model can look up exact protocol details
    instead of hallucinating them.
"""

import json
import time
import requests
from pathlib import Path
from tqdm import tqdm

RAW_NET = Path("data/raw/networking")
RAW_NET.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 1. NIT Dataset (HuggingFace)
# ──────────────────────────────────────────────
# The NIT dataset lives on HuggingFace Hub. We use the `datasets` library
# which handles download, caching, and formatting automatically.

def download_nit_dataset():
    """Download the NIT intent->config dataset and save as JSONL."""
    print("\n[NIT Dataset] Downloading from HuggingFace...")
    print("  What: 1000 pairs of (natural language intent, Juniper EX3300 config)")
    print("  Why: This is our labelled training data for the networking domain.\n")

    try:
        from datasets import load_dataset

        # 'load_dataset' fetches from HuggingFace Hub and caches locally.
        # 'split="train"' means we want the training portion.
        dataset = load_dataset("dpassaro/nit-dataset", split="train")

        out_path = RAW_NET / "nit_dataset.jsonl"
        with open(out_path, "w") as f:
            for row in tqdm(dataset, desc="Saving NIT examples"):
                # Each row has fields like 'input' (intent) and 'output' (config).
                # We normalise to a consistent schema we'll use across both domains.
                record = {
                    "domain": "networking",
                    "task": "intent_to_config",
                    "instruction": row.get("question") or row.get("input", ""),
                    "output": row.get("answer") or row.get("output", ""),
                    "source": "NIT-dataset",
                }
                f.write(json.dumps(record) + "\n")

        print(f"  Saved {len(dataset)} examples -> {out_path}")
        return len(dataset)

    except Exception as e:
        print(f"  [ERROR] Could not load NIT dataset: {e}")
        print("  Try: pip install datasets")
        return 0


# ──────────────────────────────────────────────
# 2. IETF RFCs (plain text, sample)
# ──────────────────────────────────────────────
# RFCs are numbered documents. We fetch a curated list of the most
# important networking protocol RFCs. Each RFC is a plain-text file
# available directly from the IETF website at no cost.

# These are the RFCs most relevant to networking tasks we'll train on:
# BGP, OSPF, MPLS, VLAN (802.1Q), STP, IS-IS, IPv6, etc.
IMPORTANT_RFCS = [
    ("RFC4271", 4271, "BGP-4 — Border Gateway Protocol"),
    ("RFC2328", 2328, "OSPFv2 — Open Shortest Path First"),
    ("RFC3031", 3031, "MPLS Architecture"),
    ("RFC4364", 4364, "BGP/MPLS IP VPNs"),
    ("RFC2460", 2460, "IPv6 Specification"),
    ("RFC4861", 4861, "Neighbor Discovery for IPv6"),
    ("RFC5340", 5340, "OSPFv3 for IPv6"),
    ("RFC7938", 7938, "BGP in Large-Scale Data Centers"),
    ("RFC1918", 1918, "Private IP Address Space"),
    ("RFC3748", 3748, "Extensible Authentication Protocol (EAP)"),
    ("RFC5246", 5246, "TLS 1.2"),
    ("RFC8446", 8446, "TLS 1.3"),
    ("RFC2865", 2865, "RADIUS Authentication"),
    ("RFC3164", 3164, "Syslog Protocol"),
    ("RFC5424", 5424, "Syslog Protocol (updated)"),
]


def download_rfcs(rfc_list=None, max_rfcs=15):
    """Download IETF RFCs as plain text files."""
    if rfc_list is None:
        rfc_list = IMPORTANT_RFCS[:max_rfcs]

    rfc_dir = RAW_NET / "rfcs"
    rfc_dir.mkdir(exist_ok=True)

    print(f"\n[IETF RFCs] Downloading {len(rfc_list)} key protocol RFCs...")
    print("  What: Plain-text RFC documents from ietf.org")
    print("  Why: We index these in ChromaDB so the model can retrieve exact")
    print("       protocol specs during inference (RAG) instead of guessing.\n")

    downloaded = []
    for name, number, description in tqdm(rfc_list, desc="Downloading RFCs"):
        out_path = rfc_dir / f"{name}.txt"
        if out_path.exists():
            downloaded.append(name)
            continue

        url = f"https://www.rfc-editor.org/rfc/rfc{number}.txt"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                out_path.write_text(resp.text, encoding="utf-8", errors="replace")
                downloaded.append(name)
            else:
                print(f"  [WARN] {name} returned HTTP {resp.status_code}")
        except Exception as e:
            print(f"  [WARN] Failed to download {name}: {e}")

        time.sleep(0.5)  # polite delay — don't hammer the IETF server

    print(f"  Downloaded {len(downloaded)}/{len(rfc_list)} RFCs -> {rfc_dir}")
    return downloaded


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Week 1 — Networking Data Collection")
    print("=" * 60)

    nit_count = download_nit_dataset()
    rfc_list = download_rfcs()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  NIT examples:  {nit_count}")
    print(f"  RFCs saved:    {len(rfc_list)}")
    print(f"  Location:      data/raw/networking/")
    print("\n  Next step: run src/data/download_finance.py")
