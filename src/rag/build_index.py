"""
RAG — Step 1: Build the Index
================================

EXPLAIN LIKE I'M 5:
  Imagine you have a giant library with thousands of index cards.
  Each card has a short piece of text on it (a chunk from an RFC or a 10-K).

  To find the right card quickly, we give EVERY card a secret code —
  a list of 384 numbers that describes what the card is "about".
  Cards about similar topics get similar codes.

  This script reads all our text chunks and gives each one its secret code.
  Then it files all the cards into ChromaDB — our library cabinet.

  Later, when the model gets a question, we turn the question into a code too,
  and find the cards with the closest matching codes. Those are the relevant ones.

  The secret codes are called EMBEDDINGS.
  The process of finding closest codes is called VECTOR SEARCH.
"""

import json
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# ── Where our processed chunks live ──────────────────────────────────────────
PROC_NET = Path("data/processed/networking")
PROC_FIN = Path("data/processed/finance")

# ── Where ChromaDB will save the index (on disk, so it persists) ─────────────
CHROMA_PATH = "./chroma_db"


def get_chroma_client():
    """
    Create (or reopen) the ChromaDB database on disk.

    Think of this like opening a filing cabinet.
    If it already exists, we just open it.
    If it's new, we create it.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client


def get_embedding_function():
    """
    This is the function that turns text into its secret number-code (embedding).

    We use 'all-MiniLM-L6-v2' — a small, fast model that's good at capturing
    meaning. It's only 80MB and runs on your CPU. No GPU needed.

    Why this model?
      - Free and open-source
      - Runs locally (no API calls)
      - Fast enough for our ~1,400 chunks
      - Good at finding semantic similarity (meaning, not just matching words)
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def load_chunks(filepath: Path, task_filter: str = "rag_chunk") -> list[dict]:
    """
    Load chunks from a JSONL file.
    We only want chunks tagged as 'rag_chunk' — not the training examples.

    Remember our two types of data:
      rag_chunk     → goes into ChromaDB (searchable reference)
      intent_to_config / sentiment_classification → goes into fine-tuning
    """
    chunks = []
    with open(filepath) as f:
        for line in f:
            record = json.loads(line)
            if record.get("task") == task_filter:
                chunks.append(record)
    return chunks


def index_collection(client, collection_name: str, chunks: list[dict], embed_fn):
    """
    Add a batch of chunks into a ChromaDB collection.

    A "collection" is like a labelled drawer in the filing cabinet.
    We use two drawers:
      - "networking_kb" for RFCs
      - "finance_kb" for 10-K sections

    EXPLAIN LIKE I'M 5:
      Imagine putting index cards into a magical drawer.
      The drawer automatically reads each card and figures out its secret code.
      Later you can ask the drawer: "find me cards about BGP" and it returns
      the 5 most relevant ones, even if you didn't use the exact right words.
    """
    # Delete and recreate if it already exists (clean rebuild)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},  # cosine = measure similarity by angle
    )

    # ChromaDB wants three parallel lists: ids, texts, and metadata
    ids, documents, metadatas = [], [], []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{collection_name}_{i}"
        text = chunk.get("text", "")

        if not text.strip():
            continue

        # Metadata = extra info stored alongside the text (not searched, just returned)
        meta = {
            "source_doc": chunk.get("source_doc", chunk.get("ticker", "unknown")),
            "section":    chunk.get("section", ""),
            "chunk_id":   str(chunk.get("chunk_id", i)),
            "domain":     chunk.get("domain", ""),
        }

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(meta)

    # Add in batches of 100 — avoids memory spikes
    BATCH = 100
    total = len(ids)
    for start in range(0, total, BATCH):
        end = min(start + BATCH, total)
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"    Indexed {end}/{total} chunks...", end="\r")

    print(f"    Indexed {total}/{total} chunks — done.   ")
    return total


def search_demo(client, collection_name: str, embed_fn, query: str, top_k: int = 3):
    """
    Demo: run a search query and show what gets retrieved.

    EXPLAIN LIKE I'M 5:
      You walk up to the magic filing cabinet and ask it a question.
      It thinks for a second, then hands you the 3 most relevant index cards.
      You didn't have to use the exact words on the cards — it understood what you meant.
    """
    collection = client.get_collection(collection_name, embedding_function=embed_fn)
    results = collection.query(query_texts=[query], n_results=top_k)

    print(f"\n  Query: '{query}'")
    print(f"  Top {top_k} results from '{collection_name}':")
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        source = meta.get("source_doc", "?")
        section = meta.get("section", "")
        label = f"{source} / {section}" if section else source
        print(f"\n  [{i+1}] Source: {label}")
        print(f"       Text: {doc[:180]}...")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Building RAG Index")
    print("=" * 55)
    print("""
  What's happening:
    1. We load all RFC and 10-K chunks
    2. An embedding model turns each chunk into 384 numbers
    3. ChromaDB stores those numbers + the original text on disk
    4. We run a demo search to prove it works
    """)

    client   = get_chroma_client()
    embed_fn = get_embedding_function()

    # ── Networking: index RFC chunks ──────────────────────────────────────────
    print("\n[1/2] Indexing networking knowledge base (RFCs)...")
    rfc_chunks = load_chunks(PROC_NET / "rfc_chunks.jsonl")
    print(f"      Loaded {len(rfc_chunks)} RFC chunks")
    net_count = index_collection(client, "networking_kb", rfc_chunks, embed_fn)

    # ── Finance: index EDGAR chunks ───────────────────────────────────────────
    print("\n[2/2] Indexing finance knowledge base (10-K filings)...")
    edgar_chunks = load_chunks(PROC_FIN / "edgar_chunks.jsonl")
    print(f"      Loaded {len(edgar_chunks)} EDGAR chunks")
    fin_count = index_collection(client, "finance_kb", edgar_chunks, embed_fn)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Index built!")
    print(f"  networking_kb : {net_count} chunks")
    print(f"  finance_kb    : {fin_count} chunks")
    print(f"  Saved to      : {CHROMA_PATH}/")
    print(f"{'='*55}")

    # ── Demo searches ─────────────────────────────────────────────────────────
    print("\n--- Demo: Does the search actually work? ---")

    search_demo(client, "networking_kb", embed_fn,
        query="How does BGP handle path selection between multiple routes?")

    search_demo(client, "finance_kb", embed_fn,
        query="What are the main risks Apple faces in its business?")
