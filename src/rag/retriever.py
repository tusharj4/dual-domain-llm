"""
RAG — Step 2: The Retriever
=============================

EXPLAIN LIKE I'M 5:
  The index we built is the library.
  The retriever is the librarian.

  You ask the librarian a question.
  The librarian goes into the library, finds the 3-5 most helpful cards,
  and hands them back to you so you can read them before answering.

  In our system: the model calls retrieve() before answering any question.
  The returned text gets added to the top of the model's prompt as extra context.

  This is the core idea behind RAG:
    [User question] + [Retrieved context] → Model → [Better answer]
"""

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path


CHROMA_PATH = "./chroma_db"


class Retriever:
    """
    A simple retriever that searches the right knowledge base
    based on the domain (networking or finance).
    """

    def __init__(self, chroma_path: str = CHROMA_PATH):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        # Load both collections (think: two drawers of the filing cabinet)
        self._collections = {}

    def _get_collection(self, name: str):
        """Lazy-load collections so we don't open both at startup."""
        if name not in self._collections:
            self._collections[name] = self.client.get_collection(
                name=name,
                embedding_function=self.embed_fn
            )
        return self._collections[name]

    def retrieve(self, query: str, domain: str = "auto", top_k: int = 4) -> list[dict]:
        """
        Find the most relevant chunks for a given query.

        domain: "networking", "finance", or "auto" (we guess from the query)
        top_k:  how many chunks to return (3-5 is usually best)

        Returns a list of dicts with keys: text, source, section
        """
        collection_name = self._pick_collection(query, domain)
        collection = self._get_collection(collection_name)

        results = collection.query(query_texts=[query], n_results=top_k)

        chunks = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append({
                "text":    doc,
                "source":  meta.get("source_doc", "unknown"),
                "section": meta.get("section", ""),
                "domain":  meta.get("domain", ""),
            })
        return chunks

    def _pick_collection(self, query: str, domain: str) -> str:
        """
        Decide which knowledge base to search.

        If domain="auto", we guess based on keywords in the query.
        Not perfect, but good enough for our purposes.
        """
        if domain == "networking":
            return "networking_kb"
        if domain == "finance":
            return "finance_kb"

        # Auto-detect: look for financial keywords
        finance_words = {
            "revenue", "profit", "loss", "earnings", "stock", "shares",
            "sec", "10-k", "filing", "risk", "balance sheet", "cash flow",
            "investment", "ipo", "acquisition", "merger", "dividend", "equity",
            "bank", "financial", "market", "trading", "portfolio"
        }
        query_lower = query.lower()
        if any(word in query_lower for word in finance_words):
            return "finance_kb"

        # Default to networking
        return "networking_kb"

    def format_context(self, chunks: list[dict]) -> str:
        """
        Turn retrieved chunks into a clean context string
        to prepend to the model's prompt.

        EXPLAIN LIKE I'M 5:
          Imagine you're about to answer a question on a test.
          Someone hands you 3 sticky notes with relevant info.
          You read the sticky notes, then write your answer.
          This function creates those sticky notes.
        """
        if not chunks:
            return ""

        lines = ["--- Relevant context ---"]
        for i, chunk in enumerate(chunks, 1):
            source = chunk["source"]
            section = f" / {chunk['section']}" if chunk["section"] else ""
            lines.append(f"[{i}] Source: {source}{section}")
            lines.append(chunk["text"].strip())
            lines.append("")  # blank line between chunks

        lines.append("--- End of context ---")
        return "\n".join(lines)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    r = Retriever()

    print("Test 1: Networking query (auto-detect)")
    chunks = r.retrieve("Configure OSPF in area 0 on an interface", domain="auto")
    print(r.format_context(chunks))

    print("\nTest 2: Finance query (auto-detect)")
    chunks = r.retrieve("What risks does Microsoft face from cloud competition?", domain="auto")
    print(r.format_context(chunks))

    print("\nTest 3: How the full RAG prompt looks")
    question = "How does BGP choose between two routes to the same destination?"
    chunks = r.retrieve(question, domain="networking")
    context = r.format_context(chunks)

    full_prompt = f"""{context}

Question: {question}

Answer:"""
    print(full_prompt[:800], "...")
