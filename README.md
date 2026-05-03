# Dual-Domain LLM: Networking + Investment Banking

A domain-specialized LLM that handles both **computer networking** tasks (config generation, intent translation, fault diagnosis) and **investment banking** tasks (SEC filing QA, risk summarization, financial sentiment).

Built entirely with **free and open-source tools**.

## Architecture

- **Base model:** LLaMA-3.1 8B Instruct (Meta, open-weight)
- **Fine-tuning:** QLoRA via Unsloth (runs on free T4 GPU — Kaggle/Colab)
- **RAG:** ChromaDB + SentenceTransformers (`all-MiniLM-L6-v2`)
- **Data:** IETF RFCs, NIT dataset, SEC EDGAR filings, Financial PhraseBank
- **Memory:** ChromaDB vector store + AES-256 encryption
- **Eval:** HuggingFace `evaluate` library

## Project Structure

```
dual-domain-llm/
├── data/
│   ├── raw/
│   │   ├── networking/      # RFCs, NIT dataset, device configs
│   │   └── finance/         # SEC filings, Financial PhraseBank, loan data
│   └── processed/
│       ├── networking/      # Cleaned, chunked, JSONL format
│       └── finance/         # Cleaned, chunked, JSONL format
├── notebooks/               # Exploratory analysis and experiments
├── src/
│   ├── data/                # Data download and preprocessing scripts
│   ├── training/            # QLoRA fine-tuning code
│   ├── rag/                 # ChromaDB indexing and retrieval
│   ├── memory/              # Encrypted session memory system
│   └── evaluation/          # Metrics and benchmarking
├── scripts/                 # One-off utility scripts
├── models/                  # Saved LoRA adapter weights
├── configs/                 # Hyperparameter and model configs
└── tests/                   # Unit tests
```

## Week-by-Week Plan

| Week | Focus |
|------|-------|
| 1 | Data pipeline — scrape RFCs, EDGAR filings, NIT dataset |
| 2 | Model setup — QLoRA fine-tuning scaffold + RAG framework |
| 3 | Experiments — ablation studies, memory system |
| 4 | Evaluation, paper writing, artifact packaging |

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Download networking data
python src/data/download_networking.py

# Download finance data
python src/data/download_finance.py
```

## Free Resources Used

- [HuggingFace](https://huggingface.co) — model hosting + datasets
- [Kaggle Notebooks](https://kaggle.com) — 30 hrs/week free T4 GPU
- [SEC EDGAR](https://efts.sec.gov/LATEST/search-index) — free financial filings API
- [Unsloth](https://github.com/unslothai/unsloth) — 2x faster QLoRA training
