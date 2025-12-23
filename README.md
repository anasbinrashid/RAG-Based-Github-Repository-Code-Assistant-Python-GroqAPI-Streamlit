# Code RAG Assistant

**Code RAG Assistant** is an opinionated, production-ready Reference-and-Generation (RAG) system tailored to explore, index, and answer questions about codebases. It combines AST-aware chunking, a hybrid semantic + BM25 retriever, and LLM-based generation to provide accurate, contextual answers and source citations.

---

## Features

- AST-aware chunking for Python (and line-based chunking for other languages)
- Hybrid retrieval: semantic embeddings (ChromaDB) + BM25 lexical search
- Cross-encoder reranking for higher precision
- Streamlit UI and interactive CLI for querying and repository management
- Parallel repository processing and batching for scalable ingestion
- Demo scripts for common workflows and comparisons

---

## Quick Start

### Requirements
- Python 3.10+ (or compatible environment)
- A Groq API key (set `GROQ_API_KEY` in environment or `.env`)
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Set environment variables

Copy the example and set your key:

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY
```

> **Security:** Do **not** commit your real `.env` file. If you accidentally committed secrets, rotate them immediately (revoke keys) and follow repository history purge steps below.

### Run the Streamlit UI

```bash
streamlit run src/app.py
```

Open the UI in your browser and use the **Repository Manager** tab to add and process repos, then query in the **Query** tab.

### Use the CLI

```bash
python -m src.cli
```

Common commands:
- `process <repo_url>` — clone, parse and index a repository
- `search <query>` — quick BM25 + semantic search (no LLM)
- Type a query directly to ask the LLM-backed assistant

### Run demos

```bash
python src/example.py        # interactive demo runner
python src/example.py basic  # run a specific demo
```

---

## Project layout

- `src/` — core package and entry points (`app.py`, `cli.py`, `core.py`, `processor.py`, `example.py`)
- `data/` — default storage for processed repositories and local ChromaDB data
- `requirements.txt` & `packages.txt` — Python deps and system packages
- `README.md` — this file

---

## Implementation notes

- Chunking: `ASTAwareChunker` extracts functions/classes for Python and falls back to line-based chunks for other languages.
- Hybrid Retrieval: `HybridRetriever` performs Chroma semantic queries and a BM25 lexical search, merges results with RRF, and can rerank with a cross-encoder.
- Engine: `EnhancedRAGEngine` wraps the retrieval + LLM generation (Groq client) and exposes `query()` and `get_stats()`.
- Processing: `RepositoryProcessor` handles cloning, file discovery (filtered by extensions), parallel chunking and batch insertion into Chroma.

---

## Contribution

Contributions welcome — please open issues or PRs. Preferred workflow:
1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Add tests and documentation
4. Open a PR with a clear description

Please follow existing coding style and add tests for non-trivial changes.

---

## Tips & Next steps

- Add CI to run linting and tests
- Consider adding a `Makefile` or `tox` for reproducible dev tasks
- Add example notebooks or exportable queries for onboarding
