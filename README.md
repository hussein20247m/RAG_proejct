# Document-Based RAG API

A production-ready **Retrieval-Augmented Generation** service built with **FastAPI**, **LangChain**, **ChromaDB**, and **Ollama**.

Drop documents into the `data/` folder — the system discovers them recursively, extracts text, chunks it, embeds it, and stores the vectors. Ask questions through the API and get grounded answers with **source attribution** (file, page, chunk, score).

---

## Features

- **Drop-zone ingestion** — put files in `data/` (recursively scanned, including subfolders)
- **Multiple formats** — PDF, TXT, Markdown, LOG, CSV, DOCX, HTML
- **Incremental indexing** — SHA-256 manifest in SQLite; unchanged files are **never re-embedded**, modified files are replaced, deleted files are removed from the index
- **Grounded answers** — the LLM answers strictly from retrieved context and explicitly refuses when the answer is not in the documents
- **Source attribution** — every answer returns sources: filename, relative path, document type, page number, chunk index, and cosine similarity score
- **Persistent vector store** — single Chroma collection (`documents`, cosine space) under `backend/data/vectors/`
- **Chat sessions** — queries and their sources are persisted in SQLite
- **Resilient scanning** — a corrupt or unsupported file is logged and skipped; it never crashes the scan

## Tech Stack

| Component | Choice |
|---|---|
| API | FastAPI + Uvicorn |
| LLM framework | LangChain 1.x |
| Embeddings | Ollama `nomic-embed-text:latest` (768 dimensions) |
| Chat model | Ollama `qwen2.5:0.5b` |
| Vector database | Chroma (persistent, cosine) |
| Metadata store | SQLite (SQLAlchemy, WAL mode) |
| Tests | pytest |

## Requirements

- Python 3.14+ (developed on 3.14.7)
- [Ollama](https://ollama.com) running locally with the required models:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:0.5b
```

## Installation

```bash
# from the repository root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Settings are read from environment variables and `backend/app/.env` (no secrets required — everything is local defaults):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text:latest` | Embedding model |
| `OLLAMA_MODEL` | `qwen2.5:0.5b` | Chat model |
| `INDEX_ON_STARTUP` | `true` | Run an incremental scan at startup |
| `LOG_LEVEL` | `INFO` | Logging level |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K` | `6` | Default retrieved chunks |
| `MIN_SIMILARITY` | `0.30` | Cosine-score floor (below → refusal) |
| `EMBED_BATCH_SIZE` | `32` | Chunks per embedding request |
| `MAX_FILE_SIZE_MB` | `50` | Per-file size limit |
| `INGEST_EXTENSIONS` | `.pdf,.txt,.md,.markdown,.log,.csv,.docx,.html,.htm` | Comma-separated allow-list |

## Running

```bash
# from the repository root (NOT from backend/)
uvicorn backend.app.main:app --reload
```

- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/api/v1/health

## Usage

### 1. Add documents

Either drop files directly into `data/`:

```
data/
├── notes.txt
├── manual.md
├── book.docx
├── reports/
│   ├── report1.pdf
│   └── report2.pdf
└── uploads/          # files uploaded via the API
```

or upload through the API:

```bash
curl -X POST http://localhost:8000/api/v1/pdfs/upload \
  -F "file=@notes.txt"
```

### 2. Index

Indexing runs automatically at startup (`INDEX_ON_STARTUP=true`). Trigger it manually:

```bash
curl -X POST http://localhost:8000/api/v1/index/scan
```

A scan only processes **new / modified / deleted** files — a warm start costs a few `stat()` calls and embeds nothing.

Check status:

```bash
curl http://localhost:8000/api/v1/index/status
```

### 3. Ask

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the manual say about setup?"}'
```

Response:

```json
{
  "answer": "The manual states that setup requires ... [1]",
  "sources": [
    {
      "pdf_name": "manual.md",
      "pdf_id": "8f1c2a...",
      "chunk_index": 3,
      "page": null,
      "score": 0.72,
      "source_path": "manual.md",
      "file_type": "md"
    }
  ],
  "metadata": { "top_score": 0.72, "min_score_hit": true, "elapsed_ms": 412 },
  "session_id": "...",
  "message_id": 1
}
```

If nothing relevant is retrieved, the API refuses honestly:

```json
{ "answer": "I don't have enough information in the indexed documents to answer that.", "sources": [] }
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/api/v1/health` | Health (Ollama, collections, indexed docs) |
| `GET` | `/api/v1/models` | Available Ollama chat models |
| `POST` | `/api/v1/pdfs/upload` | Upload + index a single file |
| `GET` | `/api/v1/pdfs` | List uploaded files (paginated) |
| `DELETE` | `/api/v1/pdfs/{pdf_id}` | Delete an upload (file + vectors + manifest) |
| `POST` | `/api/v1/query` | Ask a question → answer + sources |
| `POST` | `/api/v1/index/scan` | Trigger incremental scan |
| `GET` | `/api/v1/index/status` | Index statistics |
| `GET` | `/api/v1/documents` | List all indexed documents (paginated) |
| `DELETE` | `/api/v1/documents/{file_id}` | Remove a document from index + disk |
| `GET` | `/api/v1/sessions` | List chat sessions |
| `GET` | `/api/v1/sessions/{id}/messages` | Messages of a session (with sources) |
| `DELETE` | `/api/v1/sessions/{id}` | Delete a session |

Interactive documentation for all endpoints is available at **`/docs`**.

## How Incremental Indexing Works

For every file under `data/`, the system stores a manifest row in SQLite (`indexed_files`): relative path, content SHA-256, size, mtime, status, chunk/page counts.

| Event | Detection | Action |
|---|---|---|
| **New file** | path not in manifest | parse → chunk → embed → add vectors |
| **Unchanged** | same size + mtime | **skip (no embedding)** |
| **Touched** | size/mtime changed, same hash | update stats only |
| **Modified** | content hash changed | delete old chunks → index new content |
| **Deleted** | manifest row without file | remove vectors + manifest row |
| **Model change** | `index_meta.embedding_model` mismatch | full rebuild (dimension lock-in guard) |

A failed file keeps its old vectors until a successful re-index, so transient read errors never destroy a working index.

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app, lifespan, router registration
│   ├── config.py            # pydantic-settings (env + .env, anchored paths)
│   ├── database.py          # SQLAlchemy models: pdfs, sessions, messages, indexed_files
│   ├── dependencies.py      # cached service providers
│   ├── schemas.py           # Pydantic request/response models
│   ├── routers/             # health, models, pdfs, query, indexing, sessions
│   └── services/            # ingestion_service, pdf_services, reg_services
├── core/
│   ├── document.py          # loaders (per format), cleaning, chunking
│   ├── embeddings.py        # Chroma wrapper (add/delete/search, cosine)
│   ├── llm.py               # ChatOllama + grounded prompt
│   └── reg.py               # retrieve → threshold → generate pipeline
└── data/                    # internal storage: api.db, vectors/
```

Documents live separately in the top-level **`data/`** directory of the repository.
