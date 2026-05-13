# RAG Pinecone — Production-Ready Retrieval-Augmented Generation

A production-ready RAG system built with **FastAPI**, **Pinecone Serverless**,
**Google Gemini Embeddings**, **FlashRank** reranking, and **Groq LLM** via LiteLLM.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT REQUEST                              │
│                     POST /query or /ingest                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │   FastAPI   │
                    │  + Rate     │
                    │  Limiter    │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐  ┌─────▼──────┐  ┌──────▼──────┐
   │  /query     │  │  /ingest   │  │  /health    │
   └──────┬──────┘  └─────┬──────┘  └─────────────┘
          │               │
          │         ┌─────▼──────┐
          │         │   Celery   │──── Upstash Redis (TLS)
          │         │   Worker   │
          │         └─────┬──────┘
          │               │
          │    ┌──────────┼──────────┐
          │    │ Loader → Chunker → │
          │    │ Embedder → Upserter│
          │    └──────────┬─────────┘
          │               │
   ┌──────▼──────┐  ┌─────▼──────┐
   │  Semantic   │  │  Pinecone  │
   │  Cache      │  │  (768-dim) │
   │  (Upstash)  │  │  cosine    │
   └──────┬──────┘  └─────┬──────┘
          │               │
   ┌──────▼───────────────▼──────┐
   │  HyDE + Multi-Query Search  │
   │  → FlashRank Reranking      │
   │  → Prompt Builder           │
   │  → Groq LLM (LiteLLM)      │
   │  → Citation Parser          │
   └──────┬──────────────────────┘
          │
   ┌──────▼──────┐
   │  Langfuse   │
   │  Tracing    │
   └─────────────┘
```

---

## Quick Start

### 1. Clone & Configure

```bash
git clone <repo-url>
cd rag-pinecone
cp .env.example .env
# Fill in all API keys in .env
```

### 2. Install Dependencies

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start the Celery Worker (separate terminal)

```bash
celery -A app.workers.celery_app worker --loglevel=info --concurrency=2
```

### 5. Using Docker Compose

```bash
docker-compose up --build
```

> **Note:** FlashRank model weights (~22 MB) are downloaded automatically on
> first startup and cached in the `flashrank_cache` Docker volume.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `PINECONE_API_KEY` | ✅ | Pinecone API key |
| `PINECONE_HOST` | ✅ | Pinecone index host URL |
| `PINECONE_INDEX_NAME` | ✅ | Pinecone index name |
| `PINECONE_CLOUD` | ❌ | Cloud provider (default: `aws`) |
| `PINECONE_REGION` | ❌ | Region (default: `us-east-1`) |
| `GEMINI_API_KEY` | ✅ | Google Gemini API key |
| `GROQ_API_KEY` | ✅ | Groq API key |
| `UPSTASH_REDIS_REST_URL` | ✅ | Upstash Redis REST endpoint |
| `UPSTASH_REDIS_REST_TOKEN` | ✅ | Upstash Redis REST token |
| `LANGFUSE_SECRET_KEY` | ✅ | Langfuse secret key |
| `LANGFUSE_PUBLIC_KEY` | ✅ | Langfuse public key |
| `LANGFUSE_BASE_URL` | ❌ | Langfuse host (default: `https://us.cloud.langfuse.com`) |

---

## API Reference

### `GET /health`

Health check endpoint. No auth required.

**Response (200):**
```json
{
  "status": "ok",
  "checks": {
    "pinecone": "ok",
    "redis": "ok",
    "celery": "ok"
  }
}
```
Returns **503** if any check fails.

---

### `POST /ingest`

Dispatch an async document ingestion task.

**Request:**
```json
{
  "source_url_or_path": "https://example.com/paper.pdf",
  "namespace": "my-project"
}
```

**Response (200):**
```json
{
  "task_id": "abc123-...",
  "status": "queued"
}
```

---

### `GET /ingest/{task_id}/status`

Poll the status of an ingestion task.

**Response:**
```json
{
  "task_id": "abc123-...",
  "status": "PROGRESS",
  "progress": {"step": "embedding", "progress": 50},
  "result": null,
  "error": null
}
```

---

### `POST /query`

Ask a question against your ingested documents.

**Request:**
```json
{
  "question": "What is machine learning?",
  "namespace": "my-project",
  "stream": false
}
```

**Response (200):**
```json
{
  "answer": "Machine learning is a subset of AI that...[1]",
  "no_answer": false,
  "sources": [
    {"text": "...", "source": "ml.pdf", "page": 5, "score": 0.95}
  ],
  "cached": false,
  "latency_ms": 1420
}
```

**No-answer response:**
```json
{
  "answer": null,
  "no_answer": true,
  "sources": [],
  "cached": false,
  "latency_ms": 850
}
```

**SSE Streaming (`stream: true`):**
- Events with `event: token` stream answer tokens
- Final `event: done` contains the full response JSON with sources

---

## How to Swap the LLM

Only change the `LLM_MODEL` setting — nothing else:

```env
# In .env — change from Groq to OpenAI, Anthropic, etc.
LLM_MODEL=groq/llama-3.3-70b-versatile     # default
LLM_MODEL=gpt-4o                            # OpenAI
LLM_MODEL=claude-3-5-sonnet-20241022        # Anthropic
```

LiteLLM handles the routing. Add the corresponding API key
(e.g., `OPENAI_API_KEY`) to `.env`.

---

## Multi-Tenancy via Namespaces

Every `/ingest` and `/query` request includes a `namespace` parameter.
Namespaces provide **complete data isolation** within a single Pinecone
index — documents in namespace `"project-a"` are invisible to queries
targeting namespace `"project-b"`.

Use cases:
- Per-user document stores
- Per-project knowledge bases
- Environment separation (dev/staging/prod)

---

## Hybrid Search Roadmap

This build uses **dense-only search** (Gemini 768-dim embeddings) +
FlashRank reranking. To add hybrid search in a future phase:

1. **Install** `pinecone-text` for sparse encoding:
   ```bash
   pip install pinecone-text
   ```

2. **Extend `embedder.py`** to generate sparse vectors:
   ```python
   from pinecone_text.sparse import BM25Encoder
   bm25 = BM25Encoder()
   bm25.fit(corpus)  # fit on your document corpus
   sparse_values = bm25.encode_documents(text)
   ```

3. **Extend `upserter.py`** to include `sparse_values` in each upsert record:
   ```python
   {"id": ..., "values": dense, "sparse_values": sparse, "metadata": ...}
   ```

4. **Update `searcher.py`** to pass both `vector` and `sparse_vector`
   in query calls.

---

## Architecture Details

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API Framework | FastAPI + Uvicorn | HTTP API with async support |
| Vector DB | Pinecone Serverless | Dense vector storage (768-dim, cosine) |
| Embeddings | Google Gemini `text-embedding-004` | 768-dim document/query embeddings |
| Reranker | FlashRank (`ms-marco-TinyBERT-L-2-v2`) | Local CPU cross-encoder reranking |
| LLM | Groq `llama-3.3-70b-versatile` via LiteLLM | Answer generation with citations |
| Cache | Upstash Redis (REST SDK) | Semantic query cache (cosine > 0.92) |
| Queue | Celery + Upstash Redis (TLS) | Async document ingestion |
| Observability | Langfuse | Query/ingestion tracing |

---

## Running Tests

```bash
# Run all tests (unit tests work without API keys)
pytest tests/ -v

# Run only unit tests (no API keys needed)
pytest tests/ -v -k "not skipif"

# Run integration tests (requires API keys in .env)
pytest tests/ -v --tb=short
```

---

## License

MIT
