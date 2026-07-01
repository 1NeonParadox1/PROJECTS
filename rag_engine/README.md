# RAG Knowledge Engine

A production-grade Retrieval-Augmented Generation service: hybrid
retrieval (dense vector + BM25 keyword search) fused with Reciprocal Rank
Fusion, cross-encoder reranking, and citation-grounded generation via
Claude — served behind a FastAPI API.

## Architecture

```
                     ┌─────────────┐
   documents ───────▶│  Ingestion  │  load → chunk (recursive, token-aware)
   (pdf/docx/txt/md)  └──────┬──────┘
                              │ embed (local or OpenAI)
                              ▼
                    ┌───────────────────┐
                    │  Chroma (vectors) │◀────┐
                    └───────────────────┘     │ rebuilt on ingest
                    ┌───────────────────┐     │
                    │  BM25 (keywords)  │◀────┘
                    └───────────────────┘
                              │
         query ──▶  vector search + BM25 search
                              │
                    Reciprocal Rank Fusion
                              │
                    Cross-encoder reranking
                              │
                    top-N chunks → Claude
                              │
                    grounded answer + citations
```

**Why this design:**
- **Hybrid retrieval** — dense embeddings catch semantic matches, BM25 catches exact terms (IDs, acronyms) that embeddings often miss. Neither alone is reliable enough for production.
- **RRF fusion** — combines two very differently-scaled ranking signals without needing score normalization.
- **Reranking** — a cross-encoder scores (query, chunk) pairs jointly, which is far more precise than bi-encoder similarity alone. This is the highest-leverage step for cutting irrelevant context.
- **Citation-grounded generation** — the system prompt forces the model to cite sources and say "not found" when the context is insufficient, which is the main defense against hallucination.
- **Swappable components** — embeddings, vector store, and reranker are each isolated behind a single file/class, so scaling up (e.g. Chroma → Qdrant, local embeddings → OpenAI) touches one module.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY and API_KEY at minimum
```

## Ingest documents

CLI (batch):
```bash
python -m app.ingest /path/to/docs_directory
```

Or via API (see below).

## Run the API

```bash
uvicorn app.api:app --reload
```

Or with Docker:
```bash
docker compose up --build
```

## API usage

**Ingest a file:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@handbook.pdf"
```

**Query:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our refund policy?"}'
```

Response:
```json
{
  "answer": "Refunds are issued within 30 days of purchase... [1]",
  "sources": [{"index": 1, "source": "handbook.pdf", "chunk_id": "...", "text_preview": "..."}],
  "retrieved_chunks": 6,
  "latency_ms": 812
}
```

**Delete a document:**
```bash
curl -X DELETE http://localhost:8000/documents/handbook.pdf \
  -H "Authorization: Bearer $API_KEY"
```

**Health check:**
```bash
curl http://localhost:8000/health
```

## Production considerations / next steps

- **Scale**: Chroma + in-memory BM25 comfortably handles up to ~hundreds of thousands of chunks on a single node. Beyond that, move to a managed vector DB (Qdrant/Weaviate/pgvector) and a real search engine (OpenSearch) for BM25.
- **Eval**: add a retrieval eval set (query → expected source) and track recall@k / MRR before and after any pipeline change. Log `fused_score`/`rerank_score` to catch retrieval drift.
- **Caching**: add a query-embedding cache (e.g. Redis) if the same questions repeat often.
- **Streaming**: `generator.py` can be switched to `client.messages.stream(...)` for token-by-token responses to the client.
- **Multi-tenancy**: add a `tenant_id` metadata field and filter on it in both vector and BM25 search if serving multiple customers from one deployment.
- **Observability**: wrap `/query` with request tracing (OpenTelemetry) — retrieval and generation latency should be logged separately.
- **Security**: the current auth is a single static bearer token — fine for internal use; use per-client API keys or OAuth for external exposure.
