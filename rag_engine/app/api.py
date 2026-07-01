"""
FastAPI service layer. Exposes:
  POST /ingest  - upload a document (pdf/docx/txt/md) into the knowledge base
  POST /query   - ask a question, get a grounded answer with citations
  GET  /health  - liveness/readiness probe
  DELETE /documents/{source} - remove a document and its chunks

Auth is a simple bearer API key suitable for internal/service-to-service
use; swap for OAuth/JWT if exposing this externally.
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.bm25_index import get_bm25_index
from app.config import get_settings
from app.generator import generate_answer
from app.ingest import SUPPORTED_EXTENSIONS, ingest_file
from app.logging_config import configure_logging, get_logger
from app.retriever import retrieve
from app.vector_store import get_vector_store

configure_logging()
logger = get_logger(__name__)

app = FastAPI(title="RAG Knowledge Engine", version="1.0.0")


def require_api_key(authorization: str = Header(default="")) -> None:
    settings = get_settings()
    expected = f"Bearer {settings.api_key}"
    if not settings.api_key or authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    source_filter: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    retrieved_chunks: int
    latency_ms: int


@app.get("/health")
def health() -> dict:
    store = get_vector_store()
    return {"status": "ok", "indexed_chunks": store.count()}


@app.post("/ingest", dependencies=[Depends(require_api_key)])
async def ingest_endpoint(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / file.filename
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            chunk_count = ingest_file(tmp_path)
        except Exception as e:
            logger.exception("Ingestion failed for %s", file.filename)
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return {"filename": file.filename, "chunks_indexed": chunk_count}


@app.delete("/documents/{source}", dependencies=[Depends(require_api_key)])
def delete_document(source: str) -> dict:
    store = get_vector_store()
    store.delete_by_source(source)
    get_bm25_index().build(store)
    return {"deleted": source}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query_endpoint(req: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    where = {"source": req.source_filter} if req.source_filter else None

    try:
        chunks = retrieve(req.question, where=where)
        if not chunks:
            return QueryResponse(
                answer="I couldn't find any relevant information in the knowledge base to answer this question.",
                sources=[],
                retrieved_chunks=0,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
        result = generate_answer(req.question, chunks)
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        retrieved_chunks=len(chunks),
        latency_ms=int((time.perf_counter() - start) * 1000),
    )
