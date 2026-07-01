"""
Ingestion pipeline. Can be run as a CLI for batch loading a directory, or
called from the API for single-file uploads. Always rebuilds the BM25
index after writing to the vector store so keyword search stays in sync.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.bm25_index import get_bm25_index
from app.chunking import chunk_document
from app.config import get_settings
from app.logging_config import configure_logging, get_logger
from app.vector_store import get_vector_store

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def ingest_file(path: str | Path, extra_metadata: dict | None = None) -> int:
    settings = get_settings()
    store = get_vector_store()
    chunks = chunk_document(
        path, settings.chunk_size, settings.chunk_overlap, extra_metadata
    )
    count = store.upsert_chunks(chunks)
    get_bm25_index().build(store)  # keep keyword index in sync
    return count


def ingest_directory(directory: str | Path) -> int:
    directory = Path(directory)
    total = 0
    for path in directory.rglob("*"):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                n = ingest_file(path)
                logger.info("Ingested %s (%d chunks)", path.name, n)
                total += n
            except Exception:
                logger.exception("Failed to ingest %s", path)
    return total


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base")
    parser.add_argument("path", help="File or directory to ingest")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_dir():
        total = ingest_directory(target)
    else:
        total = ingest_file(target)
    logger.info("Done. %d total chunks ingested.", total)
