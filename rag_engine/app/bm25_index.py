"""
Keyword (BM25) index used alongside vector search for hybrid retrieval.
Dense embeddings are great at semantic similarity but weak on exact terms
(IDs, acronyms, product codes, rare proper nouns) -- BM25 covers that gap.
Kept in-memory and rebuilt from the vector store's contents; cheap enough
to rebuild on ingest for corpora up to roughly hundreds of thousands of
chunks. At larger scale, swap for a real search engine (e.g. OpenSearch).
"""
from __future__ import annotations

import re
import threading

from rank_bm25 import BM25Okapi

from app.logging_config import get_logger
from app.vector_store import VectorStore

logger = get_logger(__name__)

_token_re = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _token_re.findall(text.lower())


class BM25Index:
    def __init__(self):
        self._lock = threading.Lock()
        self._bm25: BM25Okapi | None = None
        self._docs: list[dict] = []

    def build(self, store: VectorStore) -> None:
        docs = store.all_documents()
        with self._lock:
            self._docs = docs
            corpus = [_tokenize(d["text"]) for d in docs]
            self._bm25 = BM25Okapi(corpus) if corpus else None
        logger.info("BM25 index built over %d chunks", len(docs))

    def search(self, query: str, top_k: int) -> list[dict]:
        with self._lock:
            if self._bm25 is None:
                return []
            scores = self._bm25.get_scores(_tokenize(query))
            docs = self._docs
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"id": d["id"], "text": d["text"], "metadata": d["metadata"], "score": float(s)}
            for d, s in ranked
            if s > 0
        ]


_index = BM25Index()


def get_bm25_index() -> BM25Index:
    return _index
