"""
Cross-encoder reranking. Vector + BM25 retrieval optimizes for recall over
a broad candidate set; a cross-encoder then scores (query, chunk) pairs
jointly for much higher precision on the final top-N shown to the LLM.
This is the single highest-leverage step for reducing irrelevant context.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class Reranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker model: %s", model_name)
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]


@lru_cache
def get_reranker() -> Reranker | None:
    settings = get_settings()
    if not settings.use_reranker:
        return None
    return Reranker(settings.reranker_model)
