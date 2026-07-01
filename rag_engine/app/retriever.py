"""
Hybrid retrieval pipeline:
  1. Run dense vector search and BM25 keyword search in parallel candidate sets.
  2. Fuse the two rankings with Reciprocal Rank Fusion (RRF) -- robust to the
     very different score scales of cosine similarity vs. BM25 scores.
  3. Optionally rerank the fused candidates with a cross-encoder for a final
     precision pass before handing chunks to the LLM.
"""
from __future__ import annotations

from app.bm25_index import get_bm25_index
from app.config import get_settings
from app.logging_config import get_logger
from app.reranker import get_reranker
from app.vector_store import get_vector_store

logger = get_logger(__name__)


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]], k: int
) -> list[dict]:
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item["id"]] = scores.get(item["id"], 0.0) + 1.0 / (k + rank + 1)
            payload[item["id"]] = item
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out = []
    for doc_id, score in fused:
        item = dict(payload[doc_id])
        item["fused_score"] = score
        out.append(item)
    return out


def retrieve(query: str, where: dict | None = None) -> list[dict]:
    settings = get_settings()
    store = get_vector_store()
    bm25 = get_bm25_index()

    vector_hits = store.query(query, top_k=settings.top_k_vector, where=where)
    keyword_hits = bm25.search(query, top_k=settings.top_k_bm25)

    fused = _reciprocal_rank_fusion([vector_hits, keyword_hits], k=settings.rrf_k)

    reranker = get_reranker()
    if reranker is not None:
        # Rerank a generous slice of the fused list, not just top_k_final,
        # so the cross-encoder can recover good chunks RRF ranked lower.
        candidate_pool = fused[: max(settings.top_k_final * 4, 20)]
        final = reranker.rerank(query, candidate_pool, top_k=settings.top_k_final)
    else:
        final = fused[: settings.top_k_final]

    logger.info(
        "Retrieved: %d vector, %d bm25 -> %d fused -> %d final",
        len(vector_hits), len(keyword_hits), len(fused), len(final),
    )
    return final
