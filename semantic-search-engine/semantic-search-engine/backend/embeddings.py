"""
Text embedding layer using a local sentence-transformers model
(all-MiniLM-L6-v2: 384-dim, fast, strong general-purpose semantic embeddings).

Kept local/offline-friendly on purpose (no OpenAI API key required) so the
project is trivially runnable and free to demo -- but swapping in
`openai.embeddings.create(...)` is a one-function change (see embed_texts below)
if you'd rather show off API integration on a resume/demo.
"""
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a batch of texts -> (N, EMBEDDING_DIM) float32 array, L2-normalized
    so that cosine similarity == dot product (what Qdrant uses under COSINE distance
    anyway, but normalizing keeps scores stable/comparable)."""
    model = get_embedder()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
