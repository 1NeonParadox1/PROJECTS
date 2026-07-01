"""
Embedding provider abstraction. Swappable between a local sentence-transformers
model (default, no API key / no network needed at inference time after the
model is cached) and OpenAI's embeddings API. Both expose the same interface
so the rest of the pipeline doesn't care which is in use.
"""
from __future__ import annotations

from functools import lru_cache

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class LocalEmbeddingProvider(EmbeddingProvider):
    """Runs entirely on-device via sentence-transformers. Good default for
    production because it has no per-call cost, no external dependency at
    query time, and predictable latency."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading local embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )
        return vectors.tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str, api_key: str):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model_name

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
    def embed(self, texts: list[str]) -> list[list[float]]:
        # OpenAI recommends batches of <= 2048 inputs; we also cap payload
        # size defensively for very large ingest jobs.
        out: list[list[float]] = []
        for i in range(0, len(texts), 256):
            batch = texts[i : i + 256]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        return OpenAIEmbeddingProvider(settings.embedding_model, settings.openai_api_key)
    return LocalEmbeddingProvider(settings.embedding_model)
