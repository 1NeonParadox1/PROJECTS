"""
Persistent vector store wrapper. Chroma is used because it's embeddable
(no separate server to operate), persists to disk, and supports metadata
filtering out of the box -- a good default until scale demands something
like Qdrant/pgvector/Pinecone, at which point only this file needs to change.
"""
from __future__ import annotations

from functools import lru_cache

import chromadb

from app.chunking import Chunk
from app.config import get_settings
from app.embeddings import get_embedding_provider
from app.logging_config import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._embedder = get_embedding_provider()

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self._embedder.embed([c.text for c in chunks])
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        logger.info("Upserted %d chunks into collection", len(chunks))
        return len(chunks)

    def delete_by_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def query(
        self, query_text: str, top_k: int, where: dict | None = None
    ) -> list[dict]:
        vector = self._embedder.embed_query(query_text)
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist, doc_id in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            result["ids"][0],
        ):
            # cosine distance -> similarity score in [0, 1]
            hits.append(
                {"id": doc_id, "text": doc, "metadata": meta, "score": 1 - dist}
            )
        return hits

    def all_documents(self) -> list[dict]:
        """Fetch every stored chunk -- used to build/refresh the BM25 index."""
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            {"id": doc_id, "text": doc, "metadata": meta}
            for doc_id, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    def count(self) -> int:
        return self._collection.count()


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(settings.chroma_persist_dir, settings.collection_name)
