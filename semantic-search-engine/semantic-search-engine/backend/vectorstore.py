"""
Vector database layer backed by Qdrant.

Supports two modes, controlled purely by the QDRANT_URL env var:
  - Local/embedded mode (default): Qdrant runs in-process and persists to disk
    at storage/qdrant_db. Zero setup, perfect for a CV demo or local dev.
  - Remote mode: set QDRANT_URL=http://localhost:6333 (see docker-compose.yml)
    to talk to a real Qdrant server -- this is what you'd use in production /
    when you want to show you understand client-server vector DB deployments.

Each point stored in the collection carries a payload with everything needed to
render a search result: filename, the chunk text, and the start/end timestamps
in the source media.
"""
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import config


class VectorStore:
    def __init__(self):
        if config.QDRANT_URL:
            self.client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        else:
            self.client = QdrantClient(path=config.QDRANT_LOCAL_PATH)
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if config.QDRANT_COLLECTION not in collections:
            self.client.create_collection(
                collection_name=config.QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=config.EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            # Index file_id so we can filter/delete per-file efficiently
            self.client.create_payload_index(
                collection_name=config.QDRANT_COLLECTION,
                field_name="file_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

    def upsert_chunks(self, file_id: str, filename: str, vectors, chunks) -> None:
        points = []
        for vec, chunk in zip(vectors, chunks):
            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec.tolist(),
                    payload={
                        "file_id": file_id,
                        "filename": filename,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "start_time": chunk.start_time,
                        "end_time": chunk.end_time,
                    },
                )
            )
        self.client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)

    def search(self, query_vector, top_k: int, file_id: Optional[str] = None):
        query_filter = None
        if file_id:
            query_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(key="file_id", match=qmodels.MatchValue(value=file_id))]
            )
        return self.client.search(
            collection_name=config.QDRANT_COLLECTION,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
        )

    def delete_file(self, file_id: str) -> None:
        self.client.delete(
            collection_name=config.QDRANT_COLLECTION,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="file_id", match=qmodels.MatchValue(value=file_id))]
                )
            ),
        )

    def list_files(self) -> List[dict]:
        """Scroll through all points and aggregate per-file metadata for the library view."""
        files = {}
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=config.QDRANT_COLLECTION,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                fid = p.payload["file_id"]
                if fid not in files:
                    files[fid] = {
                        "file_id": fid,
                        "filename": p.payload["filename"],
                        "num_chunks": 0,
                        "duration_seconds": 0.0,
                    }
                files[fid]["num_chunks"] += 1
                files[fid]["duration_seconds"] = max(files[fid]["duration_seconds"], p.payload["end_time"])
            if next_offset is None:
                break
        return list(files.values())


_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
