from typing import List, Optional
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    file_id: str
    filename: str
    duration_seconds: float
    num_chunks: int
    status: str = "indexed"


class SearchHit(BaseModel):
    file_id: str
    filename: str
    chunk_id: int
    text: str
    start_time: float
    end_time: float
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchHit]


class LibraryItem(BaseModel):
    file_id: str
    filename: str
    duration_seconds: float
    num_chunks: int
    media_url: str


class LibraryResponse(BaseModel):
    items: List[LibraryItem]


class JobStatus(BaseModel):
    file_id: str
    status: str               # "processing" | "transcribing" | "embedding" | "done" | "error"
    progress: float = 0.0     # 0.0 - 1.0
    message: Optional[str] = None
