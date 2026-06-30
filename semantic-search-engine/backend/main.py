"""
FastAPI backend for the Real-Time Semantic Search Engine for Video/Audio.

Pipeline (ingest_pipeline):
  upload -> ffmpeg extract 16kHz mono wav -> faster-whisper ASR (timestamped
  segments) -> merge into text chunks -> sentence-transformers embeddings ->
  upsert into Qdrant with timestamp payload.

Search simply embeds the query and does a cosine-similarity nearest-neighbor
search in Qdrant, returning the original media file + timestamp so the
frontend can seek the <video>/<audio> element straight there.

Run with:  uvicorn main:app --reload --port 8000
"""
import shutil
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from audio_utils import extract_wav, get_duration_seconds
from chunking import build_chunks
from embeddings import embed_query, embed_texts
from models import IngestResponse, JobStatus, LibraryItem, LibraryResponse, SearchHit, SearchResponse
from transcribe import transcribe_audio
from vectorstore import get_store

app = FastAPI(
    title="Real-Time Semantic Search Engine for Video/Audio",
    description="Upload audio/video, transcribe with Whisper, and search by meaning instead of keywords.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job tracker. For a multi-worker production deployment this would move
# to Redis, but for a single-process demo this is simple and effective.
JOBS: Dict[str, JobStatus] = {}
JOBS_LOCK = threading.Lock()

# Map file_id -> stored media path, so we can stream it back to the <video> tag
FILE_PATHS: Dict[str, Path] = {}


def _set_job(file_id: str, **kwargs):
    with JOBS_LOCK:
        current = JOBS.get(file_id)
        if current is None:
            current = JobStatus(file_id=file_id, status="processing")
        data = current.model_dump()
        data.update(kwargs)
        JOBS[file_id] = JobStatus(**data)


def ingest_pipeline(file_id: str, media_path: Path, filename: str):
    """Runs in a background thread so the upload request returns immediately."""
    try:
        _set_job(file_id, status="processing", progress=0.05, message="Extracting audio")
        wav_path = config.AUDIO_DIR / f"{file_id}.wav"
        extract_wav(media_path, wav_path)
        duration = get_duration_seconds(media_path)

        _set_job(file_id, status="transcribing", progress=0.25, message="Running Whisper ASR")
        segments = transcribe_audio(wav_path)

        _set_job(file_id, status="chunking", progress=0.55, message="Building text chunks")
        chunks = build_chunks(segments)
        if not chunks:
            _set_job(file_id, status="error", progress=1.0, message="No speech detected in file")
            return

        _set_job(file_id, status="embedding", progress=0.7, message="Generating embeddings")
        vectors = embed_texts([c.text for c in chunks])

        _set_job(file_id, status="indexing", progress=0.9, message="Writing to vector store")
        store = get_store()
        store.upsert_chunks(file_id, filename, vectors, chunks)

        _set_job(
            file_id,
            status="done",
            progress=1.0,
            message=f"Indexed {len(chunks)} chunks ({duration:.1f}s media)",
        )
        # Clean up intermediate wav to save disk; keep original media for playback
        wav_path.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client
        _set_job(file_id, status="error", progress=1.0, message=str(exc))


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}")

    file_id = uuid.uuid4().hex[:12]
    dest_path = config.UPLOAD_DIR / f"{file_id}{ext}"
    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    FILE_PATHS[file_id] = dest_path
    _set_job(file_id, status="queued", progress=0.0, message="Queued for processing")

    thread = threading.Thread(target=ingest_pipeline, args=(file_id, dest_path, file.filename), daemon=True)
    thread.start()

    duration = get_duration_seconds(dest_path)
    return IngestResponse(file_id=file_id, filename=file.filename, duration_seconds=duration, num_chunks=0, status="queued")


@app.get("/api/status/{file_id}", response_model=JobStatus)
async def status(file_id: str):
    job = JOBS.get(file_id)
    if job is None:
        raise HTTPException(404, "Unknown file_id")
    return job


@app.get("/api/search", response_model=SearchResponse)
async def search(q: str = Query(..., min_length=1), file_id: Optional[str] = None, top_k: int = config.MAX_SEARCH_RESULTS):
    store = get_store()
    query_vec = embed_query(q)
    raw_hits = store.search(query_vec, top_k=top_k, file_id=file_id)

    results = [
        SearchHit(
            file_id=h.payload["file_id"],
            filename=h.payload["filename"],
            chunk_id=h.payload["chunk_id"],
            text=h.payload["text"],
            start_time=h.payload["start_time"],
            end_time=h.payload["end_time"],
            score=h.score,
        )
        for h in raw_hits
    ]
    return SearchResponse(query=q, results=results)


@app.get("/api/library", response_model=LibraryResponse)
async def library():
    store = get_store()
    files = store.list_files()
    items = [
        LibraryItem(
            file_id=f["file_id"],
            filename=f["filename"],
            duration_seconds=f["duration_seconds"],
            num_chunks=f["num_chunks"],
            media_url=f"/api/media/{f['file_id']}",
        )
        for f in files
    ]
    return LibraryResponse(items=items)


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str):
    store = get_store()
    store.delete_file(file_id)
    path = FILE_PATHS.pop(file_id, None)
    if path and path.exists():
        path.unlink()
    JOBS.pop(file_id, None)
    return {"deleted": file_id}


@app.get("/api/media/{file_id}")
async def get_media(file_id: str):
    path = FILE_PATHS.get(file_id)
    if path is None:
        # Fall back to scanning uploads dir in case the server restarted
        matches = list(config.UPLOAD_DIR.glob(f"{file_id}.*"))
        if not matches:
            raise HTTPException(404, "File not found")
        path = matches[0]
        FILE_PATHS[file_id] = path
    return FileResponse(path)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve the frontend (single-page app) at the root, after all /api routes are defined.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
