# Resonance — Real-Time Semantic Search Engine for Video/Audio

Search a library of video/audio files **by meaning**, not keywords. Upload a file,
the system transcribes it with Whisper, chunks and embeds the transcript, and lets
you ask things like *"where did they talk about budget cuts"* — returning the exact
timestamp, with a player that jumps straight there.

```
Upload  ─▶  FFmpeg (extract 16kHz mono audio)
        ─▶  faster-whisper ASR (timestamped transcript segments)
        ─▶  Chunking (merge segments into ~350-char semantic chunks)
        ─▶  sentence-transformers (384-dim embeddings, all-MiniLM-L6-v2)
        ─▶  Qdrant (vector upsert with timestamp payload)

Search  ─▶  Embed query  ─▶  Qdrant cosine similarity search  ─▶  ranked
            timestamped results  ─▶  click to seek video/audio player
```

## Why this project

This is a small but complete example of a **Retrieval-Augmented** style pipeline
built end-to-end: ASR, an embedding model, a real vector database, an async
ingestion pipeline with background jobs/progress polling, and a UI that ties
search results back to playable timestamps. It demonstrates:

- **AI/ML**: ASR (Whisper), text embeddings, semantic similarity search, RAG-style retrieval
- **Backend/SDE**: FastAPI, background job processing, FFmpeg audio pipelines, vector DB integration (Qdrant), clean REST API design
- **Frontend**: a dependency-free single-page UI with upload progress, scoped search, and a synced media player

## Project layout

```
semantic-search-engine/
├── backend/
│   ├── main.py            # FastAPI app & routes
│   ├── config.py          # central settings
│   ├── models.py          # Pydantic schemas
│   ├── audio_utils.py     # FFmpeg extraction
│   ├── transcribe.py      # faster-whisper ASR
│   ├── chunking.py        # segment -> chunk merging
│   ├── embeddings.py      # sentence-transformers wrapper
│   ├── vectorstore.py     # Qdrant wrapper
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── docker-compose.yml      # optional standalone Qdrant server
└── README.md
```

## Quickstart (local, zero external services)

**Prerequisites:** Python 3.10+, and `ffmpeg`/`ffprobe` on your PATH.

```bash
# 1. System dependency
sudo apt-get install ffmpeg        # macOS: brew install ffmpeg

# 2. Python environment
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run it
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — the FastAPI app serves the frontend directly, no
separate dev server needed. Drag in an MP3/MP4, wait for the progress bar to hit
"done", then search.

> First run will download the Whisper (`base`, ~140MB) and embedding
> (`all-MiniLM-L6-v2`, ~90MB) models automatically — that's the only "setup" needed.
> No API keys required anywhere in the default configuration.

## Running with a real Qdrant server ("production mode")

By default the vector store runs **embedded** (in-process, persisted to
`storage/qdrant_db/`) — convenient, but single-process only. To run against a real
Qdrant server (closer to a production deployment, and lets multiple backend workers
share one index):

```bash
docker compose up -d
export QDRANT_URL=http://localhost:6333
uvicorn main:app --reload --port 8000
```

## API reference

| Method | Path                  | Description                                      |
|--------|-----------------------|---------------------------------------------------|
| POST   | `/api/ingest`          | Upload a file (multipart), kicks off background processing, returns `file_id` |
| GET    | `/api/status/{file_id}`| Poll ingestion progress (`queued → transcribing → embedding → indexing → done`) |
| GET    | `/api/search?q=...`    | Semantic search across the library (optional `&file_id=` to scope to one file) |
| GET    | `/api/library`         | List all indexed files with chunk counts/durations |
| DELETE | `/api/files/{file_id}` | Remove a file and its vectors from the index      |
| GET    | `/api/media/{file_id}` | Stream the original media file (used by the `<video>` player) |

## Configuration

All tunable via environment variables (see `backend/.env.example`):

- `WHISPER_MODEL_SIZE` — `tiny`/`base`/`small`/`medium`/`large-v3`. Bigger = more accurate, slower.
- `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` — set `WHISPER_DEVICE=cuda` if you have a GPU.
- `EMBEDDING_MODEL_NAME` — swap to any sentence-transformers model.
- `QDRANT_URL` — unset = embedded mode; set to use a remote/dockerized Qdrant.
- `CHUNK_MAX_CHARS` — controls how much context lands in each embedded chunk.

## Design notes / things worth highlighting in an interview

- **Why faster-whisper over openai-whisper**: ~4x faster CPU inference via
  CTranslate2, with built-in VAD filtering to skip silence — both matter when you
  don't have a GPU lying around for a demo.
- **Why chunk merging instead of embedding raw Whisper segments**: raw segments are
  often sub-sentence fragments; embedding them individually loses context and hurts
  retrieval quality. Chunks are built by greedily merging consecutive segments up to
  a character budget while preserving true start/end timestamps.
- **Why Qdrant in embedded mode by default**: zero infra to demo the project, but
  the exact same code path (`vectorstore.py`) talks to a real Qdrant server just by
  setting `QDRANT_URL` — a clean illustration of designing for both dev and prod.
- **Background jobs without Celery/Redis**: a simple in-memory job tracker + daemon
  thread per upload keeps the project dependency-light while still demonstrating
  async/non-blocking ingestion with progress polling. Documented as the first thing
  to swap for a task queue (Celery/RQ) in a real multi-worker deployment.

## Possible extensions (good "what would you add next" answers)

- Swap in `openai.embeddings.create()` to demonstrate hosted-API integration alongside the local model
- Add diarization (`pyannote.audio`) to label *who* said what, not just *when*
- Add a re-ranking step (cross-encoder) on top of the initial vector search
- Stream transcript chunks to the frontend via WebSocket as they're generated
- Multi-tenant auth + per-user collections in Qdrant
