"""
Central configuration for the Semantic Search Engine backend.
Reads from environment variables where useful, with sane local defaults
so the project runs out-of-the-box on a laptop with zero external services.
"""
import os
from pathlib import Path

# ---- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR = BASE_DIR / "storage" / "audio"
QDRANT_LOCAL_PATH = str(BASE_DIR / "storage" / "qdrant_db")

for d in (UPLOAD_DIR, AUDIO_DIR, BASE_DIR / "storage"):
    d.mkdir(parents=True, exist_ok=True)

# ---- ASR (Whisper) -----------------------------------------------------------
# Model sizes: tiny, base, small, medium, large-v3 (bigger = more accurate, slower)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")          # "cuda" if GPU available
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # fast + low-memory on CPU

# ---- Embeddings --------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # must match the embedding model's output dimension

# ---- Chunking -----------------------------------------------------------------
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", 350))   # ~ a few sentences per chunk
CHUNK_OVERLAP_SECONDS = float(os.getenv("CHUNK_OVERLAP_SECONDS", 2.0))

# ---- Vector store (Qdrant) -----------------------------------------------------
# Two modes:
#  1) Embedded/local mode (default) - no server needed, persists to disk. Great for demos.
#  2) Remote mode - point at a running Qdrant server (e.g. via docker-compose.yml)
QDRANT_URL = os.getenv("QDRANT_URL")              # e.g. "http://localhost:6333" -> remote mode
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")      # for Qdrant Cloud, optional
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "media_chunks")

# ---- Misc ----------------------------------------------------------------------
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", 10))
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".mp3", ".wav", ".m4a", ".flac", ".webm"}
