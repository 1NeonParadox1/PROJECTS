"""
Document loading + chunking.

Chunking strategy: recursive character splitting that tries progressively
finer separators (section breaks -> paragraphs -> sentences -> words) so
splits fall on natural boundaries whenever possible, with a token-based
size budget (tiktoken) and configurable overlap so context isn't lost at
chunk edges.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")

_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _token_len(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_on_separator(text: str, separators: list[str]) -> list[str]:
    if not separators:
        return [text]
    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split_on_separator(text, rest)
    parts = [p for p in text.split(sep) if p.strip()]
    return parts if parts else _split_on_separator(text, rest)


def recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks <= chunk_size tokens, merging small pieces
    back together and carrying `chunk_overlap` tokens of trailing context
    forward into the next chunk."""
    raw_pieces = _split_on_separator(text, _SEPARATORS)

    chunks: list[str] = []
    current = ""
    for piece in raw_pieces:
        candidate = (current + " " + piece).strip() if current else piece
        if _token_len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # piece itself may exceed chunk_size (e.g. huge paragraph) -> hard split
            if _token_len(piece) > chunk_size:
                tokens = _ENCODING.encode(piece)
                for i in range(0, len(tokens), chunk_size - chunk_overlap):
                    sub = _ENCODING.decode(tokens[i : i + chunk_size])
                    chunks.append(sub)
                current = ""
            else:
                current = piece
    if current:
        chunks.append(current)

    # add overlap by prefixing each chunk (after the first) with the tail
    # of the previous chunk
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            prev_tokens = _ENCODING.encode(prev)
            tail = _ENCODING.decode(prev_tokens[-chunk_overlap:])
            overlapped.append((tail + " " + cur).strip())
        chunks = overlapped

    return chunks


def load_document(path: str | Path) -> str:
    """Load raw text from pdf, docx, md, or txt files."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        import docx

        d = docx.Document(str(path))
        return "\n\n".join(p.text for p in d.paragraphs if p.text.strip())

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_document(
    path: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    extra_metadata: dict | None = None,
) -> list[Chunk]:
    path = Path(path)
    text = load_document(path)
    text = re.sub(r"[ \t]+", " ", text)  # normalize whitespace, keep newlines

    doc_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    pieces = recursive_split(text, chunk_size, chunk_overlap)

    chunks = []
    for idx, piece in enumerate(pieces):
        meta = {"source": path.name, "chunk_index": idx, **(extra_metadata or {})}
        chunks.append(
            Chunk(
                id=f"{doc_id}-{idx}-{uuid.uuid4().hex[:8]}",
                text=piece,
                doc_id=doc_id,
                source=path.name,
                chunk_index=idx,
                metadata=meta,
            )
        )
    return chunks
