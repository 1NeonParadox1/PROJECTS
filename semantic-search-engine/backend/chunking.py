"""
Chunking layer.

Whisper produces many tiny segments (often single sentences or even sentence
fragments). Embedding each tiny segment individually hurts semantic search quality
because there's not enough context per vector. Instead we greedily merge consecutive
segments into chunks of up to CHUNK_MAX_CHARS characters, while keeping track of the
true start/end timestamps of the merged span -- this is what lets us point the user
to the exact moment in the video.
"""
from dataclasses import dataclass
from typing import List

import config
from transcribe import TranscriptSegment


@dataclass
class TextChunk:
    chunk_id: int
    text: str
    start_time: float
    end_time: float


def build_chunks(segments: List[TranscriptSegment]) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    if not segments:
        return chunks

    buffer_text = []
    buffer_start = segments[0].start
    buffer_end = segments[0].end
    chunk_id = 0

    def flush():
        nonlocal buffer_text, buffer_start, buffer_end, chunk_id
        if buffer_text:
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=" ".join(buffer_text).strip(),
                    start_time=buffer_start,
                    end_time=buffer_end,
                )
            )
            chunk_id += 1
        buffer_text = []

    for seg in segments:
        candidate_len = sum(len(t) for t in buffer_text) + len(seg.text)
        if buffer_text and candidate_len > config.CHUNK_MAX_CHARS:
            flush()
            buffer_start = seg.start
        if not buffer_text:
            buffer_start = seg.start
        buffer_text.append(seg.text)
        buffer_end = seg.end

    flush()
    return chunks
