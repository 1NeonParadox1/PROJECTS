"""
Automatic Speech Recognition (ASR) layer.

Uses faster-whisper (a CTranslate2 reimplementation of OpenAI's Whisper) because it
is 4x faster and uses far less memory than the original openai-whisper package on
CPU, which matters a lot for a self-hosted demo project. The model is loaded once
and reused across requests (singleton pattern) to avoid reloading weights per call.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

from faster_whisper import WhisperModel

import config


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@lru_cache(maxsize=1)
def get_model() -> WhisperModel:
    """Lazily load and cache the Whisper model (singleton)."""
    return WhisperModel(
        config.WHISPER_MODEL_SIZE,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )


def transcribe_audio(wav_path: Path) -> List[TranscriptSegment]:
    """
    Transcribe a WAV file and return time-aligned segments.
    faster-whisper natively returns segments with start/end timestamps,
    which is exactly what we need to later jump the video player to the
    right moment.
    """
    model = get_model()
    segments_iter, info = model.transcribe(
        str(wav_path),
        beam_size=5,
        vad_filter=True,           # skip silence -> faster + cleaner transcript
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    segments: List[TranscriptSegment] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append(TranscriptSegment(start=seg.start, end=seg.end, text=text))
    return segments
