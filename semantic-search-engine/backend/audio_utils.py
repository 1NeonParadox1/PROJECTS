"""
Extracts a clean mono 16kHz WAV track from any uploaded video/audio file using FFmpeg.
Whisper expects 16kHz mono audio for best performance, and pre-converting once
keeps the transcription step fast and format-agnostic (mp4, mkv, mp3, etc. all
get normalized the same way).
"""
import subprocess
from pathlib import Path


def extract_wav(input_path: Path, output_path: Path) -> None:
    """
    Run ffmpeg to convert any audio/video input into a 16kHz mono WAV file.
    Raises RuntimeError with ffmpeg's stderr if conversion fails.
    """
    cmd = [
        "ffmpeg",
        "-y",                 # overwrite output
        "-i", str(input_path),
        "-vn",                 # drop video stream
        "-acodec", "pcm_s16le",
        "-ar", "16000",        # 16kHz sample rate (Whisper's native rate)
        "-ac", "1",            # mono
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-2000:]}")


def get_duration_seconds(input_path: Path) -> float:
    """Use ffprobe to fetch media duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0
