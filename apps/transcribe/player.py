"""Audio-source helpers for the Kirtan Player tab.

Supports:
  - Local audio files (wav / mp3 / m4a / flac / ogg) via `soundfile`.
  - YouTube URLs via `yt-dlp` (CLI or Python module, whichever is present).

The player module only prepares the audio as a numpy array and a local
filepath. Real-time sync (transcribe while browser plays) happens in the
app's generator handler.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

TARGET_SR = 16_000
CACHE_DIR = Path(tempfile.gettempdir()) / "surt_kirtan_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _resample_to(audio: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    if sr_in == sr_out or audio.size == 0:
        return audio.astype(np.float32)
    n = int(round(len(audio) * sr_out / sr_in))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def load_audio_16k(path: str | os.PathLike) -> tuple[np.ndarray, int]:
    """Decode `path` to mono float32 @ 16 kHz. Returns (audio, original_sr)."""
    import soundfile as sf  # lazy: only needed when the player tab is used
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    return _resample_to(data, sr, TARGET_SR), sr


def _yt_dlp_available() -> Optional[str]:
    if shutil.which("yt-dlp"):
        return "cli"
    try:
        import yt_dlp  # noqa: F401
        return "module"
    except ImportError:
        return None


def download_youtube_audio(url: str) -> Path:
    """Download `url` as an audio file into the local cache and return its path.

    Raises RuntimeError if yt-dlp is not available or download fails.
    """
    mode = _yt_dlp_available()
    if not mode:
        raise RuntimeError(
            "yt-dlp not installed. Install with `pip install yt-dlp` to use "
            "YouTube URLs."
        )
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    # yt-dlp writes with its own extension; use a template
    out_tpl = str(CACHE_DIR / f"yt_{key}.%(ext)s")

    # Try cached result first
    for existing in CACHE_DIR.glob(f"yt_{key}.*"):
        if existing.is_file() and existing.stat().st_size > 1024:
            return existing

    if mode == "cli":
        cmd = [
            "yt-dlp", "-x", "--audio-format", "mp3",
            "-o", out_tpl, url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {proc.stderr.strip()[:400]}")
    else:
        import yt_dlp  # type: ignore
        opts = {
            "format": "bestaudio/best",
            "outtmpl": out_tpl,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "160",
            }],
            "quiet": True, "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

    for cand in CACHE_DIR.glob(f"yt_{key}.*"):
        if cand.is_file() and cand.stat().st_size > 1024:
            return cand
    raise RuntimeError("yt-dlp produced no output file")


def prepare_source(url: str | None, file_path: str | None) -> tuple[Path, np.ndarray]:
    """Return (playable_path, audio_16k_mono). URL takes precedence if given."""
    if url and url.strip():
        path = download_youtube_audio(url.strip())
    elif file_path:
        path = Path(file_path)
    else:
        raise ValueError("No source provided — enter a URL or upload a file.")
    audio, _ = load_audio_16k(path)
    return path, audio
