"""Streaming audio from a YouTube URL — no wait for the full download.

Two outputs from one ffmpeg process:
  1. PCM s16le @ 16 kHz mono on stdout  → consumed by Python for ASR
  2. HLS segments + playlist on disk    → consumed by the browser for playback

                yt-dlp -f bestaudio -o -   (webm/opus native container)
                           │
                           ▼ stdin pipe
                      ffmpeg -i pipe:0
                       │              │
         -map 0:a -f s16le pipe:1     -map 0:a -c:a libmp3lame -f hls …
                       │                        │
                       ▼                        ▼
           numpy float32 chunks    /tmp/surt_hls_<id>/playlist.m3u8
                  (ASR)                + seg_*.ts (browser via hls.js)

Caller paces the PCM transcription to wall-clock so STTM pushes match what
the browser is actually playing. The HLS `event` playlist type lets the
browser see segments appear incrementally and keep buffering.

URL time offsets (`&t=2128s` for "start 35 min in") are parsed and passed to
yt-dlp via `--download-sections *start-end`.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

TARGET_SR = 16_000
BYTES_PER_SEC = TARGET_SR * 2  # int16 mono
DEFAULT_MAX_SECONDS = int(os.environ.get("SURT_STREAM_MAX_SECONDS", "1800"))  # 30 min cap
READ_CHUNK_BYTES = BYTES_PER_SEC // 2  # ~0.5 s per read for responsive UI


@dataclass
class StreamMeta:
    downloaded_s: float
    total_chunk_s: float
    done: bool
    error: Optional[str] = None


def parse_url_time_offset(url: str) -> int:
    """Extract `&t=<N>` or `&t=<N>s` from a YouTube URL. Returns 0 if absent."""
    m = re.search(r"[?&]t=(\d+)(?:s|m|h)?", url or "")
    return int(m.group(1)) if m else 0


def _tools_available() -> tuple[bool, bool]:
    return (shutil.which("yt-dlp") is not None, shutil.which("ffmpeg") is not None)


def stream_audio_16k(
    url: str,
    *,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    from_seconds: Optional[int] = None,
    hls_dir: Optional[str] = None,
) -> Iterator[tuple[np.ndarray, StreamMeta]]:
    """Yield numpy float32 chunks at 16 kHz mono as yt-dlp streams from URL.

    Each yield is ~0.5 s of audio; the caller batches as needed.
    If `hls_dir` is given, ffmpeg also writes an HLS playlist there so the
    browser can play the same audio via hls.js in parallel.
    """
    if not url or not url.strip():
        yield np.zeros(0, dtype=np.float32), StreamMeta(0, 0, True, "empty url")
        return

    yt_ok, ff_ok = _tools_available()
    if not yt_ok:
        yield (
            np.zeros(0, dtype=np.float32),
            StreamMeta(0, 0, True, "yt-dlp not installed (brew install yt-dlp)"),
        )
        return
    if not ff_ok:
        yield (
            np.zeros(0, dtype=np.float32),
            StreamMeta(0, 0, True, "ffmpeg not installed (brew install ffmpeg)"),
        )
        return

    t_start_offset = (
        from_seconds if from_seconds is not None else parse_url_time_offset(url)
    )
    t_end_offset = t_start_offset + max_seconds

    # Run yt-dlp and ffmpeg as two processes connected by an OS pipe so we can
    # reap both if the generator is abandoned (Gradio does this on new click).
    yt_cmd = [
        "yt-dlp",
        "-q",
        "-f", "bestaudio",
        "--download-sections", f"*{t_start_offset}-{t_end_offset}",
        "--force-keyframes-at-cuts",
        "--no-playlist",
        "-o", "-",
        url,
    ]
    ff_cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel", "error",
        "-i", "pipe:0",
        # Output 1: PCM to stdout for ASR
        "-map", "0:a",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-f", "s16le",
        "pipe:1",
    ]
    if hls_dir:
        os.makedirs(hls_dir, exist_ok=True)
        # Output 2: HLS to disk for browser playback.
        # hls_time=2 → snappy start (~2-4 s of audio before first segment lands).
        # append_list + event playlist_type → growing playlist; client keeps polling.
        ff_cmd += [
            "-map", "0:a",
            "-c:a", "libmp3lame",
            "-b:a", "128k",
            "-f", "hls",
            "-hls_time", "2",
            "-hls_list_size", "0",
            "-hls_playlist_type", "event",
            "-hls_flags", "append_list",
            "-hls_segment_filename", os.path.join(hls_dir, "seg_%05d.ts"),
            os.path.join(hls_dir, "playlist.m3u8"),
        ]

    # Put both children in their own process group so we can SIGTERM the whole
    # pipeline cleanly on abort (Gradio cancelling the generator).
    yt = subprocess.Popen(
        yt_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    ff = subprocess.Popen(
        ff_cmd, stdin=yt.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    # Close the parent's copy so ffmpeg gets EOF when yt-dlp exits.
    if yt.stdout is not None:
        yt.stdout.close()

    t0 = time.time()
    bytes_read = 0
    try:
        while True:
            buf = ff.stdout.read(READ_CHUNK_BYTES)
            if not buf:
                break
            if len(buf) % 2 == 1:
                buf = buf[:-1]
            if not buf:
                continue
            samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
            bytes_read += len(buf)
            yield samples, StreamMeta(
                downloaded_s=bytes_read / BYTES_PER_SEC,
                total_chunk_s=len(samples) / TARGET_SR,
                done=False,
            )
        # Surface any ffmpeg stderr so load failures are visible.
        err_tail = ""
        try:
            err = ff.stderr.read() if ff.stderr else b""
            err_tail = err.decode("utf-8", errors="ignore").strip().splitlines()
            err_tail = err_tail[-1] if err_tail else ""
        except Exception:  # noqa: BLE001
            pass
        yield np.zeros(0, dtype=np.float32), StreamMeta(
            downloaded_s=bytes_read / BYTES_PER_SEC,
            total_chunk_s=0.0,
            done=True,
            error=err_tail or None,
        )
    finally:
        _kill_tree(yt)
        _kill_tree(ff)


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=2)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
