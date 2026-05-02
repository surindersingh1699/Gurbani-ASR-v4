"""Live-vs-file parity benchmark for the Surt transcribe backend.

Feeds the SAME audio through three decode paths that the transcribe app
uses at runtime, so we can isolate which path is causing the observed
"file/link is accurate, live mic misses it" gap.

    file-mode        : 10 s commit windows + 2 s carry, merged
                       (this is apps/transcribe/app.py::_chunked_decode,
                        the on_upload / on_stream_url path)

    live-unlocked    : one decode per buffer fill, up to max_window_s
                       (apps/transcribe/app.py::on_stream when
                        locked_shabad_id is None)

    live-LOCKED      : fast-pointer path — 1.5 s tail decoded every
                       throttle window with no carry-over. THIS is the
                       regime the live mic lives in during real use
                       because lock engages instantly.

Usage:
    python scripts/bench_live_vs_file.py <path-to-wav-or-mp3>

The WAV should be recorded via the live mic chain (raw, no NS/AGC, ~30-60 s
of Gurbani) — see README below the script for how to capture one.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Make apps/transcribe importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from apps.transcribe import backend as _backend  # noqa: E402
from apps.transcribe.app import (  # noqa: E402
    _merge_committed,
    _resample_hq,
    _to_mono_float32,
    _suppress_repeat_hallucination,
)

TARGET_SR = 16000

# Mic-mode settings — these mirror StreamState defaults in app.py.
COMMIT_S = 10.0
MAX_WINDOW_S = 12.0
CARRY_OVER_S = 2.0
MIN_TRANSCRIBE_S = 3.0
THROTTLE_S = 1.2

# Fast-pointer (locked) path.
FAST_POINTER_MIN_S = 1.2
FAST_POINTER_THROTTLE_S = 0.6
FAST_TAIL_S = 1.5  # the exact tail length the live app uses in locked mode

# Chunk size the browser Gradio stream feeds in — close enough to reality.
SIM_CHUNK_MS = 100


def _load_audio(path: str) -> np.ndarray:
    import soundfile as sf  # type: ignore

    data, sr = sf.read(path, always_2d=False)
    y = _to_mono_float32(np.asarray(data))
    y = _resample_hq(y, sr, TARGET_SR)
    return y


def _transcribe(backend, audio: np.ndarray) -> str:
    try:
        text = backend.transcribe(audio, TARGET_SR, vad_filter=True) or ""
    except TypeError:
        text = backend.transcribe(audio, TARGET_SR) or ""
    return _suppress_repeat_hallucination(text)


def _rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(y.astype(np.float32) ** 2)))


def run_file_mode(backend, audio: np.ndarray) -> tuple[str, float]:
    """Replicate app.py::_chunked_decode exactly."""
    commit_samples = int(COMMIT_S * TARGET_SR)
    carry_samples = int(CARRY_OVER_S * TARGET_SR)
    min_samples = int(MIN_TRANSCRIBE_S * TARGET_SR)

    merged = ""
    cursor = 0
    t0 = time.time()
    while cursor + commit_samples <= audio.size:
        window = audio[cursor:cursor + commit_samples]
        text = _transcribe(backend, window)
        if text:
            merged = _merge_committed(merged, text)
        cursor += commit_samples - carry_samples

    remainder = audio[cursor:]
    if remainder.size >= min_samples:
        text = _transcribe(backend, remainder)
        if text:
            merged = _merge_committed(merged, text)
    elapsed = time.time() - t0
    return merged, elapsed


def run_live_unlocked(backend, audio: np.ndarray) -> tuple[str, float]:
    """Replicate the unlocked tentative+commit branch of on_stream.

    Feeds audio in SIM_CHUNK_MS-sized chunks so buffer timing matches
    what the real Gradio mic stream does.
    """
    commit_samples = int(COMMIT_S * TARGET_SR)
    max_samples = int(MAX_WINDOW_S * TARGET_SR)
    carry_samples = int(CARRY_OVER_S * TARGET_SR)
    min_samples = int(MIN_TRANSCRIBE_S * TARGET_SR)

    buffer = np.zeros(0, dtype=np.float32)
    committed = ""
    last_tentative = ""
    last_call_t = 0.0
    # Use real wall-clock for the throttle so behavior matches live use.
    start = time.time()

    chunk_samples = int(SIM_CHUNK_MS / 1000.0 * TARGET_SR)
    cur = 0
    t0 = time.time()
    while cur < audio.size:
        end = min(cur + chunk_samples, audio.size)
        buffer = np.concatenate([buffer, audio[cur:end]])
        cur = end

        # commit-and-carry
        if buffer.size > max_samples:
            commit_slice = buffer[:commit_samples]
            if _rms(commit_slice) >= 0.005:
                txt = _transcribe(backend, commit_slice)
                if txt:
                    committed = (committed + " " + txt).strip()
            buffer = buffer[-carry_samples:].copy()
            last_tentative = ""

        now = time.time() - start
        if (buffer.size >= min_samples
                and (now - last_call_t) >= THROTTLE_S
                and _rms(buffer) >= 0.005):
            last_call_t = now
            last_tentative = _transcribe(backend, buffer)
    final = (committed + " " + last_tentative).strip()
    return final, time.time() - t0


def run_live_locked(
    backend,
    audio: np.ndarray,
    *,
    fast_tail_s: float = FAST_TAIL_S,
    initial_prompt_mode: str = "none",
) -> tuple[str, float]:
    """Replicate the LOCKED fast-pointer branch of on_stream.

    This is the regime live mic is actually in during real use. Every
    `fast_pointer_throttle_s` seconds of wall-clock, we decode the last
    `fast_tail_s` seconds of buffer with NO carry-over.

    `initial_prompt_mode` lets the bench compare the proposed fix:
      - "none"     : decode each tail with no prompt (current behavior)
      - "previous" : pass the previous tail text as initial_prompt
    """
    buffer = np.zeros(0, dtype=np.float32)
    fast_min_samples = int(FAST_POINTER_MIN_S * TARGET_SR)
    fast_tail_samples = int(fast_tail_s * TARGET_SR)
    last_fast_t = -1e9
    tails: list[str] = []
    prev_tail_text = ""
    start = time.time()

    chunk_samples = int(SIM_CHUNK_MS / 1000.0 * TARGET_SR)
    cur = 0
    t0 = time.time()
    while cur < audio.size:
        end = min(cur + chunk_samples, audio.size)
        buffer = np.concatenate([buffer, audio[cur:end]])
        cur = end

        now = time.time() - start
        if buffer.size < fast_min_samples:
            continue
        if (now - last_fast_t) < FAST_POINTER_THROTTLE_S:
            continue
        if _rms(buffer[-fast_min_samples:]) < 0.005:
            continue
        last_fast_t = now
        tail = (buffer[-fast_tail_samples:]
                if buffer.size > fast_tail_samples else buffer)
        prompt = prev_tail_text if initial_prompt_mode == "previous" else None
        tail_text = _transcribe_with_prompt(backend, tail, prompt)
        if tail_text:
            tails.append(tail_text)
            prev_tail_text = tail_text
    # The live app displays `tentative` (last tail) + advances pointer on each
    # tail. For a head-to-head ASR quality comparison, return every unique
    # tail in order — this is what Whisper actually produced.
    joined = _dedup_concat(tails)
    return joined, time.time() - t0


def _transcribe_with_prompt(backend, audio, prompt):
    """Call backend.transcribe with optional initial_prompt.

    Falls back gracefully if the backend doesn't accept the kwarg yet —
    lets the bench run against the current (unpatched) backend.
    """
    try:
        text = backend.transcribe(
            audio, TARGET_SR, vad_filter=True, initial_prompt=prompt,
        ) or ""
    except TypeError:
        try:
            text = backend.transcribe(audio, TARGET_SR, vad_filter=True) or ""
        except TypeError:
            text = backend.transcribe(audio, TARGET_SR) or ""
    return _suppress_repeat_hallucination(text)


def _dedup_concat(texts: list[str]) -> str:
    """Join fast-pointer tails the way a downstream rerank would see them.

    Adjacent tails overlap by ~50% in real use; _merge_committed handles
    that at the word level.
    """
    merged = ""
    for t in texts:
        merged = _merge_committed(merged, t)
    return merged


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    path = sys.argv[1]
    if not Path(path).exists():
        print(f"audio not found: {path}")
        sys.exit(2)

    audio = _load_audio(path)
    dur = audio.size / TARGET_SR
    print(f"[bench] loaded {path} — {dur:.1f}s @ 16 kHz mono, "
          f"rms={_rms(audio):.4f}")

    backend = _backend.load_backend()
    print(f"[bench] backend={backend.name}")

    print("\n=== FILE MODE (10 s commit + 2 s carry) ===")
    text_file, t_file = run_file_mode(backend, audio)
    print(f"[t={t_file:.1f}s  rtf={t_file/dur:.2f}×]")
    print(text_file)

    print("\n=== LIVE UNLOCKED (up to 12 s buffer) ===")
    text_u, t_u = run_live_unlocked(backend, audio)
    print(f"[t={t_u:.1f}s]")
    print(text_u)

    print("\n=== LIVE LOCKED — current (1.5 s tail, NO prompt) ===")
    text_l, t_l = run_live_locked(
        backend, audio, fast_tail_s=1.5, initial_prompt_mode="none",
    )
    print(f"[t={t_l:.1f}s]")
    print(text_l)

    print("\n=== LIVE LOCKED — proposed (3.0 s tail + previous-text prompt) ===")
    text_lp, t_lp = run_live_locked(
        backend, audio, fast_tail_s=3.0, initial_prompt_mode="previous",
    )
    print(f"[t={t_lp:.1f}s]")
    print(text_lp)

    print("\n---- summary ----")
    print(f"file           : {len(text_file.split()):4d} words, {t_file:5.1f}s")
    print(f"live unlocked  : {len(text_u.split()):4d} words, {t_u:5.1f}s")
    print(f"live LOCKED old: {len(text_l.split()):4d} words, {t_l:5.1f}s")
    print(f"live LOCKED new: {len(text_lp.split()):4d} words, {t_lp:5.1f}s")


if __name__ == "__main__":
    main()
