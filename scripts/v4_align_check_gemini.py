#!/usr/bin/env python3
"""Gemini sample-based alignment check for v4 auto-caption harvest.

For each video that survived `pilot_yt_caption_chunks.py`:
  1. Sample N random clips (default 10) from the manifest.
  2. Send all N FLAC clips in ONE Gemini call (gemini-2.5-flash-lite).
  3. For each clip: compare first 3 Gurmukhi words of Gemini output vs
     the same prefix of the YouTube caption text.
  4. If matches/total >= --threshold (default 0.6), the video PASSES;
     its manifest is left as-is and pushed normally.
  5. If it fails, the manifest is renamed to `manifest.jsonl.rejected`
     so the additive HF push step naturally skips it.

Every sample (passed or failed) is recorded to `alignment_checks` in
the v4 SQLite DB for later audit.

Pattern copied from `scripts/kirtan_bulk_transcribe.py` (batching multi-
audio in one `generate_content` call, JSON-array response).

Used as:
    from v4_align_check_gemini import check_video_alignment
    passed = check_video_alignment(video_id, video_dir, db_conn, ...)

Or standalone:
    python scripts/v4_align_check_gemini.py \\
        --video-dir /root/yt_v4/staging/abc123 \\
        --db /root/yt_v4/harvest.sqlite
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

# Local import (same dir).
sys.path.insert(0, str(Path(__file__).parent))
import v4_harvest_db as db


GURMUKHI = re.compile(r"[\u0A00-\u0A7F]+")
WS = re.compile(r"\s+")


def _first_n_words(text: str, n: int) -> str:
    """First n whitespace-separated tokens, lowercased & ZW-stripped."""
    if not text:
        return ""
    cleaned = text.replace("\u200c", "").replace("\u200d", "")
    cleaned = WS.sub(" ", cleaned).strip()
    return " ".join(cleaned.split()[:n])


def _first_letters(text: str, n: int | None = None) -> str:
    """First-letter-of-each-word skeleton. With Gurmukhi this is the only
    stable signal between captions and ASR \u2014 matras and word boundaries
    drift constantly, but the leading consonant of each word is the
    project's "anchor first letter" metric (branch
    feat/anchor-first-letter-v1).

    Returns a string of the first chars of each word, ZW stripped.
    If `n` is given, only the first `n` words contribute.
    """
    if not text:
        return ""
    cleaned = text.replace("\u200c", "").replace("\u200d", "")
    cleaned = WS.sub(" ", cleaned).strip()
    words = cleaned.split()
    if n is not None:
        words = words[:n]
    return "".join(w[0] for w in words if w)


def _looks_gurmukhi(text: str, min_chars: int = 3) -> bool:
    """Quick filter so we don't compare romanised junk."""
    chars = "".join(GURMUKHI.findall(text or ""))
    return len(chars) >= min_chars


def _init_client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY in env "
            "(see kirtan_bulk_transcribe.py for the same pattern)."
        )
    return genai.Client(api_key=key)


# HF fallback: surt-small-v3 (current SOTA Whisper-Small fine-tune).
# Loaded lazily and cached at module scope so 1000h harvests don't
# re-load the 244M-param model per video.
_HF_CACHE: dict = {}
HF_FALLBACK_MODEL_ID = "surindersinghssj/surt-small-v3"


def _init_hf_model(model_id: str = HF_FALLBACK_MODEL_ID):
    """Lazily load Whisper model + processor. Caches on model_id."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    if model_id in _HF_CACHE:
        return _HF_CACHE[model_id]
    proc = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train(False)
    # Force decoder to emit Punjabi tokens. Whisper's `language` kwarg
    # accepts the full English name internally per project memory.
    forced_ids = proc.get_decoder_prompt_ids(language="punjabi", task="transcribe")
    _HF_CACHE[model_id] = (proc, model, device, forced_ids)
    return _HF_CACHE[model_id]


def _hf_transcribe_batch(clip_audio_paths: list[Path],
                         model_id: str = HF_FALLBACK_MODEL_ID) -> list[str]:
    """Whisper inference for the same N clips. Same return shape as Gemini path."""
    import soundfile as sf
    import torch
    proc, model, device, forced_ids = _init_hf_model(model_id)
    out_texts: list[str] = []
    for p in clip_audio_paths:
        try:
            audio, sr = sf.read(str(p))
            if sr != 16000:
                # FLACs from pilot_yt_caption_chunks.py are already 16k mono,
                # so this branch is defensive only.
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                gen = model.generate(
                    inputs.input_features.to(device),
                    forced_decoder_ids=forced_ids,
                    max_new_tokens=128,
                )
            txt = proc.batch_decode(gen, skip_special_tokens=True)[0]
            out_texts.append(txt or "")
        except Exception as e:
            print(f"[hf] clip {p.name} fail: {e!r}", file=sys.stderr)
            out_texts.append("")
    return out_texts


def _sample_clips(manifest_fp: Path, n: int, min_dur_s: float = 4.0,
                  seed: int | None = None) -> list[dict]:
    rows: list[dict] = []
    for line in manifest_fp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        dur = m.get("duration_s") or (m.get("end_s", 0) - m.get("start_s", 0))
        if dur < min_dur_s:
            continue
        if not _looks_gurmukhi(m.get("text", "")):
            continue
        rows.append(m)
    if not rows:
        return []
    rng = random.Random(seed)
    if len(rows) <= n:
        return rows
    return rng.sample(rows, n)


def _gemini_transcribe_batch(client, clip_audio_paths: list[Path]) -> list[str]:
    """Send all clips in one call, get back a JSON array of strings."""
    from google.genai import types
    n = len(clip_audio_paths)
    prompt = (
        f"You are a Gurbani/Punjabi transcription expert. This is Sikh kirtan.\n\n"
        f"I am sending {n} short audio clips. For each clip:\n"
        f"- Transcribe ONLY the Gurmukhi text being sung\n"
        f"- If a clip has no singing, return empty string \"\"\n\n"
        f"Return a JSON array of exactly {n} strings, one per clip in order.\n"
        f"Return ONLY the JSON array."
    )
    contents: list = [types.Part.from_text(text=prompt)]
    for i, p in enumerate(clip_audio_paths):
        contents.append(types.Part.from_text(text=f"Clip {i + 1}:"))
        # FLAC produced by pilot_yt_caption_chunks.py is 16kHz mono.
        contents.append(types.Part.from_bytes(
            data=p.read_bytes(), mime_type="audio/flac"))

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            raw = (response.text or "").strip()
            texts = json.loads(raw)
            if isinstance(texts, dict):
                texts = list(texts.values())
            if not isinstance(texts, list):
                return [""] * n
            # Pad/truncate to expected count.
            if len(texts) < n:
                texts = list(texts) + [""] * (n - len(texts))
            elif len(texts) > n:
                texts = texts[:n]
            return [str(t or "") for t in texts]
        except Exception as e:
            if attempt == 2:
                print(f"[gemini] failed after 3 attempts: {e!r}", file=sys.stderr)
                return [""] * n
    return [""] * n


def check_video_alignment(
    video_id: str,
    video_dir: Path,
    conn,
    *,
    n_samples: int = 10,
    threshold: float = 0.6,
    client=None,
    seed: int | None = None,
    backend: str = "gemini",
    hf_model_id: str = HF_FALLBACK_MODEL_ID,
) -> bool:
    """Sample-aligner. Returns True iff video passes (and is therefore kept).

    Writes per-sample rows to `alignment_checks` and updates the video's
    `alignment_*` / `status` columns. On fail, renames
    `manifest.jsonl` -> `manifest.jsonl.rejected` so the HF pusher skips
    it without us having to teach push_additive about the DB.
    """
    manifest_fp = video_dir / "manifest.jsonl"
    if not manifest_fp.exists():
        db.mark_video_aligned(conn, video_id, 0, 0, threshold)
        return False

    samples = _sample_clips(manifest_fp, n_samples, seed=seed)
    if not samples:
        # No usable clips at all -> treat as misaligned/empty.
        passed = db.mark_video_aligned(conn, video_id, 0, 0, threshold)
        if not passed:
            manifest_fp.rename(manifest_fp.with_suffix(".jsonl.rejected"))
        return passed

    # Resolve audio paths (manifest stores them relative to the video dir).
    audio_paths: list[Path] = []
    valid_samples: list[dict] = []
    for m in samples:
        rel = m.get("audio_path") or ""
        ap = video_dir / rel
        if ap.exists():
            audio_paths.append(ap)
            valid_samples.append(m)
    if not audio_paths:
        passed = db.mark_video_aligned(conn, video_id, 0, 0, threshold)
        if not passed:
            manifest_fp.rename(manifest_fp.with_suffix(".jsonl.rejected"))
        return passed

    if backend == "gemini":
        if client is None:
            client = _init_client()
        gemini_texts = _gemini_transcribe_batch(client, audio_paths)
    elif backend == "hf":
        gemini_texts = _hf_transcribe_batch(audio_paths, model_id=hf_model_id)
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    matches = 0
    for m, g_text in zip(valid_samples, gemini_texts):
        cap_text = m.get("text", "") or m.get("raw_text", "")
        # First-letter-of-each-word anchor (matra-insensitive). Matras drift
        # heavily between captions and ASR; first consonant of each word is
        # the stable signal — and matches the project's "anchor first letter"
        # metric (branch feat/anchor-first-letter-v1).
        cap_fl = _first_letters(cap_text, 5)
        gem_fl = _first_letters(g_text, 5)
        matched = (
            len(cap_fl) >= 3 and len(gem_fl) >= 3 and (
                cap_fl == gem_fl
                or cap_fl in gem_fl
                or gem_fl in cap_fl
                # Same length, at most one substitution (ASR mis-hears one
                # leading consonant, e.g. ਹ vs ਧ).
                or (len(cap_fl) == len(gem_fl)
                    and sum(a != b for a, b in zip(cap_fl, gem_fl)) <= 1)
            )
        )
        db.record_alignment_check(
            conn, video_id,
            float(m.get("start_s", 0.0)),
            float(m.get("end_s", 0.0)),
            cap_text, g_text, cap_fl, gem_fl, matched,
        )
        if matched:
            matches += 1

    passed = db.mark_video_aligned(
        conn, video_id, matches, len(valid_samples), threshold
    )
    if not passed:
        # Rename so push_additive's glob skips it.
        try:
            manifest_fp.rename(manifest_fp.with_suffix(".jsonl.rejected"))
        except OSError as e:
            print(f"[align] rename fail {video_id}: {e!r}", file=sys.stderr)
    return passed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, required=True,
                    help="Directory containing manifest.jsonl + clip FLACs.")
    ap.add_argument("--db", type=Path, required=True,
                    help="Path to harvest.sqlite (created if missing).")
    ap.add_argument("--video-id", default=None,
                    help="Override video_id (default: video-dir's basename).")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    video_id = args.video_id or args.video_dir.name
    conn = db.connect(args.db)
    # Ensure a videos row exists so foreign-key + update works.
    db.upsert_video(conn, video_id)
    passed = check_video_alignment(
        video_id, args.video_dir, conn,
        n_samples=args.n_samples, threshold=args.threshold, seed=args.seed,
    )
    print(f"video_id={video_id} passed={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
