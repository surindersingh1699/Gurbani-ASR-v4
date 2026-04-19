#!/usr/bin/env python3
"""Empirically measure the YT pa-orig caption-to-audio offset.

Send the first 60s and last 60s of the downloaded FLAC to Gemini 2.5 Flash Lite
and ask for per-word timestamps, then match each Gemini word against a pa-orig
word with the same normalized Gurmukhi form within ±5s. Report median delta
per segment.

Sign convention (matches pilot_yt_caption_chunks.py):
    delta_s = median( paorig_time - gemini_time )
    `--caption-offset-s` to feed into the chunker = -delta_s
    (i.e. if pa-orig is LATE by 1.3s vs audio, offset should be -1.3)

If the start and end deltas differ by more than 0.3s, drift is flagged — a
constant offset won't fix the whole video and a linear model is needed.

Example:
    python scripts/pilot_yt_caption_delta.py \\
        --video-id goHzUu4v8mU \\
        --workdir /tmp/ytpilot
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


# Gurmukhi normalization: strip diacritics and zero-width joiners so that
# spelling variance between YT ASR and Gemini ASR doesn't prevent matching.
_STRIP_CHARS = re.compile(
    r"[\u0a01\u0a02\u0a03\u0a3c\u0a4d\u0a70\u0a71\u200c\u200d\u0964\u0965।॥]"
)
_WS = re.compile(r"\s+")


def normalize_word(w: str) -> str:
    w = _WS.sub("", w.strip())
    w = _STRIP_CHARS.sub("", w)
    return w


def parse_paorig_words(json3_path: Path):
    """Yield (abs_time_s, word) per word segment in pa-orig json3."""
    data = json.loads(json3_path.read_text(encoding="utf-8"))
    for ev in data.get("events", []):
        t_start = ev.get("tStartMs") or 0
        for seg in ev.get("segs") or []:
            raw = seg.get("utf8") or ""
            w = raw.strip()
            if not w or w.startswith("[") or w in ("\n",):
                continue
            t_abs_ms = t_start + (seg.get("tOffsetMs") or 0)
            yield t_abs_ms / 1000.0, w


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def cut_clip(src: Path, dst: Path, start_s: float, dur_s: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", f"{start_s:.3f}", "-i", str(src),
         "-t", f"{dur_s:.3f}",
         "-ac", "1", "-ar", "16000", "-c:a", "flac", str(dst)],
        check=True,
    )


GEMINI_PROMPT = """\
You are given a short audio clip of sung Gurbani (Sikh religious singing in \
Gurmukhi / Punjabi). Transcribe each sung word with its START time in \
seconds from the beginning of THIS clip.

Return ONLY a JSON array, no prose, no code fences. Format:
[{"t": 0.82, "w": "ਵਾਹਿਗੁਰੂ"}, {"t": 1.34, "w": "ਜੀ"}]

Where:
- `t` is the start time in seconds (float, in the range [0, clip_length]).
- `w` is a single Gurmukhi word (do not merge multiple words into one entry).

Transcribe only sung words. Skip background music, applause, and spoken \
explanations (katha). If the entire clip is instrumental, return [].
"""


def gemini_client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set; `set -a; source .env; set +a`")
    return genai.Client(api_key=key)


def gemini_transcribe_words(client, audio_bytes: bytes, label: str) -> list[dict]:
    from google.genai import types
    contents = [
        types.Part.from_text(text=GEMINI_PROMPT),
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/flac"),
    ]
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            raw = (resp.text or "").strip()
            data = json.loads(raw)
            if isinstance(data, dict):
                data = list(data.values())
            if not isinstance(data, list):
                raise ValueError(f"Gemini returned {type(data).__name__}, not list")
            # Normalize entries
            out = []
            for it in data:
                if isinstance(it, dict) and "t" in it and "w" in it:
                    try:
                        out.append({"t": float(it["t"]),
                                    "w": str(it["w"]).strip()})
                    except (TypeError, ValueError):
                        continue
            print(f"[gemini/{label}] got {len(out)} words", file=sys.stderr)
            return out
        except Exception as e:
            print(f"[gemini/{label}] attempt {attempt+1} failed: {e}",
                  file=sys.stderr)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise


def measure_deltas(gemini_words: list[dict], paorig_words: list[tuple[float, str]],
                   clip_start_abs_s: float, window_s: float = 5.0) -> tuple[list[float], list[dict]]:
    """Return (deltas, matches) where delta_i = paorig_t - gemini_abs_t."""
    # Pa-orig word index by normalized form within this segment only.
    # clip_start..clip_end defines the segment; we only match pa-orig words
    # inside that region (with a ±window buffer).
    seg_start = clip_start_abs_s - window_s
    seg_end = clip_start_abs_s + 60.0 + window_s
    po_by_norm: dict[str, list[tuple[float, str]]] = {}
    for t, w in paorig_words:
        if not (seg_start <= t <= seg_end):
            continue
        n = normalize_word(w)
        if len(n) < 2:
            continue
        po_by_norm.setdefault(n, []).append((t, w))

    deltas: list[float] = []
    matches: list[dict] = []
    for gw in gemini_words:
        n = normalize_word(gw["w"])
        if len(n) < 2:
            continue
        cands = po_by_norm.get(n)
        if not cands:
            continue
        g_abs = clip_start_abs_s + gw["t"]
        close = [(t, w) for t, w in cands if abs(t - g_abs) <= window_s]
        if not close:
            continue
        best_t, best_w = min(close, key=lambda tw: abs(tw[0] - g_abs))
        deltas.append(best_t - g_abs)
        matches.append({
            "gemini_word": gw["w"],
            "gemini_t_abs": round(g_abs, 3),
            "paorig_word": best_w,
            "paorig_t": round(best_t, 3),
            "delta": round(best_t - g_abs, 3),
        })
    return deltas, matches


def summarize(deltas: list[float], label: str) -> dict:
    if not deltas:
        return {"label": label, "n": 0, "median": None, "iqr": None}
    ds = sorted(deltas)
    median = statistics.median(ds)
    if len(ds) >= 4:
        q1 = statistics.median(ds[: len(ds) // 2])
        q3 = statistics.median(ds[-(len(ds) // 2):])
        iqr = q3 - q1
    else:
        iqr = None
    return {"label": label, "n": len(ds),
            "median": round(median, 3),
            "iqr": round(iqr, 3) if iqr is not None else None,
            "min": round(ds[0], 3), "max": round(ds[-1], 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--workdir", type=Path, required=True,
                    help="Dir containing <video_id>.flac and <video_id>.pa-orig.json3")
    ap.add_argument("--probe-len-s", type=float, default=60.0)
    ap.add_argument("--drift-threshold-s", type=float, default=0.3,
                    help="If |start_median - end_median| exceeds this, flag drift.")
    ap.add_argument("--keep-clips", action="store_true")
    args = ap.parse_args()

    wd = args.workdir / args.video_id if (args.workdir / args.video_id).exists() else args.workdir
    master = wd / f"{args.video_id}.flac"
    cap_candidates = list(wd.glob(f"{args.video_id}.pa-orig*.json3"))
    if not master.exists() or not cap_candidates:
        raise SystemExit(f"Missing master FLAC or pa-orig json3 under {wd}")
    cap_path = cap_candidates[0]
    duration = probe_duration(master)
    print(f"[info] master={master} dur={duration:.1f}s cap={cap_path.name}",
          file=sys.stderr)

    paorig = list(parse_paorig_words(cap_path))
    print(f"[info] pa-orig word-segs: {len(paorig)}", file=sys.stderr)

    probes = [
        ("first", 0.0),
        ("last", max(0.0, duration - args.probe_len_s)),
    ]

    client = gemini_client()
    clips_dir = wd / "delta_probes"
    clips_dir.mkdir(exist_ok=True)

    all_summaries = []
    all_matches: dict[str, list[dict]] = {}
    for label, start in probes:
        probe_flac = clips_dir / f"{label}_{int(start)}s_{int(args.probe_len_s)}s.flac"
        if not probe_flac.exists():
            cut_clip(master, probe_flac, start, args.probe_len_s)
        audio_bytes = probe_flac.read_bytes()
        gwords = gemini_transcribe_words(client, audio_bytes, label=label)
        deltas, matches = measure_deltas(gwords, paorig, clip_start_abs_s=start)
        all_matches[label] = matches
        s = summarize(deltas, label)
        all_summaries.append(s)
        print(f"[{label:5}] start={start:.0f}s  gemini_words={len(gwords)}  "
              f"matched={s['n']}  median_delta={s['median']}s  iqr={s['iqr']}",
              file=sys.stderr)

    # Report
    print()
    print("================ DELTA MEASUREMENT ================")
    print(f"Video            : {args.video_id}")
    print(f"Probe length     : {args.probe_len_s:.0f}s (first + last)")
    for s in all_summaries:
        if s["n"] == 0:
            print(f"[{s['label']:5}] n=0 — no word matches; check audio/captions")
            continue
        print(f"[{s['label']:5}] n={s['n']:3}  median_delta={s['median']:+.3f}s"
              f"  range=[{s['min']:+.3f}, {s['max']:+.3f}]  iqr={s['iqr']}")

    meds = [s["median"] for s in all_summaries if s["median"] is not None]
    if len(meds) == 2:
        drift = meds[1] - meds[0]
        print(f"Drift (end-start): {drift:+.3f}s")
        if abs(drift) > args.drift_threshold_s:
            print(f"⚠ DRIFT DETECTED (>|{args.drift_threshold_s}|s). "
                  f"A constant offset will not fix the whole video. "
                  f"Consider linear model offset(t) = {meds[0]:+.3f} + "
                  f"{drift/duration:+.6f}*t")
        else:
            avg = sum(meds) / 2
            print(f"✓ No significant drift. "
                  f"Recommended: --caption-offset-s {-avg:+.2f}")

    # Dump matches for audit
    out_path = wd / "delta_matches.json"
    out_path.write_text(json.dumps({
        "video_id": args.video_id,
        "duration_s": duration,
        "probes": [{"label": l, "start_s": s, "len_s": args.probe_len_s}
                   for l, s in probes],
        "summaries": all_summaries,
        "matches": all_matches,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Match details  : {out_path}")

    if not args.keep_clips:
        for f in clips_dir.glob("*.flac"):
            f.unlink()
        try:
            clips_dir.rmdir()
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
