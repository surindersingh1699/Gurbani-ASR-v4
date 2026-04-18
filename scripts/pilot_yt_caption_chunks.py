#!/usr/bin/env python3
"""Pilot: YouTube audio + pa-orig auto-captions -> fixed 20s FLAC clips -> HF dataset.

Hypothesis: YouTube's `pa-orig` Punjabi auto-caption is usable as a noisy label
if we slice the audio into fixed 20s windows and drop instrumental-only windows
(captioned `[sangit]` / `[Music]` / similar by YouTube).

Pipeline:
  1. yt-dlp: download FLAC (16 kHz mono) + `pa-orig` json3 captions
  2. Slice audio into fixed 20s windows [0,20), [20,40), ...
     Final partial window: pad to 20s if >=5s else drop.
  3. For each window, concat pa-orig cues whose MIDPOINT falls in [start,end).
  4. Drop a window if — after stripping `[bracketed]` sound-tags, `॥`/`॥੧॥`
     verse markers, and whitespace — nothing is left.
  5. Write manifest.jsonl (kept) + dropped.jsonl (audit trail).
  6. Optionally push to a private HF dataset for inspection.

Example:
    python scripts/pilot_yt_caption_chunks.py \\
        --video-id goHzUu4v8mU \\
        --out /root/v3_data/pilot_yt_captions \\
        --repo surindersinghssj/gurbani-yt-caption-pilot
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path


BRACKETED = re.compile(r"\[[^\]]*\]")
PAREN_TAGS = re.compile(r"\(\s*(music|sangit|applause|laughter|instrumental)\s*\)", re.I)
VERSE_MARK = re.compile(r"॥[੦-੯0-9]*॥|॥|।")
ZW = re.compile(r"[\u200c\u200d]")
WS = re.compile(r"\s+")


def strip_for_drop_check(text: str) -> str:
    """Return text with sound-tags + verse markers removed. Empty => drop window."""
    t = BRACKETED.sub(" ", text)
    t = PAREN_TAGS.sub(" ", t)
    t = VERSE_MARK.sub(" ", t)
    t = ZW.sub("", t)
    t = WS.sub(" ", t).strip()
    return t


def normalize_kept(text: str) -> str:
    """Normalization applied to kept clips' final `text` field."""
    t = BRACKETED.sub(" ", text)
    t = PAREN_TAGS.sub(" ", t)
    t = VERSE_MARK.sub(" ", t)
    t = ZW.sub("", t)
    t = WS.sub(" ", t).strip()
    return t


def parse_json3(path: Path):
    """Yield (start_s, end_s, text) per non-empty caption event."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for ev in data.get("events", []):
        segs = ev.get("segs") or []
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs)
        text = text.replace("\n", " ").strip()
        if not text:
            continue
        start = (ev.get("tStartMs") or 0) / 1000.0
        dur = (ev.get("dDurationMs") or 0) / 1000.0
        yield start, start + dur, text


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), file=sys.stderr)
    return subprocess.run(cmd, check=True, **kw)


def download_audio_and_captions(video_id: str, workdir: Path, lang: str,
                                pot_provider: str | None = None,
                                cookies: Path | None = None) -> tuple[Path, Path]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    workdir.mkdir(parents=True, exist_ok=True)
    master = workdir / f"{video_id}.flac"
    cap_glob = lambda: sorted(workdir.glob(f"{video_id}.{lang}.*.json3")) + \
                       sorted(workdir.glob(f"{video_id}.{lang}.json3"))

    if master.exists() and cap_glob():
        return master, cap_glob()[0]

    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "flac",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "--write-auto-subs",
        "--sub-langs", lang,
        "--sub-format", "json3",
        "--no-playlist",
        "-o", str(workdir / f"{video_id}.%(ext)s"),
    ]
    if pot_provider:
        cmd += ["--extractor-args", f"youtube:pot_provider={pot_provider}"]
    if cookies and cookies.exists():
        cmd += ["--cookies", str(cookies)]
    cmd.append(url)
    run(cmd)

    caps = cap_glob()
    if not caps:
        raise SystemExit(f"No {lang} captions downloaded for {video_id}")
    if not master.exists():
        raise SystemExit(f"Audio FLAC not produced for {video_id}")
    return master, caps[0]


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def slice_clip(master: Path, out: Path, start_s: float, dur_s: float,
               pad_to_s: float | None = None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-i", str(master),
        "-t", f"{dur_s:.3f}",
    ]
    if pad_to_s is not None and pad_to_s > dur_s + 1e-3:
        # Pad tail with silence up to pad_to_s
        pad = pad_to_s - dur_s
        cmd += ["-af", f"apad=pad_dur={pad:.3f}"]
    cmd += ["-ac", "1", "-ar", "16000", "-c:a", "flac", str(out)]
    subprocess.run(cmd, check=True)


def build_summary(video_id: str, duration: float, kept: list[dict],
                  dropped: list[dict], seed: int = 42) -> str:
    total = len(kept) + len(dropped)
    rng = random.Random(seed)
    lines = []
    p = lines.append
    p("================ PILOT SUMMARY ================")
    p(f"Video            : {video_id}")
    p(f"Duration         : {duration:.0f}s ({duration/60:.1f} min)")
    p(f"Total windows    : {total}")
    if total:
        p(f"Kept             : {len(kept)}  ({len(kept)/total*100:.1f}%)")
        p(f"Dropped          : {len(dropped)}  ({len(dropped)/total*100:.1f}%)")
    p("")
    p("--- 10 random KEPT samples ---")
    for m in rng.sample(kept, min(10, len(kept))):
        p(f"[{m['clip_id']}] {m['start_s']:>5.0f}s-{m['end_s']:<5.0f}s  "
          f"cues={m['n_cues']}  {m['audio_path']}")
        p(f"    {m['text'][:140]}")
    p("")
    p("--- 5 random DROPPED samples ---")
    for d in rng.sample(dropped, min(5, len(dropped))):
        reason = d["drop_reason"]
        raw = d["raw_text"]
        p(f"[win_{d['window_idx']:05d}] {d['start_s']:>5.0f}s-{d['end_s']:<5.0f}s "
          f" reason={reason:<16} raw={raw!r}")
    p("===============================================")
    return "\n".join(lines)


def push_to_hf(repo: str, workdir: Path, manifest: list[dict],
               video_id: str, lang: str, summary: str, private: bool) -> str:
    from datasets import Audio, Dataset
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Build dataset rows with absolute audio paths.
    rows = []
    for m in manifest:
        rows.append({
            "audio": str(workdir / m["audio_path"]),
            "text": m["text"],
            "raw_text": m["raw_text"],
            "clip_id": m["clip_id"],
            "start_s": m["start_s"],
            "end_s": m["end_s"],
            "n_cues": m["n_cues"],
            "video_id": video_id,
            "caption_lang": lang,
        })

    ds = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000))
    ds.push_to_hub(repo, private=private, token=token)

    api = HfApi(token=token)
    for fname in ("manifest.jsonl", "dropped.jsonl", "summary.txt"):
        fp = workdir / fname
        if fp.exists():
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=fname,
                repo_id=repo,
                repo_type="dataset",
            )

    readme = (
        f"# Pilot: YouTube pa-orig captions as noisy labels\n\n"
        f"Video: https://www.youtube.com/watch?v={video_id}\n\n"
        f"Caption lang: `{lang}` (YouTube auto-generated original-language captions).\n\n"
        f"Pipeline: fixed 20s windows; per-window text = concat of cues whose midpoint\n"
        f"falls in the window; drop window if — after stripping `[sangit]`/`[Music]`/etc.\n"
        f"and verse markers — nothing is left.\n\n"
        f"## Summary\n\n```\n{summary}\n```\n"
    )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{repo}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output root; creates <out>/<video_id>/...")
    ap.add_argument("--lang", default="pa-orig",
                    help="Caption language code (default: pa-orig)")
    ap.add_argument("--window-s", type=float, default=20.0)
    ap.add_argument("--min-tail-s", type=float, default=5.0,
                    help="Drop final partial window if shorter than this; else pad.")
    ap.add_argument("--pot-provider", default="http://127.0.0.1:4416",
                    help="yt-dlp PO-token provider URL (empty = disable).")
    ap.add_argument("--cookies", type=Path, default=Path("/root/cookies.txt"))
    ap.add_argument("--repo", default="",
                    help="HF dataset repo to push to (empty = skip upload).")
    ap.add_argument("--public", action="store_true",
                    help="Push as public (default: private).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    workdir = args.out / args.video_id
    clips_dir = workdir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    pot = args.pot_provider if args.pot_provider else None
    master, cap_path = download_audio_and_captions(
        args.video_id, workdir, args.lang,
        pot_provider=pot, cookies=args.cookies,
    )

    duration = probe_duration(master)
    cues = list(parse_json3(cap_path))
    print(f"[info] duration={duration:.1f}s  cues={len(cues)}  caption={cap_path.name}",
          file=sys.stderr)

    W = args.window_s
    n_full = int(duration // W)
    tail = duration - n_full * W
    include_tail = tail >= args.min_tail_s
    n_windows = n_full + (1 if include_tail else 0)

    buckets: list[list[tuple[float, float, str]]] = [[] for _ in range(n_windows)]
    for cstart, cend, ctext in cues:
        mid = (cstart + cend) / 2.0
        idx = int(mid // W)
        if 0 <= idx < n_windows:
            buckets[idx].append((cstart, cend, ctext))

    kept: list[dict] = []
    dropped: list[dict] = []
    for i, bucket in enumerate(buckets):
        start = i * W
        nominal_end = min(start + W, duration)
        raw_text = " ".join(t for _, _, t in sorted(bucket)).strip()
        cleaned = strip_for_drop_check(raw_text)
        if not cleaned:
            dropped.append({
                "window_idx": i,
                "start_s": round(start, 3),
                "end_s": round(nominal_end, 3),
                "raw_text": raw_text,
                "drop_reason": "sound_tag_only" if raw_text else "empty",
                "n_cues": len(bucket),
            })
            continue
        clip_name = f"clip_{i:05d}.flac"
        clip_path = clips_dir / clip_name
        dur_s = nominal_end - start
        pad_to = W if (i == n_full and include_tail and dur_s < W) else None
        slice_clip(master, clip_path, start, dur_s, pad_to_s=pad_to)
        kept.append({
            "clip_id": f"{args.video_id}_clip_{i:05d}",
            "audio_path": f"clips/{clip_name}",
            "start_s": round(start, 3),
            "end_s": round(nominal_end, 3),
            "text": normalize_kept(raw_text),
            "raw_text": raw_text,
            "n_cues": len(bucket),
        })

    (workdir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in kept) + "\n",
        encoding="utf-8",
    )
    (workdir / "dropped.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in dropped) + "\n",
        encoding="utf-8",
    )

    summary = build_summary(args.video_id, duration, kept, dropped, seed=args.seed)
    (workdir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)

    if args.repo:
        url = push_to_hf(args.repo, workdir, kept, args.video_id,
                         args.lang, summary, private=not args.public)
        print(f"[hf] pushed -> {url}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
