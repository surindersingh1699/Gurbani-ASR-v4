#!/usr/bin/env python3
"""Build an (audio, text) training dataset from a YouTube video's own pa-orig
auto-captions. Defaults reproduce YouTube's transcript-panel lines exactly.

See `memory/feedback_yt_caption_pipeline.md` for the full rationale and the
two non-obvious rules that make this work.

Default pipeline (clip-mode=caption, caption-offset-s=0):

  1. yt-dlp: download audio in its NATIVE format (opus in webm, or m4a) —
     NOT re-encoded to FLAC at download time. Re-encoding at download
     introduces a ~0.04% linear drift vs YouTube's player timeline.
     Also grabs `pa-orig` json3 captions.
  2. Build clip boundaries to match YouTube's UI transcript panel:
        - filter pa-orig events to CONTENT-only (drop `[ਸੰਗੀਤ]`/
          `[Music]`/etc. sound-tag events)
        - group contiguous content events (gap < 2s) so pairs never
          bridge a music break
        - pair consecutive content events (0,1), (2,3), … within each
          group — each pair is one UI transcript line / one clip
        - cap each clip's end at the next clip's start for non-overlap
  3. Per clip: ffmpeg slices from native webm, resamples to 16 kHz mono,
     encodes FLAC. Because ffmpeg decodes opus honouring container
     timestamps, the clip audio aligns to the caption `tStartMs` exactly.
  4. Write manifest.jsonl + dropped.jsonl + summary.txt.
  5. Optionally push to HF with an `audio` column so clips are playable
     inline in the Hub preview.

Legacy mode: `--clip-mode fixed --window-s 20` keeps the earlier fixed-
window behaviour for comparison/debugging.

Example (run from repo root, after `set -a; source .env; set +a`):
    python scripts/pilot_yt_caption_chunks.py \\
        --video-id goHzUu4v8mU \\
        --out /tmp/ytpilot \\
        --repo surindersinghssj/gurbani-yt-caption-pilot --public \\
        --pot-provider ""      # keep default URL on Hetzner instead

Validated on goHzUu4v8mU (2026-04-18): caption-mode clip boundaries matched
all 14 visible UI transcript-panel timestamps exactly (0:02 … 1:52), and
text coalescing matched the UI to the character.
"""
from __future__ import annotations

import argparse
import concurrent.futures
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


def build_ui_transcript_lines(cues: list[tuple[float, float, str]],
                              group_gap_s: float = 2.0,
                              duration: float | None = None
                              ) -> list[tuple[float, float, str, int]]:
    """Reproduce YouTube's transcript-panel line coalescing.

    YouTube's transcript UI pairs consecutive pa-orig events into "reading
    units" — the text the viewer sees on one line. We replicate that rule:
      1. Filter to CONTENT events (text remains after stripping
         `[bracketed]` sound-tags and `॥`/`॥੧॥` verse markers).
      2. Group contiguous content events (gap < `group_gap_s`); reset
         grouping across long music/silence breaks so the pairing does not
         bridge a sound-tag gap and produce a weird 30s clip.
      3. Within each group, pair events (0,1), (2,3), ... Odd trailing event
         is a standalone 1-event clip.
      4. `start = first.tStart`, text = first.text + " " + second.text.
         `end = second.end` (or standalone event's end), then cap at the
         next clip's start for clean non-overlapping tiling.

    Validated against the UI transcript panel on `goHzUu4v8mU`: all 14
    visible UI timestamps (0:02 … 1:52) match exactly. See
    memory/feedback_yt_caption_offset.md.

    Returns a list of (start_s, end_s, text, n_cues) tuples.
    """
    content = [(s, e, t) for (s, e, t) in cues
               if strip_for_drop_check(t)]

    if not content:
        return []

    groups: list[list[tuple[float, float, str]]] = [[content[0]]]
    for ev in content[1:]:
        if ev[0] - groups[-1][-1][1] < group_gap_s:
            groups[-1].append(ev)
        else:
            groups.append([ev])

    clips: list[tuple[float, float, str, int]] = []
    for g in groups:
        i = 0
        while i < len(g):
            if i + 1 < len(g):
                a = g[i]
                b = g[i + 1]
                text = f"{a[2]} {b[2]}"
                clips.append((a[0], b[1], text, 2))
                i += 2
            else:
                a = g[i]
                clips.append((a[0], a[1], a[2], 1))
                i += 1

    # Cap each clip's end at the next clip's start so audio tiles cleanly
    # without overlap (also matches the UI — each line plays until the next).
    for i in range(len(clips) - 1):
        s, e, t, n = clips[i]
        ns = clips[i + 1][0]
        if ns < e:
            clips[i] = (s, ns, t, n)

    # Cap the final clip at the master duration if known.
    if duration is not None and clips:
        s, e, t, n = clips[-1]
        if e > duration:
            clips[-1] = (s, duration, t, n)

    return clips


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), file=sys.stderr)
    return subprocess.run(cmd, check=True, **kw)


def _find_master(workdir: Path, video_id: str) -> Path | None:
    """Return preferred master audio (webm > m4a > flac) if present."""
    for ext in ("webm", "m4a", "opus", "flac"):
        p = workdir / f"{video_id}.{ext}"
        if p.exists():
            return p
    return None


def download_audio_and_captions(video_id: str, workdir: Path, lang: str,
                                pot_provider: str | None = None,
                                cookies: Path | None = None) -> tuple[Path, Path]:
    """Download audio in NATIVE format (no resample) + pa-orig captions.

    Native format preserves YouTube's exact timeline — opus/webm pre-skip,
    DASH segments, and variable-rate frames are all handled correctly by
    ffmpeg at slice time. This is what the player uses, so caption
    timestamps match audio exactly (offset=0 works).
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    workdir.mkdir(parents=True, exist_ok=True)
    cap_glob = lambda: sorted(workdir.glob(f"{video_id}.{lang}.*.json3")) + \
                       sorted(workdir.glob(f"{video_id}.{lang}.json3"))

    existing = _find_master(workdir, video_id)
    if existing and cap_glob():
        return existing, cap_glob()[0]

    cmd = [
        "yt-dlp",
        "-f", "bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio",
        "--write-auto-subs",
        "--sub-langs", lang,
        "--sub-format", "json3",
        "--no-playlist",
        # --force-ipv4: YouTube's googlevideo CDN was observed blocking
        # Hetzner's IPv6 address (2a01:4f8:c014:48bb::1) with 403 while
        # accepting the same request on IPv4. Pin to IPv4 to avoid this.
        "--force-ipv4",
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
    master = _find_master(workdir, video_id)
    if not master:
        raise SystemExit(f"No audio master file produced for {video_id}")
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
                  dropped: list[dict], caption_offset_s: float = 0.0,
                  seed: int = 42) -> str:
    total = len(kept) + len(dropped)
    rng = random.Random(seed)
    lines = []
    p = lines.append
    p("================ PILOT SUMMARY ================")
    p(f"Video            : {video_id}")
    p(f"Duration         : {duration:.0f}s ({duration/60:.1f} min)")
    p(f"Caption offset   : {caption_offset_s:+.3f}s "
      f"(applied to cue midpoint before window assignment)")
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
            "duration_s": m.get("duration_s", m["end_s"] - m["start_s"]),
            "n_cues": m["n_cues"],
            "clip_mode": m.get("clip_mode", "fixed"),
            "caption_offset_s": m.get("caption_offset_s", 0.0),
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
    ap.add_argument("--clip-mode", choices=["fixed", "caption"],
                    default="caption",
                    help="'caption' (default): one clip per YouTube UI "
                         "transcript-panel line (variable length, pair "
                         "consecutive content events). 'fixed': legacy "
                         "fixed-window mode using --window-s.")
    ap.add_argument("--caption-offset-s", type=float, default=0.0,
                    help="Seconds to add to every cue time before window "
                         "assignment (fixed mode only). With native-webm "
                         "download pipeline, 0 is correct — no drift.")
    ap.add_argument("--pot-provider", default="http://127.0.0.1:4416",
                    help="yt-dlp PO-token provider URL (empty = disable).")
    ap.add_argument("--cookies", type=Path, default=Path("/root/cookies.txt"))
    ap.add_argument("--repo", default="",
                    help="HF dataset repo to push to (empty = skip upload).")
    ap.add_argument("--public", action="store_true",
                    help="Push as public (default: private).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--slice-workers", type=int, default=8,
                    help="Number of parallel ffmpeg slice workers. Each "
                         "slice is an independent subprocess reading the "
                         "shared webm master, so this scales linearly with "
                         "CPU cores (Hetzner has 16). No effect on YouTube "
                         "request rate — pure local CPU parallelism.")
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

    kept: list[dict] = []
    dropped: list[dict] = []
    offset = args.caption_offset_s

    # slice_tasks collects (master, clip_path, start_s, dur_s, pad_to_s)
    # tuples so we can run ffmpeg slices in parallel AFTER the drop/keep
    # decisions are made. Each slice is an independent subprocess, safe to
    # parallelize.
    slice_tasks: list[tuple[Path, Path, float, float, float | None]] = []

    if args.clip_mode == "caption":
        lines = build_ui_transcript_lines(cues, duration=duration)
        print(f"[info] caption mode: {len(lines)} UI transcript lines",
              file=sys.stderr)
        for i, (start, end, raw_text, n_cues) in enumerate(lines):
            cleaned = strip_for_drop_check(raw_text)
            if not cleaned:
                dropped.append({
                    "window_idx": i,
                    "start_s": round(start, 3),
                    "end_s": round(end, 3),
                    "raw_text": raw_text,
                    "drop_reason": "sound_tag_only" if raw_text else "empty",
                    "n_cues": n_cues,
                })
                continue
            clip_name = f"clip_{i:05d}.flac"
            clip_path = clips_dir / clip_name
            dur_s = end - start
            if dur_s <= 0.05:
                dropped.append({
                    "window_idx": i,
                    "start_s": round(start, 3),
                    "end_s": round(end, 3),
                    "raw_text": raw_text,
                    "drop_reason": "zero_duration",
                    "n_cues": n_cues,
                })
                continue
            slice_tasks.append((master, clip_path, start, dur_s, None))
            kept.append({
                "clip_id": f"{args.video_id}_clip_{i:05d}",
                "audio_path": f"clips/{clip_name}",
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "duration_s": round(dur_s, 3),
                "text": normalize_kept(raw_text),
                "raw_text": raw_text,
                "n_cues": n_cues,
                "caption_offset_s": offset,
                "clip_mode": "caption",
            })
    else:
        W = args.window_s
        n_full = int(duration // W)
        tail = duration - n_full * W
        include_tail = tail >= args.min_tail_s
        n_windows = n_full + (1 if include_tail else 0)

        buckets: list[list[tuple[float, float, str]]] = [[] for _ in range(n_windows)]
        for cstart, cend, ctext in cues:
            mid = (cstart + cend) / 2.0 + offset
            idx = int(mid // W)
            if 0 <= idx < n_windows:
                buckets[idx].append((cstart, cend, ctext))

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
            slice_tasks.append((master, clip_path, start, dur_s, pad_to))
            kept.append({
                "clip_id": f"{args.video_id}_clip_{i:05d}",
                "audio_path": f"clips/{clip_name}",
                "start_s": round(start, 3),
                "end_s": round(nominal_end, 3),
                "duration_s": round(dur_s, 3),
                "text": normalize_kept(raw_text),
                "raw_text": raw_text,
                "n_cues": len(bucket),
                "caption_offset_s": offset,
                "clip_mode": "fixed",
            })

    # Parallel ffmpeg slicing. Each ffmpeg call is an independent
    # subprocess reading the shared webm master and writing its own FLAC,
    # so ThreadPoolExecutor parallelizes safely. With max_workers=8 on
    # Hetzner's 16-core box this is the biggest wall-time win we can make
    # on the chunker without touching YouTube request rates.
    if slice_tasks:
        def _do_slice(t: tuple[Path, Path, float, float, float | None]) -> None:
            slice_clip(*t)
        n_workers = max(1, args.slice_workers)
        print(f"[info] slicing {len(slice_tasks)} clips with "
              f"{n_workers} parallel workers", file=sys.stderr)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            # materialize to force errors to raise here, not silently
            list(ex.map(_do_slice, slice_tasks))

    (workdir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in kept) + "\n",
        encoding="utf-8",
    )
    (workdir / "dropped.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in dropped) + "\n",
        encoding="utf-8",
    )

    summary = build_summary(args.video_id, duration, kept, dropped,
                            caption_offset_s=offset, seed=args.seed)
    (workdir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)

    if args.repo:
        url = push_to_hf(args.repo, workdir, kept, args.video_id,
                         args.lang, summary, private=not args.public)
        print(f"[hf] pushed -> {url}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
