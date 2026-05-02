#!/usr/bin/env python3
"""Produce a benchmark prediction file from one wav, by driving the live
STTM pipeline end-to-end.

Pipeline
--------
    wav  →  python -m apps.transcribe.app --bench-wav <wav>
         →  parse `[sttm] pushed audio_t=… banidb_gurmukhi=…` events
         →  pred JSON in the public benchmark schema

The pred captures audience-honest behaviour: every `STTMController.push_hit()`
becomes a segment painted from its `audio_t` until the next push (or end of
audio). That's exactly what the projector showed at each second.

Score the pred with the public benchmark — it owns the GT, the wav download
recipe, and the scorer:

    https://github.com/karanbirsingh/live-gurbani-captioning-benchmark-v1

Usage
-----
    # 1. clone the benchmark repo and grab the audio (one-time)
    git clone https://github.com/karanbirsingh/live-gurbani-captioning-benchmark-v1
    cd live-gurbani-captioning-benchmark-v1
    yt-dlp -x --audio-format wav -o "%(id)s.wav" "https://youtube.com/watch?v=IZOsmkdmmcg"
    ffmpeg -y -i IZOsmkdmmcg.wav -ar 16000 -ac 1 IZOsmkdmmcg_16k.wav

    # 2. produce a pred from this repo's pipeline
    python eval/bench_one.py \\
        --wav /path/to/IZOsmkdmmcg_16k.wav \\
        --out preds/IZOsmkdmmcg.json

    # 3. score it (in the benchmark repo)
    python eval.py --pred /path/to/preds/IZOsmkdmmcg.json \\
                   --gt   test/IZOsmkdmmcg.json --collar 2

Iterate
-------
1. tweak something in `apps/transcribe/`
2. rerun step 2
3. rerun step 3
4. compare the headline % against the previous run
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
SURT_REPO = EVAL_DIR.parent  # repo root

# Format emitted by `_try_auto_push` in apps/transcribe/app.py:
#   [sttm] pushed audio_t=12.34 sid=4377 verseId=58823 line_idx=2 score=0.78 banidb_gurmukhi='ਪੋਥੀ …'
# Trailing field is repr()-quoted in the source so it survives any internal
# punctuation/whitespace — `ast.literal_eval` recovers the original unicode.
_PUSH_RE = re.compile(
    r"^\[sttm\] pushed "
    r"audio_t=(?P<audio_t>[-\d.]+)\s+"
    r"sid=(?P<sid>\d+)\s+"
    r"verseId=(?P<vid>\d+)\s+"
    r"line_idx=(?P<line_idx>-?\d+)\s+"
    r"score=(?P<score>\S+)\s+"
    r"banidb_gurmukhi=(?P<banidb_gurmukhi>.+)$"
)


def _run_bench(wav: Path, log_path: Path) -> int:
    """Run the bench subprocess and tee stdout to a log file. Returns rc."""
    cmd = [
        sys.executable, "-u",  # unbuffered for live tee
        "-m", "apps.transcribe.app",
        "--bench-wav", str(wav),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SURT_REPO) + os.pathsep + env.get("PYTHONPATH", "")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {wav.name} → {log_path}")
    t0 = time.time()
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd, cwd=str(SURT_REPO), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            # Echo high-signal events to our stdout for live tailing.
            if line.startswith(("[bench]", "[sttm] pushed", "[lock]")):
                sys.stdout.write(line)
                sys.stdout.flush()
        rc = proc.wait()
    print(f"[run] done · rc={rc} · wall={time.time() - t0:.1f}s")
    return rc


def _parse_pushes(log_path: Path) -> list[dict]:
    """Extract `[sttm] pushed` events from the bench log.

    Each event becomes `{audio_t, banidb_gurmukhi}`. We deliberately drop
    the app's `line_idx` (skips sirlekh, different convention from GT)
    and `verseId` (internal sqlite rowid, not BaniDB canonical). The
    pangti text resolves uniquely against GT via canonical
    `banidb_gurmukhi`, so it's the only key the scorer needs.
    """
    events: list[dict] = []
    for raw in log_path.read_text().splitlines():
        m = _PUSH_RE.match(raw.strip())
        if not m:
            continue
        try:
            banidb_gurmukhi = ast.literal_eval(m.group("banidb_gurmukhi"))
        except (SyntaxError, ValueError) as e:
            raise RuntimeError(
                f"could not parse banidb_gurmukhi from line: {raw!r}: {e}"
            ) from e
        events.append({
            "audio_t": float(m.group("audio_t")),
            "banidb_gurmukhi": banidb_gurmukhi,
        })
    return events


def _events_to_segments(
    events: list[dict], total_duration: float,
) -> list[dict]:
    """Convert a stream of pushes into non-overlapping `(start, end, text)` segments.

    Each push paints from its `audio_t` until the next push (or end of
    audio). Adjacent identical-text pushes collapse — defends against the
    rare case where a single tick fires both the commit window and the
    fast-pointer scan with the same pangti.
    """
    if not events:
        return []
    segs: list[dict] = []
    for i, ev in enumerate(events):
        start = ev["audio_t"]
        end = events[i + 1]["audio_t"] if i + 1 < len(events) else total_duration
        if end <= start:
            continue
        if segs and segs[-1].get("banidb_gurmukhi") == ev["banidb_gurmukhi"]:
            segs[-1]["end"] = end
            continue
        segs.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "banidb_gurmukhi": ev["banidb_gurmukhi"],
        })
    return segs


def _audio_duration_s(wav: Path) -> float:
    """Get the duration of a wav file in seconds.

    Uses the standard library so this script has zero dependencies beyond
    whatever `apps.transcribe.app` itself already needs.
    """
    import wave
    with wave.open(str(wav), "rb") as f:
        return f.getnframes() / float(f.getframerate())


def _default_video_id(wav: Path) -> str:
    """Derive video_id from the wav stem, stripping the `_16k` resampling suffix."""
    return wav.stem.removesuffix("_16k")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Produce a benchmark prediction file from one wav.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Score the resulting pred with the public benchmark:\n"
            "  https://github.com/karanbirsingh/live-gurbani-captioning-benchmark-v1"
        ),
    )
    ap.add_argument("--wav", required=True, type=Path,
                    help="path to a 16 kHz mono wav (download via the benchmark repo's yt-dlp recipe)")
    ap.add_argument("--out", required=True, type=Path,
                    help="path to write the pred JSON (e.g. preds/<video_id>.json)")
    ap.add_argument("--video-id", default=None,
                    help="video_id to embed in the pred JSON "
                         "(default: --wav stem with _16k stripped)")
    ap.add_argument("--log", default=None, type=Path,
                    help="path for the raw bench stdout log (default: <out>.log)")
    ap.add_argument("--no-rerun", action="store_true",
                    help="skip the bench subprocess if --log already exists (just re-parse it)")
    args = ap.parse_args()

    if not args.wav.exists():
        print(f"[err] wav not found: {args.wav}", file=sys.stderr)
        sys.exit(2)

    video_id = args.video_id or _default_video_id(args.wav)
    log_path = args.log or args.out.with_suffix(args.out.suffix + ".log")
    total_duration = _audio_duration_s(args.wav)

    # 1. Run the live pipeline (or reuse a cached log)
    if not (args.no_rerun and log_path.exists()):
        rc = _run_bench(args.wav, log_path)
        if rc != 0:
            print(f"[warn] bench exited rc={rc} — emitting pred from whatever pushes did land",
                  file=sys.stderr)

    # 2. Parse pushes → segments → pred JSON
    events = _parse_pushes(log_path)
    segments = _events_to_segments(events, total_duration)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "video_id": video_id,
        "total_duration": round(total_duration, 2),
        "segments": segments,
    }, indent=2, ensure_ascii=False))

    print(f"[pred] pushes={len(events)} segments={len(segments)} → {args.out}")
    if not segments:
        print("[err] no [sttm] pushed events parsed from log — pred is empty",
              file=sys.stderr)
        sys.exit(1)

    # 3. Tell the user how to score it
    print()
    print("Score against the public benchmark:")
    print(f"  python eval.py --pred {args.out} --gt test/{video_id}.json --collar 2")
    print()
    print("(eval.py + test/ live in https://github.com/karanbirsingh/live-gurbani-captioning-benchmark-v1)")


if __name__ == "__main__":
    main()
