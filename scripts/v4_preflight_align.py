"""Pre-flight alignment gate.

Decides whether to keep a YouTube video for the full chunker pass WITHOUT
downloading the full audio. Steps:

  1. Download just the pa-orig caption .json3 (a few hundred KB at most).
  2. Pick N random caption windows from the middle of the video.
  3. yt-dlp --download-sections only those N small slices.
  4. Run the HF aligner on the slices vs caption text.
  5. PASS iff matches/total >= threshold.

Exit codes:
  0 -> PASS (call full chunker)
  1 -> REJECT (skip full chunker + downloads; ~99% bandwidth saved)
  2 -> ERROR / NO_CAPTION (fall back to current pipeline behavior)
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pilot_yt_caption_chunks import parse_json3, _gurmukhi_only  # noqa: E402
from v4_align_check_gemini import (  # noqa: E402
    _hf_transcribe_batch, _first_letters, HF_FALLBACK_MODEL_ID,
)
import v4_harvest_db as db  # noqa: E402


def fetch_captions(video_id: str, pot: str, cookies: Path | None,
                   out_dir: Path) -> Path | None:
    cmd = [
        "yt-dlp", "--quiet", "--no-warnings", "--skip-download",
        "--write-auto-subs", "--sub-langs", "pa-orig",
        "--sub-format", "json3", "--no-playlist",
        "--extractor-args", f"youtube:pot_provider={pot}",
        "-o", str(out_dir / "%(id)s.%(ext)s"),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    if cookies and cookies.exists():
        cmd += ["--cookies", str(cookies)]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    fp = out_dir / f"{video_id}.pa-orig.json3"
    return fp if fp.exists() else None


def pick_samples(captions_fp: Path, n: int,
                 seed: int | None = None) -> list[tuple[float, float, str]]:
    rng = random.Random(seed)
    all_windows = [
        (s, e, _gurmukhi_only(t))
        for s, e, t in parse_json3(captions_fp)
    ]
    all_windows = [w for w in all_windows if w[2] and len(w[2]) >= 5]
    if not all_windows:
        return []
    lo = int(len(all_windows) * 0.10)
    hi = int(len(all_windows) * 0.90)
    pool = all_windows[lo:hi] or all_windows
    n = min(n, len(pool))
    chosen = rng.sample(pool, n)
    # Expand short windows so the aligner has enough context.
    expanded: list[tuple[float, float, str]] = []
    for s, e, t in chosen:
        if e - s < 4.0:
            mid = (s + e) / 2
            s = max(0.0, mid - 2.5)
            e = mid + 2.5
        expanded.append((s, e, t))
    return expanded


def download_section(video_id: str, start: float, end: float,
                     out_stem: Path, pot: str,
                     cookies: Path | None) -> Path | None:
    section = f"*{start:.2f}-{end:.2f}"
    cmd = [
        "yt-dlp", "--quiet", "--no-warnings",
        "--extractor-args", f"youtube:pot_provider={pot}",
        "--download-sections", section, "--force-keyframes-at-cuts",
        "-f", "bestaudio",
        "--extract-audio", "--audio-format", "flac",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", f"{out_stem}.%(ext)s",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    if cookies and cookies.exists():
        cmd += ["--cookies", str(cookies)]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    fp = out_stem.with_suffix(".flac")
    return fp if fp.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pot-provider", default="http://127.0.0.1:4416")
    ap.add_argument("--cookies", type=Path, default=None)
    ap.add_argument("--hf-model", default=HF_FALLBACK_MODEL_ID)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    captions_fp = fetch_captions(args.video_id, args.pot_provider,
                                 args.cookies, args.out_dir)
    if captions_fp is None:
        print(f"PREFLIGHT_NO_CAPTION {args.video_id}", file=sys.stderr)
        return 2

    samples = pick_samples(captions_fp, args.n_samples, seed=args.seed)
    if not samples:
        print(f"PREFLIGHT_NO_SAMPLES {args.video_id}", file=sys.stderr)
        try:
            captions_fp.unlink()
        except OSError:
            pass
        return 2

    flac_paths: list[Path] = []
    valid_samples: list[tuple[float, float, str]] = []
    for i, (s, e, t) in enumerate(samples):
        stem = args.out_dir / f"{args.video_id}_pre{i}"
        fp = download_section(args.video_id, s, e, stem,
                              args.pot_provider, args.cookies)
        if fp is not None:
            flac_paths.append(fp)
            valid_samples.append((s, e, t))
    if not flac_paths:
        print(f"PREFLIGHT_DL_FAIL {args.video_id}", file=sys.stderr)
        try:
            captions_fp.unlink()
        except OSError:
            pass
        return 2

    asr_texts = _hf_transcribe_batch(flac_paths, model_id=args.hf_model)

    conn = db.connect(args.db)
    db.upsert_video(conn, args.video_id)
    matches = 0
    for (s, e, cap_text), asr_text in zip(valid_samples, asr_texts):
        cap_fl = _first_letters(cap_text, 5)
        asr_fl = _first_letters(asr_text, 5)
        matched = (
            len(cap_fl) >= 3 and len(asr_fl) >= 3 and (
                cap_fl == asr_fl
                or cap_fl in asr_fl
                or asr_fl in cap_fl
            )
        )
        db.record_alignment_check(
            conn, args.video_id, float(s), float(e),
            cap_text, asr_text, cap_fl, asr_fl, matched,
        )
        if matched:
            matches += 1

    passed = db.mark_video_aligned(
        conn, args.video_id, matches, len(valid_samples), args.threshold
    )

    for fp in flac_paths:
        try:
            fp.unlink()
        except OSError:
            pass
    try:
        captions_fp.unlink()
    except OSError:
        pass

    verdict = "PASS" if passed else "REJECT"
    print(f"PREFLIGHT_{verdict} {args.video_id} "
          f"{matches}/{len(valid_samples)}", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
