#!/usr/bin/env python3
"""Bulk driver: enumerate YT channels -> per-video caption pipeline ->
accumulate one combined HF dataset.

Wraps `scripts/pilot_yt_caption_chunks.py`. For each video:
  1. Probe for pa-orig captions (skip if absent or geo-blocked).
  2. Delegate download + chunking to pilot_yt_caption_chunks.py (caption
     mode, offset=0, webm-native).
  3. Append that video's kept clips to the combined manifest.
  4. Delete the webm master immediately to cap disk usage.
  5. Log + record done.

Pushes the accumulated dataset to HF after every `--push-every` videos
(snapshotting so a crash doesn't lose progress) and once more at the end.
Resumable: re-running picks up from `done.txt`.

Example:
    python scripts/bulk_yt_caption_pipeline.py \\
        --target-hours 300 \\
        --out /root/yt_300h \\
        --repo surindersinghssj/gurbani-kirtan-yt-captions-300h \\
        --public
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_SEEDS: list[tuple[str, str]] = [
    # (channel URL, short slug used for grouping)
    ("https://www.youtube.com/@BestRecords/videos", "BestRecords"),
    ("https://www.youtube.com/@amrittsaagar/videos", "amrittsaagar"),
    ("https://www.youtube.com/@GurbaniMediaCentre/videos", "GurbaniMediaCentre"),
    ("https://www.youtube.com/@NirbaanKeertan/videos", "NirbaanKeertan"),
    ("https://www.youtube.com/@sikhnet/videos", "sikhnet"),
]


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(fp: Path, msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with fp.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def enumerate_channel(url: str, logfp: Path, pot: str,
                      cookies: Path | None) -> list[dict]:
    """yt-dlp --flat-playlist. Returns list of {id, title, duration_s}."""
    cmd = [
        "yt-dlp", "--flat-playlist", "--skip-download",
        "--print", "%(id)s|%(duration)s|%(title)s",
        "--extractor-args", f"youtube:pot_provider={pot}",
    ]
    if cookies and cookies.exists():
        cmd += ["--cookies", str(cookies)]
    cmd.append(url)
    log(logfp, f"enum {url}")
    try:
        r = run(cmd, timeout=600)
    except subprocess.TimeoutExpired:
        log(logfp, f"  TIMEOUT enumerating {url}")
        return []
    if r.returncode != 0:
        log(logfp, f"  ENUM FAIL rc={r.returncode} stderr={r.stderr[:200]}")
    out = []
    for line in (r.stdout or "").splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        vid, dur_s, title = parts
        try:
            dur = int(float(dur_s)) if dur_s and dur_s != "NA" else 0
        except ValueError:
            dur = 0
        out.append({"id": vid, "duration_s": dur, "title": title.strip()})
    log(logfp, f"  enum got {len(out)} videos from {url}")
    return out


RATE_LIMIT_MARKERS = (
    "rate-limited by YouTube",
    "rate-limited",
    "content isn't available, try again later",
    "This content isn't available",
    "Sign in to confirm you're not a bot",
)


def _is_rate_limited(text: str) -> bool:
    return any(m in text for m in RATE_LIMIT_MARKERS)


def try_fetch_caption(video_id: str, workdir: Path, lang: str, pot: str,
                      cookies: Path | None, logfp: Path
                      ) -> tuple[str, Path | None]:
    """Download ONLY the pa-orig caption (no audio). Returns (status, path).

    status: "ok" | "no_caption" | "rate_limited" | "unavailable" | "error"
    path:   caption file path if status == "ok"

    This replaces the separate --list-subs probe AND caches the caption so
    the main chunker can reuse it (no duplicate request).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp", "--skip-download",
        "--write-auto-subs", "--sub-langs", lang, "--sub-format", "json3",
        "--no-playlist",
        "--force-ipv4",          # YT CDN blocks Hetzner IPv6 (see chunker docstring)
        "--sleep-requests", "1", "--min-sleep-interval", "1",
        "--max-sleep-interval", "3",
        "--extractor-args", f"youtube:pot_provider={pot}",
        "-o", str(workdir / f"{video_id}.%(ext)s"),
    ]
    if cookies and cookies.exists():
        cmd += ["--cookies", str(cookies)]
    cmd.append(f"https://www.youtube.com/watch?v={video_id}")
    try:
        r = run(cmd, timeout=120)
    except subprocess.TimeoutExpired:
        return "error", None

    out = (r.stdout or "") + (r.stderr or "")
    if _is_rate_limited(out):
        return "rate_limited", None

    caps = sorted(workdir.glob(f"{video_id}.{lang}*.json3"))
    if caps:
        return "ok", caps[0]

    # No caption file; check whether the video itself is unavailable.
    if ("UNPLAYABLE" in out or "Video unavailable" in out or
            "This content isn't available" in out):
        return "unavailable", None
    return "no_caption", None


def run_chunker(script: Path, venv_python: Path, video_id: str,
                out_root: Path, pot: str, cookies: Path | None,
                logfp: Path) -> tuple[str, Path]:
    """Returns status: "ok" | "rate_limited" | "fail" | "timeout"."""
    cmd = [
        str(venv_python), str(script),
        "--video-id", video_id,
        "--out", str(out_root),
        "--clip-mode", "caption",
        "--pot-provider", pot,
    ]
    if cookies and cookies.exists():
        cmd += ["--cookies", str(cookies)]
    try:
        r = run(cmd, timeout=3600)
    except subprocess.TimeoutExpired:
        log(logfp, f"  CHUNKER TIMEOUT {video_id}")
        return "timeout", out_root / video_id
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-600:]
        if _is_rate_limited(tail):
            log(logfp, f"  CHUNKER RATE-LIMITED {video_id}")
            return "rate_limited", out_root / video_id
        log(logfp, f"  CHUNKER FAIL {video_id} rc={r.returncode} tail={tail!r}")
        return "fail", out_root / video_id
    return "ok", out_root / video_id


def cleanup_video_dir(workdir: Path, logfp: Path) -> None:
    """Delete webm master to cap disk usage. Keep clips + manifest + json3."""
    for patt in ("*.webm", "*.m4a", "*.opus"):
        for f in workdir.glob(patt):
            try:
                f.unlink()
            except OSError as e:
                log(logfp, f"  unlink fail {f}: {e}")


def load_done(done_fp: Path) -> set[str]:
    if not done_fp.exists():
        return set()
    return {l.strip() for l in done_fp.read_text().splitlines() if l.strip()}


def push_snapshot(staging: Path, repo: str, public: bool,
                  logfp: Path) -> str | None:
    """Build combined HF dataset from all per-video manifests and push."""
    try:
        from datasets import Audio, Dataset
        from huggingface_hub import HfApi
    except ImportError as e:
        log(logfp, f"  PUSH FAIL (import): {e}")
        return None
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        log(logfp, "  PUSH FAIL: no HF_TOKEN in env")
        return None

    rows: list[dict] = []
    for manifest_fp in sorted(staging.glob("*/manifest.jsonl")):
        vid = manifest_fp.parent.name
        video_root = manifest_fp.parent
        for line in manifest_fp.read_text().splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            audio_rel = m.get("audio_path") or ""
            audio_abs = video_root / audio_rel
            if not audio_abs.exists():
                continue
            rows.append({
                "audio": str(audio_abs),
                "text": m.get("text", ""),
                "raw_text": m.get("raw_text", ""),
                "clip_id": m.get("clip_id", ""),
                "start_s": m.get("start_s", 0.0),
                "end_s": m.get("end_s", 0.0),
                "duration_s": m.get("duration_s", m.get("end_s", 0.0) - m.get("start_s", 0.0)),
                "n_cues": m.get("n_cues", 0),
                "clip_mode": m.get("clip_mode", "caption"),
                "caption_offset_s": m.get("caption_offset_s", 0.0),
                "video_id": vid,
                "caption_lang": "pa-orig",
            })

    if not rows:
        log(logfp, "  PUSH SKIP (no rows)")
        return None

    total_s = sum(r["duration_s"] for r in rows)
    log(logfp, f"  pushing {len(rows)} clips ({total_s/3600:.1f} h) to {repo}")

    try:
        ds = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000))
        ds.push_to_hub(repo, private=(not public), token=token)
    except Exception as e:
        log(logfp, f"  PUSH FAIL (push_to_hub): {e!r}")
        return None

    # Write a quick README.
    try:
        api = HfApi(token=token)
        readme = (
            f"# Gurbani kirtan — YouTube pa-orig caption pipeline\n\n"
            f"Built by `scripts/bulk_yt_caption_pipeline.py` using the "
            f"native-webm + UI-transcript-line caption-mode pipeline "
            f"(see `feedback_yt_caption_pipeline.md` in the repo's "
            f"agent memory).\n\n"
            f"- clips: **{len(rows)}**\n"
            f"- total audio: **{total_s/3600:.1f} h**\n"
            f"- sources: YouTube channels — see `video_id` column for provenance.\n"
            f"- last snapshot: {now()}\n"
        )
        api.upload_file(
            path_or_fileobj=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
        )
    except Exception as e:
        log(logfp, f"  README upload fail: {e!r}")

    return f"https://huggingface.co/datasets/{repo}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--target-hours", type=float, default=300.0)
    ap.add_argument("--min-video-min", type=float, default=30.0,
                    help="Skip videos shorter than this many minutes.")
    ap.add_argument("--max-video-min", type=float, default=360.0,
                    help="Skip videos longer than this (likely live streams).")
    ap.add_argument("--push-every", type=int, default=15,
                    help="Push a dataset snapshot to HF after this many videos.")
    ap.add_argument("--inter-video-sleep", type=float, default=8.0,
                    help="Seconds to sleep between successful videos "
                         "(gentle pacing to avoid YouTube rate limits).")
    ap.add_argument("--channels", default="",
                    help="Comma-separated YouTube URLs. If empty, use DEFAULT_SEEDS.")
    ap.add_argument("--pot-provider", default="http://127.0.0.1:4416")
    ap.add_argument("--cookies", type=Path, default=Path("/root/cookies.txt"))
    ap.add_argument("--chunker-script", type=Path,
                    default=Path(__file__).parent / "pilot_yt_caption_chunks.py")
    ap.add_argument("--venv-python", type=Path, default=Path("/root/venv/bin/python"))
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    staging = out / "staging"
    staging.mkdir(exist_ok=True)
    logfp = out / "run.log"
    done_fp = out / "done.txt"
    skipped_fp = out / "skipped.jsonl"
    candidates_fp = out / "candidates.jsonl"

    log(logfp, f"=== BULK RUN START target={args.target_hours}h "
               f"repo={args.repo} public={args.public} ===")

    # Enumerate channels (cached).
    if not candidates_fp.exists():
        seeds = []
        if args.channels:
            for url in args.channels.split(","):
                url = url.strip()
                if url:
                    slug = url.rstrip("/").split("/")[-2] if "/" in url else url
                    seeds.append((url, slug))
        else:
            seeds = DEFAULT_SEEDS
        log(logfp, f"enumerating {len(seeds)} seeds")
        videos: list[dict] = []
        for url, slug in seeds:
            vids = enumerate_channel(url, logfp, args.pot_provider, args.cookies)
            for v in vids:
                v["channel_slug"] = slug
                v["channel_url"] = url
            videos.extend(vids)
        with candidates_fp.open("w") as f:
            for v in videos:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
        log(logfp, f"candidates: {len(videos)} -> {candidates_fp}")
    else:
        log(logfp, f"using cached candidates {candidates_fp}")

    # Load candidates, filter by duration, shuffle within channel for diversity.
    candidates: list[dict] = []
    for line in candidates_fp.read_text().splitlines():
        if line.strip():
            candidates.append(json.loads(line))
    min_s = args.min_video_min * 60
    max_s = args.max_video_min * 60
    filtered = [v for v in candidates
                if min_s <= (v.get("duration_s") or 0) <= max_s]
    log(logfp, f"filtered to {len(filtered)} videos "
               f"(duration in [{args.min_video_min}, {args.max_video_min}] min)")

    # Interleave channels for diversity: round-robin across channel_slug.
    from collections import defaultdict
    by_chan: dict[str, list[dict]] = defaultdict(list)
    for v in filtered:
        by_chan[v.get("channel_slug", "?")].append(v)
    # Deterministic interleave.
    interleaved: list[dict] = []
    while any(by_chan.values()):
        for slug in sorted(by_chan):
            if by_chan[slug]:
                interleaved.append(by_chan[slug].pop(0))

    done = load_done(done_fp)
    log(logfp, f"already done: {len(done)}")

    # Compute current hours from existing manifests.
    def current_hours() -> float:
        total_s = 0.0
        for mf in staging.glob("*/manifest.jsonl"):
            for line in mf.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    m = json.loads(line)
                    total_s += float(m.get("duration_s") or
                                     (m.get("end_s", 0) - m.get("start_s", 0)))
                except json.JSONDecodeError:
                    pass
        return total_s / 3600.0

    processed_since_push = 0
    consecutive_rate_limits = 0
    for v in interleaved:
        vid = v["id"]
        if vid in done:
            continue

        hrs = current_hours()
        if hrs >= args.target_hours:
            log(logfp, f"target reached: {hrs:.1f}h >= {args.target_hours}h")
            break

        log(logfp, f"[{hrs:.1f}h/{args.target_hours}h] trying {vid} "
                   f"({v.get('duration_s',0)/60:.0f}min, "
                   f"ch={v.get('channel_slug','?')}) {v.get('title','')[:80]!r}")

        workdir = staging / vid
        status, cap_path = try_fetch_caption(vid, workdir, "pa-orig",
                                             args.pot_provider, args.cookies,
                                             logfp)

        if status == "rate_limited":
            consecutive_rate_limits += 1
            wait_s = min(60 * 10 * consecutive_rate_limits, 3600)
            log(logfp, f"  RATE-LIMITED (x{consecutive_rate_limits}); "
                       f"sleeping {wait_s//60}min. Will retry {vid} later.")
            # Do NOT mark as done — leave for retry.
            time.sleep(wait_s)
            continue
        else:
            consecutive_rate_limits = 0

        if status in ("no_caption", "unavailable", "error"):
            with skipped_fp.open("a") as f:
                f.write(json.dumps({"id": vid, "reason": status,
                                    "title": v.get("title", ""),
                                    "channel": v.get("channel_slug", "")},
                                   ensure_ascii=False) + "\n")
            with done_fp.open("a") as f:
                f.write(vid + "\n")
            done.add(vid)
            log(logfp, f"  skip ({status})")
            # Gentle pacing so we do not re-trigger rate limits.
            time.sleep(3)
            continue

        # status == "ok" -> we have the caption cached. Now run the chunker
        # (which will download the webm and reuse the cached caption via
        # pilot_yt_caption_chunks.py's _find_master/cap_glob short-circuit).
        chk_status, workdir = run_chunker(
            args.chunker_script, args.venv_python, vid, staging,
            args.pot_provider, args.cookies, logfp,
        )
        if chk_status == "rate_limited":
            consecutive_rate_limits += 1
            wait_s = min(60 * 10 * consecutive_rate_limits, 3600)
            log(logfp, f"  chunker RATE-LIMITED; sleeping {wait_s//60}min. "
                       f"Will retry {vid} later.")
            time.sleep(wait_s)
            continue
        if chk_status != "ok":
            with skipped_fp.open("a") as f:
                f.write(json.dumps({"id": vid,
                                    "reason": f"chunker_{chk_status}",
                                    "title": v.get("title", ""),
                                    "channel": v.get("channel_slug", "")},
                                   ensure_ascii=False) + "\n")
            with done_fp.open("a") as f:
                f.write(vid + "\n")
            done.add(vid)
            time.sleep(10)
            continue

        consecutive_rate_limits = 0
        cleanup_video_dir(workdir, logfp)

        manifest_fp = workdir / "manifest.jsonl"
        if manifest_fp.exists():
            n_clips = sum(1 for l in manifest_fp.read_text().splitlines() if l.strip())
            log(logfp, f"  kept {n_clips} clips")
        else:
            log(logfp, f"  no manifest produced")

        with done_fp.open("a") as f:
            f.write(vid + "\n")
        done.add(vid)
        processed_since_push += 1

        if processed_since_push >= args.push_every:
            log(logfp, f"snapshot push after {processed_since_push} videos")
            url = push_snapshot(staging, args.repo, args.public, logfp)
            if url:
                log(logfp, f"  snapshot -> {url}")
            processed_since_push = 0

        # Gentle pacing between videos so we don't re-trigger rate limits.
        time.sleep(args.inter_video_sleep)

    # Final push.
    log(logfp, f"final push; total hours = {current_hours():.1f}")
    url = push_snapshot(staging, args.repo, args.public, logfp)
    if url:
        log(logfp, f"FINAL -> {url}")

    log(logfp, "=== BULK RUN END ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
