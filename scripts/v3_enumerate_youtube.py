#!/usr/bin/env python3
"""Enumerate YouTube playlist/channel videos, filter out live streams.

Output columns match the SikhNet enumerator where sensible so the two CSVs can
be concatenated:
  sikhnet_track_id, sikhnet_shabad_id, source, url, title, name,
  artist_slug, artist_name, size_bytes, est_secs, kirtan_type,
  youtube_video_id, youtube_playlist_id

A video is kept if:
  - duration > MIN_SEC and < MAX_SEC
  - title does NOT match a live-stream keyword ("LIVE", "ਲਾਈਵ", etc.)
  - was_live != True (when yt-dlp exposes it)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp

LIVE_KEYWORDS_RE = re.compile(
    r"\b(LIVE|Live)\b|ਲਾਈਵ|लाइव|live\s*stream|livestream",
    re.IGNORECASE,
)
MIN_SEC = 120      # 2 min — drop shorts / clips
MAX_SEC = 3600 * 3  # 3 hr — drop multi-hour live VODs
UA = "Mozilla/5.0"


def flat_enumerate(playlist_url: str) -> list[dict]:
    opts = {"quiet": True, "extract_flat": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        r = ydl.extract_info(playlist_url, download=False)
    return list(r.get("entries") or [])


def probe_video(video_id: str) -> dict | None:
    """Fetch full metadata so we can see was_live reliably."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)
    except Exception as e:
        print(f"[warn] probe {video_id}: {str(e)[:80]}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--playlists", required=True,
                    help="Comma-separated YouTube playlist or channel URLs")
    ap.add_argument("--kirtan-type", required=True,
                    choices=["ragi", "akj", "sgpc"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-playlist", type=int, default=500)
    ap.add_argument("--probe-workers", type=int, default=8)
    ap.add_argument("--skip-probe", action="store_true",
                    help="Trust flat-playlist duration, skip per-video probe")
    args = ap.parse_args()

    playlists = [s.strip() for s in args.playlists.split(",") if s.strip()]
    seen: set[str] = set()
    candidates: list[tuple[str, str, str, int | None]] = []

    for pl in playlists:
        entries = flat_enumerate(pl)[: args.max_per_playlist]
        pl_id = pl.rsplit("=", 1)[-1] if "list=" in pl else pl.rsplit("/", 1)[-1]
        print(f"[{pl_id}] {len(entries)} entries")
        for e in entries:
            vid = e.get("id")
            title = e.get("title") or ""
            dur = e.get("duration")
            if not vid or vid in seen:
                continue
            if LIVE_KEYWORDS_RE.search(title):
                continue
            seen.add(vid)
            candidates.append((vid, title, pl_id, dur))

    print(f"[candidates] {len(candidates)} after live-title filter")

    kept: list[dict] = []

    def accept(vid: str, title: str, pl_id: str,
               dur: int | None, was_live: bool | None):
        if dur is None or dur < MIN_SEC or dur > MAX_SEC:
            return
        if was_live is True:
            return
        kept.append({
            "sikhnet_track_id": "",
            "sikhnet_shabad_id": "",
            "source": "youtube",
            "url": f"https://www.youtube.com/watch?v={vid}",
            "title": title[:200],
            "name": title[:200],
            "artist_slug": "",
            "artist_name": "",
            "size_bytes": 0,
            "est_secs": dur,
            "kirtan_type": args.kirtan_type,
            "youtube_video_id": vid,
            "youtube_playlist_id": pl_id,
        })

    if args.skip_probe:
        for vid, title, pl_id, dur in candidates:
            accept(vid, title, pl_id, dur, None)
    else:
        with ThreadPoolExecutor(max_workers=args.probe_workers) as ex:
            futs = {ex.submit(probe_video, c[0]): c for c in candidates}
            for fut in as_completed(futs):
                vid, title, pl_id, fdur = futs[fut]
                info = fut.result()
                if info is None:
                    continue
                dur = info.get("duration") or fdur
                was_live = info.get("was_live")
                accept(vid, title, pl_id, dur, was_live)

    fields = [
        "sikhnet_track_id", "sikhnet_shabad_id", "source", "url", "title",
        "name", "artist_slug", "artist_name", "size_bytes", "est_secs",
        "kirtan_type", "youtube_video_id", "youtube_playlist_id",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)

    total_hours = sum(r["est_secs"] for r in kept) / 3600
    print(f"[done] {len(kept)} videos, ~{total_hours:.1f}h → {args.out}")


if __name__ == "__main__":
    main()
