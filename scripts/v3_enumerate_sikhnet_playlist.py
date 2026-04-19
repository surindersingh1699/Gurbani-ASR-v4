#!/usr/bin/env python3
"""Enumerate SikhNet PLAYLIST pages (kirtan.txt ragi allowlist).

Unlike v3_enumerate_sikhnet.py (which targets /artist/<slug>/tracks), this
script targets /playlist/<slug> URLs — the format used in kirtan.txt.

Both pages embed the same Next.js __NEXT_DATA__ JSON containing track objects
with fields we care about: id, shabadId, name, pathSlug, artistName. So the
extraction logic is the same — only the fetch URL shape differs.

Output columns match v3_enumerate_sikhnet.py so the CSVs concatenate cleanly.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
TIMEOUT = 30
BITRATE_BPS = 128_000
NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>',
    re.DOTALL,
)


def fetch_playlist_tracks(playlist_url: str) -> list[dict]:
    r = requests.get(playlist_url, headers={"User-Agent": UA}, timeout=TIMEOUT)
    if r.status_code != 200:
        print(f"[warn] {playlist_url}: HTTP {r.status_code}", file=sys.stderr)
        return []
    m = NEXT_DATA_RE.search(r.text)
    if not m:
        print(f"[warn] {playlist_url}: no __NEXT_DATA__", file=sys.stderr)
        return []
    data = json.loads(m.group(1))

    seen_ids: set[int] = set()
    tracks: list[dict] = []

    def walk(obj):
        if isinstance(obj, dict):
            if (
                isinstance(obj.get("id"), int)
                and "shabadId" in obj
                and isinstance(obj.get("resource"), str)
                and "/audio/play/" in obj["resource"]
            ):
                tid = obj["id"]
                if tid not in seen_ids:
                    seen_ids.add(tid)
                    tracks.append({
                        "id": tid,
                        "shabadId": obj.get("shabadId"),
                        "name": obj.get("name") or "",
                        "artistName": obj.get("artistName") or "",
                    })
                return
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(data)
    return tracks


def resolve_mp3(track_id: int) -> tuple[str, int] | None:
    url = f"https://www.sikhnet.com/gurbani/audio/play/{track_id}"
    try:
        r = requests.head(url, headers={"User-Agent": UA},
                          allow_redirects=True, timeout=TIMEOUT)
    except requests.RequestException as e:
        print(f"[warn] track {track_id}: {e}", file=sys.stderr)
        return None
    if r.status_code != 200:
        return None
    mp3_url = r.url
    size = int(r.headers.get("content-length", "0"))
    if not mp3_url.endswith(".mp3") or size < 100_000:
        return None
    return mp3_url, size


def playlist_slug_from_url(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls-file", required=True,
                    help="Text file with one SikhNet playlist URL per line")
    ap.add_argument("--kirtan-type", required=True,
                    choices=["ragi", "akj", "sgpc"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-playlist", type=int, default=800)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    with open(args.urls_file) as f:
        urls = [ln.strip() for ln in f if ln.strip() and ln.startswith("http")]

    all_rows: list[dict] = []
    for url in urls:
        slug = playlist_slug_from_url(url)
        tracks = fetch_playlist_tracks(url)[: args.max_per_playlist]
        print(f"[{slug}] {len(tracks)} tracks")
        if not tracks:
            continue
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(resolve_mp3, t["id"]): t for t in tracks}
            for fut in as_completed(futs):
                t = futs[fut]
                res = fut.result()
                if res is None:
                    continue
                mp3_url, size = res
                est_secs = round(size * 8 / BITRATE_BPS, 1)
                title = mp3_url.rsplit("/", 1)[-1].removesuffix(".mp3")
                all_rows.append({
                    "sikhnet_track_id": t["id"],
                    "sikhnet_shabad_id": (
                        t["shabadId"] if t["shabadId"] is not None else ""
                    ),
                    "source": "sikhnet",
                    "url": mp3_url,
                    "title": title,
                    "name": t["name"],
                    "artist_slug": slug,
                    "artist_name": t["artistName"],
                    "size_bytes": size,
                    "est_secs": est_secs,
                    "kirtan_type": args.kirtan_type,
                    "youtube_video_id": "",
                    "youtube_playlist_id": "",
                })

    # dedupe by mp3 url
    seen = set()
    deduped = []
    for row in all_rows:
        if row["url"] in seen:
            continue
        seen.add(row["url"])
        deduped.append(row)

    fields = [
        "sikhnet_track_id", "sikhnet_shabad_id", "source", "url", "title",
        "name", "artist_slug", "artist_name", "size_bytes", "est_secs",
        "kirtan_type", "youtube_video_id", "youtube_playlist_id",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(deduped)

    total_hours = sum(r["est_secs"] for r in deduped) / 3600
    print(f"[done] {len(deduped)} tracks, ~{total_hours:.1f}h → {args.out}")


if __name__ == "__main__":
    main()
