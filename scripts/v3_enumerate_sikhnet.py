#!/usr/bin/env python3
"""Enumerate SikhNet tracks for given artist slugs.

Parses /artist/<slug>/tracks Next.js HTML, walks the __NEXT_DATA__ JSON to
extract every track object (id, shabadId, name, pathSlug, artistName),
then HEADs sikhnet.com/gurbani/audio/play/<id> to follow the 301 to the
direct MP3 URL and get content-length.

We estimate duration from MP3 size assuming 128 kbps (SikhNet's typical rate):
  secs = bytes * 8 / 128_000

Output columns:
  sikhnet_track_id, sikhnet_shabad_id, source, url, title, name,
  artist_slug, artist_name, size_bytes, est_secs, kirtan_type
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
BITRATE_BPS = 128_000  # SikhNet MP3s
NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>',
    re.DOTALL,
)


def fetch_tracks(artist_slug: str) -> list[dict]:
    """Return list of track dicts with id, shabadId, name, pathSlug."""
    url = f"https://play.sikhnet.com/artist/{artist_slug}/tracks"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
    if r.status_code != 200:
        print(f"[warn] {artist_slug}: HTTP {r.status_code}", file=sys.stderr)
        return []

    m = NEXT_DATA_RE.search(r.text)
    if not m:
        print(f"[warn] {artist_slug}: no __NEXT_DATA__", file=sys.stderr)
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
                        "pathSlug": obj.get("pathSlug") or "",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artists", required=True,
                    help="Comma-separated artist slugs")
    ap.add_argument("--kirtan-type", required=True,
                    choices=["ragi", "akj", "sgpc"])
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--max-per-artist", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    artists = [s.strip() for s in args.artists.split(",") if s.strip()]
    all_rows: list[dict] = []

    for slug in artists:
        tracks = fetch_tracks(slug)[: args.max_per_artist]
        print(f"[{slug}] {len(tracks)} tracks")

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(resolve_mp3, t["id"]): t for t in tracks}
            for fut in as_completed(futures):
                t = futures[fut]
                res = fut.result()
                if res is None:
                    continue
                mp3_url, size = res
                est_secs = round(size * 8 / BITRATE_BPS, 1)
                title = mp3_url.rsplit("/", 1)[-1].removesuffix(".mp3")
                all_rows.append({
                    "sikhnet_track_id": t["id"],
                    "sikhnet_shabad_id": t["shabadId"] if t["shabadId"] is not None else "",
                    "source": "sikhnet",
                    "url": mp3_url,
                    "title": title,
                    "name": t["name"],
                    "artist_slug": slug,
                    "artist_name": t["artistName"],
                    "size_bytes": size,
                    "est_secs": est_secs,
                    "kirtan_type": args.kirtan_type,
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
        "kirtan_type",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(deduped)

    total_hours = sum(r["est_secs"] for r in deduped) / 3600
    print(f"[done] {len(deduped)} tracks, ~{total_hours:.1f}h → {args.out}")


if __name__ == "__main__":
    main()
