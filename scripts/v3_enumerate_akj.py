#!/usr/bin/env python3
"""Enumerate AKJ keertan archive at akj.org/keertan.php.

The page embeds ~700 direct .mp3 links under https://akj.media/Media/Keertan/
Each filename encodes samagam, year, date, session, and singer, e.g.:
    001_BabaBakala_4April2026_SatMor_DSK_BhaiSurinderSinghJeeButar.mp3

We scrape the HTML for all https://akj.media/.../*.mp3 hrefs, HEAD each to
get Content-Length, estimate duration at 128 kbps CBR (matches AKJ encoding),
and emit a manifest row per track. Output shape matches the SikhNet
enumerators so everything concatenates.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
TIMEOUT = 30
BITRATE_BPS = 128_000
MP3_URL_RE = re.compile(r'href=["\'](https://akj\.media/[^"\']+\.mp3)["\']')


def fetch_mp3_urls(page_url: str) -> list[str]:
    r = requests.get(page_url, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    urls = sorted(set(MP3_URL_RE.findall(r.text)))
    print(f"[scrape] {page_url}: {len(urls)} unique .mp3 URLs", file=sys.stderr)
    return urls


def parse_track_meta(url: str) -> dict:
    """Pull artist/samagam/date from the filename convention."""
    name = url.rsplit("/", 1)[-1].removesuffix(".mp3")
    parts = name.split("_")
    samagam = ""
    date = ""
    singer = ""
    if len(parts) >= 5:
        samagam = parts[1]
        date = parts[2]
        singer = "_".join(parts[4:])
    # Clean singer
    singer_clean = re.sub(r"Jee|Ji|Singh", lambda m: m.group(0), singer)
    return {
        "samagam": samagam,
        "date": date,
        "singer": singer_clean or samagam,
    }


def head_size(url: str) -> int:
    try:
        r = requests.head(url, headers={"User-Agent": UA},
                          allow_redirects=True, timeout=TIMEOUT)
        if r.status_code != 200:
            return 0
        return int(r.headers.get("content-length", "0"))
    except requests.RequestException:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default="https://akj.org/keertan.php")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tracks", type=int, default=0,
                    help="Stop after this many tracks; 0 = all")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    urls = fetch_mp3_urls(args.page)
    if args.max_tracks:
        urls = urls[: args.max_tracks]

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(head_size, u): u for u in urls}
        for fut in as_completed(futs):
            u = futs[fut]
            size = fut.result()
            if size < 500_000:  # drop tiny files
                continue
            meta = parse_track_meta(u)
            est_secs = round(size * 8 / BITRATE_BPS, 1)
            title = u.rsplit("/", 1)[-1].removesuffix(".mp3")
            # Stable track id: the filename prefix number + samagam
            track_id = f"{meta['samagam']}_{title[:10]}"
            rows.append({
                "sikhnet_track_id": "",
                "sikhnet_shabad_id": "",
                "source": "akj",
                "url": u,
                "title": title,
                "name": meta["samagam"],
                "artist_slug": meta["singer"][:60],
                "artist_name": meta["singer"][:60],
                "size_bytes": size,
                "est_secs": est_secs,
                "kirtan_type": "akj",
                "youtube_video_id": "",
                "youtube_playlist_id": "",
            })

    # dedupe by URL
    seen = set()
    deduped = []
    for r in rows:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        deduped.append(r)

    fields = [
        "sikhnet_track_id", "sikhnet_shabad_id", "source", "url", "title",
        "name", "artist_slug", "artist_name", "size_bytes", "est_secs",
        "kirtan_type", "youtube_video_id", "youtube_playlist_id",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(deduped)

    total_h = sum(r["est_secs"] for r in deduped) / 3600
    print(f"[done] {len(deduped)} tracks, ~{total_h:.1f}h → {args.out}")


if __name__ == "__main__":
    main()
