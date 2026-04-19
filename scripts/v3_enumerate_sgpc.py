#!/usr/bin/env python3
"""Enumerate the SGPC recorded-kirtan archive (ragi-wise) into a v3 manifest CSV.

Source: https://sgpc.net/ragiwise/ — a directorylister.php archive organized by
ragi. The archive landing page `sgpc.net/recorded-kirtan-ragi-wise/` embeds
`/ragiwise/` in an iframe; the iframe URL is the real index.

Cloudflare note (important):
  sgpc.net sits behind Cloudflare and serves a JS challenge (`cf-mitigated:
  challenge`, HTTP 403) to plain-requests clients from many non-IN IPs,
  including the Hetzner data-prep server. Python `requests` / plain `curl`
  cannot pass it. `curl_cffi` with Chrome TLS/H2 fingerprint impersonation
  (`impersonate="chrome124"`) bypasses the challenge cleanly — no browser
  or proxy needed. See memory/reference_infra.md.

Output columns match scripts/v3_enumerate_sikhnet.py so downstream
manifest-merge / canonical-filter code does not need a schema change:
  sikhnet_track_id, sikhnet_shabad_id, source, url, title, name,
  artist_slug, artist_name, size_bytes, est_secs, kirtan_type

Ragi diversity (IMPORTANT — see PLAN.md "SGPC ragi diversity policy"):
  The SGPC archive has 201 ragi directories and is heavily skewed — a handful
  of prolific ragis have 1,000+ MP3s while most have under 100. Unbalanced
  selection would give training data dominated by 3-4 voices. For any scrape
  that feeds the training set, pass:
      --max-hours-per-ragi N     (default policy: 2h)
      --target-total-hours M     (e.g. 160h to hit the SGPC floor)
      --seed 42                   (reproducible sampling)
  The script then picks tracks with a seeded shuffle within each ragi and
  round-robins across ragis until the target is hit, so early-stop stays
  balanced.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
import urllib.parse as up
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from curl_cffi import requests as cf_req

BASE = "https://sgpc.net/ragiwise/"
IMPERSONATE = "chrome124"
TIMEOUT = 45
BITRATE_BPS = 128_000  # SGPC MP3s are 128 kbps CBR

_SIZE_UNITS = {"KB": 1_000, "MB": 1_000_000, "GB": 1_000_000_000}
_SIZE_RE = re.compile(r"([\d.]+)\s*(KB|MB|GB)", re.I)

FILE_BLOCK_RE = re.compile(
    r'<a[^>]+href="([^"]+\.mp3)"[^>]*class="clearfix"[^>]*data-name="([^"]+)"'
    r'[^>]*>(.*?)</a>',
    re.S | re.I,
)
DIR_RE = re.compile(r'href="(\?dir=[^"]+)"', re.I)
SIZE_SPAN_RE = re.compile(
    r'<span[^>]+class="file-size[^"]*"[^>]*>\s*([^<]+?)\s*</span>',
    re.I,
)


def parse_size(s: str) -> int:
    m = _SIZE_RE.search(s)
    if not m:
        return 0
    return int(float(m.group(1)) * _SIZE_UNITS[m.group(2).upper()])


def build_session() -> cf_req.Session:
    return cf_req.Session(impersonate=IMPERSONATE)


def get_html(sess: cf_req.Session, url: str, retries: int = 3) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = sess.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.text
            if r.status_code in (403, 429, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def list_ragi_dirs(sess: cf_req.Session) -> list[str]:
    html = get_html(sess, BASE)
    return sorted(set(DIR_RE.findall(html)))


def walk_dir(sess: cf_req.Session, dir_param: str) -> list[dict]:
    """DFS walk one ragi directory; return a list of {rel_path, name, size_bytes}."""
    stack = [dir_param]
    visited: set[str] = set()
    out: list[dict] = []
    while stack:
        dp = stack.pop()
        if dp in visited:
            continue
        visited.add(dp)
        html = get_html(sess, BASE + dp)
        for href, display, inner in FILE_BLOCK_RE.findall(html):
            m = SIZE_SPAN_RE.search(inner)
            size = parse_size(m.group(1)) if m else 0
            out.append({
                "rel_path": href,
                "name": display,
                "size_bytes": size,
            })
        for sub in DIR_RE.findall(html):
            if sub not in visited:
                stack.append(sub)
    return out


def ragi_name_from_dir(dp: str) -> str:
    q = up.parse_qs(dp.split("?", 1)[-1]).get("dir", [""])[0]
    return up.unquote(q)


def balance_by_ragi(
    rows: list[dict],
    max_hours_per_ragi: float,
    target_total_hours: float,
    seed: int,
) -> list[dict]:
    """Select a ragi-balanced subset of `rows`.

    Policy (see PLAN.md 'SGPC ragi diversity policy'):
      1. Group rows by artist_name. Within each ragi, shuffle deterministically.
      2. If max_hours_per_ragi > 0, cap each ragi at that many hours.
      3. If target_total_hours > 0, round-robin one track at a time across ragis
         (in a shuffled ragi order) until the target total is reached. Ragis
         that run out of tracks early are skipped; the loop terminates when
         every ragi is exhausted or the target is hit.

    Returns the selected rows, preserving the round-robin pick order (so a
    downstream downloader that processes rows top-down naturally interleaves
    ragis too).
    """
    rng = random.Random(seed)

    by_ragi: OrderedDict[str, list[dict]] = OrderedDict()
    for r in rows:
        by_ragi.setdefault(r["artist_name"], []).append(r)

    for v in by_ragi.values():
        rng.shuffle(v)

    if max_hours_per_ragi > 0:
        cap_secs = max_hours_per_ragi * 3600
        capped: OrderedDict[str, list[dict]] = OrderedDict()
        for name, tracks in by_ragi.items():
            total = 0.0
            kept: list[dict] = []
            for t in tracks:
                if total >= cap_secs:
                    break
                kept.append(t)
                total += t["est_secs"]
            capped[name] = kept
        by_ragi = capped

    if target_total_hours <= 0:
        return [t for tracks in by_ragi.values() for t in tracks]

    target_secs = target_total_hours * 3600
    ragi_names = list(by_ragi.keys())
    rng.shuffle(ragi_names)
    cursors = {name: 0 for name in ragi_names}

    selected: list[dict] = []
    total = 0.0
    exhausted: set[str] = set()
    while total < target_secs and len(exhausted) < len(ragi_names):
        for name in ragi_names:
            if name in exhausted:
                continue
            idx = cursors[name]
            if idx >= len(by_ragi[name]):
                exhausted.add(name)
                continue
            track = by_ragi[name][idx]
            cursors[name] = idx + 1
            selected.append(track)
            total += track["est_secs"]
            if total >= target_secs:
                break
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--ragi-filter", default=None,
                    help="Case-insensitive substring filter on ragi name")
    ap.add_argument("--max-per-ragi", type=int, default=0,
                    help="Hard cap on MP3 count per ragi (0 = no cap). "
                         "Prefer --max-hours-per-ragi for dataset building.")
    ap.add_argument("--max-hours-per-ragi", type=float, default=0.0,
                    help="Diversity cap: at most N hours from any single ragi "
                         "(0 = no cap). See PLAN.md 'SGPC ragi diversity policy'.")
    ap.add_argument("--target-total-hours", type=float, default=0.0,
                    help="Stop after the manifest's cumulative est_secs reaches "
                         "this many hours. Selection is round-robin across "
                         "ragis so early-stop stays balanced (0 = no target).")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for within-ragi sampling and ragi order "
                         "shuffle. Fixed so enumeration is reproducible.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent ragi-dir walkers")
    args = ap.parse_args()

    boot = build_session()
    dir_params = list_ragi_dirs(boot)
    print(f"[root] {len(dir_params)} ragi directories")

    if args.ragi_filter:
        flt = args.ragi_filter.lower()
        dir_params = [d for d in dir_params if flt in ragi_name_from_dir(d).lower()]
        print(f"[filter] {len(dir_params)} after filter '{args.ragi_filter}'")

    rows: list[dict] = []

    def worker(dp: str) -> tuple[str, list[dict]]:
        # curl_cffi Session is not thread-safe — one per worker
        ws = build_session()
        return dp, walk_dir(ws, dp)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, dp): dp for dp in dir_params}
        for fut in as_completed(futures):
            dp = futures[fut]
            artist_name = ragi_name_from_dir(dp)
            try:
                _, files = fut.result()
            except Exception as e:
                print(f"[warn] {artist_name}: {e}", file=sys.stderr)
                continue
            if args.max_per_ragi:
                files = files[: args.max_per_ragi]
            total_gb = sum(f["size_bytes"] for f in files) / 1e9
            print(f"[{artist_name}] {len(files)} mp3s, ~{total_gb:.2f} GB")
            artist_slug = re.sub(r"[^a-z0-9]+", "-", artist_name.lower()).strip("-")
            for f in files:
                mp3_url = up.urljoin(BASE, f["rel_path"])
                size = f["size_bytes"]
                est_secs = round(size * 8 / BITRATE_BPS, 1) if size else 0
                title = f["name"].rsplit(".", 1)[0]
                rows.append({
                    "sikhnet_track_id": "",
                    "sikhnet_shabad_id": "",
                    "source": "sgpc",
                    "url": mp3_url,
                    "title": title,
                    "name": title,
                    "artist_slug": artist_slug,
                    "artist_name": artist_name,
                    "size_bytes": size,
                    "est_secs": est_secs,
                    "kirtan_type": "sgpc",
                })

    seen: set[str] = set()
    deduped: list[dict] = []
    for row in rows:
        if row["url"] in seen:
            continue
        seen.add(row["url"])
        deduped.append(row)

    if args.max_hours_per_ragi or args.target_total_hours:
        deduped = balance_by_ragi(
            deduped,
            max_hours_per_ragi=args.max_hours_per_ragi,
            target_total_hours=args.target_total_hours,
            seed=args.seed,
        )

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
    total_gb = sum(r["size_bytes"] for r in deduped) / 1e9
    n_ragis = len({r["artist_name"] for r in deduped})
    print(f"[done] {len(deduped)} mp3s from {n_ragis} ragis, "
          f"~{total_hours:.1f}h, ~{total_gb:.2f} GB → {args.out}")


if __name__ == "__main__":
    main()
