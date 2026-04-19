#!/usr/bin/env python3
"""Push v3 pilot transcripts (audio + Gemini text) to an HF dataset.

Column order is deliberately chosen for easy inspection + training:

    audio, text, kirtan_type,                              (primary training view)
    primary_text, primary_model,                           (raw primary-model output)
    fallback_text, fallback_model, fallback_reviewed,      (fallback-model output)
    flag_reason,                                           (why the clip was flagged)
    item_id, clip_i, start_sec, end_sec,                   (clip location)
    source, artist_name,                                   (provenance)
    sikhnet_track_id, sikhnet_shabad_id,
    youtube_video_id, youtube_playlist_id

`text` is the final chosen transcription: fallback_text if the clip was
flagged AND fallback_reviewed is True, otherwise primary_text.

Usage:
    python scripts/v3_push_hf.py \\
        --data-root /root/v3_data \\
        --repo surindersinghssj/gurbani-kirtan-v3-pilot-v5 \\
        --source-dir transcripts_cleaned \\
        --kirtan-types ragi,akj
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


DANDA_RE = re.compile(r"॥[੦-੯]+॥|॥|।|[\u200c\u200d]")
WS_RE = re.compile(r"\s+")

SIMRAN_TOKENS = {
    "ਵਾਹਿਗੁਰੂ",
    "ਵਾਹਗੁਰੂ",
    "ਸਤਿਨਾਮੁ",
    "ਸਤਿ",
    "ਨਾਮੁ",
    "ਹਰਿ",
    "ਰਾਮ",
    "ਗੁਰੂ",
    "ਧਨੁ",
    "ਧੰਨ",
}


def is_simran(text: str) -> bool:
    """Heuristic: clip is dominated by a known simran/naam token."""
    if not text:
        return False
    norm = WS_RE.sub(" ", DANDA_RE.sub(" ", text)).strip()
    tokens = norm.split()
    if len(tokens) < 2:
        # single-word "ਵਾਹਿਗੁਰੂ" counts
        return tokens and tokens[0] in SIMRAN_TOKENS
    top_tok, top_count = Counter(tokens).most_common(1)[0]
    return top_count / len(tokens) >= 0.6 and top_tok in SIMRAN_TOKENS


ROW_COLUMN_ORDER = [
    "text",
    "kirtan_type",
    "primary_text",
    "primary_model",
    "fallback_text",
    "fallback_model",
    "fallback_reviewed",
    "flag_reason",
    "item_id",
    "clip_i",
    "start_sec",
    "end_sec",
    "source",
    "artist_name",
    "sikhnet_track_id",
    "sikhnet_shabad_id",
    "youtube_video_id",
    "youtube_playlist_id",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--kirtan-types", default="ragi,akj,sgpc")
    ap.add_argument("--source-dir", default="transcripts_cleaned")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--cap-simran-per-item", type=int, default=2,
                    help="Max simran/naam-jaap clips kept per item_id. "
                    "0 = no cap.")
    args = ap.parse_args()

    from datasets import Audio, Dataset

    data_root = Path(args.data_root)
    clips_root = data_root / "clips"
    jsonl_dir = data_root / args.source_dir
    types = {t.strip() for t in args.kirtan_types.split(",")}

    audio_paths: list[str] = []
    meta_rows: list[dict] = []
    simran_kept: dict[str, int] = defaultdict(int)
    simran_dropped = 0

    for jsonl in sorted(jsonl_dir.glob("*.jsonl")):
        # process rows in clip order so we keep the FIRST N simran clips
        rows = [json.loads(l) for l in jsonl.open()]
        rows.sort(key=lambda r: r["clip_i"])
        for r in rows:
            if r.get("kirtan_type") not in types:
                continue
            clip_path = (clips_root / r["item_id"]
                         / f"clip_{r['clip_i']:05d}.wav")
            if not clip_path.exists():
                continue

            text = r.get("text", "")
            if args.cap_simran_per_item and is_simran(text):
                if simran_kept[r["item_id"]] >= args.cap_simran_per_item:
                    simran_dropped += 1
                    continue
                simran_kept[r["item_id"]] += 1

            audio_paths.append(str(clip_path))
            meta_rows.append({
                "text": r.get("text", ""),
                "kirtan_type": r.get("kirtan_type", ""),
                "primary_text": r.get("primary_text", r.get("text", "")),
                "primary_model": r.get("primary_model", ""),
                "fallback_text": r.get("fallback_text", ""),
                "fallback_model": r.get("fallback_model", ""),
                "fallback_reviewed": bool(r.get("fallback_reviewed", False)),
                "flag_reason": r.get("flag_reason", ""),
                "item_id": r["item_id"],
                "clip_i": r["clip_i"],
                "start_sec": r["start_sec"],
                "end_sec": r["end_sec"],
                "source": r.get("source", ""),
                "artist_name": r.get("artist_name", ""),
                "sikhnet_track_id": str(r.get("sikhnet_track_id", "")),
                "sikhnet_shabad_id": str(r.get("sikhnet_shabad_id", "")),
                "youtube_video_id": r.get("youtube_video_id", ""),
                "youtube_playlist_id": r.get("youtube_playlist_id", ""),
            })

    if args.cap_simran_per_item:
        print(f"[simran-cap] kept <= {args.cap_simran_per_item}/item, "
              f"dropped {simran_dropped} simran clips")
    print(f"[build] {len(audio_paths)} rows across {sorted(types)}")
    if not audio_paths:
        print("[build] nothing to push")
        return

    # Build dict in deliberate order: audio first, then text, then the rest
    ds_dict: dict[str, list] = {"audio": audio_paths}
    for col in ROW_COLUMN_ORDER:
        ds_dict[col] = [m[col] for m in meta_rows]

    ds = Dataset.from_dict(ds_dict)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    token = os.environ.get("HF_TOKEN")
    print(f"[push] pushing to {args.repo} split={args.split}")
    ds.push_to_hub(args.repo, split=args.split, token=token,
                   private=args.private)
    print(f"[done] https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
