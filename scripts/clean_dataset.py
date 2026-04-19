"""Additive cleanup pass for a kirtan/sehaj HF dataset.

Does NOT modify the original `text` column. Does NOT drop rows. Instead adds
these columns to every row:

  - text_cleaned (string)     — text with `>>` + `<unk>` + non-Gurmukhi stripped
                                and ਵਾਹਿਗੁਰੂ-variant tokens normalized to canonical.
                                No scripture lookup. No LLM. Gurmukhi letters/matras
                                inside `text` are never modified, only surrounding junk.
  - is_simran (bool)          — ≥5 consecutive ਵਾਹਿਗੁਰੂ-skel tokens & ≥70% of tokens.
  - drop_candidate (bool)     — duration<1s OR <2 Gurmukhi tokens after cleaning.
                                Flagged but NOT dropped; downstream decides.
  - n_waheguru_normalized (int) — count of tokens rewritten by waheguru skel-match.

Usage:
  # dry-run (stats only)
  python3 scripts/clean_dataset.py --repo <src>

  # push an additive version to a new repo
  python3 scripts/clean_dataset.py --repo <src> --out-repo <dst> --push
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.canonical.gurmukhi_skeleton import tokenize
from scripts.canonical.preprocess import PreCleanConfig, should_drop_row, strip_unk_artifacts
from scripts.canonical.simran import SimranConfig, is_simran
from scripts.canonical.waheguru import normalize_waheguru_tokens

_GURMUKHI_KEEP = re.compile(r"[^\u0A00-\u0A7F\s]")


def clean_one(text: str) -> tuple[str, int]:
    """Return (cleaned_text, n_waheguru_tokens_normalized)."""
    if not text:
        return text or "", 0
    s = strip_unk_artifacts(text)
    s = _GURMUKHI_KEEP.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return "", 0
    toks = s.split()
    new_toks = normalize_waheguru_tokens(toks)
    n_norm = sum(1 for a, b in zip(toks, new_toks) if a != b)
    return " ".join(new_toks), n_norm


def process_streaming(ds_iter, preclean_cfg: PreCleanConfig,
                       simran_cfg: SimranConfig, limit: int | None = None):
    """Streaming mode: iterate rows without downloading audio, compute stats only."""
    stats = {
        "input_rows": 0,
        "rows_with_gt_markers": 0,
        "rows_with_unk_artifacts": 0,
        "total_waheguru_tokens_normalized": 0,
        "rows_flagged_simran": 0,
        "rows_flagged_drop_candidate": 0,
    }
    for r in ds_iter:
        stats["input_rows"] += 1
        if limit and stats["input_rows"] > limit:
            stats["input_rows"] -= 1
            break
        original = r.get("text", "") or ""
        if ">>" in original:
            stats["rows_with_gt_markers"] += 1
        if re.search(r"<unk|unk>|<un[^\u0A00-\u0A7F]|<k[^\u0A00-\u0A7F]", original):
            stats["rows_with_unk_artifacts"] += 1
        cleaned, n_wg = clean_one(original)
        stats["total_waheguru_tokens_normalized"] += n_wg
        if is_simran(tokenize(cleaned), simran_cfg):
            stats["rows_flagged_simran"] += 1
        if should_drop_row({"duration_s": r.get("duration_s", 0), "text": cleaned}, preclean_cfg):
            stats["rows_flagged_drop_candidate"] += 1
    return stats


def process_split(ds, preclean_cfg: PreCleanConfig, simran_cfg: SimranConfig):
    text_cleaned: list[str] = []
    is_simran_flags: list[bool] = []
    drop_cands: list[bool] = []
    n_wg_per_row: list[int] = []

    stats = {
        "input_rows": len(ds),
        "rows_with_gt_markers": 0,
        "rows_with_unk_artifacts": 0,
        "total_waheguru_tokens_normalized": 0,
        "rows_flagged_simran": 0,
        "rows_flagged_drop_candidate": 0,
    }

    for r in ds:
        original = r.get("text", "") or ""
        if ">>" in original:
            stats["rows_with_gt_markers"] += 1
        if re.search(r"<unk|unk>|<un[^\u0A00-\u0A7F]|<k[^\u0A00-\u0A7F]", original):
            stats["rows_with_unk_artifacts"] += 1

        cleaned, n_wg = clean_one(original)
        text_cleaned.append(cleaned)
        n_wg_per_row.append(n_wg)
        stats["total_waheguru_tokens_normalized"] += n_wg

        # simran detection on the CLEANED tokens
        toks = tokenize(cleaned)
        flag_s = is_simran(toks, simran_cfg)
        is_simran_flags.append(flag_s)
        if flag_s:
            stats["rows_flagged_simran"] += 1

        # drop-candidate check (runs against cleaned text; duration from row)
        probe = {"duration_s": r.get("duration_s", 0), "text": cleaned}
        flag_d = should_drop_row(probe, preclean_cfg)
        drop_cands.append(flag_d)
        if flag_d:
            stats["rows_flagged_drop_candidate"] += 1

    ds = ds.add_column("text_cleaned", text_cleaned)
    ds = ds.add_column("is_simran", is_simran_flags)
    ds = ds.add_column("drop_candidate", drop_cands)
    ds = ds.add_column("n_waheguru_normalized", n_wg_per_row)
    return ds, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Source HF dataset repo id")
    ap.add_argument("--out-repo", help="Destination repo (required for --push)")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--min-duration-s", type=float, default=1.0)
    ap.add_argument("--min-gurmukhi-tokens", type=int, default=2)
    ap.add_argument("--simran-min-reps", type=int, default=5)
    ap.add_argument("--simran-ratio", type=float, default=0.70)
    ap.add_argument("--streaming", action="store_true",
                    help="Stream rows from HF without downloading audio shards. "
                         "Only stats possible; --push disabled in this mode.")
    args = ap.parse_args()

    if args.push and not args.out_repo:
        ap.error("--push requires --out-repo")
    if args.push and args.streaming:
        ap.error("--push is incompatible with --streaming (no audio to re-upload)")

    from datasets import DatasetDict, load_dataset

    preclean_cfg = PreCleanConfig(
        min_duration_s=args.min_duration_s,
        min_gurmukhi_tokens=args.min_gurmukhi_tokens,
    )
    simran_cfg = SimranConfig(
        min_reps=args.simran_min_reps,
        ratio_threshold=args.simran_ratio,
    )

    print(f"Loading {args.repo} (streaming={args.streaming})...", file=sys.stderr)
    ds = load_dataset(args.repo, streaming=args.streaming)

    out_splits = {}
    for name, split in ds.items():
        if args.streaming:
            print(f"\n=== split: {name} (streaming, stats only) ===", file=sys.stderr)
            stats = process_streaming(split, preclean_cfg, simran_cfg,
                                      limit=args.sample or None)
            for k, v in stats.items():
                print(f"  {k.ljust(38)} {v}", file=sys.stderr)
            continue
        if args.sample:
            split = split.select(range(min(args.sample, len(split))))
        print(f"\n=== split: {name} ===", file=sys.stderr)
        enriched, stats = process_split(split, preclean_cfg, simran_cfg)
        for k, v in stats.items():
            print(f"  {k.ljust(38)} {v}", file=sys.stderr)
        out_splits[name] = enriched

    if args.push:
        out = DatasetDict(out_splits)
        print(f"\nPushing to {args.out_repo}...", file=sys.stderr)
        out.push_to_hub(args.out_repo)
        print("Done.", file=sys.stderr)
    else:
        print("\nDry-run only. --push --out-repo <dst> to upload.", file=sys.stderr)


if __name__ == "__main__":
    main()
