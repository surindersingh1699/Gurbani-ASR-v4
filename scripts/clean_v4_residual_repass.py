"""Phase 1b: re-pass residual JSONL with stronger skeleton matchers.

After Phase 1a (clean_v4_skeleton_pass.py) classifies clips using the
three base checks (single pangti, adjacent pangti pair, exact-chorus
repeat), a sizeable fraction land in the residual bucket. Empirically
(SGPC smoke, AKJ first 5k) residual sits around 45-80% — too large to
hand wholesale to Phase 2 LLM rescue.

This pass adds two more skeleton checks that catch the two most common
residual patterns seen on kirtan data:

  4. **prefix-tile repeat** — caption skel = `unit * k`-then-truncated,
     where `unit` is a prefix of some pangti skel. Catches partial
     chorus repeats where the second half of the chorus is cut short
     (e.g. caption = "ABCD ABC" while pangti = "ABCD EFG").

  5. **boundary trim** — try trimming up to ~3 chars off the left and
     up to ~3 chars off the right of the caption skel before retrying
     the single / pair checks. Catches "extra short word at the start/
     end of the caption" (e.g. caption "ਤੂੰ ਭਗਤਾ ਕੀ ਟੇਕ ..." while
     pangti starts with "ਭਗਤਾ ਕੀ ਟੇਕ ...").

Inputs: existing residual JSONL + source HF repo (for audio bytes).
Outputs: smaller residual JSONL + a clean_phase1b_<ts> split pushed to
the dest -clean repo with quality='high' for the newly matched rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from canonical.gurmukhi_skeleton import skel  # noqa: E402
from canonical.sttm_index import load_sggs  # noqa: E402


class SkelIndexV2:
    """SkelIndex with the v2 matchers (prefix-tile + boundary-trim)."""
    SEP = ""

    def __init__(self, db_path: str):
        lines, _ = load_sggs(db_path)
        self.lines = lines
        shabad_to_pangtis: dict[str, list[str]] = {}
        for ln in lines:
            shabad_to_pangtis.setdefault(ln.shabad_id, []).append(ln.skel)
        self.single_index = self.SEP.join(ln.skel for ln in lines if ln.skel)
        pair_parts = []
        for sid, ps in shabad_to_pangtis.items():
            for a, b in zip(ps, ps[1:]):
                if a and b:
                    pair_parts.append(a + b)
        self.pair_index = self.SEP.join(pair_parts)
        self.distinct_pangti_skels = list({ln.skel for ln in lines if ln.skel})

    # original step 1
    def single_pangti_substring(self, cap: str) -> bool:
        return bool(cap) and cap in self.single_index

    # original step 2
    def pair_pangti_substring(self, cap: str) -> bool:
        return bool(cap) and bool(self.pair_index) and cap in self.pair_index

    # original step 3: full chorus repeat (cap = unit * n exactly)
    def is_chorus_repeat(self, cap: str) -> bool:
        if not cap or len(cap) < 6:
            return False
        L = len(cap)
        for n in range(2, L // 3 + 1):
            if L % n:
                continue
            unit = cap[: L // n]
            if cap == unit * n and unit in self.single_index:
                return True
        return False

    # NEW step 4: prefix-tile repeat — cap = unit*k truncated to len(cap),
    # where unit is a substring of some pangti skel (i.e. unit in single_index).
    # Catches "AB AB AB A..." style partial chorus loops.
    def is_prefix_tile(self, cap: str, min_unit: int = 6) -> bool:
        if not cap or len(cap) < min_unit * 2:
            return False
        L = len(cap)
        # Try unit lengths from min_unit up to L/2. Going short→long; first
        # match wins. We avoid trivial cases by requiring k >= 2 repeats.
        for u in range(min_unit, L // 2 + 1):
            unit = cap[:u]
            # Build (unit * ceil(L/u))[:L]; faster: just check positions.
            ok = True
            for i in range(L):
                if cap[i] != unit[i % u]:
                    ok = False
                    break
            if ok and unit in self.single_index:
                return True
        return False

    # NEW step 5: boundary-trim retry — drop up to T left/right chars and
    # retry single/pair checks. Catches "extra short word at start/end".
    def matches_with_trim(self, cap: str, max_trim: int = 3, min_keep: int = 6) -> bool:
        L = len(cap)
        if L < min_keep + 1:
            return False
        for tl in range(0, max_trim + 1):
            for tr in range(0, max_trim + 1):
                if tl == 0 and tr == 0:
                    continue
                trimmed = cap[tl: L - tr] if tr else cap[tl:]
                if len(trimmed) < min_keep:
                    continue
                if trimmed in self.single_index:
                    return True
                if self.pair_index and trimmed in self.pair_index:
                    return True
        return False


def classify_v2(text: str, idx: SkelIndexV2) -> tuple[str, str]:
    cap = skel(text)
    if idx.single_pangti_substring(cap):
        return "high", "single"
    if idx.pair_pangti_substring(cap):
        return "high", "pair"
    if idx.is_chorus_repeat(cap):
        return "high", "chorus"
    if idx.is_prefix_tile(cap):
        return "high", "prefix_tile"
    if idx.matches_with_trim(cap):
        return "high", "trim"
    return "residual", "unknown"


def run(
    residual_in: Path,
    residual_out: Path,
    matched_path: Path,
    db_path: str,
    source_repo: str,
    dest_repo: str,
    public: bool,
    dry_run: bool,
    limit: int | None,
):
    print(f"[repass] loading SGGS index from {db_path} ...", flush=True)
    t0 = time.time()
    idx = SkelIndexV2(db_path)
    print(
        f"[repass] loaded {len(idx.lines)} pangtis in {time.time() - t0:.1f}s",
        flush=True,
    )

    print(f"[repass] reading residuals from {residual_in}", flush=True)
    entries: list[dict] = []
    with residual_in.open() as fp:
        for line in fp:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
            if limit and len(entries) >= limit:
                break
    print(f"[repass] {len(entries)} residuals to re-pass", flush=True)

    matched: list[dict] = []
    still_residual: list[dict] = []
    histo = Counter()
    t1 = time.time()
    for i, e in enumerate(entries):
        text = e.get("original_text") or ""
        quality, route = classify_v2(text, idx)
        histo[(quality, route)] += 1
        if quality == "high":
            matched.append({**e, "skel_route_v2": route})
        else:
            still_residual.append(e)
        if (i + 1) % 5000 == 0:
            rate = (i + 1) / (time.time() - t1)
            print(
                f"[repass] processed {i+1}/{len(entries)} ({rate:.0f}/s), "
                f"matched={len(matched)} residual={len(still_residual)}",
                flush=True,
            )

    elapsed = time.time() - t1
    print(f"[repass] done in {elapsed:.1f}s", flush=True)
    print(f"[repass] histogram:")
    for k, v in sorted(histo.items(), key=lambda kv: -kv[1]):
        print(f"    {k}: {v}")
    pct = 100.0 * len(matched) / max(len(entries), 1)
    print(f"[repass] recovered {len(matched)}/{len(entries)} ({pct:.1f}%) as high")

    print(f"[repass] writing remaining residual to {residual_out}", flush=True)
    with residual_out.open("w") as fp:
        for e in still_residual:
            fp.write(json.dumps(e, ensure_ascii=False) + "\n")
    with matched_path.open("w") as fp:
        for e in matched:
            fp.write(json.dumps(e, ensure_ascii=False) + "\n")

    if dry_run:
        print("[repass] dry-run: skipping HF push", flush=True)
        return

    if not matched:
        print("[repass] no rows to push", flush=True)
        return

    print(
        f"[repass] streaming {source_repo} to attach audio for {len(matched)} matched clips",
        flush=True,
    )
    from datasets import Audio, Dataset, load_dataset

    src = load_dataset(source_repo, split="train", streaming=True)
    if "audio" in (src.features or {}):
        try:
            src = src.cast_column("audio", Audio(decode=False))
        except Exception as e:
            print(f"[repass] cast audio decode=False failed: {e}", flush=True)

    by_id = {m["clip_id"]: m for m in matched}
    target = set(by_id)
    rows_out: list[dict] = []
    seen = 0
    for row in src:
        cid = row.get("clip_id")
        if cid not in target:
            continue
        m = by_id[cid]
        rows_out.append({
            "audio": row.get("audio"),
            "text": row.get("text"),
            "original_text": row.get("text"),
            "quality": "high",
            "video_id": row.get("video_id"),
            "start_s": row.get("start_s"),
            "end_s": row.get("end_s"),
            "duration_s": row.get("duration_s"),
            "channel": row.get("channel"),
            "clip_id": cid,
            "skel_route": m.get("skel_route_v2", "v2"),
        })
        seen += 1
        if seen >= len(target):
            break
    print(f"[repass] attached audio for {seen}/{len(target)}", flush=True)

    if not rows_out:
        return

    from huggingface_hub import create_repo

    try:
        create_repo(
            dest_repo, repo_type="dataset", private=not public,
            exist_ok=True, token=os.environ.get("HF_TOKEN"),
        )
    except Exception as e:
        print(f"[repass] create_repo: {e}", flush=True)

    ds_out = Dataset.from_list(rows_out)
    if "audio" in ds_out.column_names:
        try:
            ds_out = ds_out.cast_column("audio", Audio(sampling_rate=16000))
        except Exception as e:
            print(f"[repass] cast audio failed: {e}", flush=True)
    split_name = f"clean_phase1b_{time.strftime('%Y%m%d_%H%M%S')}"
    ds_out.push_to_hub(
        dest_repo, split=split_name, private=not public,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"[repass] pushed split={split_name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual-in", type=Path, required=True)
    ap.add_argument("--residual-out", type=Path, required=True)
    ap.add_argument("--matched", type=Path, required=True,
                    help="JSONL to write newly-matched entries for audit")
    ap.add_argument("--db", required=True)
    ap.add_argument("--source-repo", required=True)
    ap.add_argument("--dest-repo", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(
        residual_in=args.residual_in,
        residual_out=args.residual_out,
        matched_path=args.matched,
        db_path=args.db,
        source_repo=args.source_repo,
        dest_repo=args.dest_repo,
        public=not args.private,
        dry_run=args.dry_run,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
