"""Evaluate a first-letter anchor CTC model with the PLAN.md custom metric.

Reports:
  utterance_exact_match_rate
  mean_pos_match_rate
  anchor_shabad_recall@1 / @5
  top1_margin (top1 - top2 overlap score)
  per-position token-match rate
  ctc decode length histogram
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sttm_first_letter_map import (  # noqa: E402
    db_first_letters_to_search_anchor,
    training_anchor_to_search_anchor,
)


def _char_4grams(s: str) -> set[str]:
    if len(s) < 4:
        return {s} if s else set()
    return {s[i:i + 4] for i in range(len(s) - 3)}


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


def _load_db_anchors(db_path: str) -> list[tuple[str, str, set[str]]]:
    """(shabad_id, search_anchor, anchor_4gram_set) for every DB line."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = []
    for shabad_id, fl in cur.execute(
            "SELECT shabad_id, first_letters FROM lines"):
        anchor = db_first_letters_to_search_anchor(fl or "")
        if anchor:
            rows.append((shabad_id, anchor, _char_4grams(anchor)))
    conn.close()
    return rows


def _topn_shabads(query_anchor: str, db_rows: list, n: int = 5) -> list[tuple[float, str, str]]:
    q4 = _char_4grams(query_anchor)
    best_per_shabad: dict[str, tuple[float, str]] = {}
    for shabad_id, anchor, ngrams in db_rows:
        s = _overlap(q4, ngrams)
        cur = best_per_shabad.get(shabad_id)
        if cur is None or s > cur[0]:
            best_per_shabad[shabad_id] = (s, anchor)
    ranked = sorted(((s, sid, a) for sid, (s, a) in best_per_shabad.items()),
                    reverse=True)[:n]
    return ranked


def _anchor_position_match(pred: str, ref: str) -> tuple[bool, float]:
    """Position-aligned token match. Both strings are compact search anchors."""
    if not ref:
        return (pred == "", 0.0)
    n_ref = len(ref)
    n_match = 0
    for i in range(n_ref):
        if i < len(pred) and pred[i] == ref[i]:
            n_match += 1
    is_exact = (pred == ref)
    return is_exact, n_match / n_ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--db", default=str(REPO_ROOT / "database.sqlite"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap for fast smoke runs.")
    args = ap.parse_args()

    import torch
    from nemo.collections.asr.models import EncDecCTCModel

    print(f"[scoring] loading model {args.model}", flush=True)
    model = EncDecCTCModel.restore_from(args.model, map_location="cuda")
    model.freeze()
    if torch.cuda.is_available():
        model = model.cuda()

    print(f"[scoring] loading manifest {args.manifest}", flush=True)
    items = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            items.append(row)
            if args.limit and len(items) >= args.limit:
                break
    print(f"[scoring] {len(items)} items", flush=True)

    files = [it["audio_filepath"] for it in items]
    refs_anchor = [training_anchor_to_search_anchor(it["text"]) for it in items]

    print(f"[scoring] decoding (batch={args.batch_size}) ...", flush=True)
    t0 = time.time()
    hyps = model.transcribe(audio=files, batch_size=args.batch_size)
    if hyps and hasattr(hyps[0], "text"):
        hyps = [h.text for h in hyps]
    dt = time.time() - t0
    print(f"[scoring] decoded {len(hyps)} clips in {dt:.1f}s "
          f"({len(hyps)/max(dt,1e-3):.1f}/s)", flush=True)

    preds_anchor = [training_anchor_to_search_anchor(h) for h in hyps]

    db_rows = _load_db_anchors(args.db)
    print(f"[scoring] db_rows={len(db_rows)}", flush=True)

    n_exact = 0
    n_pos_total = 0
    n_pos_match = 0
    recall1 = recall5 = 0
    margin_sum = 0.0
    decode_lens: Counter = Counter()
    sample_decodes = []

    for ref, pred in zip(refs_anchor, preds_anchor):
        is_exact, pos_rate = _anchor_position_match(pred, ref)
        n_exact += int(is_exact)
        n_pos_total += max(len(ref), 1)
        n_pos_match += int(round(pos_rate * max(len(ref), 1)))
        decode_lens[len(pred)] += 1
        if len(sample_decodes) < 30:
            sample_decodes.append({"ref": ref, "pred": pred, "exact": is_exact})
        if not pred or not ref:
            continue
        ref_top = _topn_shabads(ref, db_rows, n=1)
        if not ref_top:
            continue
        ref_shabad = ref_top[0][1]
        pred_top = _topn_shabads(pred, db_rows, n=5)
        if not pred_top:
            continue
        top_ids = [sid for _, sid, _ in pred_top]
        if top_ids[0] == ref_shabad:
            recall1 += 1
        if ref_shabad in top_ids:
            recall5 += 1
        if len(pred_top) >= 2:
            margin_sum += pred_top[0][0] - pred_top[1][0]
        else:
            margin_sum += pred_top[0][0]

    n = len(items)
    report = {
        "model": args.model,
        "manifest": args.manifest,
        "n_clips": n,
        "decoding_throughput_per_s": (len(hyps) / dt) if dt else 0.0,
        "utterance_exact_match_rate": n_exact / max(n, 1),
        "mean_pos_match_rate": n_pos_match / max(n_pos_total, 1),
        "anchor_shabad_recall_at_1": recall1 / max(n, 1),
        "anchor_shabad_recall_at_5": recall5 / max(n, 1),
        "mean_top1_minus_top2_margin": margin_sum / max(n, 1),
        "pred_decode_length_histogram": dict(sorted(decode_lens.items())),
        "samples": sample_decodes,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[scoring] wrote {out_path}", flush=True)
    print(f"  utterance_exact_match_rate = {report['utterance_exact_match_rate']:.4f}")
    print(f"  mean_pos_match_rate        = {report['mean_pos_match_rate']:.4f}")
    print(f"  anchor_shabad_recall@1     = {report['anchor_shabad_recall_at_1']:.4f}")
    print(f"  anchor_shabad_recall@5     = {report['anchor_shabad_recall_at_5']:.4f}")
    print(f"  mean_top1-top2_margin      = {report['mean_top1_minus_top2_margin']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
