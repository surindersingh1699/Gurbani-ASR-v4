"""Stratified anchor eval across 5 buckets per PLAN-v2.md.

Buckets:
  seen_sehaj       gurbani-sehajpath validation
  unseen_sehaj     gurbani-sehajpath-yt-captions-eval-canonical
  ragi             gurbani-kirtan-yt-captions-eval-canonical (kirtan_type=ragi)
  akj              gurbani-kirtan-yt-captions-eval-canonical (kirtan_type=akj)
  sgpc             gurbani-kirtan-eval-pure-canonical

Reports per bucket:
  utterance_exact_match_rate
  mean_pos_match_rate (1 - WER over anchor positions)
  anchor_shabad_recall@1 / @5
  top1 - top2 margin
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

from sttm_first_letter_map import (
    db_first_letters_to_search_anchor,
    training_anchor_to_search_anchor,
)


def _char_4grams(s):
    if len(s) < 4:
        return {s} if s else set()
    return {s[i:i + 4] for i in range(len(s) - 3)}


def _overlap(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _load_db_anchors(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = []
    for sid, fl in cur.execute("SELECT shabad_id, first_letters FROM lines"):
        a = db_first_letters_to_search_anchor(fl or "")
        if a:
            rows.append((sid, a, _char_4grams(a)))
    conn.close()
    return rows


def _topn_shabads(q, db_rows, n=5):
    q4 = _char_4grams(q)
    best = {}
    for sid, a, ng in db_rows:
        s = _overlap(q4, ng)
        cur = best.get(sid)
        if cur is None or s > cur[0]:
            best[sid] = (s, a)
    return sorted(((s, sid, a) for sid, (s, a) in best.items()), reverse=True)[:n]


def _evaluate_one_bucket(model, manifest_path, db_rows, batch_size, max_clips):
    items = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
            if max_clips and len(items) >= max_clips:
                break
    if not items:
        return None
    files = [it["audio_filepath"] for it in items]
    refs = [training_anchor_to_search_anchor(it["text"]) for it in items]

    t0 = time.time()
    hyps = model.transcribe(audio=files, batch_size=batch_size)
    if hyps and hasattr(hyps[0], "text"):
        hyps = [h.text for h in hyps]
    dt = time.time() - t0
    preds = [training_anchor_to_search_anchor(h) for h in hyps]

    n_exact = n_pos_total = n_pos_match = recall1 = recall5 = 0
    margin_sum = 0.0
    decode_lens = Counter()
    samples = []
    for ref, pred in zip(refs, preds):
        if pred == ref:
            n_exact += 1
        n_pos_total += max(len(ref), 1)
        for i in range(min(len(pred), len(ref))):
            if pred[i] == ref[i]:
                n_pos_match += 1
        decode_lens[len(pred)] += 1
        if len(samples) < 20:
            samples.append({"ref": ref, "pred": pred, "exact": pred == ref})
        if not pred or not ref:
            continue
        ref_top = _topn_shabads(ref, db_rows, n=1)
        pred_top = _topn_shabads(pred, db_rows, n=5)
        if not ref_top or not pred_top:
            continue
        ref_sid = ref_top[0][1]
        top_ids = [sid for _, sid, _ in pred_top]
        if top_ids[0] == ref_sid:
            recall1 += 1
        if ref_sid in top_ids:
            recall5 += 1
        if len(pred_top) >= 2:
            margin_sum += pred_top[0][0] - pred_top[1][0]
    n = len(items)
    return {
        "n_clips": n,
        "decode_throughput_per_s": (n / dt) if dt else 0.0,
        "utterance_exact_match_rate": n_exact / n,
        "mean_pos_match_rate": n_pos_match / max(n_pos_total, 1),
        "anchor_shabad_recall_at_1": recall1 / n,
        "anchor_shabad_recall_at_5": recall5 / n,
        "mean_top1_minus_top2_margin": margin_sum / n,
        "samples": samples,
    }


BUCKETS = [
    ("seen_sehaj",   "<5%",  "v2_anchor_val_seen_sehaj.jsonl"),
    ("unseen_sehaj", "<10%", "v2_anchor_val_unseen_sehaj.jsonl"),
    ("ragi",         "<15%", "v2_anchor_val_ragi.jsonl"),
    ("akj",          "<20%", "v2_anchor_val_akj.jsonl"),
    ("sgpc",         "<25%", "v2_anchor_val_sgpc.jsonl"),
]

GATES = {"seen_sehaj": 0.05, "unseen_sehaj": 0.10, "ragi": 0.15, "akj": 0.20, "sgpc": 0.25}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifests-dir", default="/workspace/data/manifests")
    ap.add_argument("--db", default=str(REPO_ROOT / "database.sqlite"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-clips-per-bucket", type=int, default=300)
    args = ap.parse_args()

    import torch
    from nemo.collections.asr.models import EncDecCTCModel
    print(f"[scoring] loading model {args.model}", flush=True)
    model = EncDecCTCModel.restore_from(args.model, map_location="cuda")
    model.freeze()
    if torch.cuda.is_available():
        model = model.cuda()

    db_rows = _load_db_anchors(args.db)
    print(f"[scoring] db_rows={len(db_rows)}", flush=True)

    report = {"model": args.model, "buckets": {}}
    pass_count = 0
    md = Path(args.manifests_dir)
    for name, gate_label, manifest in BUCKETS:
        path = md / manifest
        if not path.exists():
            print(f"  [{name}] manifest missing: {path}", flush=True)
            continue
        print(f"\n=== bucket: {name} (gate {gate_label}) ===", flush=True)
        r = _evaluate_one_bucket(model, path, db_rows,
                                 batch_size=args.batch_size,
                                 max_clips=args.max_clips_per_bucket)
        if r is None:
            continue
        wer = 1 - r["mean_pos_match_rate"]
        gate = GATES[name]
        passed = wer < gate
        if passed:
            pass_count += 1
        r["bucket"] = name
        r["wer"] = wer
        r["gate"] = gate
        r["passed_gate"] = passed
        report["buckets"][name] = r
        print(f"  WER={wer:.4f}  gate={gate}  {'PASS' if passed else 'FAIL'}", flush=True)
        print(f"  exact={r['utterance_exact_match_rate']:.4f}  "
              f"recall@1={r['anchor_shabad_recall_at_1']:.4f}  "
              f"recall@5={r['anchor_shabad_recall_at_5']:.4f}  "
              f"margin={r['mean_top1_minus_top2_margin']:.4f}", flush=True)

    report["n_buckets_passed"] = pass_count
    report["n_buckets_total"] = len([b for b in BUCKETS if (md / b[2]).exists()])
    report["decision"] = (
        "GO" if pass_count == report["n_buckets_total"]
        else ("PARTIAL" if pass_count >= 3 else "NO-GO"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[scoring] wrote {out_path}", flush=True)
    print(f"\n=== SUMMARY: {pass_count}/{report['n_buckets_total']} gates passed -> {report['decision']} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
