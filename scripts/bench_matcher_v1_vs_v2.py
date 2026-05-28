"""Benchmark matcher v1 (4-gram overlap, current production) vs matcher v2
(beam + window voting + edit-distance + continuity bias) on the same
v2.0 model and kirtan eval clips.

Usage:
    python scripts/bench_matcher_v1_vs_v2.py \
        --model /workspace/.../anchor-large-v2.nemo \
        --manifest /workspace/data/manifests/v2_anchor_val_kirtan_mixed.jsonl \
        --db /workspace/Gurbani-ASR-v4/database.sqlite \
        --out /workspace/eval/bench_matchers.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sttm_first_letter_map import (
    db_first_letters_to_search_anchor,
    training_anchor_to_search_anchor,
)
from anchor_matcher_v2 import AnchorMatcherV2, _char_4grams, _overlap


def _v1_topn(query_anchor, db_rows, n=5):
    """v1 matcher: exact 4-gram overlap, best-per-shabad."""
    q4 = _char_4grams(query_anchor)
    best_per_shabad = {}
    for shabad_id, anchor, ngrams in db_rows:
        s = _overlap(q4, ngrams)
        cur = best_per_shabad.get(shabad_id)
        if cur is None or s > cur[0]:
            best_per_shabad[shabad_id] = (s, anchor)
    return sorted(((s, sid, a) for sid, (s, a) in best_per_shabad.items()),
                  reverse=True)[:n]


def _load_db_anchors_v1(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = []
    for sid, fl in cur.execute("SELECT shabad_id, first_letters FROM lines"):
        a = db_first_letters_to_search_anchor(fl or "")
        if a:
            rows.append((sid, a, _char_4grams(a)))
    conn.close()
    return rows


def _try_beam_decode(model, files, batch_size=8, beam_size=5):
    """Try to get top-K CTC hypotheses via NeMo beam search.
    Falls back to greedy if beam-search API isn't available.
    """
    try:
        from nemo.collections.asr.parts.submodules.ctc_decoding import (
            CTCDecodingConfig)
        from omegaconf import OmegaConf

        # Switch decoding to beam search
        decoding_cfg = OmegaConf.create(model.cfg.decoding) if "decoding" in model.cfg else OmegaConf.create({})
        decoding_cfg.strategy = "beam"
        if "beam" not in decoding_cfg:
            decoding_cfg.beam = OmegaConf.create({})
        decoding_cfg.beam.beam_size = beam_size
        decoding_cfg.beam.return_best_hypothesis = False
        model.change_decoding_strategy(decoding_cfg)
        hyps = model.transcribe(audio=files, batch_size=batch_size,
                                return_hypotheses=True)
        # NeMo returns nested list of hypotheses when return_best_hypothesis=False
        out = []
        for h in hyps:
            if isinstance(h, list):
                out.append([x.text if hasattr(x, "text") else x for x in h])
            elif hasattr(h, "text"):
                out.append([h.text])
            else:
                out.append([h])
        return out, True
    except Exception as e:
        print(f"[beam] not available ({e}); falling back to greedy top-1", flush=True)
        hyps = model.transcribe(audio=files, batch_size=batch_size)
        if hyps and hasattr(hyps[0], "text"):
            hyps = [h.text for h in hyps]
        return [[h] for h in hyps], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--window-size", type=int, default=8)
    ap.add_argument("--window-stride", type=int, default=4)
    args = ap.parse_args()

    import torch
    from nemo.collections.asr.models import EncDecCTCModel

    print(f"[bench] loading model {args.model}", flush=True)
    model = EncDecCTCModel.restore_from(args.model, map_location="cuda")
    model.freeze()
    if torch.cuda.is_available():
        model = model.cuda()

    print(f"[bench] loading manifest {args.manifest}", flush=True)
    items = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
            if args.max_clips and len(items) >= args.max_clips:
                break
    print(f"[bench] {len(items)} clips", flush=True)
    files = [it["audio_filepath"] for it in items]
    refs = [training_anchor_to_search_anchor(it["text"]) for it in items]

    print(f"[bench] decoding with beam (size={args.beam_size})", flush=True)
    t0 = time.time()
    hyp_lists, beam_ok = _try_beam_decode(model, files, args.batch_size, args.beam_size)
    print(f"[bench] decoded {len(hyp_lists)} clips in {time.time()-t0:.1f}s, "
          f"beam={'yes' if beam_ok else 'no (greedy only)'}", flush=True)

    print(f"[bench] loading DB", flush=True)
    db_v1 = _load_db_anchors_v1(args.db)
    matcher_v2 = AnchorMatcherV2(args.db,
                                 w_overlap=0.5, w_edit=0.5,
                                 continuity_bonus=0.15,
                                 continuity_window=10)
    print(f"[bench] db_rows={len(db_v1)}", flush=True)

    # Score v1 (greedy top-1 + exact overlap)
    # Score v2 (beam + edit-distance + window voting + continuity)
    v1_r1 = v1_r5 = 0
    v2_r1 = v2_r5 = 0
    v2nc_r1 = v2nc_r5 = 0  # v2 without continuity
    sample_dec = []
    matcher_v2.reset_lock()

    n = 0
    for hyps, ref in zip(hyp_lists, refs):
        if not ref:
            continue
        # Convert each hypothesis (delimited) to compact form
        candidates = [training_anchor_to_search_anchor(h) for h in hyps if h]
        if not candidates:
            continue
        pred_greedy = candidates[0]
        beam_alt = candidates[1:] if len(candidates) > 1 else []
        n += 1

        # Ground-truth top-1 shabad (from ref anchor)
        ref_top = _v1_topn(ref, db_v1, n=1)
        if not ref_top:
            continue
        ref_shabad = ref_top[0][1]

        # ----- v1 matcher -----
        v1_top = _v1_topn(pred_greedy, db_v1, n=5)
        v1_ids = [sid for _, sid, _ in v1_top]
        if v1_ids and v1_ids[0] == ref_shabad:
            v1_r1 += 1
        if ref_shabad in v1_ids:
            v1_r5 += 1

        # ----- v2 matcher with continuity -----
        v2_top = matcher_v2.search_with_windows(
            pred_greedy, window_size=args.window_size,
            stride=args.window_stride, n_best=5,
            beam_anchors=beam_alt if beam_ok else None)
        v2_ids = [c.shabad_id for c in v2_top]
        if v2_ids and v2_ids[0] == ref_shabad:
            v2_r1 += 1
        if ref_shabad in v2_ids:
            v2_r5 += 1
        # Update lock state for next clip
        matcher_v2.update_lock(v2_top[0] if v2_top else None,
                               lock_threshold=0.4, min_margin=0.05,
                               all_candidates=v2_top)

        # ----- v2 matcher without continuity (for ablation) -----
        matcher_v2.reset_lock()
        v2nc_top = matcher_v2.search_with_windows(
            pred_greedy, window_size=args.window_size,
            stride=args.window_stride, n_best=5,
            beam_anchors=beam_alt if beam_ok else None)
        matcher_v2.locked_shabad = (
            v2_top[0].shabad_id if v2_top and v2_top[0].score >= 0.4 else None)
        matcher_v2.locked_order_id = None  # rough — we don't recover order
        v2nc_ids = [c.shabad_id for c in v2nc_top]
        if v2nc_ids and v2nc_ids[0] == ref_shabad:
            v2nc_r1 += 1
        if ref_shabad in v2nc_ids:
            v2nc_r5 += 1
        if len(sample_dec) < 30:
            sample_dec.append({
                "ref": ref, "pred_greedy": pred_greedy,
                "beam_alt": beam_alt[:3],
                "ref_shabad": ref_shabad,
                "v1_top1": v1_ids[0] if v1_ids else "",
                "v2_top1": v2_ids[0] if v2_ids else "",
            })
    matcher_v2.reset_lock()

    report = {
        "model": args.model,
        "manifest": args.manifest,
        "n_clips": n,
        "beam_search": beam_ok,
        "beam_size": args.beam_size,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "matcher_v1_recall_at_1": v1_r1 / max(n, 1),
        "matcher_v1_recall_at_5": v1_r5 / max(n, 1),
        "matcher_v2_recall_at_1": v2_r1 / max(n, 1),
        "matcher_v2_recall_at_5": v2_r5 / max(n, 1),
        "matcher_v2_no_continuity_recall_at_1": v2nc_r1 / max(n, 1),
        "matcher_v2_no_continuity_recall_at_5": v2nc_r5 / max(n, 1),
        "samples": sample_dec,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("\n=== HEAD-TO-HEAD ===")
    print(f"  n_clips                          = {n}")
    print(f"  beam_search                      = {beam_ok}  (size={args.beam_size})")
    print(f"  matcher v1 (exact 4-gram only)  recall@1={report['matcher_v1_recall_at_1']:.4f}  recall@5={report['matcher_v1_recall_at_5']:.4f}")
    print(f"  matcher v2 (+ window + edit + continuity)")
    print(f"                                  recall@1={report['matcher_v2_recall_at_1']:.4f}  recall@5={report['matcher_v2_recall_at_5']:.4f}")
    print(f"  matcher v2 (without continuity)  recall@1={report['matcher_v2_no_continuity_recall_at_1']:.4f}  recall@5={report['matcher_v2_no_continuity_recall_at_5']:.4f}")
    print(f"\n[bench] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
