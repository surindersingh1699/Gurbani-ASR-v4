"""Kirtan-only eval for the IndicConformer-pa hybrid checkpoint.

Restores a .nemo, transcribes the kirtan eval manifest, and reports
WER + CER (primary: CER) on normalized Gurmukhi. Sehaj is intentionally
NOT scored here -- the /goal is "show evals for kirtan only".

Usage:
    python scripts/eval_kirtan_nemo.py \
        --nemo /workspace/runs/<ts>/checkpoints/<best>.nemo \
        --manifest /workspace/data/manifests/val_kirtan.jsonl \
        --decoder rnnt \
        --out /workspace/runs/<ts>/kirtan_eval.json
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path


def normalize_gurbani_text(text: str) -> str:
    """Match training-target normalization: strip the danda verse markers."""
    text = re.sub(r'॥[੦-੯]+॥', '', text or '')
    text = re.sub(r'॥', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Gurmukhi matra / vowel-sign + nasal codepoints stripped for the
# "normalized Gurbani CER" headline (matra diffs are not worth counting).
_MATRAS = ''.join(chr(c) for c in [
    0x0A3C, 0x0A41, 0x0A42, 0x0A47, 0x0A48, 0x0A4B, 0x0A4C, 0x0A4D,
    0x0A3E, 0x0A3F, 0x0A40, 0x0A70, 0x0A71, 0x0A02, 0x0A01,
])
_MATRA_RE = re.compile('[' + re.escape(_MATRAS) + ']')


def strip_matras(text: str) -> str:
    return _MATRA_RE.sub('', text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True, help="path to fine-tuned .nemo")
    ap.add_argument("--manifest", required=True, help="kirtan eval manifest jsonl")
    ap.add_argument("--decoder", default="rnnt", choices=["rnnt", "ctc"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="kirtan_eval.json")
    args = ap.parse_args()

    import torch
    from jiwer import wer as jwer, cer as jcer
    import nemo.collections.asr as nemo_asr

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    paths = [r["audio_filepath"] for r in rows]
    refs = [normalize_gurbani_text(r["text"]) for r in rows]
    print(f"[eval] {len(rows)} kirtan clips from {args.manifest}", flush=True)

    print(f"[eval] restoring {args.nemo}", flush=True)
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
        args.nemo, map_location="cuda" if torch.cuda.is_available() else "cpu")
    try:
        model.change_decoding_strategy(decoder_type=args.decoder)
    except Exception as e:
        print(f"[eval] change_decoding_strategy({args.decoder}) failed: {e}", flush=True)
    model = model.eval()

    t = time.time()
    hyps = model.transcribe(paths, batch_size=args.batch_size, num_workers=4)
    if isinstance(hyps, tuple):          # hybrid can return (best, all)
        hyps = hyps[0]
    hyps = [h.text if hasattr(h, "text") else h for h in hyps]
    hyps = [normalize_gurbani_text(h) for h in hyps]
    print(f"[eval] transcribed in {time.time()-t:.0f}s", flush=True)

    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    refs2 = [r for r, _ in pairs]; hyps2 = [h for _, h in pairs]
    W = jwer(refs2, hyps2) * 100
    C = jcer(refs2, hyps2) * 100
    Cm = jcer([strip_matras(r) for r in refs2],
              [strip_matras(h) for h in hyps2]) * 100

    print("\n=== KIRTAN EVAL (decoder=%s) ===" % args.decoder)
    print(f"  clips        : {len(refs2)}")
    print(f"  WER          : {W:.2f}%")
    print(f"  CER          : {C:.2f}%")
    print(f"  CER (matra-normalized, headline): {Cm:.2f}%")
    for i in range(min(4, len(refs2))):
        print(f"  REF: {refs2[i][:80]}")
        print(f"  HYP: {hyps2[i][:80]}\n")

    out = {
        "decoder": args.decoder, "n_clips": len(refs2),
        "kirtan_wer": round(W, 3), "kirtan_cer": round(C, 3),
        "kirtan_cer_matra_normalized": round(Cm, 3),
        "nemo": args.nemo, "manifest": args.manifest,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[eval] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
