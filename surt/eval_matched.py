"""
Standalone eval for the surt-small-turbo-baseline-v0 student.

Runs three eval sets to give the apples-to-apples comparison with the teacher:

1. Sehaj matched canonical (`gurbani-sehajpath-yt-captions-eval-canonical`)
   - Same set used for the teacher's headline 16.31% WER.
2. Kirtan matched canonical (`gurbani-kirtan-yt-captions-eval-canonical`)
   - Same set used for the teacher's headline 54.80% WER.
3. Kirtan pure canonical (`gurbani-kirtan-eval-pure-canonical`)
   - The format-mismatched set used during training; reported for continuity.

Usage:
    python -m surt.eval_matched \\
        --model-id surindersinghssj/surt-small-turbo-baseline-v0 \\
        --batch-size 16

Or with a local checkpoint:
    python -m surt.eval_matched --model-path /workspace/surt/distill_turbo/checkpoint-1000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jiwer
import torch
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from surt.config import GENERATION_MAX_LENGTH


EVAL_SETS = [
    {
        "label": "sehaj_matched",
        "repo": "surindersinghssj/gurbani-sehajpath-yt-captions-eval-canonical",
        "split": "train",
        "text_col": "final_text",
        "teacher_wer": 16.31,
        "teacher_cer": 5.25,
    },
    {
        "label": "kirtan_matched",
        "repo": "surindersinghssj/gurbani-kirtan-yt-captions-eval-canonical",
        "split": "train",
        "text_col": "final_text",
        "teacher_wer": 54.80,
        "teacher_cer": 28.00,
    },
    {
        "label": "kirtan_pure",
        "repo": "surindersinghssj/gurbani-kirtan-eval-pure-canonical",
        "split": "eval",
        "text_col": "final_text",
        "teacher_wer": None,  # known to be label-format-mismatched, no canonical teacher number
        "teacher_cer": None,
    },
]


def load_model(model_id: str | None, model_path: str | None, dtype):
    src = model_path or model_id
    print(f"[eval] loading model from: {src}")
    model = WhisperForConditionalGeneration.from_pretrained(
        src, torch_dtype=dtype, attn_implementation="sdpa"
    )
    model.generation_config.language = "punjabi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_length = GENERATION_MAX_LENGTH
    model.train(False)
    processor = WhisperProcessor.from_pretrained(
        src, language="punjabi", task="transcribe", use_fast=False,
    )
    return model, processor


def eval_one(model, processor, *, repo: str, split: str, text_col: str, batch_size: int, device: str):
    print(f"\n[eval] === {repo} (split={split}) ===")
    t0 = time.time()
    ds = load_dataset(repo, split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"[eval]   loaded {len(ds)} examples in {time.time()-t0:.1f}s")

    all_preds: list[str] = []
    all_refs: list[str] = []
    n_processed = 0
    t_inf = time.time()

    for i in range(0, len(ds), batch_size):
        batch = ds[i : i + batch_size]
        arrays = [a["array"] for a in batch["audio"]]
        refs = batch[text_col]

        feats = processor(
            arrays, sampling_rate=16000, return_tensors="pt", padding=True
        ).input_features.to(device=device, dtype=next(model.parameters()).dtype)

        with torch.no_grad():
            # Reserve 4 tokens for Whisper's special start sequence
            # (decoder_start, language, task, no_timestamps).
            ids = model.generate(feats, num_beams=1, max_new_tokens=GENERATION_MAX_LENGTH - 4)
        preds = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)

        all_preds.extend([p.strip() for p in preds])
        all_refs.extend([r.strip() for r in refs])
        n_processed += len(refs)
        if (i // batch_size) % 5 == 0:
            print(f"[eval]   {n_processed}/{len(ds)}", flush=True)

    wer = 100 * jiwer.wer(all_refs, all_preds)
    cer = 100 * jiwer.cer(all_refs, all_preds)
    print(
        f"[eval]   WER={wer:.2f}  CER={cer:.2f}  "
        f"({len(ds)} examples, {time.time()-t_inf:.0f}s inference)"
    )
    return {"wer": wer, "cer": cer, "n": len(ds)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=None, help="HF repo id (e.g. surindersinghssj/surt-small-turbo-baseline-v0)")
    p.add_argument("--model-path", default=None, help="Local checkpoint path (overrides --model-id)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default="/workspace/surt/distill_turbo/eval_matched.json")
    args = p.parse_args()

    if not args.model_id and not args.model_path:
        raise SystemExit("provide --model-id or --model-path")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model, processor = load_model(args.model_id, args.model_path, dtype)
    model.to(device)

    results = {}
    for spec in EVAL_SETS:
        try:
            res = eval_one(
                model, processor,
                repo=spec["repo"], split=spec["split"], text_col=spec["text_col"],
                batch_size=args.batch_size, device=device,
            )
            res["teacher_wer"] = spec["teacher_wer"]
            res["teacher_cer"] = spec["teacher_cer"]
            results[spec["label"]] = res
        except Exception as e:
            print(f"[eval] FAILED on {spec['label']}: {e}")
            results[spec["label"]] = {"error": str(e)}

    print("\n=== SUMMARY ===")
    print(f"{'set':<22} {'WER':>8} {'CER':>8}  {'teacher WER':>12} {'teacher CER':>12}  {'gap':>10}")
    for label, r in results.items():
        if "error" in r:
            print(f"{label:<22} ERROR: {r['error']}")
            continue
        tw = r.get("teacher_wer")
        tc = r.get("teacher_cer")
        gap = (r["wer"] - tw) if tw is not None else None
        gap_s = f"+{gap:.2f}" if gap is not None else "n/a"
        tw_s = f"{tw:.2f}" if tw is not None else "n/a"
        tc_s = f"{tc:.2f}" if tc is not None else "n/a"
        print(f"{label:<22} {r['wer']:>8.2f} {r['cer']:>8.2f}  {tw_s:>12} {tc_s:>12}  {gap_s:>10}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
