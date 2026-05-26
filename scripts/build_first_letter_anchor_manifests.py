"""Build NeMo JSONL manifests for first-letter anchor CTC training.

PLAN.md M1.1 — train target = delimited STTM-ASCII first-letter sequence.

For v1 we use ONLY surindersinghssj/gurbani-sehajpath (clean, well-aligned,
~66h, ~63k clips). Kirtan is deferred to a follow-up.

Output:
  {AUDIO_OUTDIR}/{clip_id}.flac
  data/manifests/anchor_first_letter_train.jsonl
  data/manifests/anchor_first_letter_val.jsonl
  data/manifests/anchor_first_letter_eval_by_domain.jsonl

Each row:
  {"audio_filepath": "...", "duration": 7.34,
   "text": "s|n|k|p", "search_anchor": "snkp",
   "source": "sehajpath", "speaker_id": "..."}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sttm_first_letter_map import (  # noqa: E402
    gurmukhi_text_to_training_anchor,
    training_anchor_to_search_anchor,
)
from scripts.build_indicconformer_manifests import (  # noqa: E402
    normalize_gurbani_text,
    passes_simran_filter,
)


def _iter_dataset(name: str, split: str, audio_outdir: Path, source_tag: str,
                  text_cols: tuple[str, ...] = ("gurmukhi_text", "transcription", "text"),
                  min_duration: float = 0.5, max_duration: float = 18.0):
    """Decode each HF dataset row to a local FLAC, yield (path, dur, anchor, search, source, speaker)."""
    import soundfile as sf
    from datasets import load_dataset, Audio

    audio_outdir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}:{split}] loading ...", flush=True)
    t0 = time.time()
    ds = load_dataset(name, split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"[{name}:{split}] loaded in {time.time()-t0:.1f}s, {len(ds)} rows", flush=True)

    kept = dropped_simran = dropped_empty = dropped_dur = dropped_anchor = decoded = 0
    last_log = time.time()
    for i, ex in enumerate(ds):
        text = ""
        for col in text_cols:
            v = ex.get(col)
            if v:
                text = str(v)
                break
        text = normalize_gurbani_text(text)
        if not text:
            dropped_empty += 1
            continue
        if not passes_simran_filter(text):
            dropped_simran += 1
            continue

        anchor = gurmukhi_text_to_training_anchor(text)
        if not anchor:
            dropped_anchor += 1
            continue

        clip_id = ex.get("clip_id") or ex.get("id") or f"{source_tag}_{split}_{i:07d}"
        flac_path = audio_outdir / f"{clip_id}.flac"
        audio = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        duration = float(audio.shape[0]) / float(sr)
        if duration < min_duration or duration > max_duration:
            dropped_dur += 1
            continue
        if not flac_path.exists():
            try:
                sf.write(str(flac_path), audio.astype("float32"), sr,
                         format="FLAC", subtype="PCM_16")
                decoded += 1
            except Exception as e:
                print(f"  [warn] decode failed for {clip_id}: {e}", flush=True)
                continue
        speaker = str(ex.get("speaker_id") or "")
        yield (str(flac_path), duration, anchor,
               training_anchor_to_search_anchor(anchor), source_tag, speaker)
        kept += 1
        if time.time() - last_log > 30:
            print(f"  [{name}:{split}] {i+1}/{len(ds)} kept={kept} decoded={decoded} "
                  f"dropped: simran={dropped_simran} empty={dropped_empty} "
                  f"dur={dropped_dur} anchor={dropped_anchor}", flush=True)
            last_log = time.time()
    print(f"[{name}:{split}] FINAL kept={kept} decoded={decoded} "
          f"simran={dropped_simran} empty={dropped_empty} dur={dropped_dur} "
          f"anchor={dropped_anchor}", flush=True)


def _write_jsonl(rows, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for wav, dur, anchor, search, source, speaker in rows:
            f.write(json.dumps({
                "audio_filepath": wav,
                "duration": dur,
                "text": anchor,
                "search_anchor": search,
                "source": source,
                "speaker_id": speaker,
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspace/data")
    ap.add_argument("--audio-root", default="/workspace/data/audio_anchor")
    ap.add_argument("--dataset", default="surindersinghssj/gurbani-sehajpath",
                    help="HF dataset id for training+validation source.")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation")
    args = ap.parse_args()

    audio_root = Path(args.audio_root)
    manifests_dir = Path(args.data_root) / "manifests"

    train_path = manifests_dir / "anchor_first_letter_train.jsonl"
    val_path = manifests_dir / "anchor_first_letter_val.jsonl"

    print("=== TRAIN ===")
    n_train = _write_jsonl(
        _iter_dataset(args.dataset, args.train_split,
                      audio_root / "sehajpath_train", "sehajpath"),
        train_path,
    )
    print(f"[train] wrote {n_train} rows -> {train_path}")

    print("=== VAL ===")
    n_val = _write_jsonl(
        _iter_dataset(args.dataset, args.val_split,
                      audio_root / "sehajpath_val", "sehajpath"),
        val_path,
    )
    print(f"[val] wrote {n_val} rows -> {val_path}")

    # eval-by-domain == val for now (sehaj only). Kept distinct so the
    # NeMo trainer's test_ds path is wired and we can swap a real domain
    # split in later without changing the config.
    eval_path = manifests_dir / "anchor_first_letter_eval_by_domain.jsonl"
    eval_path.write_text(val_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[eval] copied val -> {eval_path}")

    # Vocab snapshot — record all unique chars seen in anchor text (excluding delim).
    chars: set[str] = set()
    for line in train_path.read_text(encoding="utf-8").splitlines():
        anchor = json.loads(line)["text"]
        chars.update(c for c in anchor if c != "|")
    vocab_path = manifests_dir / "anchor_first_letter_vocab.json"
    vocab_path.write_text(json.dumps({
        "delim": "|",
        "chars": sorted(chars),
        "vocab_size_with_delim_and_blank": len(chars) + 2,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[vocab] {len(chars)} unique anchor chars -> {vocab_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
