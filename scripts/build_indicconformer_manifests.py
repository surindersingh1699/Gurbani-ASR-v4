"""Materialize NeMo-style JSONL manifests for IndicConformer-pa fine-tune.

Reads HF datasets (already cached locally on the network volume), decodes the
embedded audio bytes to FLAC files ONCE on the volume, and writes a NeMo
manifest:

    {"audio_filepath": "/workspace/data/audio/<dataset>/<clip_id>.flac",
     "duration": 7.34,
     "text": "ਵਾਹਿਗੁਰੂ ..."}

Idempotent: if the FLAC already exists, skips re-decode. Drops video_ids
that leak into eval; applies the v3 simran/repeat filter (REPEAT_THRESHOLD=10).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# normalize_gurbani_text needs numpy/torch chain — duplicate inline to avoid import.
import re

REPEAT_THRESHOLD = 10


def normalize_gurbani_text(text: str) -> str:
    text = re.sub(r'॥[੦-੯]+॥', '', text)
    text = re.sub(r'॥', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def passes_simran_filter(text: str) -> bool:
    if not text or not text.strip():
        return False
    words = text.split()
    if len(words) < 2:
        return True
    run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            if run > REPEAT_THRESHOLD:
                return False
        else:
            run = 1
    return True


def materialize(name: str, audio_outdir: Path, leaked: set[str], text_col: str = "final_text"):
    """Decode each clip's audio bytes to FLAC and yield (path, duration, text)."""
    import soundfile as sf
    from datasets import load_dataset, Audio

    audio_outdir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] loading from local cache (already on volume) ...", flush=True)
    t0 = time.time()
    ds = load_dataset(name, split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"[{name}] loaded in {time.time()-t0:.1f}s, {len(ds)} rows", flush=True)

    kept = dropped_leak = dropped_simran = dropped_empty = decoded = 0
    last_log = time.time()
    for i, ex in enumerate(ds):
        vid = ex.get("video_id", "")
        if vid in leaked:
            dropped_leak += 1; continue

        # Get text — try canonical text col first, fall back to a few alternates
        text = (ex.get(text_col) or ex.get("transcription") or ex.get("text") or "").strip()
        text = normalize_gurbani_text(text)
        if not text:
            dropped_empty += 1; continue
        if not passes_simran_filter(text):
            dropped_simran += 1; continue

        clip_id = ex.get("clip_id") or f"{vid}_{i:06d}"
        flac_path = audio_outdir / f"{clip_id}.flac"
        audio = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        if not flac_path.exists():
            try:
                sf.write(str(flac_path), audio.astype("float32"), sr, format="FLAC", subtype="PCM_16")
                decoded += 1
            except Exception as e:
                print(f"  [warn] decode failed for {clip_id}: {e}")
                continue
        duration = float(audio.shape[0]) / float(sr)
        yield str(flac_path), duration, text
        kept += 1
        if time.time() - last_log > 30:
            print(f"  [{name}] {i+1}/{len(ds)} done — kept={kept} decoded_this_run={decoded}", flush=True)
            last_log = time.time()

    print(f"[{name}] FINAL: kept={kept}  decoded_this_run={decoded}  "
          f"dropped: leak={dropped_leak} simran={dropped_simran} empty={dropped_empty}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspace/data")
    ap.add_argument("--audio-root", default="/workspace/data/audio",
                    help="Where to write decoded FLAC files (one subdir per dataset)")
    ap.add_argument("--leaked-file", default="/workspace/runs/latest/train_dropped_video_ids.txt")
    args = ap.parse_args()

    leaked: set[str] = set()
    if Path(args.leaked_file).exists():
        leaked = set(Path(args.leaked_file).read_text().strip().splitlines())
    print(f"[leak-list] dropping {len(leaked)} video_ids from training", flush=True)

    manifests_dir = Path(args.data_root) / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    audio_root = Path(args.audio_root)

    # train sources, sampling weights expressed as integer repeats
    sources = [
        ("surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical", "final_text", "kirtan",         5),  # 50%
        ("surindersinghssj/gurbani-sehajpath-yt-captions-canonical",   "final_text", "sehajpath_yt",   3),  # 30%
        ("surindersinghssj/gurbani-sehajpath",                         "transcription", "sehajpath_orig", 2),  # 20%
    ]

    train_path = manifests_dir / "train.jsonl"
    n_total = 0
    with train_path.open("w", encoding="utf-8") as f:
        for hf_id, text_col, subdir, repeats in sources:
            entries = list(materialize(hf_id, audio_root / subdir, leaked, text_col=text_col))
            for wav, dur, text in entries * repeats:
                f.write(json.dumps({"audio_filepath": wav, "duration": dur, "text": text},
                                   ensure_ascii=False) + "\n")
                n_total += 1
            print(f"  [train] → {hf_id}: {len(entries)} unique × {repeats} repeats = "
                  f"{len(entries)*repeats} entries", flush=True)
    print(f"[train] {n_total} total entries → {train_path}", flush=True)

    # eval (no leak filter, no simran filter)
    for hf_id, out_name, subdir in [
        ("surindersinghssj/gurbani-kirtan-yt-captions-eval-canonical",   "val_kirtan.jsonl",    "eval_kirtan"),
        ("surindersinghssj/gurbani-sehajpath-yt-captions-eval-canonical", "val_sehajpath.jsonl", "eval_sehajpath"),
    ]:
        out_path = manifests_dir / out_name
        with out_path.open("w", encoding="utf-8") as f:
            for wav, dur, text in materialize(hf_id, audio_root / subdir, leaked=set(), text_col="final_text"):
                f.write(json.dumps({"audio_filepath": wav, "duration": dur, "text": text},
                                   ensure_ascii=False) + "\n")
        print(f"[eval] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
