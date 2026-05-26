"""Build v2 NeMo manifests from all public canonical Gurbani datasets.

PLAN-v2.md M2.1 — diversity-filtered ~250h mix.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sttm_first_letter_map import (
    gurmukhi_text_to_training_anchor,
    training_anchor_to_search_anchor,
)
from scripts.build_indicconformer_manifests import (
    normalize_gurbani_text,
    passes_simran_filter,
)

SEED = 42
random.seed(SEED)


def _emit_row(audio_root: Path, ex, text, source_tag, kirtan_type=""):
    import soundfile as sf
    audio = ex["audio"]["array"]
    sr = ex["audio"]["sampling_rate"]
    duration = float(audio.shape[0]) / float(sr)
    if not 0.5 <= duration <= 18.0:
        return None
    anchor = gurmukhi_text_to_training_anchor(text)
    search = training_anchor_to_search_anchor(anchor)
    if len(search) < 3:
        return None
    if not 0.15 <= duration / max(len(search), 1) <= 1.5:
        return None
    clip_id = (ex.get("clip_id") or
               f"{source_tag}_{ex.get('video_id', 'x')}_{hash(text) % 10**9:09d}")
    flac_path = audio_root / source_tag / f"{clip_id}.flac"
    flac_path.parent.mkdir(parents=True, exist_ok=True)
    if not flac_path.exists():
        try:
            sf.write(str(flac_path), audio.astype("float32"), sr,
                     format="FLAC", subtype="PCM_16")
        except Exception:
            return None
    return {
        "audio_filepath": str(flac_path),
        "duration": duration,
        "text": anchor,
        "search_anchor": search,
        "source": source_tag,
        "kirtan_type": kirtan_type,
        "video_id": ex.get("video_id", ""),
        "speaker_id": ex.get("speaker_id", ""),
    }


def _stream_sehajpath_orig(audio_root: Path, max_hours: float):
    from datasets import load_dataset, Audio
    ds = load_dataset("surindersinghssj/gurbani-sehajpath", split="train",
                      streaming=True).cast_column("audio", Audio(sampling_rate=16000))
    print(f"[sehaj-orig] streaming, target {max_hours}h", flush=True)
    secs = 0
    seen = kept = 0
    for ex in ds:
        seen += 1
        text = normalize_gurbani_text(ex.get("gurmukhi_text") or ex.get("transcription") or "")
        if not text or not passes_simran_filter(text):
            continue
        row = _emit_row(audio_root, ex, text, "sehaj_orig")
        if not row:
            continue
        kept += 1
        secs += row["duration"]
        yield row
        if secs >= max_hours * 3600:
            break
        if seen % 5000 == 0:
            print(f"  [sehaj-orig] seen={seen} kept={kept} hours={secs/3600:.1f}", flush=True)
    print(f"[sehaj-orig] DONE seen={seen} kept={kept} hours={secs/3600:.1f}", flush=True)


def _stream_canonical(audio_root: Path, hf_id: str, source_tag: str,
                      max_hours: float, max_hours_per_video: float,
                      max_hours_per_kirtan_type=None,
                      min_match_score: float = 0.70):
    from datasets import load_dataset, Audio
    ds = load_dataset(hf_id, split="train",
                      streaming=True).cast_column("audio", Audio(sampling_rate=16000))
    print(f"[{source_tag}] streaming, target {max_hours}h, cap/video {max_hours_per_video}h", flush=True)
    per_video_secs = defaultdict(float)
    per_kt_secs = defaultdict(float)
    total_secs = 0
    seen = kept = dropped_score = dropped_video = dropped_kt = dropped_text = 0
    for ex in ds:
        seen += 1
        if ex.get("is_simran", False):
            dropped_text += 1; continue
        score = ex.get("canonical_match_score", ex.get("match_score", 1.0))
        if score is not None and score < min_match_score:
            dropped_score += 1; continue
        if ex.get("training_usable") is False:
            dropped_score += 1; continue
        text = normalize_gurbani_text(
            ex.get("final_text") or ex.get("stage2_reviewer_final") or
            ex.get("text") or ex.get("gurmukhi_text") or "")
        if not text or not passes_simran_filter(text):
            dropped_text += 1; continue
        vid = ex.get("video_id", "")
        if vid and per_video_secs[vid] >= max_hours_per_video * 3600:
            dropped_video += 1; continue
        kt = ex.get("kirtan_type", "")
        if max_hours_per_kirtan_type and kt and \
                per_kt_secs[kt] >= max_hours_per_kirtan_type.get(kt, max_hours) * 3600:
            dropped_kt += 1; continue
        row = _emit_row(audio_root, ex, text, source_tag, kirtan_type=kt)
        if not row:
            continue
        kept += 1
        total_secs += row["duration"]
        per_video_secs[vid] += row["duration"]
        if kt:
            per_kt_secs[kt] += row["duration"]
        yield row
        if total_secs >= max_hours * 3600:
            break
        if seen % 5000 == 0:
            print(f"  [{source_tag}] seen={seen} kept={kept} hours={total_secs/3600:.1f} "
                  f"videos={len(per_video_secs)}", flush=True)
    print(f"[{source_tag}] DONE seen={seen} kept={kept} hours={total_secs/3600:.1f} "
          f"dropped: score={dropped_score} video={dropped_video} kt={dropped_kt} text={dropped_text}",
          flush=True)
    if per_kt_secs:
        print(f"  kirtan_type breakdown: " +
              " ".join(f"{k}={v/3600:.1f}h" for k, v in per_kt_secs.items()), flush=True)


def _build_test_manifest(audio_root: Path, manifests_dir: Path, hf_id: str, name: str,
                         kirtan_type_filter=None, max_clips: int = 500):
    from datasets import load_dataset, Audio
    print(f"[test:{name}] streaming from {hf_id}", flush=True)
    ds = load_dataset(hf_id, split="train",
                      streaming=True).cast_column("audio", Audio(sampling_rate=16000))
    out_path = manifests_dir / f"v2_anchor_val_{name}.jsonl"
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            if n >= max_clips:
                break
            if kirtan_type_filter and ex.get("kirtan_type", "") != kirtan_type_filter:
                continue
            text = normalize_gurbani_text(
                ex.get("final_text") or ex.get("stage2_reviewer_final") or
                ex.get("text") or ex.get("gurmukhi_text") or ex.get("transcription") or "")
            if not text:
                continue
            row = _emit_row(audio_root, ex, text, f"test_{name}",
                            kirtan_type=ex.get("kirtan_type", ""))
            if not row:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"[test:{name}] wrote {n} rows -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspace/data")
    ap.add_argument("--audio-root", default="/workspace/data/audio_v2")
    ap.add_argument("--sehaj-orig-hours", type=float, default=30)
    ap.add_argument("--sehaj-yt-hours", type=float, default=30)
    ap.add_argument("--sehaj-yt-per-video-hours", type=float, default=1.5)
    ap.add_argument("--kirtan-total-hours", type=float, default=180)
    ap.add_argument("--kirtan-per-video-hours", type=float, default=0.5)
    args = ap.parse_args()

    audio_root = Path(args.audio_root)
    manifests_dir = Path(args.data_root) / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    train_path = manifests_dir / "v2_anchor_train.jsonl"
    n_total = 0
    rows = []

    print("=== TRAIN ===", flush=True)
    t0 = time.time()

    for row in _stream_sehajpath_orig(audio_root, args.sehaj_orig_hours):
        rows.append(row); n_total += 1
    print(f"  cum total {n_total} rows after sehaj-orig ({(time.time()-t0)/60:.1f}m)", flush=True)

    for row in _stream_canonical(
            audio_root,
            "surindersinghssj/gurbani-sehajpath-yt-captions-canonical",
            "sehaj_yt", max_hours=args.sehaj_yt_hours,
            max_hours_per_video=args.sehaj_yt_per_video_hours):
        rows.append(row); n_total += 1
    print(f"  cum total {n_total} rows after sehaj-yt ({(time.time()-t0)/60:.1f}m)", flush=True)

    kirtan_kt_caps = {"ragi": 80.0, "akj": 60.0, "sgpc": 50.0}
    for row in _stream_canonical(
            audio_root,
            "surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical",
            "kirtan_yt", max_hours=args.kirtan_total_hours,
            max_hours_per_video=args.kirtan_per_video_hours,
            max_hours_per_kirtan_type=kirtan_kt_caps):
        rows.append(row); n_total += 1
    print(f"  cum total {n_total} rows after kirtan ({(time.time()-t0)/60:.1f}m)", flush=True)

    random.shuffle(rows)
    with train_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[train] wrote {n_total} rows -> {train_path}", flush=True)

    print("=== VAL/TEST MANIFESTS ===", flush=True)
    _build_test_manifest(audio_root, manifests_dir,
                "surindersinghssj/gurbani-sehajpath",
                "seen_sehaj", max_clips=300)
    _build_test_manifest(audio_root, manifests_dir,
                "surindersinghssj/gurbani-sehajpath-yt-captions-eval-canonical",
                "unseen_sehaj", max_clips=400)
    _build_test_manifest(audio_root, manifests_dir,
                "surindersinghssj/gurbani-kirtan-yt-captions-eval-canonical",
                "ragi", kirtan_type_filter="ragi", max_clips=300)
    _build_test_manifest(audio_root, manifests_dir,
                "surindersinghssj/gurbani-kirtan-yt-captions-eval-canonical",
                "akj", kirtan_type_filter="akj", max_clips=300)
    _build_test_manifest(audio_root, manifests_dir,
                "surindersinghssj/gurbani-kirtan-eval-pure-canonical",
                "sgpc", max_clips=400)

    chars = set()
    for r in rows:
        chars.update(c for c in r["text"] if c != "|")
    vocab_path = manifests_dir / "v2_anchor_vocab.json"
    vocab_path.write_text(json.dumps({
        "delim": "|", "chars": sorted(chars),
        "vocab_size_with_delim_and_blank": len(chars) + 2,
        "n_train_rows": n_total,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[vocab] {len(chars)} unique chars -> {vocab_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
