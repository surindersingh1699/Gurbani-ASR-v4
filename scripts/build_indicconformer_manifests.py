"""Materialize NeMo-style JSONL manifests for IndicConformer-pa fine-tune.

Reads HF datasets for the *labels and clip_ids*, then resolves each clip to
an existing FLAC (or WAV) on the network volume — no audio re-encoding.
Drops video_ids that leak into eval, applies the v3 simran/repeat filter
(REPEAT_THRESHOLD=10), and writes a NeMo manifest:

    {"audio_filepath": "/workspace/data/audio/.../<clip_id>.flac",
     "duration": 7.34,
     "text": "ਵਾਹਿਗੁਰੂ ..."}

NeMo's data loader uses soundfile/torchaudio, both of which decode FLAC at
load time. Training reads FLAC directly — saves disk + decode wall-clock.

Layout assumption (configurable via --audio-root-* flags):
    /workspace/data/audio/kirtan/<video_id>/<clip_id>.flac
    /workspace/data/audio/sehajpath_yt/<video_id>/<clip_id>.flac
    /workspace/data/audio/sehajpath_orig/<clip_id>.flac

If the layout is different, override per-source with --audio-root-<source>
or --layout {flat|by_video}.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from surt.data import normalize_gurbani_text  # noqa: E402

REPEAT_THRESHOLD = 10  # v3 pipeline: drop clips with >10 consecutive repeats of same word


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


def resolve_audio_path(audio_root: Path, clip_id: str, video_id: str, layout: str) -> Path | None:
    """Find the existing FLAC/WAV file for this clip on the network volume.

    Tries (in order): .flac then .wav, with both flat and by-video layouts.
    Returns the first existing path or None.
    """
    candidates: list[Path] = []
    if layout in ("by_video", "auto"):
        for ext in (".flac", ".wav"):
            candidates.append(audio_root / video_id / f"{clip_id}{ext}")
    if layout in ("flat", "auto"):
        for ext in (".flac", ".wav"):
            candidates.append(audio_root / f"{clip_id}{ext}")
    for c in candidates:
        if c.exists():
            return c
    return None


def materialize(
    name: str,
    audio_root: Path,
    leaked: set[str],
    layout: str = "auto",
    text_col: str = "final_text",
    duration_col: str = "duration_s",
):
    """Yield (audio_path, duration, text) for each kept clip."""
    from datasets import load_dataset

    print(f"[{name}] loading metadata (no audio decode) ...")
    # NB: do NOT cast `audio` column — that triggers a decode pass we don't need.
    ds = load_dataset(name, split="train")

    kept = dropped_leak = dropped_simran = dropped_empty = dropped_missing = 0
    missing_examples: list[str] = []
    for ex in ds:
        vid = ex.get("video_id", "")
        if vid in leaked:
            dropped_leak += 1; continue
        text = (ex.get(text_col) or "").strip()
        text = normalize_gurbani_text(text)
        if not text:
            dropped_empty += 1; continue
        if not passes_simran_filter(text):
            dropped_simran += 1; continue
        clip_id = ex.get("clip_id") or f"{vid}_{kept:06d}"
        audio_path = resolve_audio_path(audio_root, clip_id, vid, layout)
        if audio_path is None:
            dropped_missing += 1
            if len(missing_examples) < 3:
                missing_examples.append(clip_id)
            continue
        # Use duration_s from the dataset if present (cheaper than reading the audio)
        duration = float(ex.get(duration_col) or 0.0)
        if duration <= 0:
            import soundfile as sf
            with sf.SoundFile(str(audio_path)) as f:
                duration = len(f) / f.samplerate
        yield str(audio_path), duration, text
        kept += 1
    print(f"[{name}] kept={kept}  dropped: leak={dropped_leak} simran={dropped_simran} "
          f"empty={dropped_empty} missing_audio={dropped_missing}")
    if missing_examples:
        print(f"  [warn] examples of missing audio (first 3): {missing_examples}")
        print(f"  [warn] expected layout: {audio_root}/<video_id>/<clip_id>.flac")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspace/data",
                    help="Root for manifests output. Audio is read from --audio-root-* below.")
    ap.add_argument("--audio-root-kirtan", default="/workspace/data/audio/kirtan",
                    help="Where kirtan FLAC clips live")
    ap.add_argument("--audio-root-sehajpath-yt", default="/workspace/data/audio/sehajpath_yt",
                    help="Where sehajpath YT-captions FLAC clips live")
    ap.add_argument("--audio-root-sehajpath-orig", default="/workspace/data/audio/sehajpath_orig",
                    help="Where original sehajpath FLAC clips live")
    ap.add_argument("--audio-root-eval-kirtan", default="/workspace/data/audio/eval_kirtan",
                    help="Where eval kirtan FLAC clips live")
    ap.add_argument("--audio-root-eval-sehajpath", default="/workspace/data/audio/eval_sehajpath",
                    help="Where eval sehajpath FLAC clips live")
    ap.add_argument("--layout", default="auto", choices=["auto", "by_video", "flat"],
                    help="Filesystem layout: by_video = <root>/<video_id>/<clip_id>.flac, "
                         "flat = <root>/<clip_id>.flac, auto = try both")
    ap.add_argument("--leaked-file", default="bench_results/train_dropped_video_ids.txt")
    args = ap.parse_args()

    leaked: set[str] = set()
    if Path(args.leaked_file).exists():
        leaked = set(Path(args.leaked_file).read_text().strip().splitlines())
    print(f"[leak-list] dropping {len(leaked)} video_ids from training")

    manifests_dir = Path(args.data_root) / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # --- train sources, sampling weights expressed as integer repeats ---
    sources = [
        # (HF id, text col, audio root, repeats — repeats give us 5:3:2 mix ≈ 50/30/20%)
        ("surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical",
            "final_text", Path(args.audio_root_kirtan), 5),
        ("surindersinghssj/gurbani-sehajpath-yt-captions-canonical",
            "final_text", Path(args.audio_root_sehajpath_yt), 3),
        ("surindersinghssj/gurbani-sehajpath",
            "transcription", Path(args.audio_root_sehajpath_orig), 2),
    ]

    train_path = manifests_dir / "train.jsonl"
    n_total = 0
    with train_path.open("w", encoding="utf-8") as f:
        for hf_id, text_col, audio_root, repeats in sources:
            entries = list(materialize(hf_id, audio_root, leaked, args.layout, text_col=text_col))
            for wav, dur, text in entries * repeats:
                f.write(json.dumps({"audio_filepath": wav, "duration": dur, "text": text},
                                   ensure_ascii=False) + "\n")
                n_total += 1
            print(f"  → {hf_id}: {len(entries)} unique × {repeats} repeats")
    print(f"[train] {n_total} total entries (with sampling weights) → {train_path}")

    # --- eval (no leak filter, no simran filter — eval as-is) ---
    for hf_id, audio_root, out_name in [
        ("surindersinghssj/gurbani-kirtan-yt-captions-eval-canonical",
            Path(args.audio_root_eval_kirtan), "val_kirtan.jsonl"),
        ("surindersinghssj/gurbani-sehajpath-yt-captions-eval-canonical",
            Path(args.audio_root_eval_sehajpath), "val_sehajpath.jsonl"),
    ]:
        out_path = manifests_dir / out_name
        with out_path.open("w", encoding="utf-8") as f:
            for wav, dur, text in materialize(
                hf_id, audio_root, leaked=set(), layout=args.layout, text_col="final_text"
            ):
                f.write(json.dumps({"audio_filepath": wav, "duration": dur, "text": text},
                                   ensure_ascii=False) + "\n")
        print(f"[eval] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
