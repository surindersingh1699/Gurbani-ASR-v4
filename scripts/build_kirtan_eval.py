#!/usr/bin/env python3
"""
Build clean kirtan eval set with canonical SGGS labels.

Pipeline per source (matches kirtan_bulk_transcribe.py approach):
  1. Download audio via yt-dlp → 16kHz mono wav
  2. Optional trim to max_duration_sec (omit field to use full audio)
  3. Pre-split into fixed 20s clips (we control timestamps locally)
  4. Batch-transcribe via Gemini Flash Lite (15 clips/call ~= 5 min audio)
  5. Snap each transcription to canonical SGGS using FAISS tuk index
  6. CLI review (afplay): keep / skip / edit each segment
  7. Push as HF DatasetDict with one split per style (ragi/akj/sgpc)

Usage:
    # First run (download + transcribe + review + push):
    python scripts/build_kirtan_eval.py --manifest scripts/kirtan_eval_manifest.yaml

    # Re-review without re-paying API (uses cached transcriptions):
    python scripts/build_kirtan_eval.py --manifest ... --resume

    # Skip review, auto-keep canonical-snapped clips only:
    python scripts/build_kirtan_eval.py --manifest ... --skip-review

    # Dry run (no HF push):
    python scripts/build_kirtan_eval.py --manifest ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import faiss
import numpy as np
import soundfile as sf
import yaml

# Reuse the proven Gemini bulk pipeline from the existing kirtan script.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from kirtan_bulk_transcribe import init_gemini, split_audio, transcribe_all  # noqa: E402

SAMPLE_RATE = 16000
PROJECT_ROOT = SCRIPT_DIR.parent
TUK_INDEX_PATH = PROJECT_ROOT / "index" / "sggs_tuk.faiss"
TUKS_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "tuks.json"
MURIL_MODEL = "google/muril-base-cased"


# --------------------------------------------------------------------------- #
# Download / trim                                                             #
# --------------------------------------------------------------------------- #

def download_audio(url: str, dest_wav: Path) -> Path:
    """Download URL → 16kHz mono wav via yt-dlp. Caches the result."""
    if dest_wav.exists():
        print(f"[download] cached: {dest_wav.name}")
        return dest_wav

    dest_wav.parent.mkdir(parents=True, exist_ok=True)
    out_template = str(dest_wav.with_suffix(".%(ext)s"))

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "-ar 16000 -ac 1",
        "-o", out_template,
        url,
    ]
    print(f"[download] {url}")
    subprocess.run(cmd, check=True)

    if not dest_wav.exists():
        candidates = list(dest_wav.parent.glob(dest_wav.stem + "*.wav"))
        if not candidates:
            raise RuntimeError(f"yt-dlp did not produce a wav for {url}")
        candidates[0].rename(dest_wav)
    print(f"[download] wrote {dest_wav}")
    return dest_wav


def maybe_trim(src: Path, max_sec: float | None, dst: Path) -> Path:
    """If max_sec is set and shorter than audio, trim. Otherwise reuse src."""
    info = sf.info(str(src))
    if max_sec is None or info.duration <= max_sec + 0.5:
        if dst != src and not dst.exists():
            shutil.copy2(src, dst)
        elif dst == src:
            return src
        print(f"[trim] {src.name}: using full {info.duration:.1f}s ({info.duration / 60:.1f}min)")
        return dst

    audio, sr = sf.read(str(src))
    n = int(max_sec * sr)
    sf.write(str(dst), audio[:n], sr)
    print(f"[trim] {src.name}: {info.duration:.1f}s → {max_sec:.0f}s")
    return dst


# --------------------------------------------------------------------------- #
# Snap to canonical SGGS via FAISS tuk index                                  #
# --------------------------------------------------------------------------- #

class CanonicalSnapper:
    """Replace noisy Gemini transcriptions with exact SGGS tuk text when
    cosine similarity (MuRIL embedding, normalized) exceeds threshold.

    Loads metadata from data/processed/tuks.json (same source-of-truth that
    01_build_tuk_index.py used) — index position i corresponds to tuks[i].
    """

    def __init__(self, threshold: float = 0.85):
        if not TUK_INDEX_PATH.exists():
            raise RuntimeError(
                f"Missing FAISS index at {TUK_INDEX_PATH}. "
                "Run scripts/01_build_tuk_index.py first."
            )
        if not TUKS_JSON_PATH.exists():
            raise RuntimeError(f"Missing tuks json at {TUKS_JSON_PATH}.")

        from sentence_transformers import SentenceTransformer

        print(f"[snap] loading FAISS tuk index ({TUK_INDEX_PATH.name})")
        self.index = faiss.read_index(str(TUK_INDEX_PATH))
        with open(TUKS_JSON_PATH, encoding="utf-8") as f:
            self.tuks = json.load(f)
        if len(self.tuks) != self.index.ntotal:
            raise RuntimeError(
                f"tuks.json ({len(self.tuks)}) and FAISS index "
                f"({self.index.ntotal}) row counts disagree — rebuild index"
            )
        print(f"[snap] {self.index.ntotal:,} canonical tuks loaded")

        print(f"[snap] loading MuRIL ({MURIL_MODEL}) — first run downloads ~700MB")
        self.model = SentenceTransformer(MURIL_MODEL)
        self.threshold = threshold

    def snap(self, text: str) -> tuple[str, float, str]:
        """Return (canonical_text, similarity, source_tag).

        canonical_text is the nearest SGGS tuk text only when similarity
        crosses the threshold — otherwise empty string. Gemini text is
        preserved separately by the caller and is never discarded.

        source_tag is 'canonical' if we trust the snap, 'gemini' if we
        fell back to the raw Gemini text, 'empty' if Gemini returned nothing.
        """
        if not text or not text.strip():
            return "", 0.0, "empty"

        vec = self.model.encode([text], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(vec, k=1)
        sim = float(D[0][0])
        idx = int(I[0][0])
        canonical = self.tuks[idx]["text"]

        if sim >= self.threshold:
            return canonical, sim, "canonical"
        return "", sim, "gemini"


# --------------------------------------------------------------------------- #
# Per-source pipeline                                                          #
# --------------------------------------------------------------------------- #

def transcribe_source(
    *,
    source: dict,
    cache_dir: Path,
    settings: dict,
    snapper: CanonicalSnapper,
    resume: bool,
) -> list[dict]:
    """Run download -> trim -> split -> transcribe -> snap for one source.

    Caches intermediate artifacts under cache_dir/<style>/ so --resume
    can skip the Gemini API calls on subsequent runs.
    """
    style = source["style"]
    work_dir = cache_dir / style
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_wav = work_dir / "raw.wav"
    trimmed_wav = work_dir / "trimmed.wav"
    transcribe_cache = work_dir / "transcribed.json"

    download_audio(source["url"], raw_wav)
    audio_path = maybe_trim(raw_wav, source.get("max_duration_sec"), trimmed_wav)

    if resume and transcribe_cache.exists():
        print(f"[transcribe] cached: {transcribe_cache}")
        with open(transcribe_cache, encoding="utf-8") as f:
            cached = json.load(f)
        segments = []
        for s in cached:
            audio, _ = sf.read(str(work_dir / s["audio_path"]))
            segments.append({
                "audio_array": audio,
                "transcription": s["transcription"],
                "duration_sec": s["duration_sec"],
                "start_sec": s["start_sec"],
                "end_sec": s["end_sec"],
            })
    else:
        client = init_gemini()
        clips = split_audio(str(audio_path), settings["segment_secs"])
        if not clips:
            raise RuntimeError(f"No clips produced for {style}")
        gemini_segs = transcribe_all(client, clips, settings["batch_size"])

        clip_dir = work_dir / "clips"
        clip_dir.mkdir(exist_ok=True)
        cache_payload = []
        segments = []
        for i, s in enumerate(gemini_segs):
            clip_path = clip_dir / f"{i:03d}.wav"
            sf.write(str(clip_path), s["audio"]["array"], SAMPLE_RATE)
            cache_payload.append({
                "audio_path": str(clip_path.relative_to(work_dir)),
                "transcription": s["transcription"],
                "duration_sec": s["duration_sec"],
                "start_sec": s["start_sec"],
                "end_sec": s["end_sec"],
            })
            segments.append({
                "audio_array": s["audio"]["array"],
                "transcription": s["transcription"],
                "duration_sec": s["duration_sec"],
                "start_sec": s["start_sec"],
                "end_sec": s["end_sec"],
            })
        with open(transcribe_cache, "w", encoding="utf-8") as f:
            json.dump(cache_payload, f, ensure_ascii=False, indent=2)
        print(f"[transcribe] cached -> {transcribe_cache}")

    snapped = []
    for seg in segments:
        canonical, sim, source_tag = snapper.snap(seg["transcription"])
        snapped.append({
            **seg,
            # gemini_text is always preserved as its own column.
            # canonical_text is filled only when snap passed threshold.
            "gemini_text": seg["transcription"],
            "canonical_text": canonical,
            "snap_similarity": sim,
            "snap_source": source_tag,
            "style": style,
            "source_url": source["url"],
            "source_title": source["title"],
        })

    n_canon = sum(1 for s in snapped if s["snap_source"] == "canonical")
    n_gem = sum(1 for s in snapped if s["snap_source"] == "gemini")
    n_emp = sum(1 for s in snapped if s["snap_source"] == "empty")
    print(
        f"[snap] {style}: canonical={n_canon} gemini={n_gem} empty={n_emp} "
        f"(threshold={snapper.threshold:.2f})"
    )
    return snapped


# --------------------------------------------------------------------------- #
# Review CLI (Mac afplay)                                                      #
# --------------------------------------------------------------------------- #

def review_segments(segments: list[dict], cache_dir: Path) -> list[dict]:
    """Interactive review. For each segment: play audio, show texts, prompt.

    Commands:
      Enter / k = keep snapped/canonical text (or raw if no snap)
      s         = skip (drop from eval set)
      e         = edit text manually
      r         = replay
      g         = use raw Gemini text instead of snapped
      q         = quit (keep everything decided so far, drop the rest)
    """
    if not shutil.which("afplay"):
        print("[review] WARN: afplay not found (Mac-only). Audio playback disabled.")

        def play(_arr: np.ndarray) -> None:
            return None
    else:
        def play(arr: np.ndarray) -> None:
            tmp = cache_dir / "_review_tmp.wav"
            sf.write(str(tmp), arr, SAMPLE_RATE)
            subprocess.run(["afplay", str(tmp)], check=False)

    kept: list[dict] = []
    for i, seg in enumerate(segments):
        print("\n" + "=" * 80)
        print(
            f"[{i + 1}/{len(segments)}] {seg['style']} | "
            f"{seg['start_sec']:.0f}-{seg['end_sec']:.0f}s | "
            f"sim={seg['snap_similarity']:.3f} ({seg['snap_source']})"
        )
        print(f"  gemini:    {seg['gemini_text']}")
        print(f"  canonical: {seg['canonical_text'] or '(not snapped)'}")

        while True:
            play(seg["audio_array"])
            cmd = input(
                "  [Enter=keep / s=skip / e=edit / r=replay / g=use-gemini / q=quit] "
            ).strip().lower()

            if cmd in ("", "k"):
                final = (
                    seg["canonical_text"]
                    if seg["snap_source"] == "canonical"
                    else seg["raw_text"]
                )
                if not final.strip():
                    print("  → empty text, auto-skipping")
                    break
                kept.append({**seg, "final_text": final, "review_action": "kept"})
                break
            if cmd == "s":
                break
            if cmd == "g":
                if not seg["raw_text"].strip():
                    print("  → empty raw text, can't keep")
                    continue
                kept.append({**seg, "final_text": seg["raw_text"], "review_action": "kept_gemini"})
                break
            if cmd == "e":
                edited = input("  enter Gurmukhi text: ").strip()
                if not edited:
                    print("  → empty, treating as skip")
                    break
                kept.append({**seg, "final_text": edited, "review_action": "edited"})
                break
            if cmd == "r":
                continue
            if cmd == "q":
                print(f"[review] quitting, kept {len(kept)} so far")
                return kept
            print("  unknown command")

    return kept


def auto_keep_segments(segments: list[dict]) -> list[dict]:
    """No-review mode: keep only segments that snapped to canonical."""
    kept = [
        {**s, "final_text": s["canonical_text"], "review_action": "auto"}
        for s in segments
        if s["snap_source"] == "canonical"
    ]
    print(f"[auto] kept {len(kept)}/{len(segments)} (canonical-only)")
    return kept


# --------------------------------------------------------------------------- #
# HF push                                                                      #
# --------------------------------------------------------------------------- #

def push_to_hub(kept_by_style: dict[str, list[dict]], repo_id: str) -> None:
    from datasets import Audio, Dataset, DatasetDict

    splits = {}
    for style, segs in kept_by_style.items():
        if not segs:
            print(f"[push] {style}: no segments, skipping split")
            continue
        ds = Dataset.from_dict({
            "audio": [
                {"array": s["audio_array"], "sampling_rate": SAMPLE_RATE} for s in segs
            ],
            # `transcription` is the label to use for WER/CER — canonical SGGS
            # when snapped, otherwise Gemini or the reviewer's edit.
            "transcription": [s["final_text"] for s in segs],
            # `gemini_text` and `canonical_text` are ALWAYS both present so
            # the dataset preserves the full provenance. canonical_text is ""
            # when the clip did not snap to a canonical SGGS tuk.
            "gemini_text": [s["gemini_text"] for s in segs],
            "canonical_text": [s["canonical_text"] for s in segs],
            "duration_sec": [s["duration_sec"] for s in segs],
            "snap_similarity": [s["snap_similarity"] for s in segs],
            "snap_source": [s["snap_source"] for s in segs],
            "review_action": [s["review_action"] for s in segs],
            "source_url": [s["source_url"] for s in segs],
            "source_title": [s["source_title"] for s in segs],
            "style": [s["style"] for s in segs],
        })
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
        splits[style] = ds
        total_sec = sum(s["duration_sec"] for s in segs)
        print(f"[push] {style}: {len(segs)} segs, {total_sec / 60:.1f}min")

    if not splits:
        print("[push] no splits to push, aborting")
        return

    print(f"[push] uploading to {repo_id}")
    DatasetDict(splits).push_to_hub(repo_id)
    print(f"[push] done: https://huggingface.co/datasets/{repo_id}")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Build clean kirtan eval set")
    parser.add_argument("--manifest", required=True, help="Path to YAML manifest")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse cached transcriptions (skip Gemini API calls)")
    parser.add_argument("--skip-review", action="store_true",
                        help="Auto-keep canonical-snapped segments only (no CLI review)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats and skip HF push")
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    settings = manifest["settings"]
    cache_dir = Path(settings["cache_dir"]).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    snapper = CanonicalSnapper(threshold=settings["snap_threshold"])

    kept_by_style: dict[str, list[dict]] = {}
    for source in manifest["sources"]:
        style = source["style"]
        print("\n" + "#" * 80)
        print(f"# {style.upper()} — {source['title']}")
        print("#" * 80)

        snapped = transcribe_source(
            source=source,
            cache_dir=cache_dir,
            settings=settings,
            snapper=snapper,
            resume=args.resume,
        )
        kept = (
            auto_keep_segments(snapped) if args.skip_review
            else review_segments(snapped, cache_dir)
        )
        kept_by_style[style] = kept

    print("\n" + "=" * 80)
    print("REVIEW SUMMARY")
    print("=" * 80)
    grand_total_sec = 0.0
    for style, segs in kept_by_style.items():
        sec = sum(s["duration_sec"] for s in segs)
        grand_total_sec += sec
        print(f"  {style:8s}: {len(segs)} segments, {sec / 60:.1f} min")
    print(f"  TOTAL    : {grand_total_sec / 60:.1f} min")

    if args.dry_run:
        print("\n[dry-run] skipping HF push")
        return

    push_to_hub(kept_by_style, settings["out_dataset"])


if __name__ == "__main__":
    main()
