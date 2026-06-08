"""Parallel parquet→FLAC decoder + manifest builder for IndicConformer fine-tune.

Same output shape as build_indicconformer_manifests.py but:
  - Uses ProcessPoolExecutor with N workers (default = cpu_count - 1)
  - Each worker decodes its slice of the dataset to FLAC files
  - Idempotent: skips FLAC files that already exist on disk
  - Main thread aggregates results into a single JSONL manifest

Designed to saturate the pod's 8-9 vCPU during decode (vs 1 in the sequential version).
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPEAT_THRESHOLD = 10
# ~1/VAL_HOLDOUT_MOD of kirtan VIDEOS are reserved (video-level, no clip leak)
# as a diverse held-out val_kirtan — the caption eval is only 1 video, too thin
# to early-stop / select checkpoints on. bani is never held out (eval is
# kirtan-only). Deterministic by video_id hash so it's reproducible.
VAL_HOLDOUT_MOD = 25


def is_val_video(video_id: str) -> bool:
    h = int(hashlib.md5((video_id or "").encode()).hexdigest(), 16)
    return h % VAL_HOLDOUT_MOD == 0


def subsample_to_hours(entries, target_h):
    """Cap a source to ~target_h by keeping an EVEN portion of each video's
    clips (not all) — every video stays represented, clips evenly spaced across
    its timeline, so voice/raga/recording diversity is preserved while hours drop.
    entries: list of (wav, dur, text, vid). Returns the kept subset."""
    from collections import defaultdict
    total_h = sum(e[1] for e in entries) / 3600.0
    if total_h <= target_h or total_h == 0:
        return entries
    frac = target_h / total_h
    by_vid = defaultdict(list)
    for e in entries:
        by_vid[e[3]].append(e)
    kept = []
    for vid, clips in by_vid.items():
        # Bresenham-style even pick: keep clip i iff floor((i+1)*frac) > floor(i*frac)
        for i, e in enumerate(clips):
            if int((i + 1) * frac) > int(i * frac):
                kept.append(e)
    kept_h = sum(e[1] for e in kept) / 3600.0
    print(f"  [subsample] {len(entries)}→{len(kept)} clips, {total_h:.1f}h→{kept_h:.1f}h "
          f"(target {target_h}h, frac {frac:.3f}, {len(by_vid)} videos)", flush=True)
    return kept


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


def _worker_decode_shard(args):
    """Decode one shard of the dataset. Runs in a child process."""
    name, text_col, audio_outdir, leaked, shard_idx, num_shards, quality_filter = args
    import soundfile as sf
    from datasets import load_dataset, Audio
    audio_outdir = Path(audio_outdir)
    audio_outdir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(name, split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    n = len(ds)
    # Slice this shard
    start = shard_idx * n // num_shards
    end = (shard_idx + 1) * n // num_shards
    sub = ds.select(range(start, end))

    entries = []
    kept = dropped = decoded = 0
    for i, ex in enumerate(sub):
        vid = ex.get("video_id", "")
        if vid in leaked:
            dropped += 1; continue
        # quality gate: kirtan sources pass quality_filter='high' (skeleton-clean
        # verbatim only); bani passes quality_filter=None (full pass-through).
        if quality_filter is not None and (ex.get("quality") or "") != quality_filter:
            dropped += 1; continue
        text = (ex.get(text_col) or ex.get("transcription") or ex.get("text") or "").strip()
        text = normalize_gurbani_text(text)
        if not text or not passes_simran_filter(text):
            dropped += 1; continue

        clip_id = ex.get("clip_id") or f"{vid}_{start+i:08d}"
        flac_path = audio_outdir / f"{clip_id}.flac"
        audio = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        if not flac_path.exists():
            try:
                sf.write(str(flac_path), audio.astype("float32"), sr, format="FLAC", subtype="PCM_16")
                decoded += 1
            except Exception as e:
                print(f"  [shard {shard_idx}] decode fail {clip_id}: {e}", flush=True)
                dropped += 1; continue
        duration = float(audio.shape[0]) / float(sr)
        entries.append((str(flac_path), duration, text, vid))
        kept += 1
    return entries, kept, dropped, decoded, shard_idx


def materialize_parallel(name: str, audio_outdir: Path, leaked: set[str],
                          text_col: str, num_workers: int,
                          quality_filter: str | None = None):
    """Decode the whole dataset in parallel, return list of manifest entries."""
    from datasets import load_dataset
    print(f"[{name}] sizing dataset (quality_filter={quality_filter}) ...", flush=True)
    ds = load_dataset(name, split="train")
    n = len(ds)
    print(f"[{name}] {n} rows; sharding across {num_workers} workers", flush=True)
    del ds  # free metadata before forking

    args_list = [
        (name, text_col, str(audio_outdir), leaked, i, num_workers, quality_filter)
        for i in range(num_workers)
    ]
    all_entries = []
    total_kept = total_dropped = total_decoded = 0
    t0 = time.time()
    last_log = t0

    ctx = mp.get_context("spawn")  # cleaner with HF datasets / soundfile
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
        futures = {ex.submit(_worker_decode_shard, a): a[4] for a in args_list}
        done = 0
        for fut in as_completed(futures):
            shard_idx = futures[fut]
            try:
                entries, kept, dropped, decoded, _ = fut.result()
                all_entries.extend(entries)
                total_kept += kept
                total_dropped += dropped
                total_decoded += decoded
                done += 1
                elapsed = time.time() - t0
                rate = total_kept / max(elapsed, 1)
                print(f"  [{name}] shard {shard_idx} done — kept {kept} (decoded {decoded}). "
                      f"Total kept={total_kept}, decoded={total_decoded}. "
                      f"Workers done {done}/{num_workers}. Rate={rate:.0f} clips/s",
                      flush=True)
            except Exception as e:
                print(f"  [{name}] shard {shard_idx} FAILED: {e}", flush=True)
    print(f"[{name}] FINAL: kept={total_kept} decoded={total_decoded} "
          f"dropped={total_dropped} in {time.time()-t0:.1f}s", flush=True)
    return all_entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspace/data")
    ap.add_argument("--audio-root", default="/workspace/data/audio")
    ap.add_argument("--leaked-file", default="/workspace/runs/latest/train_dropped_video_ids.txt")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    leaked: set[str] = set()
    if Path(args.leaked_file).exists():
        leaked = set(Path(args.leaked_file).read_text().strip().splitlines())
    print(f"[leak-list] dropping {len(leaked)} video_ids from training", flush=True)
    print(f"[parallel] using {args.workers} workers (cpu_count={os.cpu_count()})", flush=True)

    manifests_dir = Path(args.data_root) / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    audio_root = Path(args.audio_root)

    # v4-clean training pool (set by /goal): kirtan = HIGH-only skeleton-clean
    # verbatim from the 3 cleanv2 sources (×2 to stay dominant); bani-v4 = FULL
    # pass-through (text is clean continuous gurbani, quality gate would wrongly
    # drop 83% of valid Sukhmani-style spans). Text column is `text` on cleanv2.
    # bani is capped to ~50h via even per-video subsample (keep a PORTION of each
    # video's clips, not all) so all 174 bani videos stay represented but bani is a
    # light clean regularizer, not 318h. kirtan = no cap (target_hours=None).
    #   (repo, text_col, subdir, repeats, quality_filter, target_hours)
    sources = [
        ("surindersinghssj/gurbani-kirtan-yt-captions-300h-cleanv2", "text", "kirtan_300h",  2, "high", None),
        ("surindersinghssj/gurbani-kirtan-v4-sgpc-cleanv2",          "text", "kirtan_sgpc",  2, "high", None),
        ("surindersinghssj/gurbani-kirtan-v4-1000h-cleanv2",         "text", "kirtan_1000h", 2, "high", None),
        ("surindersinghssj/gurbani-bani-v4-cleanv2",                 "text", "bani_v4",      1, None,   50),
    ]

    def jline(wav, dur, text):
        # IndicConformer's aggregate (multilingual) tokenizer requires `lang` per entry
        return json.dumps({"audio_filepath": wav, "duration": dur, "text": text, "lang": "pa"},
                          ensure_ascii=False) + "\n"

    from collections import Counter
    VAL_CLIPS_PER_VIDEO = 40            # cap so val stays ~1-2k clips (diverse, not huge)
    val_vid_counts: Counter = Counter()

    train_path = manifests_dir / "train.jsonl"
    heldout_path = manifests_dir / "val_kirtan.jsonl"   # PRIMARY val = diverse held-out kirtan
    n_total = n_val = 0
    with train_path.open("w", encoding="utf-8") as f, heldout_path.open("w", encoding="utf-8") as fv:
        for hf_id, text_col, subdir, repeats, quality_filter, target_hours in sources:
            entries = materialize_parallel(hf_id, audio_root / subdir, leaked, text_col,
                                           args.workers, quality_filter=quality_filter)
            if target_hours:
                entries = subsample_to_hours(entries, target_hours)
            is_kirtan = quality_filter == "high"   # kirtan sources only; bani has quality_filter=None
            tr = vl = 0
            for wav, dur, text, vid in entries:
                if is_kirtan and is_val_video(vid):
                    # held-out val video; cap clips/video so val stays small + diverse.
                    # over-cap held-out clips are simply dropped (not trained on — no leak).
                    if val_vid_counts[vid] < VAL_CLIPS_PER_VIDEO:
                        fv.write(jline(wav, dur, text)); n_val += 1; vl += 1
                        val_vid_counts[vid] += 1
                else:
                    for _ in range(repeats):
                        f.write(jline(wav, dur, text)); n_total += 1
                    tr += 1
            print(f"  [train] → {hf_id}: {tr} train-videos×{repeats} + {vl} held-out-val clips", flush=True)
    print(f"[train] {n_total} train entries → {train_path}", flush=True)
    print(f"[val]   {n_val} held-out kirtan clips → {heldout_path}", flush=True)

    # Secondary references: the designated caption evals (kirtan = 1 video, sehaj = 2).
    for hf_id, out_name, subdir in [
        ("surindersinghssj/gurbani-kirtan-yt-captions-eval-canonical",   "val_kirtan_caption.jsonl", "eval_kirtan"),
        ("surindersinghssj/gurbani-sehajpath-yt-captions-eval-canonical", "val_sehajpath.jsonl",      "eval_sehajpath"),
    ]:
        out_path = manifests_dir / out_name
        with out_path.open("w", encoding="utf-8") as f:
            entries = materialize_parallel(hf_id, audio_root / subdir, leaked=set(),
                                           text_col="final_text", num_workers=args.workers)
            for wav, dur, text, vid in entries:
                f.write(jline(wav, dur, text))
        print(f"[eval] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
