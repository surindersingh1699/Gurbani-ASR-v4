#!/usr/bin/env python3
"""v3 pilot orchestrator: download → split → transcribe → push.

Reads a unified manifest CSV (from v3_enumerate_sikhnet.py or
v3_enumerate_youtube.py), selects enough rows to hit target_hours per
kirtan_type, then runs three parallel stages:

  1. Download        (concurrent HTTP / yt-dlp)    → data/raw_audio/<type>/<id>.wav
  2. Split into 20s  (multiprocessing)              → data/clips/<id>/clip_{i:05d}.wav
  3. Transcribe      (concurrent Gemini, 5-min batches)
       → data/gemini_raw/<id>/batch_{k:04d}.json
       → data/transcripts/<id>.jsonl

All stages are idempotent: existing outputs cause the stage to skip that item.

Each row keeps all enumerator columns plus a stable `item_id` used for all
output paths:  sikhnet-<track_id> | yt-<video_id>
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

SAMPLE_RATE = 16000
CLIP_SEC = 20.0
BATCH_CLIPS = 15     # 15 × 20s = 5-min batches per Gemini call
GEMINI_MODEL = "gemini-2.5-flash-lite"
UA = "Mozilla/5.0"

SYSTEM_INSTRUCTION = (
    "You are a Gurbani/Punjabi transcription expert specialized in Sikh kirtan.\n\n"
    "Rules you must follow on every clip, in priority order:\n"
    "1. Output MUST be in Gurmukhi script only. Never output Roman letters, "
    "Devanagari, or any other script.\n"
    "2. Transcribe word by word, with correct matras. Include EVERY repetition — "
    "if a line is sung three times, write it three times. Do not deduplicate.\n"
    "3. If the clip ends mid-word or mid-phrase, stop exactly where the audio "
    "stops. Do NOT complete the phrase from memory or from knowing the shabad.\n"
    "4. If the clip starts mid-word, start from the first word you actually hear.\n"
    '5. Return "" (empty string) if no Gurbani is being sung in the clip — for '
    "any reason (instrumental, wordless alaap/humming, spoken katha, speech, "
    "silence, etc.). Do NOT fabricate Gurbani when none is sung."
)


def _slug_from_url(url: str, max_len: int = 80) -> str:
    """Stable filesystem-safe id from an MP3 URL's filename stem."""
    import re
    stem = url.rsplit("/", 1)[-1].removesuffix(".mp3")
    # keep alnum, dash, underscore; collapse everything else to underscore
    clean = re.sub(r"[^A-Za-z0-9\-_]+", "_", stem).strip("_")
    return clean[:max_len] or "track"


def item_id(row: dict) -> str:
    if row["source"] == "sikhnet":
        return f"sikhnet-{row['sikhnet_track_id']}"
    if row["source"] == "youtube":
        return f"yt-{row['youtube_video_id']}"
    if row["source"] == "akj":
        return f"akj-{_slug_from_url(row['url'])}"
    if row["source"] == "sgpc":
        return f"sgpc-{_slug_from_url(row['url'])}"
    raise ValueError(f"unknown source: {row['source']}")


def select_rows(rows: list[dict], target_hours: float) -> list[dict]:
    """Take rows in order until est_secs sums to target_hours."""
    target = target_hours * 3600
    total = 0.0
    out = []
    for r in rows:
        est = float(r.get("est_secs") or 0)
        if est <= 0:
            continue
        out.append(r)
        total += est
        if total >= target:
            break
    return out


def download_one(row: dict, data_root: Path) -> Path | None:
    iid = item_id(row)
    ktype = row["kirtan_type"]
    out = data_root / "raw_audio" / ktype / f"{iid}.wav"
    if out.exists() and out.stat().st_size > 100_000:
        return out

    out.parent.mkdir(parents=True, exist_ok=True)

    def _http_mp3_then_ffmpeg(use_curl_cffi: bool) -> Path | None:
        """Download MP3 then convert to 16kHz mono WAV. curl_cffi for Cloudflare."""
        tmp_mp3 = out.with_suffix(".mp3.part")
        try:
            if use_curl_cffi:
                from curl_cffi import requests as cf_req
                with cf_req.Session(impersonate="chrome124") as s:
                    with s.get(row["url"], stream=True, timeout=120) as r:
                        r.raise_for_status()
                        with open(tmp_mp3, "wb") as f:
                            for chunk in r.iter_content(256 * 1024):
                                f.write(chunk)
            else:
                with requests.get(row["url"], headers={"User-Agent": UA},
                                  stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(tmp_mp3, "wb") as f:
                        for chunk in r.iter_content(256 * 1024):
                            f.write(chunk)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_mp3),
                 "-ar", str(SAMPLE_RATE), "-ac", "1", str(out)],
                check=True,
            )
            tmp_mp3.unlink(missing_ok=True)
            return out
        except Exception as e:
            print(f"[dl-fail] {iid}: {str(e)[:120]}", file=sys.stderr)
            return None

    if row["source"] == "sikhnet":
        return _http_mp3_then_ffmpeg(use_curl_cffi=False)
    if row["source"] == "akj":
        return _http_mp3_then_ffmpeg(use_curl_cffi=False)
    if row["source"] == "sgpc":
        return _http_mp3_then_ffmpeg(use_curl_cffi=True)

    if row["source"] == "youtube":
        try:
            subprocess.run(
                ["yt-dlp", "-x", "--audio-format", "wav",
                 "--postprocessor-args", f"-ar {SAMPLE_RATE} -ac 1",
                 "-o", str(out.with_suffix("")) + ".%(ext)s",
                 "--no-progress", "--quiet", "--retries", "5",
                 row["url"]],
                check=True,
            )
            return out
        except Exception as e:
            print(f"[dl-fail] {iid}: {str(e)[:120]}", file=sys.stderr)
            return None


def split_one(args):
    wav_path, clips_dir, max_secs = args
    wav_path = Path(wav_path)
    clips_dir = Path(clips_dir)
    if clips_dir.exists() and any(clips_dir.iterdir()):
        return str(clips_dir), sum(1 for _ in clips_dir.glob("clip_*.wav"))

    clips_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(str(wav_path))
    if sr != SAMPLE_RATE:
        # shouldn't happen — we downsampled at download — but guard anyway
        import librosa  # type: ignore
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # Per-item length cap: truncate audio before splitting. Long kirtan tracks
    # repeat refrains; first ~25 min has the distinct content.
    if max_secs and len(audio) > max_secs * sr:
        audio = audio[: int(max_secs * sr)]

    seg_samples = int(CLIP_SEC * sr)
    i = 0
    offset = 0
    while offset < len(audio):
        end = min(offset + seg_samples, len(audio))
        clip = audio[offset:end]
        if len(clip) / sr < 5.0:  # skip trailing short clip
            break
        sf.write(
            str(clips_dir / f"clip_{i:05d}.wav"),
            clip.astype(np.float32), sr, subtype="PCM_16",
        )
        i += 1
        offset = end
    return str(clips_dir), i


def gemini_batch(client, batch_paths: list[Path], start_sec: int,
                 end_sec: int, artist_name: str = "",
                 kirtan_type: str = "") -> list[str]:
    """Send one batch of clips, return len(batch_paths) text strings."""
    from google.genai import types

    n = len(batch_paths)
    artist_line = ""
    if artist_name:
        artist_line = (
            f"This recording is by {artist_name}, who sings Gurbani kirtan "
            f"({kirtan_type or 'traditional'} style). Expect real shabads from "
            "SGGS, Dasam Granth, or Bhai Gurdas.\n\n"
        )
    user_prompt = (
        f"{artist_line}"
        f"I am sending {n} audio clips (each ~20 seconds). Transcribe each "
        f"according to the rules in your system instructions.\n\n"
        f"Return a JSON array of exactly {n} strings, one per clip in order. "
        f"Each string is the Gurmukhi transcription (or \"\" if no Gurbani "
        f"is being sung in that clip)."
    )
    contents = [types.Part.from_text(text=user_prompt)]
    for i, p in enumerate(batch_paths):
        contents.append(types.Part.from_text(text=f"Clip {i + 1}:"))
        contents.append(types.Part.from_bytes(
            data=p.read_bytes(), mime_type="audio/wav"))

    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "array",
                        "items": {"type": "string"},
                    },
                ),
            )
            raw = resp.text.strip()
            texts = json.loads(raw)
            if isinstance(texts, dict):
                texts = list(texts.values())
            if not isinstance(texts, list):
                return [""] * n
            if len(texts) < n:
                texts += [""] * (n - len(texts))
            elif len(texts) > n:
                texts = texts[:n]
            return [t if isinstance(t, str) else "" for t in texts]
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 2 ** attempt
                print(f"    [rate-limit] waiting {wait}s ({start_sec}-{end_sec}s)",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            if attempt == 3:
                print(f"    [err] batch {start_sec}-{end_sec}: {err[:120]}",
                      file=sys.stderr)
                return [""] * n
            time.sleep(1)
    return [""] * n


def transcribe_one(item_id: str, clips_dir: Path, data_root: Path,
                   batch_workers: int, row: dict):
    """Transcribe all batches for one item. Idempotent via raw JSON checkpoints."""
    from google import genai

    jsonl_path = data_root / "transcripts" / f"{item_id}.jsonl"
    if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
        # already done
        return sum(1 for _ in open(jsonl_path))
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    raw_dir = data_root / "gemini_raw" / item_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(clips_dir.glob("clip_*.wav"))
    if not clips:
        return 0

    # build batches
    batches: list[tuple[int, list[Path]]] = []
    for b in range(0, len(clips), BATCH_CLIPS):
        batches.append((b // BATCH_CLIPS, clips[b:b + BATCH_CLIPS]))

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    artist_name = row.get("artist_name", "") or ""
    kirtan_type = row.get("kirtan_type", "") or ""

    def do_batch(args):
        k, batch_paths = args
        raw_path = raw_dir / f"batch_{k:04d}.json"
        first_idx = int(batch_paths[0].stem.split("_")[1])
        start = first_idx * int(CLIP_SEC)
        end = start + len(batch_paths) * int(CLIP_SEC)
        if raw_path.exists():
            try:
                data = json.loads(raw_path.read_text())
                if isinstance(data, list) and len(data) == len(batch_paths):
                    return k, batch_paths, data
            except Exception:
                pass
        texts = gemini_batch(client, batch_paths, start, end,
                             artist_name, kirtan_type)
        raw_path.write_text(json.dumps(texts, ensure_ascii=False))
        return k, batch_paths, texts

    all_results: dict[int, tuple[list[Path], list[str]]] = {}
    with ThreadPoolExecutor(max_workers=batch_workers) as ex:
        futs = {ex.submit(do_batch, b): b[0] for b in batches}
        for fut in as_completed(futs):
            k, batch_paths, texts = fut.result()
            all_results[k] = (batch_paths, texts)

    # write jsonl in order
    n_written = 0
    tmp = jsonl_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for k in sorted(all_results):
            batch_paths, texts = all_results[k]
            for p, t in zip(batch_paths, texts):
                clip_i = int(p.stem.split("_")[1])
                if not t or not t.strip():
                    continue
                rec = {
                    "item_id": item_id,
                    "clip_i": clip_i,
                    "start_sec": clip_i * CLIP_SEC,
                    "end_sec": (clip_i + 1) * CLIP_SEC,
                    "text": t.strip(),
                    "source": row["source"],
                    "kirtan_type": row["kirtan_type"],
                    "sikhnet_track_id": row.get("sikhnet_track_id", ""),
                    "sikhnet_shabad_id": row.get("sikhnet_shabad_id", ""),
                    "youtube_video_id": row.get("youtube_video_id", ""),
                    "youtube_playlist_id": row.get("youtube_playlist_id", ""),
                    "artist_name": row.get("artist_name", ""),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
    tmp.rename(jsonl_path)
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="Combined CSV from enumerators")
    ap.add_argument("--data-root", default="/root/v3_data")
    ap.add_argument("--target-hours", type=float, default=5.0,
                    help="Hours per kirtan_type")
    ap.add_argument("--download-workers", type=int, default=8)
    ap.add_argument("--split-workers", type=int, default=16)
    ap.add_argument("--item-workers", type=int, default=8,
                    help="Concurrent items during transcription")
    ap.add_argument("--batch-workers", type=int, default=6,
                    help="Concurrent Gemini batches per item")
    ap.add_argument("--stage", default="all",
                    choices=["all", "download", "split", "transcribe"])
    ap.add_argument("--max-item-secs", type=int, default=0,
                    help="Cap per-track audio before splitting (0 = no cap). "
                    "Long kirtan tracks repeat refrains; cap ~1500 (25 min).")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        all_rows = list(csv.DictReader(f))

    # select per kirtan_type
    by_type: dict[str, list[dict]] = {}
    for r in all_rows:
        by_type.setdefault(r["kirtan_type"], []).append(r)

    selected: list[dict] = []
    for ktype, rows in by_type.items():
        chosen = select_rows(rows, args.target_hours)
        hours = sum(float(r.get("est_secs") or 0) for r in chosen) / 3600
        print(f"[select] {ktype}: {len(chosen)} items, {hours:.2f}h")
        selected.extend(chosen)

    # persist selection
    sel_path = data_root / "selection.csv"
    if selected:
        with open(sel_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
            w.writeheader()
            w.writerows(selected)
    print(f"[select] total: {len(selected)} items → {sel_path}")

    # 1. Download
    if args.stage in ("all", "download"):
        print(f"\n=== stage 1: download ({args.download_workers} workers) ===")
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=args.download_workers) as ex:
            futs = {ex.submit(download_one, r, data_root): r for r in selected}
            for fut in as_completed(futs):
                r = futs[fut]
                out = fut.result()
                done += 1
                if done % 10 == 0 or done == len(selected):
                    print(f"  [dl] {done}/{len(selected)} (+{time.time() - t0:.0f}s)")
        print(f"[download] {done}/{len(selected)} complete in {time.time() - t0:.0f}s")

    # 2. Split
    if args.stage in ("all", "split"):
        print(f"\n=== stage 2: split ({args.split_workers} workers) ===")
        t0 = time.time()
        work = []
        for r in selected:
            iid = item_id(r)
            wav = data_root / "raw_audio" / r["kirtan_type"] / f"{iid}.wav"
            if not wav.exists():
                continue
            clips_dir = data_root / "clips" / iid
            work.append((str(wav), str(clips_dir), args.max_item_secs))
        with Pool(args.split_workers) as p:
            for j, (cdir, n) in enumerate(p.imap_unordered(split_one, work), 1):
                if j % 10 == 0 or j == len(work):
                    print(f"  [split] {j}/{len(work)} (+{time.time() - t0:.0f}s)")
        print(f"[split] {len(work)} items complete in {time.time() - t0:.0f}s")

    # 3. Transcribe
    if args.stage in ("all", "transcribe"):
        print(f"\n=== stage 3: transcribe "
              f"({args.item_workers} items × {args.batch_workers} batches) ===")
        t0 = time.time()
        done = 0
        total_segs = 0

        def worker(r):
            iid = item_id(r)
            clips_dir = data_root / "clips" / iid
            if not clips_dir.exists():
                return iid, 0
            n = transcribe_one(iid, clips_dir, data_root, args.batch_workers, r)
            return iid, n

        with ThreadPoolExecutor(max_workers=args.item_workers) as ex:
            futs = {ex.submit(worker, r): r for r in selected}
            for fut in as_completed(futs):
                iid, n = fut.result()
                done += 1
                total_segs += n
                if done % 5 == 0 or done == len(selected):
                    elapsed = time.time() - t0
                    print(f"  [tr] {done}/{len(selected)} items, "
                          f"{total_segs} segs (+{elapsed:.0f}s)")
        print(f"[transcribe] {total_segs} segments in {time.time() - t0:.0f}s")

    print("\n[done]")


if __name__ == "__main__":
    main()
