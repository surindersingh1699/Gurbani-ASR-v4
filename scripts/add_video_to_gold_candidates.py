#!/usr/bin/env python3
"""Add clips from one YouTube video to the GOLD-eval labeling queue
(apps/gold_eval), using Punjabi AUTO-captions as proposed labels.

Measured on W2v3sgZz6MU (raag kirtan): 63% of auto-caption lines land within
skeleton lev<=8 of an SGGS line — noisy but anchorable. Flow per clip:
auto-caption text → SGGS skeleton retrieval (top-3 candidates) → LLM
pre-correction (scripts/llm_precorrect_gold.py) → human confirms by ear.

Audio is downloaded in NATIVE container (webm/opus) so caption timestamps
align with the audio timeline (see feedback_yt_caption_pipeline). Caption
events are grouped into 5–18 s clips (gap > 4 s breaks a group), then
--num-clips groups are sampled uniformly across the video.

Appends to <out>/gold_candidates.jsonl + writes <out>/audio/<clip_id>.flac.
Idempotent per clip_id.

Usage:
  python3 scripts/add_video_to_gold_candidates.py \
      --video W2v3sgZz6MU --out ./gold_eval --db ./database.sqlite \
      --num-clips 80 --source ragi --seed 42
"""
from __future__ import annotations
import argparse, json, random, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_gold_eval_candidates import SggsRetriever, norm   # noqa: E402

MIN_S, MAX_S, GAP_S = 5.0, 18.0, 4.0


def sh(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


def load_auto_caption_events(json3_fp: Path) -> list[dict]:
    d = json.loads(json3_fp.read_text(encoding="utf-8"))
    out = []
    for e in d.get("events", []):
        if not e.get("segs"):
            continue
        text = norm("".join(s.get("utf8", "") for s in e["segs"]))
        if not text:
            continue
        start = e["tStartMs"] / 1000.0
        dur = e.get("dDurationMs", 0) / 1000.0
        out.append({"start": start, "end": start + dur, "text": text})
    return out


def group_events(events: list[dict]) -> list[dict]:
    """Merge consecutive caption events into 5-18s clips; gap > 4s breaks."""
    groups, cur = [], None
    for ev in events:
        if cur is None:
            cur = dict(ev)
            continue
        gap = ev["start"] - cur["end"]
        joined = ev["end"] - cur["start"]
        if gap > GAP_S or joined > MAX_S:
            groups.append(cur); cur = dict(ev)
        else:
            cur["end"] = ev["end"]
            cur["text"] = f"{cur['text']} {ev['text']}"
    if cur:
        groups.append(cur)
    return [g for g in groups
            if MIN_S <= g["end"] - g["start"] <= MAX_S and len(g["text"].split()) >= 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="YouTube video id or URL")
    ap.add_argument("--out", default="./gold_eval")
    ap.add_argument("--db", default="./database.sqlite")
    ap.add_argument("--num-clips", type=int, default=80)
    ap.add_argument("--source", default="ragi", choices=["ragi", "akj", "sgpc", "sehaj"])
    ap.add_argument("--quality", default="residual")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pad-pre", type=float, default=2.0,
                    help="seconds of audio before caption start (auto-caption "
                         "timestamps LAG the audio: YT ASR emits text after hearing it)")
    ap.add_argument("--pad-post", type=float, default=2.0)
    ap.add_argument("--repad", action="store_true",
                    help="re-slice audio (with padding) for this video's EXISTING "
                         "rows instead of adding new clips")
    args = ap.parse_args()

    vid = args.video.split("v=")[-1].split("&")[0] if "youtube" in args.video else args.video
    url = f"https://www.youtube.com/watch?v={vid}"
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    cand_fp = out / "gold_candidates.jsonl"

    existing = set()
    if cand_fp.exists():
        for l in cand_fp.read_text(encoding="utf-8").splitlines():
            if l.strip():
                existing.add(json.loads(l)["clip_id"])

    meta = json.loads(sh(["yt-dlp", "--skip-download", "-J", url]))
    channel = meta.get("channel") or ""
    vid_dur = float(meta["duration"])
    print(f"{vid}: '{meta.get('title','')[:70]}' · {vid_dur/3600:.1f} h · {channel}")

    tmpdir = Path(tempfile.mkdtemp(prefix=f"gold_{vid}_"))

    def slice_clip(audio_fp: Path, start: float, end: float, wav: Path) -> bool:
        s = max(0.0, start - args.pad_pre)
        d = min(vid_dur, end + args.pad_post) - s
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-ss", f"{s:.2f}", "-t", f"{d:.2f}",
             "-i", str(audio_fp), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(wav)],
            capture_output=True, text=True)
        if r.returncode != 0 or not wav.exists() or wav.stat().st_size < 10_000:
            wav.unlink(missing_ok=True)
            return False
        return True

    if args.repad:
        # rewrite audio for existing rows of this video, keep rows/labels intact
        rows_all = [json.loads(l) for l in cand_fp.read_text(encoding="utf-8").splitlines() if l.strip()]
        mine = [r for r in rows_all if r["video_id"] == vid]
        if not mine:
            sys.exit(f"no existing rows for {vid}")
        print(f"repad: re-slicing {len(mine)} clips with -{args.pad_pre}/+{args.pad_post}s …")
        sh(["yt-dlp", "-f", "bestaudio", "-o", str(tmpdir / f"{vid}.audio"), url])
        audio_fp = next(p for p in tmpdir.glob(f"{vid}.audio*"))
        n = 0
        for r in mine:
            start = float(r["clip_id"].rsplit("_", 1)[-1])
            if slice_clip(audio_fp, start, start + r["duration_s"], out / r["audio_path"]):
                n += 1
        print(f"repad done: {n}/{len(mine)} rewritten")
        return

    # Punjabi auto-captions (json3)
    sh(["yt-dlp", "--skip-download", "--write-auto-subs", "--sub-langs", "pa-orig,pa",
        "--sub-format", "json3", "-o", str(tmpdir / "subs"), url])
    sub_fp = next(iter(sorted(tmpdir.glob("subs.pa*.json3"))), None)
    if not sub_fp:
        sys.exit("no Punjabi auto-captions on this video")
    groups = group_events(load_auto_caption_events(sub_fp))
    print(f"{len(groups)} caption clip-groups (5-18s)")
    if not groups:
        sys.exit("no usable caption groups")

    # native-container audio (preserve YouTube timeline — do NOT transcode at download)
    print("downloading native audio …")
    sh(["yt-dlp", "-f", "bestaudio", "-o", str(tmpdir / f"{vid}.audio"), url])
    audio_fp = next(p for p in tmpdir.glob(f"{vid}.audio*") if not p.name.endswith(".json3"))

    retr = SggsRetriever(args.db)

    rng = random.Random(args.seed)
    rng.shuffle(groups)

    rows, made = [], 0
    for g in groups:
        if made >= args.num_clips:
            break
        clip_id = f"{vid}_{int(g['start']):06d}"
        if clip_id in existing:
            made += 1
            continue
        dur = g["end"] - g["start"]
        wav = out / "audio" / f"{clip_id}.flac"
        if not slice_clip(audio_fp, g["start"], g["end"], wav):
            continue
        rows.append({
            "clip_id": clip_id, "source": args.source, "quality": args.quality,
            "video_id": vid, "channel": channel, "duration_s": round(dur, 2),
            "audio_path": f"audio/{clip_id}.flac",
            "proposed_text": g["text"], "original_text": g["text"],
            "skel_route": "yt_auto_caption", "candidates": retr.topk(g["text"], k=3),
        })
        made += 1
        if made % 10 == 0:
            print(f"  {made}/{args.num_clips}", flush=True)

    with cand_fp.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"appended {len(rows)} new clips (target {args.num_clips}) → {cand_fp}")
    print(f"temp download at {tmpdir} (delete when done)")


if __name__ == "__main__":
    main()
