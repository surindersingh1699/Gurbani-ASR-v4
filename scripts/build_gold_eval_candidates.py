#!/usr/bin/env python3
"""GATE 0 — sample 400 GOLD-eval candidate clips (stratified) + attach SGGS
candidates, for hand-verification in apps/gold_eval (Streamlit).

400 clips = 100 ragi / 100 akj / 100 sgpc / 100 sehaj, each split 50 high /
50 residual quality so the eval spans easy+hard. Video-boundary sampling with a
per-video cap keeps it diverse and leak-free; the 1-video published-eval ids are
dropped. For each clip we attach top-K SGGS candidate LINES (4-gram skeleton
TF-IDF over the STTM database.sqlite, refined by edit distance) + first-letters,
so the verifier can one-click accept / edit / reject against canonical Gurbani.

Output:
  <out>/audio/<clip_id>.flac
  <out>/gold_candidates.jsonl   # one row per clip (consumed by the labeling app)

Usage:
  python scripts/build_gold_eval_candidates.py \
      --db ./database.sqlite --out ./gold_eval --per-source 100 --seed 42
"""
from __future__ import annotations
import argparse, io, json, math, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))               # scripts/  (gurmukhi_converter)
from canonical.sttm_index import load_sggs                      # noqa: E402
from canonical.gurmukhi_skeleton import skel, lev               # noqa: E402

# repo → coarse source category. "mixed" repos are split ragi vs akj by channel.
REPOS = [
    ("surindersinghssj/gurbani-kirtan-yt-captions-300h-cleanv2", "mixed"),
    ("surindersinghssj/gurbani-kirtan-v4-1000h-cleanv2",         "mixed"),
    ("surindersinghssj/gurbani-kirtan-v4-sgpc-cleanv2",          "sgpc"),
    ("surindersinghssj/gurbani-bani-v4-cleanv2",                 "sehaj"),
]
AKJ_PAT = re.compile(r"akj|akhand\s*kirtani|a\.?k\.?j", re.I)
SOURCES = ["ragi", "akj", "sgpc", "sehaj"]
QUALITIES = ["high", "residual"]
# published 1-video evals — never sample these into GOLD
LEAK_VIDS = {"cANTWzO5P4Y", "flLNIJyGoYM", "lIXI85hmTqk"}
GURMUKHI = re.compile(r"[਀-੿]")


def norm(text: str) -> str:
    text = (text or "").replace(">>", " ")
    text = re.sub(r"॥[੦-੯]+॥", "", text)
    text = re.sub(r"[^਀-੿ ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_letters(uni: str) -> str:
    return "".join(t[0] for t in uni.split() if t)


def classify(channel: str, repo_cat: str) -> str:
    if repo_cat in ("sgpc", "sehaj"):
        return repo_cat
    return "akj" if (channel and AKJ_PAT.search(channel)) else "ragi"


# ----- SGGS line-level candidate retriever (4-gram skeleton TF-IDF) -----
def ngrams(s: str, n: int = 4):
    return [s[i:i + n] for i in range(len(s) - n + 1)] if len(s) >= n else ([s] if s else [])


class SggsRetriever:
    def __init__(self, db_path: str):
        self.lines, _ = load_sggs(db_path)
        self.sk = [ln.skel for ln in self.lines]
        self.inv: dict[str, list[int]] = defaultdict(list)
        self.df: Counter = Counter()
        for i, s in enumerate(self.sk):
            for g in set(ngrams(s)):
                self.inv[g].append(i); self.df[g] += 1
        self.N = len(self.lines)
        print(f"[sggs] {self.N} lines indexed", flush=True)

    def topk(self, text: str, k: int = 3):
        q = skel(text)
        if not q:
            return []
        sc: Counter = Counter()
        for g, tf in Counter(ngrams(q)).items():
            ids = self.inv.get(g)
            if not ids:
                continue
            idf = math.log(1 + self.N / self.df[g])
            for i in ids:
                sc[i] += tf * idf
        cand = [i for i, _ in sc.most_common(25)]
        cand.sort(key=lambda i: lev(q, self.sk[i]))   # refine by edit distance
        out = []
        for i in cand[:k]:
            ln = self.lines[i]
            out.append({"line_id": ln.line_id, "shabad_id": ln.shabad_id, "ang": ln.ang,
                        "unicode": ln.unicode, "first_letters": first_letters(ln.unicode),
                        "lev": lev(q, self.sk[i])})
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./database.sqlite")
    ap.add_argument("--out", default="./gold_eval")
    ap.add_argument("--per-source", type=int, default=100)
    ap.add_argument("--per-video-cap", type=int, default=4)
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--max-dur", type=float, default=18.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-files-per-repo", type=int, default=40,
                    help="cap parquet files scanned per repo (each has audio; few suffice)")
    ap.add_argument("--sources", default=",".join(SOURCES),
                    help="comma list of sources to sample (e.g. ragi,akj,sehaj to skip sgpc)")
    args = ap.parse_args()
    chosen_sources = {s.strip() for s in args.sources.split(",") if s.strip()}

    import random, soundfile as sf, pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    rng = random.Random(args.seed)
    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    out = Path(args.out); (out / "audio").mkdir(parents=True, exist_ok=True)

    retr = SggsRetriever(args.db)

    half = args.per_source // 2
    need = {(s, q): (half if s in chosen_sources else 0)
            for s in SOURCES for q in QUALITIES}   # 50 each for chosen sources
    kept: dict[tuple, list] = defaultdict(list)
    vid_count: Counter = Counter()
    rows_out = []

    def full(): return all(len(kept[k]) >= need[k] for k in need)

    for repo, repo_cat in REPOS:
        if full():
            break
        if repo_cat != "mixed" and all(need[(repo_cat, q)] == 0 for q in QUALITIES):
            print(f"[{repo.split('/')[-1]}] skipped (source excluded)", flush=True)
            continue
        from huggingface_hub import HfFileSystem as _HFS
        fs = _HFS(token=os.environ.get("HF_TOKEN"))   # fresh client per repo — long
        # sessions die with "client has been closed" on transient errors
        files = sorted(f for f in fs.glob(f"datasets/{repo}/**/*.parquet"))
        rng.shuffle(files)
        files = files[:args.max_files_per_repo]
        print(f"[{repo.split('/')[-1]}] scanning up to {len(files)} files", flush=True)
        for fp in files:
            cats_needed = {s for s in SOURCES
                           if repo_cat in (s, "mixed") or (repo_cat == "mixed" and s in ("ragi", "akj"))}
            if repo_cat in ("sgpc", "sehaj"):
                cats_needed = {repo_cat}
            if all(len(kept[(s, q)]) >= need[(s, q)] for s in cats_needed for q in QUALITIES):
                continue
            try:
                with fs.open(fp) as fh:
                    tbl = pq.ParquetFile(fh).read()
                rows = tbl.to_pylist()
            except Exception as e:
                print(f"  [warn] {fp.split('/')[-1]}: {type(e).__name__} {e} — recreating client, skipping file", flush=True)
                fs = _HFS(token=os.environ.get("HF_TOKEN"))
                continue
            rng.shuffle(rows)
            for ex in rows:
                vid = ex.get("video_id") or ""
                if vid in LEAK_VIDS:
                    continue
                q = (ex.get("quality") or "").lower()
                if q not in QUALITIES:
                    continue
                src = classify(ex.get("channel") or "", repo_cat)
                key = (src, q)
                if key not in need or len(kept[key]) >= need[key]:
                    continue
                if vid_count[vid] >= args.per_video_cap:
                    continue
                dur = ex.get("duration_s") or 0
                if not (args.min_dur <= dur <= args.max_dur):
                    continue
                text = norm(ex.get("text") or "")
                if len(text.split()) < 2 or len(GURMUKHI.findall(text)) < 4:
                    continue
                au = ex.get("audio")
                if not (isinstance(au, dict) and au.get("bytes")):
                    continue
                try:
                    arr, sr = sf.read(io.BytesIO(au["bytes"]), dtype="float32")
                except Exception:
                    continue
                if getattr(arr, "ndim", 1) > 1:
                    arr = arr.mean(axis=1).astype("float32")
                clip_id = ex.get("clip_id") or f"{vid}_{len(rows_out):05d}"
                wav = out / "audio" / f"{clip_id}.flac"
                try:
                    sf.write(str(wav), arr, sr, format="FLAC", subtype="PCM_16")
                except Exception:
                    continue
                rec = {
                    "clip_id": clip_id, "source": src, "quality": q, "video_id": vid,
                    "channel": ex.get("channel") or "", "duration_s": round(float(dur), 2),
                    "audio_path": f"audio/{clip_id}.flac",
                    "proposed_text": text, "original_text": ex.get("text") or "",
                    "skel_route": ex.get("skel_route") or "",
                    "candidates": retr.topk(text, k=3),
                }
                kept[key].append(clip_id)
                vid_count[vid] += 1
                rows_out.append(rec)
            print("  progress: " + " ".join(f"{s[0]}{q[0]}={len(kept[(s, q)])}/{need[(s, q)]}"
                  for s in SOURCES for q in QUALITIES), flush=True)
            if full():
                break

    rng.shuffle(rows_out)
    man = out / "gold_candidates.jsonl"
    # preserve rows added by other tools (e.g. add_video_to_gold_candidates.py)
    if man.exists():
        new_ids = {r["clip_id"] for r in rows_out}
        for l in man.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r["clip_id"] not in new_ids:
                    rows_out.append(r)
    with man.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[done] {len(rows_out)} candidates → {man}")
    dist = Counter((r["source"], r["quality"]) for r in rows_out)
    for s in SOURCES:
        print(f"  {s}: high={dist[(s,'high')]}  residual={dist[(s,'residual')]}")
    miss = {k: need[k] - len(kept[k]) for k in need if len(kept[k]) < need[k]}
    if miss:
        print(f"[warn] under-filled buckets (raise --max-files-per-repo): {miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
