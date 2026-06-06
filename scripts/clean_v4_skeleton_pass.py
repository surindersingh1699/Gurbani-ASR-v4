"""Phase 1: skeleton pre-check + chorus / pair-span detection.

For each clip, decide if the caption is already 'training-ready' under
normalized Gurbani CER by checking whether its consonant skeleton:

  1. appears as a substring of any single SGGS pangti skeleton, OR
  2. is covered by the concatenation of two adjacent pangtis of the
     same shabad (cross-pangti spans like rahao + first antra), OR
  3. is the same pangti skeleton repeated N≥2 times (chorus repeat).

If any of those hold → quality='high', text = original caption verbatim
(no swap, no matra-fix; normalized CER ignores matras anyway, and
keeping the caption verbatim avoids the algorithm's old wrong-swap
class of bugs).

Otherwise → write the clip to a residual JSONL with retrieved shabad
context for the Phase 2 LLM rescue (scripts/llm_rescue_residual.py).

Output goes to a new public HF repo (<source>-clean) AND a residual
JSONL on disk.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from canonical.gurmukhi_skeleton import skel, tokenize  # noqa: E402
from canonical.preprocess import strip_unk_artifacts  # noqa: E402
from canonical.sttm_index import (  # noqa: E402
    load_sggs, build_shabad_ngram_index,
)
from canonical.retrieval import RetrievalConfig, retrieve_shabad  # noqa: E402


# --- Pre-clean (relaxed rule from the plan) ----------------------------

def gurmukhi_tokens(text: str) -> list[str]:
    toks = tokenize(strip_unk_artifacts(text or ""))
    return [t for t in toks if any("਀" <= ch <= "੿" for ch in t)]


def should_drop_row(row: dict, min_duration_s: float = 1.0) -> bool:
    if (row.get("duration_s") or 0.0) < min_duration_s:
        return True
    gur = gurmukhi_tokens(row.get("text") or "")
    if not gur:
        return True
    if len(gur) == 1 and len(gur[0]) <= 2:
        return True
    return False


# --- Skeleton indices over the SGGS DB ----------------------------------

class SkelIndex:
    """In-memory indices over SGGS pangtis for the three skeleton checks.

    Index structures:
        pangti_skels:       list[(line_id, shabad_id, unicode, pangti_skel)]
        shabad_to_pangtis:  dict[shabad_id, list[(line_id, pangti_skel)]]
            (preserves pangti order within shabad)
        all_skels_concat:   a single huge string of all pangti skeletons
            concatenated with a separator that can never appear in a
            skeleton, so a single .find() call on this answers step 1.

    `find_substring` is the hot path; ~141k pangtis, average skeleton
    length 20 chars → ~3 MB string. Python's str.find on that is plenty
    fast (~5 µs per query).
    """
    SEP = ""  # cannot appear in Gurmukhi skeletons

    def __init__(self, db_path: str):
        lines, _ = load_sggs(db_path)
        self.lines = lines
        self.shabad_to_pangtis: dict[str, list[tuple[str, str]]] = {}
        for ln in lines:
            self.shabad_to_pangtis.setdefault(ln.shabad_id, []).append(
                (ln.line_id, ln.skel)
            )
        # Step 1 index: concat all single-pangti skels with separator
        self.single_index = self.SEP.join(ln.skel for ln in lines if ln.skel)
        # Step 2 index: pairs of adjacent pangtis within a shabad
        pair_parts = []
        for sid, pangtis in self.shabad_to_pangtis.items():
            for a, b in zip(pangtis, pangtis[1:]):
                if a[1] and b[1]:
                    pair_parts.append(a[1] + b[1])
        self.pair_index = self.SEP.join(pair_parts)
        # Distinct pangti skels (for chorus check)
        self.distinct_pangti_skels = list({ln.skel for ln in lines if ln.skel})

        # Retrieval index for Phase 2 residual context
        self.shabad_ngrams, self.shabad_lines, self.df = (
            build_shabad_ngram_index(lines, n=4)
        )

    # Step 1
    def single_pangti_substring(self, cap_skel: str) -> bool:
        if not cap_skel:
            return False
        return cap_skel in self.single_index

    # Step 2
    def pair_pangti_substring(self, cap_skel: str) -> bool:
        if not cap_skel or not self.pair_index:
            return False
        return cap_skel in self.pair_index

    # Step 3 (chorus repeat: cap_skel == pangti_skel * N for some N >= 2)
    def is_chorus_repeat(self, cap_skel: str) -> bool:
        if not cap_skel or len(cap_skel) < 6:
            return False
        # Try every possible divisor
        L = len(cap_skel)
        for n in range(2, L // 3 + 1):
            if L % n:
                continue
            unit = cap_skel[: L // n]
            if cap_skel == unit * n and unit in self.single_index:
                return True
        return False

    def retrieve_shabad_context(
        self,
        cur_tokens: list[str],
        prev_tokens: list[str],
        next_tokens: list[str],
        prev2_tokens: list[str],
        next2_tokens: list[str],
        video_hits: Counter,
    ) -> tuple[str | None, list[str]]:
        cfg = RetrievalConfig()
        sid, score, _margin, _ = retrieve_shabad(
            cur_tokens, (prev_tokens, next_tokens),
            (prev2_tokens, next2_tokens),
            self.shabad_ngrams, self.df, video_hits, None, cfg,
        )
        if not sid or score < 2.0:
            return None, []
        pangtis = [ln.unicode for ln in self.shabad_lines[sid][:8]]
        return sid, pangtis


# --- Per-clip classifier ------------------------------------------------

def classify_row(
    row: dict, idx: SkelIndex
) -> tuple[str, str]:
    """Return (quality_label, route).

    quality_label: 'high' (skeleton-canonical) | 'residual' (to Phase 2).
    route:         'single' | 'pair' | 'chorus' | 'unknown'
    """
    cap_skel = skel(row["text"])
    if idx.single_pangti_substring(cap_skel):
        return "high", "single"
    if idx.pair_pangti_substring(cap_skel):
        return "high", "pair"
    if idx.is_chorus_repeat(cap_skel):
        return "high", "chorus"
    return "residual", "unknown"


# --- Driver -------------------------------------------------------------

PUSH_CHUNK = 1500  # smaller chunks since HF datasets accumulates refs
MAX_VIDEO_BUFFER = 800  # cap cur_video_rows to bound RAM for huge videos
PUSH_LOCK_PATH = "/tmp/clean_v4_push.lock"  # serialize HF pushes across jobs


@contextmanager
def hf_push_lock(path: str = PUSH_LOCK_PATH):
    """Global flock so only ONE Phase 1 job pushes to HF at a time.

    The push step (Dataset.from_list → push_to_hub on 2500 audio rows)
    spikes RAM to 3-5GB. Three concurrent pushes blew through the 16GB
    Hetzner box on the first attempt. Holding this lock around the push
    serializes the spike while letting all jobs scan in parallel.
    """
    fp = open(path, "w")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        fp.close()


def run(
    source_repo: str,
    dest_repo: str,
    db_path: str,
    residual_path: Path,
    public: bool,
    limit: int | None,
    dry_run: bool,
    start_from: str | None,
    done_path: Path,
):
    print(f"[clean_v4] loading SGGS index from {db_path} ...", flush=True)
    t0 = time.time()
    idx = SkelIndex(db_path)
    print(
        f"[clean_v4] loaded {len(idx.lines)} pangtis "
        f"({len(idx.shabad_to_pangtis)} shabads) in {time.time() - t0:.1f}s",
        flush=True,
    )

    print(f"[clean_v4] streaming source {source_repo} ...", flush=True)
    from datasets import Audio, Dataset, Features, Value, load_dataset
    from huggingface_hub import create_repo

    # Explicit Features so type inference can't flap between chunks (e.g.
    # an all-null `channel` chunk inferring Value('null') and breaking the
    # next push's schema check against earlier string-typed splits).
    OUTPUT_FEATURES = Features({
        "audio": Audio(sampling_rate=16000),
        "text": Value("string"),
        "original_text": Value("string"),
        "quality": Value("string"),
        "video_id": Value("string"),
        "start_s": Value("float64"),
        "end_s": Value("float64"),
        "duration_s": Value("float64"),
        "channel": Value("string"),
        "clip_id": Value("string"),
        "skel_route": Value("string"),
    })

    ds = load_dataset(source_repo, split="train", streaming=True)
    # datasets >= 4 needs torchcodec to decode audio; we passthrough bytes
    # (no decoding required for skeleton-only Phase 1), so disable decode.
    if "audio" in (ds.features or {}):
        try:
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception as e:
            print(f"[clean_v4] cast audio decode=False failed: {e}", flush=True)

    done_videos = set()
    if done_path.exists():
        done_videos = {
            v.strip() for v in done_path.read_text().splitlines() if v.strip()
        }
        print(f"[clean_v4] resume: skipping {len(done_videos)} done videos")

    if not dry_run:
        try:
            create_repo(
                dest_repo, repo_type="dataset", private=not public,
                exist_ok=True, token=os.environ.get("HF_TOKEN"),
            )
        except Exception as e:
            print(f"[clean_v4] create_repo: {e}", flush=True)

    # Append-mode for both residual and done so resume continues cleanly.
    residual_fp = residual_path.open("a", encoding="utf-8")

    pending: list[dict] = []      # accumulated rows for next HF split
    pending_vids: list[str] = []  # videos fully represented in pending
    histo = Counter()
    processed = 0
    cur_video: str | None = None
    cur_video_rows: list[dict] = []
    cur_video_hits: Counter = Counter()
    chunk_seq = 0
    run_stamp = time.strftime("%Y%m%d_%H%M%S")

    def push_pending(force: bool = False):
        nonlocal chunk_seq
        if dry_run:
            return
        if not pending:
            return
        if not force and len(pending) < PUSH_CHUNK:
            return
        n = len(pending)
        split_name = f"clean_phase1_{run_stamp}_part{chunk_seq:04d}"
        print(
            f"[clean_v4] waiting for push lock ({n} rows -> {split_name}) ...",
            flush=True,
        )
        with hf_push_lock():
            print(
                f"[clean_v4] pushing {split_name} ({n} rows) -> {dest_repo}",
                flush=True,
            )
            ds_out = Dataset.from_list(pending, features=OUTPUT_FEATURES)
            last_err = None
            for attempt in range(6):
                try:
                    ds_out.push_to_hub(
                        dest_repo, split=split_name, private=not public,
                        token=os.environ.get("HF_TOKEN"),
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    wait = min(10 * (2 ** attempt), 300)
                    print(
                        f"[clean_v4] push attempt {attempt+1}/6 failed: {type(e).__name__}: {str(e)[:200]}; retrying in {wait}s",
                        flush=True,
                    )
                    time.sleep(wait)
            if last_err is not None:
                raise last_err
            # Only after push succeeds, mark these videos done.
            with done_path.open("a") as fp:
                for v in pending_vids:
                    fp.write(v + "\n")
            print(
                f"[clean_v4] pushed {split_name}, marked {len(pending_vids)} videos done",
                flush=True,
            )
            del ds_out  # release Arrow table memory before lock release
        pending.clear()
        pending_vids.clear()
        chunk_seq += 1
        gc.collect()

    def flush_video(buf_rows, vid, finalize=True):
        if not buf_rows:
            return
        for i, r in enumerate(buf_rows):
            quality, route = classify_row(r, idx)
            histo[(quality, route)] += 1
            base = {
                "audio": r.get("audio"),
                "original_text": r.get("text"),
                "text": r.get("text"),
                "quality": "high" if quality == "high" else "residual",
                "video_id": r.get("video_id"),
                "start_s": r.get("start_s"),
                "end_s": r.get("end_s"),
                "duration_s": r.get("duration_s"),
                "channel": r.get("channel"),
                "clip_id": r.get("clip_id"),
                "skel_route": route,
            }
            if quality != "high":
                cur_tokens = tokenize(r["text"])
                prev_t = tokenize(buf_rows[i - 1]["text"]) if i > 0 else []
                next_t = (
                    tokenize(buf_rows[i + 1]["text"])
                    if i + 1 < len(buf_rows) else []
                )
                prev2_t = tokenize(buf_rows[i - 2]["text"]) if i > 1 else []
                next2_t = (
                    tokenize(buf_rows[i + 2]["text"])
                    if i + 2 < len(buf_rows) else []
                )
                sid, pangtis = idx.retrieve_shabad_context(
                    cur_tokens, prev_t, next_t, prev2_t, next2_t,
                    cur_video_hits,
                )
                if sid:
                    cur_video_hits[sid] += 1
                residual_fp.write(json.dumps({
                    "clip_id": r.get("clip_id"),
                    "video_id": vid,
                    "original_text": r["text"],
                    "shabad_id": sid,
                    "shabad_pangtis": pangtis,
                    "start_s": r.get("start_s"),
                    "duration_s": r.get("duration_s"),
                    "audio_ref": "passthrough",
                }, ensure_ascii=False) + "\n")
            pending.append(base)
        residual_fp.flush()
        if finalize:
            pending_vids.append(vid)
        push_pending()

    print(f"[clean_v4] iterating clips ...", flush=True)
    t1 = time.time()
    for row in ds:
        vid = row.get("video_id") or "unknown"
        if vid in done_videos:
            continue
        if start_from and vid < start_from:
            continue
        if should_drop_row(row):
            histo[("dropped", "preclean")] += 1
            processed += 1
            if limit and processed >= limit:
                break
            continue
        if vid != cur_video:
            if cur_video is not None:
                flush_video(cur_video_rows, cur_video)
            cur_video = vid
            cur_video_rows = []
            cur_video_hits = Counter()
        cur_video_rows.append(row)
        processed += 1
        # Partial flush for huge videos so RAM stays bounded. We DON'T
        # mark the video done here — only when the video boundary is
        # crossed (finalize=True path). Boundary rows lose a sliver of
        # prev/next-token context for residual retrieval; acceptable.
        if len(cur_video_rows) >= MAX_VIDEO_BUFFER:
            flush_video(cur_video_rows, cur_video, finalize=False)
            cur_video_rows = []
            gc.collect()
        if processed % 5000 == 0:
            rate = processed / (time.time() - t1)
            print(
                f"[clean_v4] processed {processed} clips "
                f"({rate:.0f}/s), histo={dict(histo)}, pending={len(pending)}",
                flush=True,
            )
        if limit and processed >= limit:
            break

    if cur_video_rows:
        flush_video(cur_video_rows, cur_video)
    push_pending(force=True)
    residual_fp.close()

    elapsed = time.time() - t1
    print(f"\n[clean_v4] processed {processed} clips in {elapsed:.1f}s")
    print(f"[clean_v4] decision histogram:")
    for k, v in sorted(histo.items(), key=lambda kv: -kv[1]):
        print(f"    {k}: {v}")
    print(f"[clean_v4] residual rows written to: {residual_path}")
    print(f"[clean_v4] HF splits pushed: {chunk_seq}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-repo", required=True)
    ap.add_argument("--dest-repo", required=True,
                    help="HF repo for cleaned data; created public by default.")
    ap.add_argument("--db", required=True,
                    help="Path to database.sqlite")
    ap.add_argument("--residual", type=Path, required=True,
                    help="JSONL path to write Phase 2 residuals to")
    ap.add_argument("--done-file", type=Path, required=True,
                    help="Track processed video_ids for resume")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-from", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--private", action="store_true",
                    help="Default repo is PUBLIC; pass --private to override.")
    args = ap.parse_args()

    run(
        source_repo=args.source_repo,
        dest_repo=args.dest_repo,
        db_path=args.db,
        residual_path=args.residual,
        public=not args.private,
        limit=args.limit,
        dry_run=args.dry_run,
        start_from=args.start_from,
        done_path=args.done_file,
    )


if __name__ == "__main__":
    main()
