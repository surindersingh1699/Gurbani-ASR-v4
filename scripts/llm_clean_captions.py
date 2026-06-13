#!/usr/bin/env python3
"""v5 data cleaning — fix Gurbani caption labels against canonical SGGS, via Vertex Gemini.

TEXT-ONLY pass (no audio download). For every clip's caption we:
  1. norm()            strip >>, verse markers, non-Gurmukhi  (free)
  2. SggsRetriever     top-3 canonical SGGS line candidates    (free, 4-gram TF-IDF + edit dist)
  3. near-exact gate   top-1 skeleton-lev ratio <= --auto-lev  -> auto-accept canonical, NO LLM
  4. Gemini verdict    remaining clips, batched, pick-or-flag against the 3 candidates

High precision / drop-aggressive: 971K clips is surplus, so ambiguous clips are
dropped rather than guessed (avoids the v4 label-noise trap). The LLM PICKS from
candidates — it never free-generates Gurbani (that hallucinates new errors).

Output is a CLEANING MAP (small, text-only) — one row per clip:
  {clip_id, repo, video_id, raw_text, norm_text, status, gold_text,
   line_id, shabad_id, ang, lev, via}
status in {accept, corrected, drop}. The v5 training loader joins this map onto
the existing audio repos by clip_id (keep accept|corrected, drop the rest) — so
NO 280GB audio rewrite is needed to train.

Auth: Vertex ADC. Run once in your shell:  gcloud auth application-default login
Then: export GOOGLE_CLOUD_PROJECT=your-project-id

Usage:
  .venv-v5clean/bin/python scripts/llm_clean_captions.py \
      --project $GOOGLE_CLOUD_PROJECT --location us-central1 \
      --db ./database.sqlite --out ./gold_eval/clean_map.jsonl \
      --batch 30 --workers 8
  # smoke first:  add  --limit 500
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_gold_eval_candidates import SggsRetriever, norm        # noqa: E402
from canonical.gurmukhi_skeleton import skel, lev                 # noqa: E402

# Default sources = the curated -cleanv2 repos (dedup/filtered). Override with --repos.
DEFAULT_REPOS = [
    "surindersinghssj/gurbani-kirtan-yt-captions-300h-cleanv2",
    "surindersinghssj/gurbani-kirtan-v4-1000h-cleanv2",
    "surindersinghssj/gurbani-kirtan-v4-sgpc-cleanv2",
    "surindersinghssj/gurbani-bani-v4-cleanv2",
]
TEXT_COLS = ["text", "raw_text", "clip_id", "video_id"]
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def ratio(a_skel: str, b_skel: str) -> float:
    if not a_skel or not b_skel:
        return 1.0
    return lev(a_skel, b_skel) / max(len(a_skel), len(b_skel))


def vertex_client(project: str, location: str):
    """Vertex Gemini via Application Default Credentials (no API key in code)."""
    from google import genai
    return genai.Client(vertexai=True, project=project, location=location)


def iter_repo_rows(repo: str, limit: int | None = None):
    """Stream (text, raw_text, clip_id, video_id) from a HF repo's parquet shards.
    pyarrow column projection reads only text column-chunks — audio bytes are skipped."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{repo}/**/*.parquet"))
    n = 0
    for fp in files:
        with fs.open(fp) as fh:
            pf = pq.ParquetFile(fh)
            avail = set(pf.schema_arrow.names)
            cols = [c for c in TEXT_COLS if c in avail]
            for batch in pf.iter_batches(batch_size=2048, columns=cols):
                d = batch.to_pydict()
                rows = d.get("clip_id") or d.get("text") or []
                for i in range(len(rows)):
                    yield {
                        "repo": repo,
                        "clip_id": (d.get("clip_id") or [None] * len(rows))[i],
                        "video_id": (d.get("video_id") or [None] * len(rows))[i],
                        "text": (d.get("text") or [""] * len(rows))[i] or "",
                        "raw_text": (d.get("raw_text") or [""] * len(rows))[i] or "",
                    }
                    n += 1
                    if limit and n >= limit:
                        return


PROMPT_HEAD = """You are a Sri Guru Granth Sahib (Gurbani) proof-reader. Each item is one
sung-kirtan caption (Gurmukhi, possibly mis-spelled or partially mis-heard) plus up to 3
CANONICAL SGGS line candidates retrieved for it. Decide, per item, which canonical line the
caption is, if any.

Rules:
- "exact"     : the caption already equals a candidate (ignoring matra/spacing) -> choice = that candidate number.
- "corrected" : the caption is clearly the SAME line as a candidate but mis-spelled -> choice = that number.
                Use ONLY when it is the same line, NOT a shorter fragment of it.
- "drop"      : caption is a partial fragment, repeated words only, waheguru/simran-only,
                English/katha, or matches no candidate confidently -> choice = null.
Never invent Gurbani text. Pick a candidate by number or drop.

Return ONLY a JSON array of exactly {n} objects in order:
{{"i": <item number>, "verdict": "exact|corrected|drop", "choice": <1|2|3|null>}}

Items:
"""


def render_item(idx: int, row: dict) -> str:
    cands = row["cands"]
    lines = [f'Item {idx}: caption = "{row["norm"]}"']
    for j, c in enumerate(cands, 1):
        lines.append(f'  candidate {j}: "{c["unicode"]}"  (first-letters: {c["first_letters"]})')
    return "\n".join(lines)


def llm_verdict_batch(client, batch: list[dict], model: str) -> list[dict]:
    from google.genai import types
    n = len(batch)
    body = "\n".join(render_item(i + 1, r) for i, r in enumerate(batch))
    prompt = PROMPT_HEAD.format(n=n) + body
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=model, contents=[types.Part.from_text(text=prompt)],
                config=types.GenerateContentConfig(
                    temperature=0.0, response_mime_type="application/json"))
            items = json.loads(resp.text.strip())
            if isinstance(items, dict):
                items = list(items.values())
            out = {}
            for it in items:
                if isinstance(it, dict) and "i" in it:
                    out[int(it["i"])] = it
            results = []
            for i in range(1, n + 1):
                it = out.get(i, {"verdict": "drop", "choice": None})
                results.append({"verdict": str(it.get("verdict") or "drop"),
                                "choice": it.get("choice")})
            return results
        except Exception as e:
            wait = 10 * (attempt + 1)
            log(f"    [retry {attempt+1}] {type(e).__name__}: {e} — sleep {wait}s")
            time.sleep(wait)
    return [{"verdict": "drop", "choice": None}] * n


def finalize(row: dict, verdict: str, choice) -> dict:
    base = {"clip_id": row["clip_id"], "repo": row["repo"], "video_id": row["video_id"],
            "raw_text": row["text"], "norm_text": row["norm"]}
    cands = row["cands"]
    if verdict in ("exact", "corrected") and choice and 1 <= int(choice) <= len(cands):
        c = cands[int(choice) - 1]
        base.update({"status": "accept" if verdict == "exact" else "corrected",
                     "gold_text": c["unicode"], "line_id": c["line_id"],
                     "shabad_id": c["shabad_id"], "ang": c["ang"],
                     "lev": c["lev"], "via": "llm"})
    else:
        base.update({"status": "drop", "gold_text": "", "line_id": "", "shabad_id": "",
                     "ang": "", "lev": None, "via": "llm"})
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                    help="GCP project id (or set GOOGLE_CLOUD_PROJECT)")
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--db", default="./database.sqlite")
    ap.add_argument("--out", default="./clean_map.jsonl")
    ap.add_argument("--repos", default=",".join(DEFAULT_REPOS))
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--auto-lev", type=float, default=0.10,
                    help="top-1 skeleton-lev ratio at/below which we auto-accept, no LLM")
    ap.add_argument("--drop-lev", type=float, default=0.55,
                    help="if best candidate ratio worse than this, drop without spending an LLM call")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test: cap total clips")
    args = ap.parse_args()
    if not args.project:
        sys.exit("set --project or GOOGLE_CLOUD_PROJECT (and run: gcloud auth application-default login)")

    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # resume: skip clip_ids already written
    done: set[str] = set()
    if out_path.exists():
        with out_path.open() as fh:
            for ln in fh:
                try:
                    done.add(json.loads(ln)["clip_id"])
                except Exception:
                    pass
        log(f"[resume] {len(done)} clips already in {out_path}")

    retr = SggsRetriever(args.db)
    client = vertex_client(args.project, args.location)
    log(f"[vertex] project={args.project} location={args.location} model={args.model}")

    out_fh = out_path.open("a")
    write_lock = threading.Lock()
    counts = {"accept": 0, "corrected": 0, "drop": 0, "auto": 0, "llm": 0}

    def write(rec: dict) -> None:
        with write_lock:
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
            counts[rec["status"]] += 1

    pending: list[dict] = []
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = []

    def flush_pending():
        if not pending:
            return
        chunk = pending[:]
        pending.clear()
        futures.append(pool.submit(process_chunk, chunk))

    def process_chunk(chunk: list[dict]):
        verdicts = llm_verdict_batch(client, chunk, args.model)
        for row, v in zip(chunk, verdicts):
            write(finalize(row, v["verdict"], v["choice"]))
        counts["llm"] += len(chunk)

    seen = 0
    for repo in repos:
        log(f"[scan] {repo}")
        for row in iter_repo_rows(repo, args.limit):
            cid = row["clip_id"]
            if cid in done:
                continue
            seen += 1
            row["norm"] = norm(row["text"])
            if not row["norm"]:
                write({"clip_id": cid, "repo": repo, "video_id": row["video_id"],
                       "raw_text": row["text"], "norm_text": "", "status": "drop",
                       "gold_text": "", "line_id": "", "shabad_id": "", "ang": "",
                       "lev": None, "via": "norm"}); continue
            cands = retr.topk(row["norm"], 3)
            if not cands:
                write({"clip_id": cid, "repo": repo, "video_id": row["video_id"],
                       "raw_text": row["text"], "norm_text": row["norm"], "status": "drop",
                       "gold_text": "", "line_id": "", "shabad_id": "", "ang": "",
                       "lev": None, "via": "no-cand"}); continue
            row["cands"] = cands
            r1 = ratio(skel(row["norm"]), skel(cands[0]["unicode"]))
            if r1 <= args.auto_lev:                      # near-exact -> free accept
                c = cands[0]
                write({"clip_id": cid, "repo": repo, "video_id": row["video_id"],
                       "raw_text": row["text"], "norm_text": row["norm"], "status": "accept",
                       "gold_text": c["unicode"], "line_id": c["line_id"],
                       "shabad_id": c["shabad_id"], "ang": c["ang"], "lev": c["lev"],
                       "via": "auto"}); counts["auto"] += 1; continue
            if r1 > args.drop_lev:                        # hopeless -> free drop
                write({"clip_id": cid, "repo": repo, "video_id": row["video_id"],
                       "raw_text": row["text"], "norm_text": row["norm"], "status": "drop",
                       "gold_text": "", "line_id": "", "shabad_id": "", "ang": "",
                       "lev": r1, "via": "far"}); continue
            pending.append(row)
            if len(pending) >= args.batch:
                flush_pending()
            if seen % 5000 == 0:
                log(f"  seen={seen} {counts}")
    flush_pending()
    for f in as_completed(futures):
        f.result()
    pool.shutdown(wait=True)
    out_fh.close()
    log(f"[done] seen={seen} {counts}  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
