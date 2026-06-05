"""Phase 2: LLM rescue of skeleton-residual clips.

Reads the residual JSONL produced by `clean_v4_skeleton_pass.py`, groups
entries by `shabad_id` (so one shabad's pangti context grounds many
captions), batches 15 per call, runs N parallel `claude -p` workers
against `claude-sonnet-4-6`, parses the JSON response, then pulls audio
bytes from the source HF repo and pushes a `clean_phase2_<ts>` split to
the dest `*-clean` repo with `quality` ∈ {medium, low}.

Resumable via a `<dataset>_phase2_done.txt` sentinel of completed
batch IDs.

Critical prompt rule (anti wrong-swap, from PLAN.md):

  - matra differences are NEVER worth correcting (training uses
    normalized Gurbani CER which strips matras)
  - if a caption token is a valid Gurmukhi word, prefer caption verbatim
  - only replace consonant-level malformations / mixed-script garble
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SYSTEM_PROMPT = """You are correcting Gurmukhi YouTube captions of Gurbani recitation against a known shabad context. Read all rules before responding.

1. Training uses NORMALIZED Gurbani CER which strips matras (aunkar, bindi, tippi, halant, sihari, bihari, dulainkar) before scoring. Matra differences are NEVER worth correcting — only CONSONANT-level errors.

2. If a caption token is a valid Gurmukhi word recognized in any SGGS edition, Dasam Granth, or nitnem bani, prefer the caption VERBATIM. The shabad context is grounding for what's being sung, not a forced template — kirtanis often sing alternate / paraphrased / repeated lines.

3. Only replace a token when it is clearly MALFORMED: broken characters, mixed Latin/Devanagari script, ASR garble like stray single Latin letters embedded inside Gurmukhi, obvious encoding artifacts (mojibake).

4. Default action is status="keep" with verbatim text. Use status="correct" ONLY for substantive consonant fixes. Use status="reject" only if the caption is irretrievable (mostly non-Gurmukhi, total garble, off-topic English speech).

5. Output format: a JSON array, one entry per input caption, in INPUT ORDER. No prose, no commentary, no code fences.

   [{"clip_id": "<id>", "status": "keep"|"correct"|"reject", "text": "<gurmukhi or empty>"}]

   - status="keep"    → text = caption verbatim
   - status="correct" → text = your corrected Gurmukhi
   - status="reject"  → text = ""
"""


USER_TEMPLATE = """Shabad context (canonical SGGS pangtis):
{context}

Captions to evaluate ({n} clips):
{captions}

Respond ONLY with the JSON array."""


def format_user_prompt(shabad_pangtis: list[str], entries: list[dict]) -> str:
    if shabad_pangtis:
        ctx = "\n".join(f"- {p}" for p in shabad_pangtis[:8])
    else:
        ctx = "(no shabad context — judge each caption on its own merits)"
    cap_lines = []
    for e in entries:
        cid = e["clip_id"]
        cap = (e.get("original_text") or "").replace("\n", " ").strip()
        cap_lines.append(f'  {{"clip_id": {json.dumps(cid)}, "caption": {json.dumps(cap, ensure_ascii=False)}}}')
    captions = "[\n" + ",\n".join(cap_lines) + "\n]"
    return USER_TEMPLATE.format(context=ctx, captions=captions, n=len(entries))


def call_claude(prompt: str, model: str, timeout: int = 240) -> str:
    """Invoke `claude -p` non-interactively with the prompt on stdin."""
    full = SYSTEM_PROMPT + "\n\n---\n\n" + prompt
    cmd = ["claude", "-p", "--model", model]
    proc = subprocess.run(
        cmd, input=full, capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude exit={proc.returncode} stderr={proc.stderr[:400]}")
    return proc.stdout.strip()


def parse_response(raw: str) -> list[dict]:
    t = raw.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    # Sometimes assistants prepend prose; try to find the first '[' and matching ']'
    if not t.startswith("["):
        i = t.find("[")
        if i >= 0:
            t = t[i:]
    return json.loads(t)


def process_batch(entries: list[dict], pangtis: list[str], model: str) -> list[dict]:
    user = format_user_prompt(pangtis, entries)
    raw = call_claude(user, model)
    parsed = parse_response(raw)
    by_id = {r.get("clip_id"): r for r in parsed if isinstance(r, dict)}
    out = []
    for e in entries:
        r = by_id.get(e["clip_id"])
        if not r or "status" not in r:
            out.append({"clip_id": e["clip_id"], "status": "reject", "text": ""})
        else:
            out.append({
                "clip_id": r["clip_id"],
                "status": r.get("status", "reject"),
                "text": r.get("text", "") or "",
            })
    return out


def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run(
    residual_path: Path,
    source_repo: str,
    dest_repo: str,
    model: str,
    workers: int,
    batch_size: int,
    done_path: Path,
    public: bool,
    dry_run: bool,
    limit_batches: int | None,
):
    print(f"[rescue] loading residual {residual_path}", flush=True)
    entries: list[dict] = []
    with residual_path.open() as fp:
        for line in fp:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"[rescue] {len(entries)} residual entries", flush=True)

    done_batches: set[str] = set()
    if done_path.exists():
        done_batches = {ln.strip() for ln in done_path.read_text().splitlines() if ln.strip()}
    print(f"[rescue] resume: {len(done_batches)} batches already done", flush=True)

    by_shabad: dict[str, list[dict]] = defaultdict(list)
    no_shabad: list[dict] = []
    pangtis_for_shabad: dict[str, list[str]] = {}
    for e in entries:
        sid = e.get("shabad_id")
        if sid:
            by_shabad[sid].append(e)
            if sid not in pangtis_for_shabad:
                pangtis_for_shabad[sid] = e.get("shabad_pangtis") or []
        else:
            no_shabad.append(e)
    print(
        f"[rescue] {len(by_shabad)} shabads grouped + {len(no_shabad)} ungrouped",
        flush=True,
    )

    batches: list[tuple[list[str], list[dict], str]] = []
    for sid, ents in by_shabad.items():
        pangtis = pangtis_for_shabad.get(sid, [])
        for i, group in enumerate(chunk(ents, batch_size)):
            bid = f"{sid}-{i}"
            if bid in done_batches:
                continue
            batches.append((pangtis, list(group), bid))
    for i, group in enumerate(chunk(no_shabad, batch_size)):
        bid = f"NOSHABAD-{i}"
        if bid in done_batches:
            continue
        batches.append(([], list(group), bid))
    if limit_batches:
        batches = batches[:limit_batches]
    print(
        f"[rescue] {len(batches)} batches to process ({sum(len(b[1]) for b in batches)} clips)",
        flush=True,
    )

    if dry_run:
        print(f"[rescue] dry-run: stopping before LLM calls", flush=True)
        return

    results_all: list[dict] = []
    done_fp = done_path.open("a")
    processed = 0
    failed = 0
    t0 = time.time()

    def worker(b):
        pangtis, ents, bid = b
        try:
            t = time.time()
            res = process_batch(ents, pangtis, model)
            return bid, res, time.time() - t, None
        except Exception as exc:
            return bid, [], 0.0, repr(exc)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(worker, b) for b in batches]
        for fut in as_completed(futs):
            bid, res, dt, err = fut.result()
            if err:
                failed += 1
                print(f"[rescue] FAIL {bid}: {err}", flush=True)
                continue
            results_all.extend(res)
            done_fp.write(bid + "\n")
            done_fp.flush()
            processed += 1
            if processed % 25 == 0:
                rate = processed / max(time.time() - t0, 1e-3)
                eta_min = (len(batches) - processed) / max(rate, 1e-6) / 60
                print(
                    f"[rescue] batches done={processed}/{len(batches)} "
                    f"failed={failed} rate={rate:.2f}/s eta={eta_min:.0f}min",
                    flush=True,
                )
    done_fp.close()

    by_status = Counter(r.get("status", "?") for r in results_all)
    print(f"[rescue] results: {dict(by_status)}", flush=True)

    if not results_all:
        print("[rescue] nothing to push", flush=True)
        return

    # Pull audio bytes back from source repo
    clip_to_result = {r["clip_id"]: r for r in results_all}
    target_ids = set(clip_to_result)
    print(
        f"[rescue] streaming {source_repo} to attach audio for {len(target_ids)} rescued clips",
        flush=True,
    )
    from datasets import Audio, Dataset, load_dataset

    src = load_dataset(source_repo, split="train", streaming=True)
    if "audio" in (src.features or {}):
        try:
            src = src.cast_column("audio", Audio(decode=False))
        except Exception as e:
            print(f"[rescue] cast audio decode=False failed: {e}", flush=True)
    rows_out: list[dict] = []
    seen = 0
    for row in src:
        cid = row.get("clip_id")
        if cid not in target_ids:
            continue
        r = clip_to_result[cid]
        status = r["status"]
        if status == "correct":
            text = r["text"] or row.get("text", "")
            quality = "medium"
        elif status == "keep":
            text = row.get("text", "")
            quality = "medium"
        else:
            text = row.get("text", "")
            quality = "low"
        rows_out.append({
            "audio": row.get("audio"),
            "text": text,
            "original_text": row.get("text"),
            "quality": quality,
            "video_id": row.get("video_id"),
            "start_s": row.get("start_s"),
            "end_s": row.get("end_s"),
            "duration_s": row.get("duration_s"),
            "channel": row.get("channel"),
            "clip_id": cid,
            "rescue_status": status,
        })
        seen += 1
        if seen >= len(target_ids):
            break
    print(f"[rescue] attached audio for {seen}/{len(target_ids)} clips", flush=True)

    if not rows_out:
        return

    from huggingface_hub import create_repo

    try:
        create_repo(
            dest_repo, repo_type="dataset", private=not public,
            exist_ok=True, token=os.environ.get("HF_TOKEN"),
        )
    except Exception as e:
        print(f"[rescue] create_repo: {e}", flush=True)

    ds_out = Dataset.from_list(rows_out)
    if "audio" in ds_out.column_names:
        try:
            ds_out = ds_out.cast_column("audio", Audio(sampling_rate=16000))
        except Exception as e:
            print(f"[rescue] cast audio failed: {e}", flush=True)
    split_name = f"clean_phase2_{time.strftime('%Y%m%d_%H%M%S')}"
    ds_out.push_to_hub(
        dest_repo, split=split_name, private=not public,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"[rescue] pushed split={split_name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual", type=Path, required=True)
    ap.add_argument("--source-repo", required=True,
                    help="HF source repo (audio bytes pulled from here)")
    ap.add_argument("--dest-repo", required=True,
                    help="HF dest *-clean repo (new split appended)")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--done-file", type=Path, required=True)
    ap.add_argument("--private", action="store_true",
                    help="Default is PUBLIC; pass --private to override.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-batches", type=int, default=None)
    args = ap.parse_args()

    run(
        residual_path=args.residual,
        source_repo=args.source_repo,
        dest_repo=args.dest_repo,
        model=args.model,
        workers=args.workers,
        batch_size=args.batch_size,
        done_path=args.done_file,
        public=not args.private,
        dry_run=args.dry_run,
        limit_batches=args.limit_batches,
    )


if __name__ == "__main__":
    main()
