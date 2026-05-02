#!/usr/bin/env python3
"""Stage 2 Batch-API pipeline — sequential corrector-then-reviewer + HF push.

Four subcommands run in order; state persists to a JSON file between them.

    submit   Run STTM Stage 1, build corrector JSONL from Stage 2 candidates,
             upload + create OpenAI batch. Saves state file.

    poll     Poll the batch every --poll-interval seconds until it lands
             (completed / failed / expired). Updates state file.

    resolve  Download corrector JSONL, parse responses, run the sync reviewer
             on non-noop rows, apply the validator, and write the final
             parquet with `stage2_*` columns + `training_usable`.

    push     Upload the resolved parquet to HF as a new `*-canonical` repo
             (metadata-only by default, or full-merge with audio from the
             source repo if --mode full).

Typical run (kirtan):

    # 1. Submit — builds JSONL from STTM Stage 1 candidates, uploads to OpenAI.
    python scripts/stage2_batch.py submit \\
        --input-parquet kirtan_clean.parquet --dataset kirtan \\
        --stage1-parquet kirtan_stage1.parquet \\
        --state-file kirtan_batch.state.json

    # 2. Poll — waits until OpenAI finishes the batch (typically 2-8h).
    python scripts/stage2_batch.py poll \\
        --state-file kirtan_batch.state.json

    # 3. Resolve — downloads corrector output, runs sync reviewer + validator.
    python scripts/stage2_batch.py resolve \\
        --state-file kirtan_batch.state.json \\
        --output-parquet kirtan_stage2.parquet

    # 4. Push — upload to HF for training.
    python scripts/stage2_batch.py push \\
        --output-parquet kirtan_stage2.parquet \\
        --repo surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical

The `push` default (metadata-only) pushes a small text+flags parquet. Training
code joins against the source repo on `clip_id` for audio. Pass `--mode full
--source-repo <original>` to stream the source + merge audio into the output;
slower but self-contained.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from canonical.batch_submit import (
    build_corrector_jsonl,
    build_reviewer_jsonl_gemini,
    download_batch_output,
    download_gemini_batch_output,
    parse_corrector_responses,
    parse_reviewer_responses_gemini,
    poll_batch,
    poll_gemini_batch,
    submit_batch,
    submit_gemini_batch,
)
from canonical.config import get_dataset_config
from canonical.corrector import CorrectorConfig
from canonical.pipeline import CanonicalPipeline
from canonical.reviewer import ReviewerConfig, review_batch
from canonical.stage2 import (
    _pick_candidate, _shabad_lines_text, _shabad_tokens,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


USABLE_DECISIONS = {
    "sttm_exact", "sttm_matched", "sttm_replaced", "sttm_simran",
    "replaced_llm", "replaced_reviewed", "unchanged_noop",
}


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def cmd_submit(args: argparse.Namespace) -> None:
    cfg = get_dataset_config(args.dataset)
    pipe = CanonicalPipeline(cfg, db_path=Path(args.db))

    stage1_parquet = Path(args.stage1_parquet)
    if args.reuse_stage1 and stage1_parquet.exists():
        print(f"[submit] --reuse-stage1: loading cached stage1 from {stage1_parquet}",
              file=sys.stderr)
        stage1 = pd.read_parquet(stage1_parquet).to_dict("records")
        print(f"[submit] stage1 snapshot has {len(stage1)} rows", file=sys.stderr)
    else:
        df = pd.read_parquet(args.input_parquet)
        print(f"[submit] loaded {len(df)} rows from {args.input_parquet}",
              file=sys.stderr)

        if "drop_candidate" in df.columns:
            before = len(df)
            df = df[~df["drop_candidate"].fillna(False).astype(bool)].reset_index(drop=True)
            print(f"[submit] dropped {before - len(df)} drop_candidate rows",
                  file=sys.stderr)

        records = df.to_dict(orient="records")
        if args.source_col != "text":
            if args.source_col not in df.columns:
                print(f"[submit] ERROR: column '{args.source_col}' missing. "
                      f"Columns: {list(df.columns)}", file=sys.stderr)
                sys.exit(2)
            for r in records:
                r.setdefault("raw_text", r.get("text"))
                r["text"] = r[args.source_col]

        print(f"[submit] running STTM Stage 1 on {len(records)} rows...",
              file=sys.stderr)
        stage1 = pipe.run(records)
        print(f"[submit] stage1 produced {len(stage1)} rows", file=sys.stderr)

        # Persist stage1 snapshot for resolve time. Strip internal `_` fields.
        stage1_df = pd.DataFrame([
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in stage1
        ])
        stage1_df.to_parquet(stage1_parquet)
        print(f"[submit] stage1 snapshot → {stage1_parquet}", file=sys.stderr)

    dh = Counter(r["decision"] for r in stage1)
    print(f"[submit] stage1 decisions: {dict(dh)}", file=sys.stderr)

    # Filter to Stage 2 candidates: review, or rescue-unchanged with shabad.
    candidates = [
        r for r in stage1
        if (r["decision"] == "review") or
           (r["decision"] == "unchanged" and r.get("shabad_id"))
    ]
    print(f"[submit] {len(candidates)} Stage 2 candidates "
          f"({100*len(candidates)/max(len(stage1),1):.1f}% of rows)",
          file=sys.stderr)
    if not candidates:
        print("[submit] nothing to send — all rows STTM-confident", file=sys.stderr)
        state = {"status": "no_candidates", "candidates_count": 0}
        Path(args.state_file).write_text(json.dumps(state, indent=2))
        return

    shabad_lines_map = {
        sid: _shabad_lines_text(lines)
        for sid, lines in pipe.shabad_lines.items()
    }

    corrector_cfg = CorrectorConfig(
        model=args.corrector_model,
        batch_size=args.corrector_batch_size,
    )
    jsonl_path = Path(args.jsonl_path or (str(args.state_file) + ".input.jsonl"))
    custom_id_to_clips = build_corrector_jsonl(
        candidates, shabad_lines_map, jsonl_path, corrector_cfg,
    )
    print(f"[submit] wrote {len(custom_id_to_clips)} corrector prompts → {jsonl_path}",
          file=sys.stderr)

    if args.dry_run:
        print("[submit] --dry-run: not submitting to OpenAI", file=sys.stderr)
        return

    print(f"[submit] submitting to OpenAI Batch API...", file=sys.stderr)
    res = submit_batch(
        jsonl_path,
        completion_window=args.completion_window,
        metadata={"dataset": args.dataset},
    )
    print(f"[submit] batch_id={res['batch_id']} input_file={res['input_file_id']} "
          f"status={res['status']}", file=sys.stderr)

    state = {
        "dataset": args.dataset,
        "source_col": args.source_col,
        "input_parquet": str(args.input_parquet),
        "stage1_parquet": str(args.stage1_parquet),
        "jsonl_path": str(jsonl_path),
        "batch_id": res["batch_id"],
        "input_file_id": res["input_file_id"],
        "status": res["status"],
        "output_file_id": None,
        "error_file_id": None,
        "custom_id_to_clips": custom_id_to_clips,
        "candidates_count": len(candidates),
        "corrector_model": args.corrector_model,
        "corrector_batch_size": args.corrector_batch_size,
        "completion_window": args.completion_window,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(args.state_file).write_text(json.dumps(state, indent=2, ensure_ascii=False))
    print(f"[submit] state → {args.state_file}", file=sys.stderr)


# ---------------------------------------------------------------------------
# poll
# ---------------------------------------------------------------------------

def cmd_poll(args: argparse.Namespace) -> None:
    state = json.loads(Path(args.state_file).read_text())
    if state.get("status") == "no_candidates":
        print("[poll] no candidates to process — nothing to poll", file=sys.stderr)
        return

    batch_id = state["batch_id"]
    interval = args.poll_interval
    terminal = {"completed", "failed", "cancelled", "expired"}
    while True:
        info = poll_batch(batch_id)
        ts = time.strftime("%H:%M:%S", time.localtime())
        counts = info.get("request_counts") or {}
        print(f"[poll {ts}] batch {batch_id} status={info['status']} counts={counts}",
              file=sys.stderr)
        if info["status"] in terminal:
            state["status"] = info["status"]
            state["output_file_id"] = info.get("output_file_id")
            state["error_file_id"] = info.get("error_file_id")
            state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            Path(args.state_file).write_text(json.dumps(state, indent=2, ensure_ascii=False))
            print(f"[poll] terminal status={info['status']} — state updated",
                  file=sys.stderr)
            if info["status"] != "completed":
                sys.exit(3)
            return
        if args.once:
            return
        time.sleep(interval)


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------

def _build_stage2_outcome_corrector_missing(caption: str, model: str) -> dict:
    return {
        "stage2_decision": "error_fallback",
        "stage2_text": caption,
        "stage2_corrector_output": None,
        "stage2_reviewer_verdict": None,
        "stage2_reviewer_final": None,
        "stage2_changed_tokens": [],
        "stage2_reasons": ["corrector_missing_in_batch_output"],
        "stage2_corrector_model": model,
        "stage2_reviewer_model": None,
    }


def _build_stage2_outcome_noop(caption: str, corrected: str, model: str) -> dict:
    return {
        "stage2_decision": "unchanged_noop",
        "stage2_text": caption,
        "stage2_corrector_output": corrected,
        "stage2_reviewer_verdict": "APPROVE_AUTO",
        "stage2_reviewer_final": None,
        "stage2_changed_tokens": [],
        "stage2_reasons": ["corrector returned caption verbatim"],
        "stage2_corrector_model": model,
        "stage2_reviewer_model": None,
    }


def _build_stage2_outcome_passthrough(r: dict) -> dict:
    dec = r["decision"]
    if dec in ("exact", "matched", "replaced", "simran"):
        return {
            "stage2_decision": f"sttm_{dec}",
            "stage2_text": r["final_text"],
            "stage2_corrector_output": None,
            "stage2_reviewer_verdict": None,
            "stage2_reviewer_final": None,
            "stage2_changed_tokens": [],
            "stage2_reasons": [f"sttm passthrough: {dec}"],
            "stage2_corrector_model": None,
            "stage2_reviewer_model": None,
        }
    # unchanged without shabad → low-confidence passthrough, not usable.
    return {
        "stage2_decision": "sttm_unchanged_nocontext",
        "stage2_text": r["final_text"],
        "stage2_corrector_output": None,
        "stage2_reviewer_verdict": None,
        "stage2_reviewer_final": None,
        "stage2_changed_tokens": [],
        "stage2_reasons": ["no shabad retrieved"],
        "stage2_corrector_model": None,
        "stage2_reviewer_model": None,
    }


def cmd_resolve(args: argparse.Namespace) -> None:
    """Resolve a stage2 batch run.

    Two modes — auto-detected from state:
      reviewer_batch_name in state → Gemini Batch output (hands-off mode)
      else                          → run sync reviewer locally
    """
    state = json.loads(Path(args.state_file).read_text())
    if state.get("status") == "no_candidates":
        print("[resolve] no candidates were sent — nothing to resolve", file=sys.stderr)
        stage1 = pd.read_parquet(state["stage1_parquet"]).to_dict("records")
        _write_output(stage1, {}, Path(args.output_parquet))
        return

    if state.get("status") != "completed":
        print(f"[resolve] corrector batch status={state.get('status')} — run 'poll' first",
              file=sys.stderr)
        sys.exit(2)

    corr_out = Path(args.output_jsonl or (str(args.state_file) + ".output.jsonl"))
    if args.force_download or not corr_out.exists():
        print(f"[resolve] downloading corrector output → {corr_out}",
              file=sys.stderr)
        download_batch_output(state["output_file_id"], corr_out)
    else:
        print(f"[resolve] cached corrector output at {corr_out}", file=sys.stderr)

    corrected, corr_errors = parse_corrector_responses(corr_out)
    print(f"[resolve] parsed {len(corrected)} corrected clip_ids; "
          f"{len(corr_errors)} prompt-level errors", file=sys.stderr)

    stage1 = pd.read_parquet(state["stage1_parquet"]).to_dict("records")
    candidates = [
        r for r in stage1
        if (r["decision"] == "review") or
           (r["decision"] == "unchanged" and r.get("shabad_id"))
    ]
    print(f"[resolve] {len(candidates)} Stage 2 candidates", file=sys.stderr)

    pipe_cfg = get_dataset_config(state["dataset"])
    pipe = CanonicalPipeline(pipe_cfg, db_path=Path(args.db))
    shabad_lines_by_sid = pipe.shabad_lines

    corrector_model = state.get("corrector_model", "gpt-5-nano")
    stage2_outcomes: dict[str, dict] = {}
    needs_review: dict[str, list[dict]] = {}

    for r in candidates:
        cid = r["clip_id"]
        caption = r.get("text_cleaned") or r.get("text") or ""
        sid = r.get("shabad_id") or ""

        corr_text = corrected.get(cid)
        if corr_text is None:
            stage2_outcomes[cid] = _build_stage2_outcome_corrector_missing(
                caption, corrector_model,
            )
            continue
        if corr_text.strip() == caption.strip():
            stage2_outcomes[cid] = _build_stage2_outcome_noop(
                caption, corr_text, corrector_model,
            )
            continue
        needs_review.setdefault(sid, []).append({
            "clip_id": cid,
            "caption": caption,
            "corrector_output": corr_text,
        })

    total_review_items = sum(len(v) for v in needs_review.values())
    print(f"[resolve] {total_review_items} rows need reviewer; "
          f"{len(stage2_outcomes) - total_review_items} already resolved "
          f"(noop / corrector-missing)", file=sys.stderr)

    # Get reviewer outputs — either from Gemini Batch (hands-off) or sync.
    reviewed_per_shabad: dict[str, dict[str, dict]] = {}
    reviewer_model_used = args.reviewer_model

    if state.get("reviewer_status") == "completed" and state.get("reviewer_dest_file_name"):
        # --- Gemini Batch output path ---
        rev_out = Path(args.reviewer_output_jsonl or
                       (str(args.state_file) + ".reviewer.output.jsonl"))
        if args.force_download or not rev_out.exists():
            print(f"[resolve] downloading reviewer (gemini batch) output → {rev_out}",
                  file=sys.stderr)
            download_gemini_batch_output(state["reviewer_dest_file_name"], rev_out)
        else:
            print(f"[resolve] cached reviewer output at {rev_out}", file=sys.stderr)
        reviewed_flat, rev_errors = parse_reviewer_responses_gemini(rev_out)
        print(f"[resolve] parsed {len(reviewed_flat)} reviewer responses; "
              f"{len(rev_errors)} error prompts", file=sys.stderr)
        reviewer_model_used = state.get("reviewer_model", reviewer_model_used)
        # Regroup by shabad for downstream merging
        for sid, items in needs_review.items():
            out = {}
            for it in items:
                r = reviewed_flat.get(it["clip_id"])
                if r is None:
                    out[it["clip_id"]] = {
                        "verdict": "REJECT",
                        "final_text": it["caption"],
                        "changed_tokens": [],
                        "reasons": ["reviewer_missing_in_batch_output"],
                        "model": reviewer_model_used,
                        "ok": False,
                        "error": "missing",
                    }
                else:
                    out[it["clip_id"]] = {**r, "model": reviewer_model_used,
                                          "ok": True, "error": None}
            reviewed_per_shabad[sid] = out
    elif state.get("reviewer_status") == "no_reviewer_needed" or not needs_review:
        print("[resolve] no reviewer output needed", file=sys.stderr)
    else:
        # --- sync reviewer fallback ---
        reviewer_cfg = ReviewerConfig(
            model=args.reviewer_model,
            batch_size=args.reviewer_batch_size,
        )

        def _process_shabad(sid: str, items: list[dict]):
            lines_text = _shabad_lines_text(shabad_lines_by_sid.get(sid, []))
            if not lines_text:
                out = {it["clip_id"]: {
                    "verdict": "REJECT",
                    "final_text": it["caption"],
                    "changed_tokens": [],
                    "reasons": ["no_shabad_lines_at_resolve"],
                    "model": reviewer_cfg.model,
                    "ok": False,
                    "error": "no_shabad_lines",
                } for it in items}
                return sid, out
            return sid, review_batch(items, lines_text, reviewer_cfg)

        print(f"[resolve] SYNC reviewer on {len(needs_review)} shabad groups, "
              f"{args.max_workers} workers...", file=sys.stderr)
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = [ex.submit(_process_shabad, sid, items)
                    for sid, items in needs_review.items()]
            for f in futs:
                sid, result = f.result()
                reviewed_per_shabad[sid] = result
        print(f"[resolve] sync reviewer done in {time.monotonic()-t0:.1f}s",
              file=sys.stderr)

    # Merge reviewer outputs + validator
    for sid, items in needs_review.items():
        reviewed = reviewed_per_shabad.get(sid, {})
        shabad_toks = _shabad_tokens(shabad_lines_by_sid.get(sid, []))
        for item in items:
            cid = item["clip_id"]
            rreq = reviewed.get(cid, {})
            text, decision, src, log = _pick_candidate(
                caption=item["caption"],
                corrector_output=item["corrector_output"],
                reviewer_result=rreq,
                shabad_toks=shabad_toks,
            )
            stage2_outcomes[cid] = {
                "stage2_decision": decision,
                "stage2_text": text,
                "stage2_corrector_output": item["corrector_output"],
                "stage2_reviewer_verdict": rreq.get("verdict"),
                "stage2_reviewer_final": rreq.get("final_text"),
                "stage2_changed_tokens": rreq.get("changed_tokens") or [],
                "stage2_reasons": (rreq.get("reasons") or []) + log,
                "stage2_corrector_model": corrector_model,
                "stage2_reviewer_model": rreq.get("model") or reviewer_model_used,
            }

    _write_output(stage1, stage2_outcomes, Path(args.output_parquet))


# ---------------------------------------------------------------------------
# submit-reviewer — build Gemini Batch from corrector output
# ---------------------------------------------------------------------------

def cmd_submit_reviewer(args: argparse.Namespace) -> None:
    """After corrector batch completes, build the reviewer JSONL covering
    only rows where corrector_output != caption (the others are noop and
    skip reviewer). Upload + submit to Gemini Batch API."""
    state = json.loads(Path(args.state_file).read_text())
    if state.get("status") != "completed":
        print(f"[submit-reviewer] corrector batch status={state.get('status')} — "
              f"wait for poll", file=sys.stderr)
        sys.exit(2)
    if state.get("reviewer_batch_name"):
        print(f"[submit-reviewer] reviewer batch already submitted: "
              f"{state['reviewer_batch_name']}", file=sys.stderr)
        return

    # Download corrector output if not already local
    corr_out = Path(args.corrector_output_jsonl or (str(args.state_file) + ".output.jsonl"))
    if not corr_out.exists() or args.force_download_corrector:
        print(f"[submit-reviewer] downloading corrector output → {corr_out}",
              file=sys.stderr)
        download_batch_output(state["output_file_id"], corr_out)

    corrected, corr_errors = parse_corrector_responses(corr_out)
    print(f"[submit-reviewer] corrector: {len(corrected)} OK, "
          f"{len(corr_errors)} error prompts", file=sys.stderr)

    # Load stage1 to filter candidates that need review
    stage1 = pd.read_parquet(state["stage1_parquet"]).to_dict("records")
    candidates = [
        r for r in stage1
        if (r["decision"] == "review") or
           (r["decision"] == "unchanged" and r.get("shabad_id"))
    ]

    # Build reviewer batch: only non-noop rows
    needs_review: dict[str, list[dict]] = {}
    for r in candidates:
        cid = r["clip_id"]
        caption = r.get("text_cleaned") or r.get("text") or ""
        corr_text = corrected.get(cid)
        if corr_text is None:
            continue  # corrector missing → error_fallback at resolve
        if corr_text.strip() == caption.strip():
            continue  # noop, skip reviewer
        sid = r.get("shabad_id") or ""
        needs_review.setdefault(sid, []).append({
            "clip_id": cid,
            "caption": caption,
            "corrector_output": corr_text,
        })
    review_count = sum(len(v) for v in needs_review.values())
    print(f"[submit-reviewer] {review_count} rows need reviewer "
          f"across {len(needs_review)} shabad groups", file=sys.stderr)
    if review_count == 0:
        state["reviewer_status"] = "no_reviewer_needed"
        Path(args.state_file).write_text(json.dumps(state, indent=2, ensure_ascii=False))
        print("[submit-reviewer] no rows to review — marking complete",
              file=sys.stderr)
        return

    # Need shabad_lines for prompt construction
    cfg = get_dataset_config(state["dataset"])
    pipe = CanonicalPipeline(cfg, db_path=Path(args.db))
    shabad_lines_map = {
        sid: _shabad_lines_text(lines)
        for sid, lines in pipe.shabad_lines.items()
    }

    reviewer_cfg = ReviewerConfig(
        model=args.reviewer_model,
        batch_size=args.reviewer_batch_size,
    )
    jsonl_path = Path(args.reviewer_jsonl_path or
                      (str(args.state_file) + ".reviewer.input.jsonl"))
    custom_key_to_clips = build_reviewer_jsonl_gemini(
        needs_review, shabad_lines_map, jsonl_path, reviewer_cfg,
    )
    print(f"[submit-reviewer] wrote {len(custom_key_to_clips)} reviewer prompts → "
          f"{jsonl_path}", file=sys.stderr)

    if args.dry_run:
        print("[submit-reviewer] --dry-run: not submitting to Gemini", file=sys.stderr)
        return

    res = submit_gemini_batch(jsonl_path, model=reviewer_cfg.model,
                               display_name=f"stage2-reviewer-{state['dataset']}")
    state["reviewer_batch_name"] = res["batch_name"]
    state["reviewer_input_file_name"] = res["input_file_name"]
    state["reviewer_state"] = res["state"]
    state["reviewer_jsonl_path"] = str(jsonl_path)
    state["reviewer_custom_key_to_clips"] = custom_key_to_clips
    state["reviewer_model"] = reviewer_cfg.model
    state["reviewer_submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(args.state_file).write_text(json.dumps(state, indent=2, ensure_ascii=False))
    print(f"[submit-reviewer] batch={res['batch_name']} state={res['state']}",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# poll-reviewer
# ---------------------------------------------------------------------------

def cmd_poll_reviewer(args: argparse.Namespace) -> None:
    state = json.loads(Path(args.state_file).read_text())
    if state.get("reviewer_status") == "no_reviewer_needed":
        print("[poll-reviewer] no reviewer batch — nothing to poll", file=sys.stderr)
        return
    if not state.get("reviewer_batch_name"):
        print("[poll-reviewer] reviewer batch not submitted — run submit-reviewer first",
              file=sys.stderr)
        sys.exit(2)

    batch_name = state["reviewer_batch_name"]
    interval = args.poll_interval
    terminal_ok = {"JOB_STATE_SUCCEEDED"}
    terminal_fail = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    while True:
        info = poll_gemini_batch(batch_name)
        ts = time.strftime("%H:%M:%S", time.localtime())
        counts = info.get("request_counts") or {}
        print(f"[poll-reviewer {ts}] batch {batch_name} state={info['state']} "
              f"counts={counts}", file=sys.stderr)
        state["reviewer_state"] = info["state"]
        state["reviewer_dest_file_name"] = info.get("dest_file_name")
        if info["state"] in terminal_ok:
            state["reviewer_status"] = "completed"
            state["reviewer_completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            Path(args.state_file).write_text(json.dumps(state, indent=2, ensure_ascii=False))
            print(f"[poll-reviewer] ✅ completed", file=sys.stderr)
            return
        if info["state"] in terminal_fail:
            state["reviewer_status"] = info["state"]
            Path(args.state_file).write_text(json.dumps(state, indent=2, ensure_ascii=False))
            print(f"[poll-reviewer] ❌ {info['state']}", file=sys.stderr)
            sys.exit(3)
        Path(args.state_file).write_text(json.dumps(state, indent=2, ensure_ascii=False))
        if args.once:
            return
        time.sleep(interval)


# ---------------------------------------------------------------------------
# push — upload the cleaned dataset to HF for training
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS = [
    "clip_id", "video_id",
    # Source columns preserved
    "text", "text_cleaned", "duration_s", "start_s", "end_s",
    "is_simran", "drop_candidate",
    # Stage 1 outputs
    "decision", "shabad_id", "sggs_line", "final_text", "match_score",
    # Stage 2 outputs (the whole point of this pipeline)
    "stage2_decision", "stage2_text",
    "stage2_corrector_output", "stage2_reviewer_verdict",
    "stage2_reviewer_final", "stage2_changed_tokens", "stage2_reasons",
    "stage2_corrector_model", "stage2_reviewer_model",
    # Training flag
    "training_usable",
]


def cmd_push(args: argparse.Namespace) -> None:
    """Upload the stage2 output to HF. Two modes:

      --mode metadata-only  (default, fast)
        Push a text/metadata parquet to a new HF repo. No audio; training
        code joins against the source repo on `clip_id`.

      --mode full
        Stream the source HF repo, merge stage2 columns by clip_id, push
        with audio intact. Downloads audio shards — slow.
    """
    import os
    stage2_df = pd.read_parquet(args.output_parquet)
    print(f"[push] loaded {len(stage2_df)} rows from {args.output_parquet}",
          file=sys.stderr)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[push] ERROR: HF_TOKEN not set", file=sys.stderr)
        sys.exit(2)

    from datasets import Dataset

    if args.mode == "metadata-only":
        keep = [c for c in CANONICAL_COLUMNS if c in stage2_df.columns]
        meta = stage2_df[keep].copy()
        print(f"[push] metadata-only push: {len(meta)} rows, {len(keep)} columns",
              file=sys.stderr)
        ds = Dataset.from_pandas(meta, preserve_index=False)
        _hf_push(ds, args.repo, token, args.private)
        return

    # Full merge: pull audio from source, join stage2 by clip_id
    if not args.source_repo:
        print("[push] --source-repo required for full-mode merge", file=sys.stderr)
        sys.exit(2)

    from datasets import load_dataset, Audio

    print(f"[push] loading source {args.source_repo} (this can take a while)...",
          file=sys.stderr)
    src = load_dataset(args.source_repo, split=args.split)
    print(f"[push] source has {len(src)} rows", file=sys.stderr)

    stage2_by_clip = {r["clip_id"]: r for r in stage2_df.to_dict(orient="records")}
    merge_cols = [c for c in CANONICAL_COLUMNS if c not in (
        "clip_id", "video_id", "text", "duration_s", "start_s", "end_s",
    )]

    def _attach(row: dict) -> dict:
        cid = row.get("clip_id")
        stage2 = stage2_by_clip.get(cid)
        if stage2:
            for c in merge_cols:
                if c in stage2 and stage2[c] is not None:
                    row[c] = stage2[c]
        return row

    print(f"[push] merging stage2 columns...", file=sys.stderr)
    ds = src.map(_attach, desc="merge-stage2")
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    _hf_push(ds, args.repo, token, args.private)


def _hf_push(ds, repo: str, token: str, private: bool) -> None:
    print(f"[push] pushing {len(ds)} rows → {repo} (private={private})",
          file=sys.stderr)
    ds.push_to_hub(repo, token=token, private=private)
    print(f"[push] ✅ pushed to https://huggingface.co/datasets/{repo}",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# run — hands-off end-to-end orchestrator
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Fire-and-forget full pipeline: submit corrector → poll → submit
    reviewer → poll → resolve. Writes state + output parquet.

    Idempotent per phase — if interrupted, re-running resumes from the
    next phase by inspecting state file.
    """
    from types import SimpleNamespace
    state_path = Path(args.state_file)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    # Phase 1: corrector submit
    if not state.get("batch_id"):
        print("[run] phase 1: submit corrector", file=sys.stderr)
        sub_args = SimpleNamespace(
            input_parquet=args.input_parquet, dataset=args.dataset,
            db=args.db, source_col=args.source_col,
            stage1_parquet=args.stage1_parquet, state_file=args.state_file,
            jsonl_path=None, corrector_model=args.corrector_model,
            corrector_batch_size=args.corrector_batch_size,
            completion_window="24h", reuse_stage1=args.reuse_stage1,
            dry_run=False,
        )
        cmd_submit(sub_args)
        state = json.loads(state_path.read_text())
        if state.get("status") == "no_candidates":
            print("[run] no candidates — writing output parquet and exiting",
                  file=sys.stderr)
            _resolve_and_write_empty(args)
            return

    # Phase 2: poll corrector
    if state.get("status") != "completed":
        print("[run] phase 2: poll corrector batch", file=sys.stderr)
        poll_args = SimpleNamespace(
            state_file=args.state_file, poll_interval=args.poll_interval,
            once=False,
        )
        cmd_poll(poll_args)
        state = json.loads(state_path.read_text())

    # Phase 3: submit reviewer
    if not state.get("reviewer_batch_name") and \
            state.get("reviewer_status") != "no_reviewer_needed":
        print("[run] phase 3: submit reviewer", file=sys.stderr)
        sr_args = SimpleNamespace(
            state_file=args.state_file, db=args.db,
            corrector_output_jsonl=None, reviewer_jsonl_path=None,
            reviewer_model=args.reviewer_model,
            reviewer_batch_size=args.reviewer_batch_size,
            force_download_corrector=False, dry_run=False,
        )
        cmd_submit_reviewer(sr_args)
        state = json.loads(state_path.read_text())

    # Phase 4: poll reviewer
    if state.get("reviewer_status") not in ("completed", "no_reviewer_needed"):
        print("[run] phase 4: poll reviewer batch", file=sys.stderr)
        pr_args = SimpleNamespace(
            state_file=args.state_file, poll_interval=args.poll_interval,
            once=False,
        )
        cmd_poll_reviewer(pr_args)
        state = json.loads(state_path.read_text())

    # Phase 5: resolve (merge + validator + write)
    print("[run] phase 5: resolve + write output parquet", file=sys.stderr)
    res_args = SimpleNamespace(
        state_file=args.state_file, db=args.db,
        output_parquet=args.output_parquet, output_jsonl=None,
        reviewer_output_jsonl=None,
        reviewer_model=args.reviewer_model,
        reviewer_batch_size=args.reviewer_batch_size,
        max_workers=args.max_workers, force_download=False,
    )
    cmd_resolve(res_args)
    print(f"[run] ✅ done. output → {args.output_parquet}", file=sys.stderr)


def _resolve_and_write_empty(args: argparse.Namespace) -> None:
    """When no candidates needed LLM, still produce the output parquet."""
    state = json.loads(Path(args.state_file).read_text())
    stage1 = pd.read_parquet(state["stage1_parquet"]).to_dict("records")
    _write_output(stage1, {}, Path(args.output_parquet))


# ---------------------------------------------------------------------------
# run-sync — high-concurrency sync path, bypasses Batch API
# ---------------------------------------------------------------------------

def cmd_run_sync(args: argparse.Namespace) -> None:
    """Sync path for when Batch API enqueue limits force us to bypass it.
    Loads an existing stage1 parquet (no STTM rerun), runs corrector +
    reviewer per-row-batched with `--max-workers` threads."""
    from canonical.stage2 import Stage2Config, process_rows
    t0 = time.monotonic()

    stage1 = pd.read_parquet(args.stage1_parquet).to_dict("records")
    print(f"[run-sync] loaded {len(stage1)} rows from {args.stage1_parquet}",
          file=sys.stderr)
    dh = Counter(r["decision"] for r in stage1)
    print(f"[run-sync] stage1 decisions: {dict(dh)}", file=sys.stderr)

    candidates = [
        r for r in stage1
        if (r["decision"] == "review") or
           (r["decision"] == "unchanged" and r.get("shabad_id"))
    ]
    print(f"[run-sync] {len(candidates)} Stage 2 candidates "
          f"({100 * len(candidates) / max(len(stage1), 1):.1f}%)",
          file=sys.stderr)

    # Build shabad-lines index once
    cfg = get_dataset_config(args.dataset)
    pipe = CanonicalPipeline(cfg, db_path=Path(args.db))
    shabad_lines_map = pipe.shabad_lines

    stage2_cfg = Stage2Config(
        corrector=CorrectorConfig(
            model=args.corrector_model,
            batch_size=args.corrector_batch_size,
        ),
        reviewer=ReviewerConfig(
            model=args.reviewer_model,
            batch_size=args.reviewer_batch_size,
        ),
    )

    print(f"[run-sync] processing {len(candidates)} candidates with "
          f"{args.max_workers} workers...", file=sys.stderr)
    if args.checkpoint_path:
        print(f"[run-sync] checkpoint → {args.checkpoint_path}", file=sys.stderr)
    stage2_results = process_rows(
        candidates, shabad_lines_map, stage2_cfg,
        max_workers=args.max_workers,
        checkpoint_path=args.checkpoint_path,
    )
    elapsed = time.monotonic() - t0
    print(f"[run-sync] stage2 done in {elapsed:.1f}s "
          f"({len(candidates)/max(elapsed, 1):.2f} rows/s)", file=sys.stderr)

    # Build stage2_outcomes dict keyed by clip_id
    stage2_outcomes = {
        cand["clip_id"]: result
        for cand, result in zip(candidates, stage2_results)
    }

    _write_output(stage1, stage2_outcomes, Path(args.output_parquet))


def _write_output(
    stage1: list[dict],
    stage2_outcomes: dict[str, dict],
    output_parquet: Path,
) -> None:
    out_rows: list[dict] = []
    for r in stage1:
        cid = r.get("clip_id") or ""
        if cid in stage2_outcomes:
            outcome = stage2_outcomes[cid]
        else:
            outcome = _build_stage2_outcome_passthrough(r)
        merged = {**r, **outcome}
        merged["training_usable"] = outcome["stage2_decision"] in USABLE_DECISIONS
        out_rows.append(merged)

    out_df = pd.DataFrame(out_rows)
    out_df.to_parquet(output_parquet)
    print(f"[resolve] wrote {output_parquet} ({len(out_df)} rows)", file=sys.stderr)

    hist = Counter(r["stage2_decision"] for r in out_rows)
    usable = sum(1 for r in out_rows if r["training_usable"])
    print(f"\n=== Stage 2 outcome histogram ({len(out_rows)} rows) ===")
    for k, v in sorted(hist.items()):
        tag = "[USABLE]" if k in USABLE_DECISIONS else "[skip  ]"
        print(f"  {tag} {k:30s} {v}")
    pct = 100 * usable / max(len(out_rows), 1)
    print(f"\ntraining_usable: {usable}/{len(out_rows)} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit",
                       help="Run Stage 1, build corrector JSONL, submit batch")
    s.add_argument("--input-parquet", required=True)
    s.add_argument("--dataset", choices=["kirtan", "sehaj"], required=True)
    s.add_argument("--db", default="database.sqlite")
    s.add_argument("--source-col", default="text_cleaned")
    s.add_argument("--stage1-parquet", required=True,
                   help="Where to save Stage 1 output for resolve")
    s.add_argument("--state-file", required=True,
                   help="Where to save the batch state JSON")
    s.add_argument("--jsonl-path", default=None,
                   help="JSONL output path (default: <state-file>.input.jsonl)")
    s.add_argument("--corrector-model", default="gpt-5-nano")
    s.add_argument("--corrector-batch-size", type=int, default=15)
    s.add_argument("--completion-window", default="24h",
                   choices=["24h"],
                   help="OpenAI Batch API completion window")
    s.add_argument("--reuse-stage1", action="store_true",
                   help="If stage1-parquet already exists, skip re-running STTM")
    s.add_argument("--dry-run", action="store_true",
                   help="Build JSONL but don't submit to OpenAI")
    s.set_defaults(func=cmd_submit)

    p = sub.add_parser("poll", help="Poll the batch until terminal status")
    p.add_argument("--state-file", required=True)
    p.add_argument("--poll-interval", type=int, default=600,
                   help="seconds between status checks (default 600 = 10 min)")
    p.add_argument("--once", action="store_true",
                   help="Check status once and exit (for cron-style use)")
    p.set_defaults(func=cmd_poll)

    r = sub.add_parser("resolve",
                       help="Download + sync reviewer + validator + write final parquet")
    r.add_argument("--state-file", required=True)
    r.add_argument("--db", default="database.sqlite")
    r.add_argument("--output-parquet", required=True)
    r.add_argument("--output-jsonl", default=None,
                   help="Cache path for downloaded corrector output")
    r.add_argument("--reviewer-output-jsonl", default=None,
                   help="Cache path for Gemini reviewer batch output (batch mode)")
    r.add_argument("--reviewer-model", default="gemini-3-flash-lite")
    r.add_argument("--reviewer-batch-size", type=int, default=5)
    r.add_argument("--max-workers", type=int, default=10,
                   help="Concurrent shabad-groups for the sync reviewer")
    r.add_argument("--force-download", action="store_true")
    r.set_defaults(func=cmd_resolve)

    sr = sub.add_parser("submit-reviewer",
                        help="After corrector batch done, build + submit Gemini reviewer batch")
    sr.add_argument("--state-file", required=True)
    sr.add_argument("--db", default="database.sqlite")
    sr.add_argument("--corrector-output-jsonl", default=None,
                    help="Cache path for corrector output (default: <state>.output.jsonl)")
    sr.add_argument("--reviewer-jsonl-path", default=None)
    sr.add_argument("--reviewer-model", default="gemini-3-flash-lite")
    sr.add_argument("--reviewer-batch-size", type=int, default=5)
    sr.add_argument("--force-download-corrector", action="store_true")
    sr.add_argument("--dry-run", action="store_true",
                    help="Build JSONL but don't submit to Gemini")
    sr.set_defaults(func=cmd_submit_reviewer)

    pr = sub.add_parser("poll-reviewer", help="Poll the Gemini reviewer batch")
    pr.add_argument("--state-file", required=True)
    pr.add_argument("--poll-interval", type=int, default=600)
    pr.add_argument("--once", action="store_true")
    pr.set_defaults(func=cmd_poll_reviewer)

    rn = sub.add_parser("run",
                        help="Hands-off: submit → poll → submit-reviewer → poll-reviewer → resolve")
    rn.add_argument("--input-parquet", required=True)
    rn.add_argument("--dataset", choices=["kirtan", "sehaj"], required=True)
    rn.add_argument("--db", default="database.sqlite")
    rn.add_argument("--source-col", default="text_cleaned")
    rn.add_argument("--stage1-parquet", required=True)
    rn.add_argument("--state-file", required=True)
    rn.add_argument("--output-parquet", required=True)
    rn.add_argument("--corrector-model", default="gpt-5-nano")
    rn.add_argument("--corrector-batch-size", type=int, default=15)
    rn.add_argument("--reviewer-model", default="gemini-3-flash-lite")
    rn.add_argument("--reviewer-batch-size", type=int, default=5)
    rn.add_argument("--poll-interval", type=int, default=600)
    rn.add_argument("--reuse-stage1", action="store_true")
    rn.add_argument("--max-workers", type=int, default=10)
    rn.set_defaults(func=cmd_run)

    rs = sub.add_parser("run-sync",
                        help="Sync pipeline with high worker count. Skips Batch API "
                             "entirely — makes real-time sync calls through the existing "
                             "process_rows() with ThreadPoolExecutor. Use when your OpenAI "
                             "tier has low Batch-API enqueue limits.")
    rs.add_argument("--stage1-parquet", required=True,
                    help="Enriched stage1 parquet (must have text_cleaned, decision, "
                         "shabad_id, final_text columns).")
    rs.add_argument("--dataset", choices=["kirtan", "sehaj"], required=True)
    rs.add_argument("--db", default="database.sqlite")
    rs.add_argument("--output-parquet", required=True)
    rs.add_argument("--corrector-model", default="gpt-5-nano")
    rs.add_argument("--corrector-batch-size", type=int, default=15)
    rs.add_argument("--reviewer-model", default="gemini-3-flash-lite")
    rs.add_argument("--reviewer-batch-size", type=int, default=5)
    rs.add_argument("--max-workers", type=int, default=50)
    rs.add_argument("--checkpoint-path", default=None,
                    help="JSONL file. If set, completed shabad results are "
                         "appended per line. Re-running with the same path "
                         "resumes from where a prior run left off.")
    rs.set_defaults(func=cmd_run_sync)

    u = sub.add_parser("push", help="Upload stage2 output to Hugging Face")
    u.add_argument("--output-parquet", required=True,
                   help="Parquet produced by `resolve`")
    u.add_argument("--repo", required=True,
                   help="Destination HF repo, e.g. "
                        "surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical")
    u.add_argument("--mode", choices=["metadata-only", "full"],
                   default="metadata-only",
                   help="metadata-only = text + stage2 columns only (fast). "
                        "full = stream source repo + merge audio (slow).")
    u.add_argument("--source-repo", default=None,
                   help="Source HF repo for full-mode merge (needed only if "
                        "--mode full)")
    u.add_argument("--split", default="train")
    u.add_argument("--private", action="store_true",
                   help="Create destination repo as private")
    u.set_defaults(func=cmd_push)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
