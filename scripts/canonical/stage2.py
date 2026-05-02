"""Stage 2 driver: corrector → reviewer (second-layer corrector) → validator.

Flow per row:
  1. Corrector (GPT) produces text_A.
  2. If text_A == caption verbatim → fast-path `unchanged_noop`, skip reviewer.
  3. Reviewer (Gemini) always produces text_B with verdict APPROVE|IMPROVED|REJECT.
  4. Validator (pure code) checks candidates in priority order:
       [text_B, text_A, caption]
     First candidate that passes validation ships.

Two modes:
  - process_row  : one row at a time. Slow but simple; good for debugging.
  - process_rows : batched + concurrent. Groups rows by shabad_id, calls
                   corrector_batch + reviewer_batch once per group, uses a
                   ThreadPoolExecutor across groups. 5-10× faster, cheaper.

Final decision labels:
  - replaced_llm         : corrector's output shipped, reviewer APPROVED.
  - replaced_reviewed    : reviewer's improved output shipped (IMPROVED).
  - unchanged_rejected   : reviewer REJECTed and nothing safer available.
  - unchanged_noop       : corrector returned caption verbatim.
  - unchanged_nocontext  : no shabad retrieved — nothing to correct against.
  - validator_fallback   : LLM outputs failed validation — shipped caption.
  - error_fallback       : corrector or reviewer crashed — shipped caption.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .corrector import CorrectorConfig, correct, correct_batch
from .reviewer import ReviewerConfig, review, review_batch
from .sttm_index import SggsLine
from .validator import validate


@dataclass
class Stage2Config:
    corrector: CorrectorConfig = None
    reviewer: ReviewerConfig = None

    def __post_init__(self):
        if self.corrector is None:
            self.corrector = CorrectorConfig()
        if self.reviewer is None:
            self.reviewer = ReviewerConfig()


def _shabad_lines_text(lines: list[SggsLine]) -> list[str]:
    return [ln.unicode for ln in lines if ln.unicode]


def _shabad_tokens(lines: list[SggsLine]) -> set[str]:
    out: set[str] = set()
    for ln in lines:
        out.update(ln.tokens)
    return out


def process_row(
    row: dict,
    shabad_lines: list[SggsLine],
    cfg: Stage2Config,
    openai_client=None,
    gemini_client=None,
) -> dict:
    """Run one row through corrector → reviewer → validator."""
    caption = row.get("text_cleaned") or row.get("text") or ""
    sttm_hint = row.get("final_text") or row.get("canonical_text")
    lines_text = _shabad_lines_text(shabad_lines)
    shabad_toks = _shabad_tokens(shabad_lines)
    if not lines_text:
        return _no_context(caption)

    # --- Stage 2a: corrector -------------------------------------------------
    c = correct(caption, lines_text, cfg.corrector, sttm_hint=sttm_hint,
                client=openai_client)
    if not c["ok"]:
        return {
            "stage2_decision": "error_fallback",
            "stage2_text": caption,
            "stage2_corrector_output": None,
            "stage2_reviewer_verdict": None,
            "stage2_reviewer_final": None,
            "stage2_changed_tokens": [],
            "stage2_reasons": [f"corrector_error: {c['error']}"],
            "stage2_corrector_model": c["model"],
            "stage2_reviewer_model": None,
        }

    # Fast-path: corrector returned caption verbatim → nothing to review.
    if c["corrected"].strip() == caption.strip():
        return {
            "stage2_decision": "unchanged_noop",
            "stage2_text": caption,
            "stage2_corrector_output": c["corrected"],
            "stage2_reviewer_verdict": "APPROVE_AUTO",
            "stage2_reviewer_final": None,
            "stage2_changed_tokens": [],
            "stage2_reasons": ["corrector returned caption verbatim"],
            "stage2_corrector_model": c["model"],
            "stage2_reviewer_model": None,
        }

    # --- Stage 2b: reviewer --------------------------------------------------
    # Reviewer is intentionally blind to STTM's suggestion — its judgement
    # must be an independent second opinion so we get true diverse coverage.
    r = review(caption, c["corrected"], lines_text, cfg.reviewer,
               client=gemini_client)

    # --- Stage 2c: validator gate + candidate ranking ------------------------
    # Candidate ranking: prefer reviewer's final_text (has both corrector's
    # input + reviewer's expert layer), fall back to corrector, then caption.
    candidates: list[tuple[str, str, str]] = []
    if r.get("final_text"):
        # Reviewer's candidate. Label depends on verdict.
        label = {
            "APPROVE": "replaced_llm",
            "IMPROVED": "replaced_reviewed",
            "REJECT": "unchanged_rejected",
        }.get(r["verdict"], "error_fallback")
        candidates.append((r["final_text"], label, "reviewer"))
    candidates.append((c["corrected"], "replaced_llm", "corrector"))
    candidates.append((caption, "validator_fallback", "caption"))

    validation_log: list[str] = []
    for text, decision, src in candidates:
        ok, reasons = validate(caption, text, shabad_toks)
        if ok:
            return {
                "stage2_decision": decision,
                "stage2_text": text,
                "stage2_corrector_output": c["corrected"],
                "stage2_reviewer_verdict": r.get("verdict"),
                "stage2_reviewer_final": r.get("final_text"),
                "stage2_changed_tokens": r.get("changed_tokens") or [],
                "stage2_reasons": (r.get("reasons") or []) + validation_log + [
                    f"shipped_from={src}"
                ],
                "stage2_corrector_model": c["model"],
                "stage2_reviewer_model": r.get("model"),
            }
        validation_log.append(f"{src}_failed_validation: {reasons}")

    # Should not reach here — caption is always a valid candidate unless
    # the validator is buggy. Defensive fallback:
    return {
        "stage2_decision": "validator_fallback",
        "stage2_text": caption,
        "stage2_corrector_output": c["corrected"],
        "stage2_reviewer_verdict": r.get("verdict"),
        "stage2_reviewer_final": r.get("final_text"),
        "stage2_changed_tokens": r.get("changed_tokens") or [],
        "stage2_reasons": validation_log,
        "stage2_corrector_model": c["model"],
        "stage2_reviewer_model": r.get("model"),
    }


def _no_context(caption: str) -> dict:
    return {
        "stage2_decision": "unchanged_nocontext",
        "stage2_text": caption,
        "stage2_corrector_output": None,
        "stage2_reviewer_verdict": None,
        "stage2_reviewer_final": None,
        "stage2_changed_tokens": [],
        "stage2_reasons": ["no shabad_id on row"],
        "stage2_corrector_model": None,
        "stage2_reviewer_model": None,
    }


# -----------------------------------------------------------------------------
# Batched + concurrent driver
# -----------------------------------------------------------------------------


def _caption_of(row: dict) -> str:
    return row.get("text_cleaned") or row.get("text") or ""


def _sttm_hint_of(row: dict) -> str | None:
    return row.get("final_text") or row.get("canonical_text")


def _pick_candidate(
    caption: str,
    corrector_output: str,
    reviewer_result: dict,
    shabad_toks: set[str],
) -> tuple[str, str, str, list[str]]:
    """Run validator over candidate list, return (text, decision, src, log)."""
    candidates: list[tuple[str, str, str]] = []
    rv_text = reviewer_result.get("final_text")
    if rv_text:
        label = {
            "APPROVE": "replaced_llm",
            "IMPROVED": "replaced_reviewed",
            "REJECT": "unchanged_rejected",
        }.get(reviewer_result.get("verdict"), "error_fallback")
        candidates.append((rv_text, label, "reviewer"))
    candidates.append((corrector_output, "replaced_llm", "corrector"))
    candidates.append((caption, "validator_fallback", "caption"))

    log: list[str] = []
    for text, decision, src in candidates:
        ok, reasons = validate(caption, text, shabad_toks)
        if ok:
            log.append(f"shipped_from={src}")
            return text, decision, src, log
        log.append(f"{src}_failed_validation: {reasons}")
    # Should not happen — caption is trivially valid (no-op). Defensive.
    return caption, "validator_fallback", "caption", log


def _process_shabad_group(
    rows: list[tuple[int, dict]],
    shabad_lines: list[SggsLine],
    cfg: Stage2Config,
    openai_client,
    gemini_client,
) -> list[tuple[int, dict]]:
    """Returns [(row_index, stage2_fields), ...]."""
    lines_text = _shabad_lines_text(shabad_lines)
    if not lines_text:
        return [(i, _no_context(_caption_of(r))) for i, r in rows]

    # --- Corrector batch -----------------------------------------------------
    corr_items = [
        {
            "clip_id": r["clip_id"],
            "caption": _caption_of(r),
            "sttm_hint": _sttm_hint_of(r),
        }
        for _, r in rows
    ]
    corrected = correct_batch(
        corr_items, lines_text, cfg.corrector, client=openai_client,
    )

    # Split into noop (fast-path) vs needs-review
    output: list[tuple[int, dict]] = []
    review_items: list[dict] = []
    meta_by_cid: dict[str, dict] = {}
    for (i, r), item in zip(rows, corr_items):
        cid = item["clip_id"]
        cresult = corrected.get(cid, {})
        caption = item["caption"]
        if not cresult.get("ok"):
            output.append((i, {
                "stage2_decision": "error_fallback",
                "stage2_text": caption,
                "stage2_corrector_output": None,
                "stage2_reviewer_verdict": None,
                "stage2_reviewer_final": None,
                "stage2_changed_tokens": [],
                "stage2_reasons": [f"corrector_error: {cresult.get('error')}"],
                "stage2_corrector_model": cresult.get("model"),
                "stage2_reviewer_model": None,
            }))
            continue

        corrected_text = cresult["corrected"]
        if corrected_text.strip() == caption.strip():
            output.append((i, {
                "stage2_decision": "unchanged_noop",
                "stage2_text": caption,
                "stage2_corrector_output": corrected_text,
                "stage2_reviewer_verdict": "APPROVE_AUTO",
                "stage2_reviewer_final": None,
                "stage2_changed_tokens": [],
                "stage2_reasons": ["corrector returned caption verbatim"],
                "stage2_corrector_model": cresult["model"],
                "stage2_reviewer_model": None,
            }))
            continue

        review_items.append({
            "clip_id": cid,
            "caption": caption,
            "corrector_output": corrected_text,
        })
        meta_by_cid[cid] = {
            "row_index": i,
            "corrector_model": cresult["model"],
        }

    if not review_items:
        return output

    # --- Reviewer batch ------------------------------------------------------
    reviewed = review_batch(
        review_items, lines_text, cfg.reviewer, client=gemini_client,
    )

    # --- Validator gate + candidate ranking ---------------------------------
    shabad_toks = _shabad_tokens(shabad_lines)
    for item in review_items:
        cid = item["clip_id"]
        meta = meta_by_cid[cid]
        i = meta["row_index"]
        rreq = reviewed.get(cid, {})

        text, decision, src, log = _pick_candidate(
            caption=item["caption"],
            corrector_output=item["corrector_output"],
            reviewer_result=rreq,
            shabad_toks=shabad_toks,
        )
        output.append((i, {
            "stage2_decision": decision,
            "stage2_text": text,
            "stage2_corrector_output": item["corrector_output"],
            "stage2_reviewer_verdict": rreq.get("verdict"),
            "stage2_reviewer_final": rreq.get("final_text"),
            "stage2_changed_tokens": rreq.get("changed_tokens") or [],
            "stage2_reasons": (rreq.get("reasons") or []) + log,
            "stage2_corrector_model": meta["corrector_model"],
            "stage2_reviewer_model": rreq.get("model"),
        }))

    return output


def process_rows(
    rows: list[dict],
    shabad_lines_map: dict[str, list[SggsLine]],
    cfg: Stage2Config,
    max_workers: int = 5,
    openai_client=None,
    gemini_client=None,
    checkpoint_path: str | None = None,
) -> list[dict]:
    """Batched + concurrent Stage 2 over a list of rows. Groups rows by
    shabad_id, dispatches each group to a thread-pool worker, and returns
    results aligned to the input order.

    If `checkpoint_path` is set:
      - On start, loads any previously-completed shabad outputs from the
        checkpoint and skips those shabad groups.
      - After each shabad group finishes, appends its results to the
        checkpoint as a JSONL line: {"shabad_id": ..., "rows": [...]}
      - Safe to restart the full run with the same checkpoint_path to
        resume. Sensitive to shabad_id mapping — pass a stable stage1.
    """
    import json
    import threading
    from pathlib import Path as _Path

    results: list[dict | None] = [None] * len(rows)

    # No-context rows and rows without shabad_id go straight to safe output.
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, r in enumerate(rows):
        sid = r.get("shabad_id") or ""
        if not sid or sid not in shabad_lines_map:
            results[i] = _no_context(_caption_of(r))
            continue
        groups[sid].append((i, r))

    # ------------------------------------------------------------------------
    # Checkpoint load: skip shabad groups already done
    # ------------------------------------------------------------------------
    already_done: set[str] = set()
    if checkpoint_path:
        cp = _Path(checkpoint_path)
        if cp.exists():
            with cp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sid = rec.get("shabad_id")
                    rows_out = rec.get("rows") or []
                    if sid:
                        already_done.add(sid)
                        for r in rows_out:
                            idx = r.get("_row_index")
                            fields = r.get("fields")
                            if idx is not None and fields is not None:
                                results[idx] = fields
            print(f"[process_rows] resumed {len(already_done)} shabads from "
                  f"{checkpoint_path}", flush=True)
    checkpoint_lock = threading.Lock() if checkpoint_path else None

    remaining_groups = {
        sid: ir for sid, ir in groups.items() if sid not in already_done
    }
    if not remaining_groups:
        # Fill any remaining Nones defensively
        for i, r in enumerate(rows):
            if results[i] is None:
                results[i] = _no_context(_caption_of(r))
        return results  # type: ignore

    def _append_checkpoint(sid: str, outcomes: list[tuple[int, dict]]) -> None:
        if not checkpoint_path:
            return
        rec = {
            "shabad_id": sid,
            "rows": [
                {"_row_index": i, "fields": fields} for i, fields in outcomes
            ],
        }
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        assert checkpoint_lock is not None
        with checkpoint_lock:
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(line)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_sid = {
            ex.submit(
                _process_shabad_group, indexed_rows,
                shabad_lines_map[sid], cfg, openai_client, gemini_client,
            ): sid
            for sid, indexed_rows in remaining_groups.items()
        }
        for fut in as_completed(future_to_sid):
            sid = future_to_sid[fut]
            outcomes = fut.result()
            for i, fields in outcomes:
                results[i] = fields
            _append_checkpoint(sid, outcomes)

    # Any leftover None = defensive fallback (shouldn't happen).
    for i, r in enumerate(rows):
        if results[i] is None:
            results[i] = _no_context(_caption_of(r))
    return results  # type: ignore
