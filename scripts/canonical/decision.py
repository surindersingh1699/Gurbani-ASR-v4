"""Decision-label computation + safe final_text/sggs_line rendering."""
from __future__ import annotations

from dataclasses import dataclass

from .gurmukhi_skeleton import skel


@dataclass
class DecisionConfig:
    # Tightened 2026-04-19: loose 0.92/0.75 let marginal 4-of-5 rewrites through
    # with one bad fix. 0.95/0.60 demotes those to `review` for LLM pass.
    accept_threshold: float = 0.95
    review_threshold: float = 0.60
    # Boundary churn guard: if the first or last run of consecutive 1-cons
    # diff-skeleton 'fix' ops is this long, demote 'replaced' → 'review'.
    # Catches the reorder-snap pathology where NW locks onto the wrong part
    # of a repeating chorus and "fixes" caption prefix/suffix to match.
    # A value of 2 caught clip_00049 in the 50-row kirtan-eval dry-run
    # without demoting any of the other 11 legitimately-replaced rows.
    max_boundary_1cons_run: int = 2


def _is_1cons_fix(op: dict) -> bool:
    """True if op is a 'fix' with differing consonant skeletons (i.e. not a
    pure matra correction)."""
    if op["op"] != "fix":
        return False
    return skel("".join(op["cap"])) != skel("".join(op["sggs"]))


def _boundary_1cons_run(ops: list[dict]) -> int:
    """Longer of (leading, trailing) consecutive-1cons-fix runs."""
    lead = 0
    for op in ops:
        if _is_1cons_fix(op):
            lead += 1
        else:
            break
    trail = 0
    for op in reversed(ops):
        if _is_1cons_fix(op):
            trail += 1
        else:
            break
    return max(lead, trail)


def decide(ops: list[dict], used_relaxed: bool, cfg: DecisionConfig) -> str:
    n_match = sum(1 for o in ops if o["op"] == "match")
    n_fix = sum(1 for o in ops if o["op"] == "fix")
    n_merge = sum(1 for o in ops if o["op"] == "merge")
    n_split = sum(1 for o in ops if o["op"] == "split")
    n_del = sum(1 for o in ops if o["op"] == "delete")
    total = n_match + n_fix + n_merge + n_split + n_del
    if total == 0:
        return "unchanged"
    score = (n_match + n_fix + n_merge + n_split) / total
    has_change = (n_fix + n_merge + n_split) > 0

    n_matra_fix = sum(
        1 for o in ops
        if o["op"] == "fix"
        and skel("".join(o["cap"])) == skel("".join(o["sggs"]))
    )
    n_1cons_fix = n_fix - n_matra_fix

    if score >= cfg.accept_threshold and not has_change:
        return "matched"
    if score >= cfg.accept_threshold and not used_relaxed:
        if _boundary_1cons_run(ops) >= cfg.max_boundary_1cons_run:
            return "review"
        return "replaced"

    # Safe-rescue: score < 0.95 is often due to deletes (caption has words
    # flanking a correctly-matched scripture span). If every non-delete op is
    # a plain match or a matra-only fix — i.e. zero 1-cons swaps, zero
    # merges, zero splits — then caption is preserved verbatim on deletes
    # and only matra corrections are applied. That is as safe as 'replaced'
    # gets, so promote out of 'review'.
    if (
        not used_relaxed
        and n_1cons_fix == 0
        and n_merge == 0
        and n_split == 0
        and _boundary_1cons_run(ops) == 0
        and n_match + n_matra_fix > 0
    ):
        # Matra corrections applied → 'replaced'; no corrections at all →
        # 'unchanged' (caption tokens kept literally, no need to flag).
        return "replaced" if n_matra_fix > 0 else "unchanged"

    if score >= cfg.review_threshold or used_relaxed:
        return "review"
    return "unchanged"


def render_outputs(
    ops: list[dict], cap_tokens: list[str], decision: str
) -> dict:
    """Build final_text + sggs_line. On 'unchanged', final_text = caption verbatim,
    sggs_line = None. Safety fallback."""
    if decision == "unchanged":
        return {
            "final_text": " ".join(cap_tokens),
            "sggs_line": None,
            "line_ids": [],
        }
    final_parts: list[str] = []
    sggs_parts: list[str] = []
    line_ids: list[str] = []
    for op in ops:
        if op["op"] == "match":
            final_parts.extend(op["cap"])
            sggs_parts.extend(op["sggs"])
        elif op["op"] in ("fix", "merge", "split"):
            final_parts.extend(op["sggs"])
            sggs_parts.extend(op["sggs"])
        elif op["op"] == "delete":
            final_parts.extend(op["cap"])
        line_ids.extend(op.get("line_ids", []))
    sggs_line = " ".join(dict.fromkeys(sggs_parts)) or None
    return {
        "final_text": " ".join(final_parts),
        "sggs_line": sggs_line,
        "line_ids": list(dict.fromkeys(line_ids)),
    }
