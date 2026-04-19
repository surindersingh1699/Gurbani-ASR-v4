"""Decision-label computation + safe final_text/sggs_line rendering."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecisionConfig:
    accept_threshold: float = 0.92
    review_threshold: float = 0.75


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

    if score >= cfg.accept_threshold and not has_change:
        return "matched"
    if score >= cfg.accept_threshold and not used_relaxed:
        return "replaced"
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
