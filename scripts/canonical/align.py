"""Semi-global Needleman-Wunsch alignment: caption tokens vs retrieved shabad
token sequence. Supports match / fix / merge / split / insert / delete ops.

Semi-global = free skipping on the SGGS side at prefix and suffix, because
global NW's gap penalty over ~60 unused shabad tokens drowns out the match
score for short captions.
"""
from __future__ import annotations

from dataclasses import dataclass

from .gurmukhi_skeleton import lev, skel
from .sttm_index import SggsLine


@dataclass
class AlignConfig:
    max_op_edit: int = 1
    max_op_edit_relaxed: int = 2
    min_cons_len: int = 2
    max_len_delta: int = 1
    max_span: int = 3
    min_orphan_run: int = 3
    score_match: int = 10
    score_fix_matra: int = 7
    score_fix_1cons: int = 4
    score_merge_split: int = 4
    gap_cap: int = -1
    gap_sgs: int = -2
    floor: int = -10_000_000


def _fix_eligible(
    cap_skel: str, sgs_skel: str, cfg: AlignConfig, max_edit: int
) -> bool:
    if len(cap_skel) < cfg.min_cons_len or len(sgs_skel) < cfg.min_cons_len:
        return False
    if abs(len(cap_skel) - len(sgs_skel)) > max_edit:
        return False
    return lev(cap_skel, sgs_skel) <= max_edit


def _score_11(
    cap: str, cs: str, sgs: str, ss: str, cfg: AlignConfig, max_edit: int
) -> tuple[int, str | None]:
    if cap == sgs and cs == ss:
        return cfg.score_match, "match"
    if cs == ss and cs and len(cs) >= cfg.min_cons_len:
        return cfg.score_fix_matra, "fix"
    if _fix_eligible(cs, ss, cfg, max_edit):
        return cfg.score_fix_1cons, "fix"
    return cfg.floor, None


def align_nw(
    cap_tokens: list[str],
    shabad_lines: list[SggsLine],
    cfg: AlignConfig,
    max_edit: int | None = None,
) -> list[dict]:
    """Semi-global NW. Returns ops with keys op/cap/sggs/line_ids."""
    if max_edit is None:
        max_edit = cfg.max_op_edit

    stream_tok: list[str] = []
    stream_skel: list[str] = []
    stream_lid: list[str] = []
    for ln in shabad_lines:
        for t, ts in zip(ln.tokens, ln.tok_skels):
            stream_tok.append(t)
            stream_skel.append(ts)
            stream_lid.append(ln.line_id)

    m, n = len(cap_tokens), len(stream_tok)
    if m == 0 or n == 0:
        return [
            {"op": "delete", "cap": [t], "sggs": [t], "line_ids": []}
            for t in cap_tokens
        ]

    cap_skels = [skel(t) for t in cap_tokens]

    dp = [[cfg.floor] * (n + 1) for _ in range(m + 1)]
    bt: list[list] = [[None] * (n + 1) for _ in range(m + 1)]
    for j in range(n + 1):
        dp[0][j] = 0
        bt[0][j] = ("prefix_skip", 0, 0, [], [], []) if j > 0 else None
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + cfg.gap_cap
        bt[i][0] = (
            "delete",
            i - 1,
            0,
            [cap_tokens[i - 1]],
            [cap_tokens[i - 1]],
            [],
        )

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            best_score = cfg.floor
            best_bt = None

            s11, op11 = _score_11(
                cap_tokens[i - 1],
                cap_skels[i - 1],
                stream_tok[j - 1],
                stream_skel[j - 1],
                cfg,
                max_edit,
            )
            if op11 is not None:
                cand = dp[i - 1][j - 1] + s11
                if cand > best_score:
                    best_score = cand
                    sgs_tok = (
                        stream_tok[j - 1] if op11 == "fix" else cap_tokens[i - 1]
                    )
                    best_bt = (
                        op11,
                        i - 1,
                        j - 1,
                        [cap_tokens[i - 1]],
                        [sgs_tok],
                        [stream_lid[j - 1]],
                    )

            for span in range(2, cfg.max_span + 1):
                if j - span < 0:
                    break
                sj = "".join(stream_skel[j - span : j])
                if _fix_eligible(cap_skels[i - 1], sj, cfg, max_edit):
                    cand = dp[i - 1][j - span] + cfg.score_merge_split
                    if cand > best_score:
                        best_score = cand
                        best_bt = (
                            "split",
                            i - 1,
                            j - span,
                            [cap_tokens[i - 1]],
                            list(stream_tok[j - span : j]),
                            list(stream_lid[j - span : j]),
                        )

            for span in range(2, cfg.max_span + 1):
                if i - span < 0:
                    break
                cj = "".join(cap_skels[i - span : i])
                if _fix_eligible(cj, stream_skel[j - 1], cfg, max_edit):
                    cand = dp[i - span][j - 1] + cfg.score_merge_split
                    if cand > best_score:
                        best_score = cand
                        best_bt = (
                            "merge",
                            i - span,
                            j - 1,
                            list(cap_tokens[i - span : i]),
                            [stream_tok[j - 1]],
                            [stream_lid[j - 1]],
                        )

            cand = dp[i][j - 1] + cfg.gap_sgs
            if cand > best_score:
                best_score = cand
                best_bt = (
                    "insert",
                    i,
                    j - 1,
                    [],
                    [stream_tok[j - 1]],
                    [stream_lid[j - 1]],
                )

            cand = dp[i - 1][j] + cfg.gap_cap
            if cand > best_score:
                best_score = cand
                best_bt = (
                    "delete",
                    i - 1,
                    j,
                    [cap_tokens[i - 1]],
                    [cap_tokens[i - 1]],
                    [],
                )

            dp[i][j] = best_score
            bt[i][j] = best_bt

    best_j = max(range(n + 1), key=lambda j: dp[m][j])
    i, j = m, best_j
    ops: list[dict] = []
    while i > 0:
        if bt[i][j] is None:
            break
        op, bi, bj, cap_span, sgs_span, lids = bt[i][j]
        if op not in ("insert", "prefix_skip"):
            ops.append(
                {"op": op, "cap": cap_span, "sggs": sgs_span, "line_ids": lids}
            )
        i, j = bi, bj
    ops.reverse()
    return ops


def realign_orphan_runs(
    ops: list[dict], shabad_lines: list[SggsLine], cfg: AlignConfig
) -> tuple[list[dict], bool]:
    """If there are contiguous delete runs ≥ cfg.min_orphan_run tokens, try a
    fresh NW pass on that span; relax max_op_edit if the first retry also fails.
    """
    used_relaxed = False
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(ops):
            if ops[i]["op"] == "delete":
                start = i
                while i < len(ops) and ops[i]["op"] == "delete":
                    i += 1
                if i - start >= cfg.min_orphan_run:
                    orphan = [ops[k]["cap"][0] for k in range(start, i)]
                    alt = align_nw(orphan, shabad_lines, cfg, cfg.max_op_edit)
                    n_del_alt = sum(1 for o in alt if o["op"] == "delete")
                    if n_del_alt < (i - start):
                        ops = ops[:start] + alt + ops[i:]
                        changed = True
                        break
                    alt2 = align_nw(
                        orphan, shabad_lines, cfg, cfg.max_op_edit_relaxed
                    )
                    n_del_alt2 = sum(1 for o in alt2 if o["op"] == "delete")
                    if n_del_alt2 < (i - start):
                        ops = ops[:start] + alt2 + ops[i:]
                        used_relaxed = True
                        changed = True
                        break
            else:
                i += 1
    return ops, used_relaxed
