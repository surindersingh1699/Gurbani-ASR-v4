"""Load SGGS lines from STTM database.sqlite and build retrieval indices.

Public API:
  load_sggs(db_path, include_sirlekh=False) -> (list[SggsLine], global_tok_idx)
  build_shabad_ngram_index(lines, n=4) -> (shabad_ngrams, shabad_lines, doc_freq)
  next_shabad_in_sequence(lines) -> dict[shabad_id, shabad_id_next]
"""
from __future__ import annotations

import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from gurmukhi_converter import ascii_to_unicode, strip_vishraams  # noqa: E402

from .gurmukhi_skeleton import _VISHRAAM_TOKEN_RE, clean_token, skel


@dataclass(frozen=True)
class SggsLine:
    line_id: str
    shabad_id: str
    ang: int
    order_id: int
    type_id: int
    unicode: str
    skel: str
    tokens: tuple[str, ...]
    tok_skels: tuple[str, ...]


def load_sggs(
    db_path: Path | str,
    include_sirlekh: bool = False,
) -> tuple[list[SggsLine], dict[str, list[tuple[str, str, str]]]]:
    """Load SGGS content lines (+ optionally Sirlekh headers) and build a
    global token-skeleton → [(unicode_token, line_id, shabad_id)] index."""
    type_ids = (1, 2, 3, 4) if include_sirlekh else (1, 3, 4)
    placeholders = ",".join("?" * len(type_ids))

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT l.id AS line_id, l.shabad_id, l.source_page AS ang,
               l.order_id, l.type_id, l.gurmukhi
        FROM lines l
        JOIN shabads s ON l.shabad_id = s.id
        WHERE s.source_id = 1 AND l.type_id IN ({placeholders})
        ORDER BY l.order_id
        """,
        type_ids,
    ).fetchall()
    conn.close()

    out: list[SggsLine] = []
    global_tok_idx: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for r in rows:
        uni = ascii_to_unicode(strip_vishraams(r["gurmukhi"])).strip()
        if not uni or len(uni) < 3:
            continue
        raw = uni.split()
        tokens = tuple(
            clean_token(t) for t in raw if not _VISHRAAM_TOKEN_RE.match(t)
        )
        tokens = tuple(t for t in tokens if t)
        if not tokens:
            continue
        tok_skels = tuple(skel(t) for t in tokens)
        line = SggsLine(
            line_id=r["line_id"],
            shabad_id=r["shabad_id"],
            ang=r["ang"],
            order_id=r["order_id"],
            type_id=r["type_id"],
            unicode=" ".join(tokens),
            skel=skel(" ".join(tokens)),
            tokens=tokens,
            tok_skels=tok_skels,
        )
        out.append(line)
        for tok, ts in zip(tokens, tok_skels):
            if ts:
                global_tok_idx[ts].append((tok, line.line_id, line.shabad_id))
    return out, dict(global_tok_idx)


def _ngrams(s: str, n: int) -> list[str]:
    if len(s) >= n:
        return [s[i : i + n] for i in range(len(s) - n + 1)]
    return [s] if s else []


def build_shabad_ngram_index(
    lines: list[SggsLine], n: int = 4
) -> tuple[dict[str, Counter], dict[str, list[SggsLine]], Counter]:
    """Return (shabad_ngrams, shabad_lines, doc_freq)."""
    shabad_ngrams: dict[str, Counter] = defaultdict(Counter)
    shabad_lines: dict[str, list[SggsLine]] = defaultdict(list)
    for ln in lines:
        shabad_lines[ln.shabad_id].append(ln)
        for g in _ngrams(ln.skel, n):
            shabad_ngrams[ln.shabad_id][g] += 1
    df: Counter = Counter()
    for _sid, cnt in shabad_ngrams.items():
        for g in cnt:
            df[g] += 1
    return dict(shabad_ngrams), dict(shabad_lines), df


def next_shabad_in_sequence(lines: list[SggsLine]) -> dict[str, str]:
    """Map each shabad_id → the shabad_id of the next shabad by last order_id."""
    last_order: dict[str, int] = {}
    for ln in lines:
        prev = last_order.get(ln.shabad_id, -1)
        if ln.order_id > prev:
            last_order[ln.shabad_id] = ln.order_id
    ordered = sorted(last_order, key=lambda s: last_order[s])
    return {sid: ordered[i + 1] for i, sid in enumerate(ordered[:-1])}
