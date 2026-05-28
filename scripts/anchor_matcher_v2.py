"""Matcher v2 for first-letter anchor models.

Improvements over the v1 4-gram-overlap matcher:

  1. Beam-search CTC top-K: model returns multiple candidate anchor strings,
     each scored by acoustic confidence.
  2. Sliding-window voting: a long clip is chunked into overlapping windows
     (e.g. 4 words wide, stride 2). Each window votes for its top-K shabads;
     final shabad is the majority winner.
  3. Edit-distance (Levenshtein) scoring on top of 4-gram overlap. Tolerant
     to insertions/deletions caused by sangat overlap or repetition.
  4. Continuity bias: once a shabad is locked, prefer subsequent matches
     near the locked position in SGGS.

Public API:
  AnchorMatcherV2(db_path).search(anchor: str, n_best: int = 5) -> list[Candidate]
  AnchorMatcherV2.score_clip(anchors: list[str], window_size: int = 8) -> Candidate
"""
from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sttm_first_letter_map import (
    db_first_letters_to_search_anchor,
    training_anchor_to_search_anchor,
)


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def _char_4grams(s):
    if len(s) < 4:
        return {s} if s else set()
    return {s[i:i + 4] for i in range(len(s) - 3)}


def _overlap(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _levenshtein_norm(a: str, b: str) -> float:
    """Normalized 1 - edit_distance / max(len). Higher is better."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    if m < n:
        a, b = b, a
        m, n = n, m
    prev = list(range(n + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * n
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return 1.0 - (prev[n] / max(m, 1))


@dataclass
class Candidate:
    shabad_id: str
    anchor: str
    score: float
    line_id: str = ""
    components: dict = None


@dataclass
class DBRow:
    line_id: str
    shabad_id: str
    order_id: int
    anchor: str
    ngrams: set


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

class AnchorMatcherV2:
    """Matcher with edit-distance + 4-gram overlap composite scoring."""

    def __init__(self, db_path: str,
                 w_overlap: float = 0.5,
                 w_edit: float = 0.5,
                 continuity_window: int = 10,
                 continuity_bonus: float = 0.15):
        self.w_overlap = w_overlap
        self.w_edit = w_edit
        self.continuity_window = continuity_window
        self.continuity_bonus = continuity_bonus
        self._rows = self._load_db(db_path)
        # Index rows by shabad_id for shabad-best aggregation
        self._by_shabad = {}
        for r in self._rows:
            self._by_shabad.setdefault(r.shabad_id, []).append(r)
        # Locked state for continuity bias
        self.locked_shabad = None
        self.locked_order_id = None

    @staticmethod
    def _load_db(db_path: str) -> list[DBRow]:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        rows = []
        for line_id, shabad_id, order_id, fl in cur.execute(
                "SELECT id, shabad_id, order_id, first_letters FROM lines"):
            a = db_first_letters_to_search_anchor(fl or "")
            if a:
                rows.append(DBRow(line_id=line_id, shabad_id=shabad_id,
                                  order_id=order_id or 0, anchor=a,
                                  ngrams=_char_4grams(a)))
        conn.close()
        return rows

    def _row_score(self, query: str, q4: set, row: DBRow) -> float:
        ov = _overlap(q4, row.ngrams)
        ed = _levenshtein_norm(query, row.anchor) if len(query) < 60 else 0.0
        score = self.w_overlap * ov + self.w_edit * ed
        # Continuity bonus
        if self.locked_shabad is not None:
            if row.shabad_id == self.locked_shabad:
                score += self.continuity_bonus
            elif self.locked_order_id is not None and abs(
                    row.order_id - self.locked_order_id) <= self.continuity_window:
                score += self.continuity_bonus * 0.5
        return score

    def search(self, anchor: str, n_best: int = 5,
               prefilter_top_k: int = 200) -> list[Candidate]:
        """Return top-N shabad candidates for a single anchor string.

        Two-stage scoring:
          1. Fast 4-gram overlap on all 141k DB rows, take top `prefilter_top_k`.
          2. Slow composite score (overlap + edit distance + continuity) on
             those finalists only. ~700x speedup vs naive all-rows scoring,
             with negligible recall loss (true positives have high overlap).
        """
        if not anchor:
            return []
        q4 = _char_4grams(anchor)
        # Stage 1: fast 4-gram overlap pre-filter
        scored = []
        for row in self._rows:
            ov = _overlap(q4, row.ngrams)
            if ov > 0:
                scored.append((ov, row))
        scored.sort(key=lambda x: -x[0])
        candidates = scored[:prefilter_top_k]
        # Stage 2: composite score on top-K
        best_per_shabad: dict[str, tuple[float, DBRow]] = {}
        for _, row in candidates:
            s = self._row_score(anchor, q4, row)
            cur = best_per_shabad.get(row.shabad_id)
            if cur is None or s > cur[0]:
                best_per_shabad[row.shabad_id] = (s, row)
        # Stage 3: always re-score the locked shabad (continuity)
        if self.locked_shabad and self.locked_shabad not in best_per_shabad:
            for row in self._by_shabad.get(self.locked_shabad, []):
                s = self._row_score(anchor, q4, row)
                cur = best_per_shabad.get(row.shabad_id)
                if cur is None or s > cur[0]:
                    best_per_shabad[row.shabad_id] = (s, row)
        ranked = sorted(
            ((s, sid, r) for sid, (s, r) in best_per_shabad.items()),
            reverse=True)[:n_best]
        return [Candidate(shabad_id=sid, anchor=r.anchor, score=s,
                          line_id=r.line_id) for s, sid, r in ranked]

    def search_with_windows(self, anchor: str, window_size: int = 8,
                            stride: int = 4, n_best: int = 5,
                            beam_anchors: list[str] | None = None
                            ) -> list[Candidate]:
        """Score a long anchor via sliding-window voting across the DB.

        Args:
          anchor: the compact search anchor (no `|`).
          window_size: number of chars per window. ~8 chars ≈ 8 spoken words.
          stride: overlap stride between windows.
          n_best: returned candidate count.
          beam_anchors: optional list of beam-search alternatives. Each will
            also be window-scored; final votes are summed across the beam.

        Returns top-n_best Candidates ranked by total accumulated vote weight.
        """
        anchors = [anchor] + (beam_anchors or [])
        # vote_per_shabad: shabad_id -> sum of best-window scores
        vote: dict[str, float] = {}
        best_anchor_per_shabad: dict[str, tuple[str, float]] = {}
        for a in anchors:
            if not a:
                continue
            if len(a) <= window_size:
                # Single-window case (short anchor)
                for c in self.search(a, n_best=n_best * 2):
                    vote[c.shabad_id] = vote.get(c.shabad_id, 0.0) + c.score
                    cur = best_anchor_per_shabad.get(c.shabad_id)
                    if cur is None or c.score > cur[1]:
                        best_anchor_per_shabad[c.shabad_id] = (c.anchor, c.score)
                continue
            for start in range(0, len(a) - window_size + 1, stride):
                window = a[start:start + window_size]
                top = self.search(window, n_best=3)
                for rank, c in enumerate(top):
                    # Rank-weighted vote: rank 0 = 1.0, rank 1 = 0.5, rank 2 = 0.33
                    weight = c.score / (rank + 1)
                    vote[c.shabad_id] = vote.get(c.shabad_id, 0.0) + weight
                    cur = best_anchor_per_shabad.get(c.shabad_id)
                    if cur is None or c.score > cur[1]:
                        best_anchor_per_shabad[c.shabad_id] = (c.anchor, c.score)
        ranked = sorted(vote.items(), key=lambda kv: -kv[1])[:n_best]
        return [
            Candidate(shabad_id=sid, anchor=best_anchor_per_shabad[sid][0],
                      score=v) for sid, v in ranked
        ]

    def update_lock(self, candidate: Candidate, lock_threshold: float = 0.4,
                    min_margin: float = 0.05, all_candidates: list = None):
        """Update continuity-lock state based on a candidate.

        Locks if candidate score >= threshold AND top1-top2 margin >= min_margin.
        Releases lock if no candidate exceeds threshold.
        """
        if candidate is None or candidate.score < lock_threshold:
            self.locked_shabad = None
            self.locked_order_id = None
            return
        margin = candidate.score - (
            all_candidates[1].score if all_candidates and len(all_candidates) >= 2 else 0.0)
        if margin < min_margin:
            return  # ambiguous, keep prior lock if any
        rows = self._by_shabad.get(candidate.shabad_id, [])
        if not rows:
            return
        self.locked_shabad = candidate.shabad_id
        # If we have a line_id, find its order_id; else use first row of shabad
        target_row = next((r for r in rows if r.line_id == candidate.line_id),
                          rows[0])
        self.locked_order_id = target_row.order_id

    def reset_lock(self):
        self.locked_shabad = None
        self.locked_order_id = None
