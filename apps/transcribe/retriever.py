"""Local 2-layer shabad retriever for the Surt transcribe app.

Replaces the remote BaniDB API (which currently returns empty results) with
the on-disk FAISS + MuRIL pipeline described in the surt training plan:

    Layer 2 (primary):    index/sggs_tuk.faiss      56,152 tuk-level vectors
    Layer 3 (confirm):    index/sggs_shabad.faiss    5,544 shabad-level vectors

Algorithm (per query text):
    1. Embed query with MuRIL (normalized).
    2. Tuk search -> k*3 hits, group by shabad_id, sum cosine scores.
    3. Shabad-level search -> top shabad_id for this query.
    4. If tuk-top's shabad matches the shabad-level top, boost by 1.1; else 0.9.
    5. Return top-N shabads in BaniDB-compatible dict shape.

Note on pickle: `tuk_meta.pkl` / `shabad_meta.pkl` are repo-local artifacts
produced by `scripts/01_build_tuk_index.py` / `scripts/02_build_shabad_index.py`.
Same pattern already used by `apps/live_lab/tracker.py`.
"""

from __future__ import annotations

import os
import pickle  # loading repo-local index artifacts only
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = Path(os.environ.get("SURT_INDEX_DIR", REPO_ROOT / "index"))
DB_PATH = Path(os.environ.get("SURT_DB_PATH", REPO_ROOT / "database.sqlite"))
MURIL_MODEL = os.environ.get("SURT_MURIL_MODEL", "google/muril-base-cased")

TUK_SCORE_FLOOR = float(os.environ.get("SURT_TUK_FLOOR", "0.50"))
TUK_K_MULT = 3
CONFIRM_BOOST = 1.10
CONFIRM_PENALTY = 0.90

# Lock-mode thresholds. Once a shabad is locked, we only score the tail
# against that shabad's own tuks — this is the "pointer" path the UI uses
# to advance line-by-line. Score is char-4gram overlap coefficient (same
# function the literal rerank uses), so these floors are directly comparable
# to the `_literal` field on search hits.
LOCK_SCORE_FLOOR = float(os.environ.get("SURT_LOCK_SCORE", "0.80"))
# If the best in-shabad overlap falls below this for UNLOCK_MISSES consecutive
# windows, we unlock. Chosen so coincidental 4-gram overlap across unrelated
# shabads (~0.4) counts as a miss, while singing through any verse of the
# locked shabad reliably stays near 1.0.
UNLOCK_SCORE_FLOOR = float(os.environ.get("SURT_UNLOCK_FLOOR", "0.50"))
# Literal-overlap rerank: blend `LITERAL_WEIGHT * char-4gram overlap` into the
# best-cosine base. MuRIL ranks semantic paraphrases highly; without this, a
# paraphrase can outrank the exact tuk the ragi sang. We score with the
# overlap coefficient |q ∩ t| / min(|q|,|t|) — Jaccard is dominated by the
# union size, which collapses to ~0.1 when a long live-mic query contains a
# short tuk verbatim. Overlap gives 1.0 for exact containment in either
# direction, so the rerank survives long or short queries.
LITERAL_WEIGHT = float(os.environ.get("SURT_LITERAL_WEIGHT", "0.40"))

# Retrieval modes — user-selectable in the UI.
MODE_SEMANTIC_LITERAL = "semantic+literal"   # MuRIL + 2-layer FAISS + 4gram rerank (default)
MODE_SEMANTIC = "semantic"                    # MuRIL + 2-layer FAISS (no literal rerank)
MODE_LITERAL = "literal"                      # char-4gram Jaccard over all tuks (no MuRIL)
MODE_BANIDB = "banidb"                        # remote BaniDB API (online)
DEFAULT_MODE = MODE_SEMANTIC_LITERAL

MODE_LABELS: dict[str, str] = {
    MODE_SEMANTIC_LITERAL: "Semantic + literal (MuRIL + 4gram rerank) — default",
    MODE_SEMANTIC:         "Semantic only (MuRIL; no literal bonus)",
    MODE_LITERAL:          "Literal only (char-4gram Jaccard; no MuRIL)",
    MODE_BANIDB:           "BaniDB remote API (online only)",
}


def _char_4grams(s: str) -> set[str]:
    s = (s or "").strip()
    if len(s) < 4:
        return {s} if s else set()
    return {s[i : i + 4] for i in range(len(s) - 3)}


def _overlap(a: set[str], b: set[str]) -> float:
    """Overlap coefficient: |a ∩ b| / min(|a|, |b|).

    Preferred over Jaccard for query-vs-tuk scoring because live-mic queries
    are much longer than a single tuk (often 8–10× more 4-grams). Jaccard
    normalizes by union size and falls to ~0.1 even on a verbatim match;
    overlap stays at 1.0 whenever the shorter set is a subset of the longer.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


@dataclass
class ShabadRow:
    shabad_id: str
    sttm_id: Optional[int]
    first_tuk: str
    writer: str
    raag: str
    ang: int


@dataclass
class LockedHit:
    """One scored tuk from within a locked shabad.

    The top result becomes the UI's current "pointer" pangti.
    """
    shabad_id: str
    tuk_row: int
    tuk_text: str
    overlap: float
    ang: int
    writer: str
    raag: str
    sttm_id: Optional[int]


class ShabadRetriever:
    def __init__(
        self,
        index_dir: Path = INDEX_DIR,
        db_path: Path = DB_PATH,
        muril_model: str = MURIL_MODEL,
    ):
        import faiss
        from sentence_transformers import SentenceTransformer

        t0 = time.time()
        print(f"[retriever] loading tuk index ({index_dir}/sggs_tuk.faiss)...")
        self._tuk_idx = faiss.read_index(str(index_dir / "sggs_tuk.faiss"))
        self._tuk_meta: dict[int, dict] = _load_pkl(index_dir / "tuk_meta.pkl")

        print(f"[retriever] loading shabad index ({index_dir}/sggs_shabad.faiss)...")
        self._shabad_idx = faiss.read_index(str(index_dir / "sggs_shabad.faiss"))
        self._shabad_meta: dict[int, dict] = _load_pkl(index_dir / "shabad_meta.pkl")

        print(f"[retriever] loading sttm_id map from {db_path}...")
        self._sttm_id_of: dict[str, int] = _load_sttm_id_map(db_path)

        self._shabad_info: dict[str, ShabadRow] = {}
        for row in self._shabad_meta.values():
            sid = row["shabad_id"]
            self._shabad_info[sid] = ShabadRow(
                shabad_id=sid,
                sttm_id=self._sttm_id_of.get(sid),
                first_tuk=row.get("first_tuk", ""),
                writer=row.get("writer", ""),
                raag=row.get("raag", ""),
                ang=int(row.get("ang", 0) or 0),
            )

        n_tuks_no_shabad = sum(
            1 for t in self._tuk_meta.values()
            if t["shabad_id"] not in self._shabad_info
        )

        # shabad_id -> ordered list of (tuk_row, tuk_text). Used for lock-mode
        # "pointer" scoring without touching FAISS. Build is ~O(56k) dict
        # inserts so it's cheap; no need to gate behind a lazy init.
        self._tuks_by_shabad: dict[str, list[tuple[int, str]]] = {}
        for row, meta in self._tuk_meta.items():
            sid = meta["shabad_id"]
            self._tuks_by_shabad.setdefault(sid, []).append(
                (int(row), meta.get("text", ""))
            )

        print(f"[retriever] loading MuRIL: {muril_model}...")
        self._embedder = SentenceTransformer(muril_model)

        dt = time.time() - t0
        print(
            f"[retriever] ready in {dt:.1f} s | "
            f"{self._tuk_idx.ntotal:,} tuks | "
            f"{self._shabad_idx.ntotal:,} shabads | "
            f"{len(self._sttm_id_of):,} sttm_ids | "
            f"{n_tuks_no_shabad:,} orphan tuks"
        )

    @property
    def name(self) -> str:
        return "local-faiss-2layer"

    # ------- mode dispatch -------

    def search_topn(
        self, text: str, n: int = 5, mode: str = DEFAULT_MODE,
    ) -> list[dict]:
        text = (text or "").strip()
        if not text:
            return []

        if mode == MODE_BANIDB:
            from apps.transcribe.sttm_controller import (
                search_shabad_topn as banidb_search,
            )
            return banidb_search(text, n=n)

        if mode == MODE_LITERAL:
            return self._search_literal(text, n)

        # Both semantic modes use the 2-layer FAISS cascade.
        use_literal = (mode == MODE_SEMANTIC_LITERAL)
        return self._search_semantic(text, n, use_literal=use_literal)

    # ------- lock-mode pointer (score within a single shabad) -------

    def score_within_shabad(
        self, text: str, shabad_id: str,
    ) -> list[LockedHit]:
        """Score the query's 4-grams against every tuk in `shabad_id`.

        No MuRIL, no FAISS — just the overlap coefficient used by the literal
        rerank, applied to ~20-80 tuks. Intended to be called every window
        (including tentative refreshes) while the app is locked on a shabad.
        Returns an empty list if the shabad has no tuks or the query is empty.
        """
        text = (text or "").strip()
        if not text:
            return []
        tuks = self._tuks_by_shabad.get(shabad_id)
        if not tuks:
            return []
        q_grams = _char_4grams(text)
        if not q_grams:
            return []
        info = self._shabad_info.get(shabad_id)
        writer = info.writer if info else ""
        raag = info.raag if info else ""
        ang = info.ang if info else 0
        sttm_id = info.sttm_id if info else None

        out: list[LockedHit] = []
        for row, tuk_text in tuks:
            ov = _overlap(q_grams, _char_4grams(tuk_text))
            if ov <= 0.0:
                continue
            out.append(LockedHit(
                shabad_id=shabad_id,
                tuk_row=row,
                tuk_text=tuk_text,
                overlap=float(ov),
                ang=ang,
                writer=writer,
                raag=raag,
                sttm_id=sttm_id,
            ))
        out.sort(key=lambda h: h.overlap, reverse=True)
        return out

    # ------- literal-only (no MuRIL) -------

    def _ensure_literal_index(self) -> None:
        """Lazily compute a char-4gram set per tuk on first literal-mode call."""
        if getattr(self, "_tuk_4grams", None) is not None:
            return
        print("[retriever] building tuk 4gram sets for literal mode…")
        t0 = time.time()
        self._tuk_4grams: dict[int, set[str]] = {
            row: _char_4grams(meta.get("text", ""))
            for row, meta in self._tuk_meta.items()
        }
        print(f"[retriever] literal index ready in {time.time()-t0:.1f} s")

    def _search_literal(self, text: str, n: int) -> list[dict]:
        self._ensure_literal_index()
        q_grams = _char_4grams(text)
        if not q_grams:
            return []
        # Top-K tuks by overlap coefficient, then group by shabad and keep
        # each shabad's best.
        scored: list[tuple[int, float]] = []
        for row, grams in self._tuk_4grams.items():
            j = _overlap(q_grams, grams)
            if j > 0.0:
                scored.append((row, j))
        scored.sort(key=lambda t: t[1], reverse=True)
        scored = scored[: max(n * 10, 50)]

        best_per_shabad: dict[str, tuple[int, float]] = {}
        for row, j in scored:
            sid = self._tuk_meta[row]["shabad_id"]
            if sid not in best_per_shabad or j > best_per_shabad[sid][1]:
                best_per_shabad[sid] = (row, j)

        ranked = sorted(best_per_shabad.items(), key=lambda kv: kv[1][1], reverse=True)
        out: list[dict] = []
        for sid, (row, j) in ranked[:n]:
            info = self._shabad_info.get(sid)
            tm = self._tuk_meta.get(row, {})
            if info is None:
                info = ShabadRow(
                    shabad_id=sid,
                    sttm_id=self._sttm_id_of.get(sid),
                    first_tuk=tm.get("text", ""),
                    writer=tm.get("writer", ""),
                    raag=tm.get("raag", ""),
                    ang=int(tm.get("ang", 0) or 0),
                )
            out.append({
                "shabadId": info.sttm_id,
                "verseId": info.sttm_id,
                "gurmukhi": tm.get("text", info.first_tuk),
                "writer": info.writer,
                "raag": info.raag,
                "source": "SGGS",
                "ang": info.ang,
                "score": round(float(j), 3),
                "_raw_score": round(float(j), 4),
                "_tuk_score": 0.0,
                "_literal": round(float(j), 4),
                "_confirmed": False,
                "_tuk_row": int(row),
            })
        return out

    # ------- semantic (MuRIL + FAISS 2-layer, with optional literal rerank) -------

    def _search_semantic(
        self, text: str, n: int, *, use_literal: bool = True,
    ) -> list[dict]:
        q = self._embedder.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)

        k = max(n * TUK_K_MULT, 20)
        D, I = self._tuk_idx.search(q, k)
        hits = list(zip(I[0].tolist(), D[0].tolist()))
        top_tuk_score = hits[0][1] if hits else 0.0

        if top_tuk_score < TUK_SCORE_FLOOR:
            return []

        # Layer 2: group tuk hits by shabad_id, sum cosines; remember best tuk per shabad
        shabad_score: dict[str, float] = {}
        best_tuk_of: dict[str, tuple[int, float]] = {}
        for tuk_row, score in hits:
            meta = self._tuk_meta.get(int(tuk_row))
            if not meta:
                continue
            sid = meta["shabad_id"]
            shabad_score[sid] = shabad_score.get(sid, 0.0) + float(score)
            prior = best_tuk_of.get(sid)
            if prior is None or score > prior[1]:
                best_tuk_of[sid] = (int(tuk_row), float(score))

        if not shabad_score:
            return []

        # Layer 3: shabad-level confirmation
        Ds, Is = self._shabad_idx.search(q, 1)
        confirmed_sid: Optional[str] = None
        if Is.size > 0 and int(Is[0][0]) >= 0:
            row = self._shabad_meta.get(int(Is[0][0]))
            if row:
                confirmed_sid = row["shabad_id"]

        # Rerank: semantic base × confirmation multiplier, optionally blended
        # with char-4gram Jaccard literal overlap. `use_literal=False` gives
        # the pure semantic path (for A/B comparison).
        q_grams = _char_4grams(text) if use_literal else set()
        w_literal = LITERAL_WEIGHT if use_literal else 0.0
        ranked: list[tuple[str, float, float]] = []
        for sid, _sum in shabad_score.items():
            tuk_row, best_cos = best_tuk_of[sid]
            semantic = float(best_cos)
            if use_literal:
                tuk_text = self._tuk_meta.get(tuk_row, {}).get("text", "")
                literal = _overlap(q_grams, _char_4grams(tuk_text))
            else:
                literal = 0.0
            blend = (1.0 - w_literal) * semantic + w_literal * literal
            conf_mul = CONFIRM_BOOST if sid == confirmed_sid else CONFIRM_PENALTY
            ranked.append((sid, blend * conf_mul, literal))
        ranked.sort(key=lambda t: t[1], reverse=True)

        if not ranked:
            return []
        hi = ranked[0][1]
        lo = ranked[-1][1] if len(ranked) > 1 else 0.0
        span = max(hi - lo, 1e-6)

        out: list[dict] = []
        for sid, score, literal in ranked[:n]:
            info = self._shabad_info.get(sid)
            tuk_row, tuk_score = best_tuk_of.get(sid, (-1, 0.0))
            if info is None:
                tm = self._tuk_meta.get(tuk_row, {}) if tuk_row >= 0 else {}
                info = ShabadRow(
                    shabad_id=sid,
                    sttm_id=self._sttm_id_of.get(sid),
                    first_tuk=tm.get("text", ""),
                    writer=tm.get("writer", ""),
                    raag=tm.get("raag", ""),
                    ang=int(tm.get("ang", 0) or 0),
                )
            matched_tuk_text = (
                self._tuk_meta.get(tuk_row, {}).get("text", info.first_tuk)
                if tuk_row >= 0 else info.first_tuk
            )

            ui_score = 0.5 + 0.5 * (score - lo) / span
            ui_score = max(0.0, min(1.0, float(ui_score)))

            out.append({
                "shabadId": info.sttm_id,
                "verseId": info.sttm_id,
                "gurmukhi": matched_tuk_text,
                "writer": info.writer,
                "raag": info.raag,
                "source": "SGGS",
                "ang": info.ang,
                "score": round(ui_score, 3),
                "_raw_score": round(float(score), 4),
                "_tuk_score": round(float(tuk_score), 4),
                "_literal": round(float(literal), 4),
                "_confirmed": sid == confirmed_sid,
                "_tuk_row": int(tuk_row) if tuk_row >= 0 else -1,
            })
        return out


_singleton: Optional[ShabadRetriever] = None


def get_retriever() -> ShabadRetriever:
    """Lazy singleton so the app starts fast; first search pays the cost."""
    global _singleton
    if _singleton is None:
        _singleton = ShabadRetriever()
    return _singleton


def search_shabad_topn(
    query: str, n: int = 5, mode: str = DEFAULT_MODE,
) -> list[dict]:
    """Drop-in replacement for the BaniDB search used elsewhere in the app."""
    try:
        return get_retriever().search_topn(query, n=n, mode=mode)
    except Exception as e:  # noqa: BLE001
        print(f"[retriever] search failed: {e}")
        return []


def score_within_shabad(query: str, shabad_id: str) -> list[LockedHit]:
    """Pointer-mode scoring: which tuk inside this shabad does the query match?"""
    try:
        return get_retriever().score_within_shabad(query, shabad_id)
    except Exception as e:  # noqa: BLE001
        print(f"[retriever] lock-mode scoring failed: {e}")
        return []


def _load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301 - repo-local index artifacts


def _load_sttm_id_map(db_path: Path) -> dict[str, int]:
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("SELECT id, sttm_id FROM shabads WHERE sttm_id IS NOT NULL")
        return {sid: int(sttm) for sid, sttm in cur.fetchall() if sttm is not None}
    finally:
        con.close()
