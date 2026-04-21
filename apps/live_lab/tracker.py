"""Shabad tracker - matches live ASR text against the SGGS tuk index.

Design (responsive, no sticky lock):

    every window (~1-2 s of new ASR text):
        q = muril_encode(window); q /= ||q||
        D, I = tuk_faiss.search(q, k=20)
        inst[shabad_id] += sum of cosines for its hits
        ema[s] = (1-a)*ema[s] + a*inst[s]          (a ~ 0.35)
        current  = argmax(ema)
        best_tuk = top hit inside current
        alap     = best tuk cosine < floor        (freeze pointer, do not leave)

Exposes `Tracker.update(text) -> TrackerResult` and keeps a bounded history
of committed (shabad, line, text, timestamp) tuples.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = Path(os.environ.get("SURT_INDEX_DIR", REPO_ROOT / "index"))
TUKS_JSON = Path(os.environ.get("SURT_TUKS_JSON", REPO_ROOT / "data" / "processed" / "tuks.json"))
MURIL_MODEL = os.environ.get("SURT_MURIL_MODEL", "google/muril-base-cased")


def _load_meta(path: Path):
    """Load tuk_meta.pkl. Isolated helper to keep serialization details here."""
    _pk = __import__("pickle")
    with open(path, "rb") as f:
        return _pk.load(f)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class TrackerSettings:
    ema_alpha: float = 0.35
    top_k_tuk: int = 20
    top_k_shabad: int = 5
    alap_floor: float = 0.45
    history_cap: int = 40
    min_text_chars: int = 4
    decay_per_idle: float = 0.92
    hard_reset_threshold: float = 0.05


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ShabadInfo:
    shabad_id: str
    raag: str
    writer: str
    ang: int
    first_tuk: str
    tuks: list[str]
    tuk_ids: list[int]


@dataclass
class ShabadCandidate:
    shabad_id: str
    score: float
    info: ShabadInfo


@dataclass
class HistoryItem:
    ts: float
    shabad_id: str
    line_idx: int
    text: str


@dataclass
class TrackerResult:
    current_shabad: Optional[ShabadInfo]
    current_line_idx: int
    current_line_score: float
    alap: bool
    candidates: list[ShabadCandidate]
    history: list[HistoryItem]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class Tracker:
    def __init__(
        self,
        settings: Optional[TrackerSettings] = None,
        index_dir: Path = INDEX_DIR,
        tuks_json: Path = TUKS_JSON,
        muril_model: str = MURIL_MODEL,
    ):
        import faiss  # type: ignore
        import json
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.cfg = settings or TrackerSettings()
        self._faiss = faiss

        print(f"[tracker] loading SGGS tuks from {tuks_json}")
        with open(tuks_json, encoding="utf-8") as f:
            tuks = json.load(f)

        print(f"[tracker] loading tuk FAISS index from {index_dir}/sggs_tuk.faiss")
        self.index = faiss.read_index(str(index_dir / "sggs_tuk.faiss"))
        self.tuk_meta = _load_meta(index_dir / "tuk_meta.pkl")

        # Group tuks by shabad_id for the detail panel.
        shabads: dict[str, ShabadInfo] = {}
        for t in tuks:
            sid = t["shabad_id"]
            if sid not in shabads:
                shabads[sid] = ShabadInfo(
                    shabad_id=sid,
                    raag=t.get("raag") or "",
                    writer=t.get("writer") or "",
                    ang=int(t.get("ang") or 0),
                    first_tuk=t["text"],
                    tuks=[],
                    tuk_ids=[],
                )
            shabads[sid].tuks.append(t["text"])
            shabads[sid].tuk_ids.append(int(t["tuk_id"]))
        self.shabads = shabads

        # tuk_id -> (shabad_id, line_idx)
        tuk_to_shabad: dict[int, tuple[str, int]] = {}
        for sid, info in shabads.items():
            for line_idx, tid in enumerate(info.tuk_ids):
                tuk_to_shabad[tid] = (sid, line_idx)
        self._tuk_to_shabad = tuk_to_shabad

        print(f"[tracker] loading MuRIL encoder: {muril_model}")
        self.embedder = SentenceTransformer(muril_model)

        self.ema: dict[str, float] = {}
        self.history: list[HistoryItem] = []
        print(
            f"[tracker] ready - {len(tuks):,} tuks, {len(shabads):,} shabads, "
            f"dim={self.embedder.get_sentence_embedding_dimension()}"
        )

    # -- core ----------------------------------------------------------------

    def reset(self) -> None:
        self.ema.clear()
        self.history.clear()

    def _decay_only(self) -> None:
        if not self.ema:
            return
        drop: list[str] = []
        for sid in self.ema:
            self.ema[sid] *= self.cfg.decay_per_idle
            if self.ema[sid] < self.cfg.hard_reset_threshold:
                drop.append(sid)
        for sid in drop:
            self.ema.pop(sid, None)

    def update(self, text: str, record_history: bool = False) -> TrackerResult:
        text = (text or "").strip()
        if len(text) < self.cfg.min_text_chars:
            self._decay_only()
            return self._snapshot(best_tuk_score=0.0, current_line_idx=-1,
                                  current_line_score=0.0)

        q = self.embedder.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        D, I = self.index.search(q, self.cfg.top_k_tuk)
        hits = list(zip(I[0].tolist(), D[0].tolist()))

        inst: dict[str, float] = {}
        for tuk_id, score in hits:
            sid_line = self._tuk_to_shabad.get(int(tuk_id))
            if sid_line is None:
                continue
            sid, _ = sid_line
            inst[sid] = inst.get(sid, 0.0) + float(score)

        a = self.cfg.ema_alpha
        for sid in set(self.ema) | set(inst):
            self.ema[sid] = (1.0 - a) * self.ema.get(sid, 0.0) + a * inst.get(sid, 0.0)

        for sid in list(self.ema):
            if self.ema[sid] < self.cfg.hard_reset_threshold:
                self.ema.pop(sid, None)

        current_sid: Optional[str] = max(self.ema, key=self.ema.get) if self.ema else None

        current_line_idx = -1
        current_line_score = 0.0
        if current_sid is not None:
            best = None
            for tuk_id, score in hits:
                sid_line = self._tuk_to_shabad.get(int(tuk_id))
                if sid_line is None or sid_line[0] != current_sid:
                    continue
                if best is None or score > best[1]:
                    best = (sid_line[1], float(score))
            if best is not None:
                current_line_idx, current_line_score = best

        top_raw = hits[0][1] if hits else 0.0

        if record_history and current_sid is not None and current_line_idx >= 0 \
                and top_raw >= self.cfg.alap_floor:
            self.history.append(
                HistoryItem(
                    ts=time.time(),
                    shabad_id=current_sid,
                    line_idx=current_line_idx,
                    text=text,
                )
            )
            if len(self.history) > self.cfg.history_cap:
                self.history = self.history[-self.cfg.history_cap :]

        return self._snapshot(
            best_tuk_score=top_raw,
            current_line_idx=current_line_idx,
            current_line_score=current_line_score,
        )

    def _snapshot(
        self,
        best_tuk_score: float,
        current_line_idx: int = -1,
        current_line_score: float = 0.0,
    ) -> TrackerResult:
        current_sid: Optional[str] = max(self.ema, key=self.ema.get) if self.ema else None
        current_info = self.shabads.get(current_sid) if current_sid else None
        alap = best_tuk_score < self.cfg.alap_floor

        ranked = sorted(self.ema.items(), key=lambda kv: kv[1], reverse=True)
        candidates: list[ShabadCandidate] = []
        for sid, score in ranked[: self.cfg.top_k_shabad]:
            info = self.shabads.get(sid)
            if info is not None:
                candidates.append(ShabadCandidate(shabad_id=sid, score=float(score), info=info))

        return TrackerResult(
            current_shabad=current_info,
            current_line_idx=current_line_idx,
            current_line_score=current_line_score,
            alap=alap,
            candidates=candidates,
            history=list(self.history),
        )
