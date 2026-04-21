"""EMA smoother for retriever outputs on streaming inputs.

The stateless retriever (`apps.transcribe.retriever`) scores each query in
isolation. For live mic and play-sync, consecutive ASR windows carry partial,
overlapping text — a single noisy window shouldn't bounce the top match.
This wrapper keeps an exponentially-weighted moving average over the
per-shabad scores so the UI stabilises on the true shabad over 2-3 windows.

Only used on live paths. Manual search, file upload, and contenteditable
corrections call the retriever directly (fresh, no history).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievalEMA:
    alpha: float = 0.35   # new-weight; higher = more responsive, less smooth
    decay: float = 0.92   # per-call decay applied to absent shabads
    floor: float = 0.05   # drop shabads that decay below this
    # state
    scores: dict[int, float] = field(default_factory=dict)
    last_hit: dict[int, dict] = field(default_factory=dict)

    def reset(self) -> None:
        self.scores.clear()
        self.last_hit.clear()

    def update(self, hits: list[dict], top_n: int = 5) -> list[dict]:
        """Merge new retriever hits with the running EMA and return a fresh top-N.

        Hits must have `shabadId` (hashable) and `score` in [0, 1].
        Metadata fields (gurmukhi, writer, raag, ang, verseId) are taken from
        the latest hit seen for each shabad — the retriever returns the
        closest-matched tuk's text, which is what we want to display.
        """
        # Decay existing scores. Shabads that dropped off the new top-N still
        # get smoothed down gradually rather than disappearing instantly.
        for sid in list(self.scores):
            self.scores[sid] *= self.decay
            if self.scores[sid] < self.floor:
                self.scores.pop(sid, None)
                self.last_hit.pop(sid, None)

        # Fold in the new scores. If a shabad isn't in the new hits, it keeps
        # decaying from the loop above.
        for h in hits:
            sid = h.get("shabadId")
            if sid is None:
                continue
            new_s = float(h.get("score", 0.0))
            old_s = self.scores.get(sid, 0.0)
            self.scores[sid] = (1.0 - self.alpha) * old_s + self.alpha * new_s
            self.last_hit[sid] = h  # keep latest matched-tuk metadata

        # Rank and return top-N. Clamp the smoothed score to [0, 1] so the UI
        # bar behaves.
        ranked = sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)
        out: list[dict] = []
        for sid, smoothed in ranked[:top_n]:
            base = self.last_hit.get(sid)
            if not base:
                continue
            merged = dict(base)
            merged["score"] = round(min(1.0, max(0.0, float(smoothed))), 3)
            out.append(merged)
        return out
