"""Main pipeline driver. Composes preprocess → simran → retrieval →
align → realign → decide → render.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import DatasetConfig
from .preprocess import (
    PreCleanConfig, should_drop_row, strip_unk_artifacts,
)
from .simran import SimranConfig, is_simran, apply_simran_quota
from .waheguru import normalize_waheguru_tokens
from .sirlekh import normalize_sirlekh, split_multi_shabad
from .retrieval import RetrievalConfig, retrieve_shabad
from .align import AlignConfig, align_nw, realign_orphan_runs
from .decision import DecisionConfig, decide, render_outputs
from .sttm_index import (
    load_sggs, build_shabad_ngram_index, next_shabad_in_sequence,
)
from .gurmukhi_skeleton import tokenize


@dataclass
class PipelineResult:
    """Single-row output. Lines up with dataset columns."""
    clip_id: str
    video_id: str
    text: str                    # passed through unchanged
    sggs_line: str | None
    final_text: str
    decision: str
    is_simran: bool
    # audit (written to sidecar, not dataset)
    shabad_id: str | None = None
    line_ids: list[str] | None = None
    match_score: float | None = None
    op_counts: dict | None = None
    retrieval_margin: float | None = None


class CanonicalPipeline:
    def __init__(self, cfg: DatasetConfig, db_path: Path | str):
        self.cfg = cfg
        self.preclean_cfg = PreCleanConfig()
        self.simran_cfg = SimranConfig()
        self.retrieval_cfg = RetrievalConfig(
            sequential_current_boost=0.8 if cfg.sequential_shabad_retrieval else 0.0,
            sequential_next_boost=0.5 if cfg.sequential_shabad_retrieval else 0.0,
        )
        self.align_cfg = AlignConfig()
        self.decision_cfg = DecisionConfig()

        self.lines, self.global_tok_idx = load_sggs(
            db_path, include_sirlekh=cfg.include_sirlekh,
        )
        self.shabad_ngrams, self.shabad_lines, self.df = build_shabad_ngram_index(
            self.lines, n=self.retrieval_cfg.ngram_n,
        )
        self.nxt_shabad = (
            next_shabad_in_sequence(self.lines)
            if cfg.sequential_shabad_retrieval else None
        )

    def run(self, rows: list[dict]) -> list[dict]:
        """Process a list of HF-dataset rows. Returns list of dicts with
        all original fields + the new canonical columns."""
        # Phase -1: drop dead rows
        kept = [r for r in rows if not should_drop_row(r, self.preclean_cfg)]

        # Phase 0a: normalize text (doesn't modify row['text'] in-place)
        for r in kept:
            r["_clean_text"] = r["text"]
            if self.cfg.strip_unk_artifacts:
                r["_clean_text"] = strip_unk_artifacts(r["_clean_text"])
            if self.cfg.include_sirlekh:
                r["_clean_text"] = normalize_sirlekh(r["_clean_text"])

        # Phase 0b: simran detection
        if self.cfg.simran_detection:
            for r in kept:
                toks = tokenize(r["_clean_text"])
                r["is_simran"] = is_simran(toks, self.simran_cfg)
            kept = apply_simran_quota(kept, self.simran_cfg)
        else:
            for r in kept:
                r["is_simran"] = False

        # Phase 1-4: per-row processing
        out: list[dict] = []
        by_video_hits: dict[str, Counter[str]] = {}
        for i, r in enumerate(kept):
            video_hits = by_video_hits.setdefault(r["video_id"], Counter())
            result = self._process_row(r, i, kept, video_hits)
            if result["decision"] in ("matched", "replaced", "review"):
                video_hits[result["shabad_id"]] += 1
            out.append(result)
        return out

    def _neighbors(self, rows, i):
        prev_t = tokenize(rows[i - 1]["_clean_text"]) if i > 0 else []
        next_t = tokenize(rows[i + 1]["_clean_text"]) if i + 1 < len(rows) else []
        prev2_t = tokenize(rows[i - 2]["_clean_text"]) if i > 1 else []
        next2_t = tokenize(rows[i + 2]["_clean_text"]) if i + 2 < len(rows) else []
        return (prev_t, next_t), (prev2_t, next2_t)

    def _process_row(self, r, i, rows, video_hits) -> dict:
        clean_text = r["_clean_text"]
        # Simran short-circuit
        if r.get("is_simran"):
            toks = normalize_waheguru_tokens(tokenize(clean_text))
            return {
                **r, "sggs_line": None,
                "final_text": " ".join(toks),
                "decision": "simran", "is_simran": True,
                "shabad_id": None, "line_ids": [], "match_score": None,
                "op_counts": {}, "retrieval_margin": None,
            }

        # Multi-shabad split (if enabled)
        if self.cfg.split_multi_shabad_rows:
            parts = split_multi_shabad(clean_text)
        else:
            parts = [clean_text]

        # Process each sub-part independently, concatenate results
        per_part_results = []
        for part in parts:
            cap_tokens = tokenize(part)
            # Always apply waheguru normalization to caption tokens before alignment
            cap_tokens = normalize_waheguru_tokens(cap_tokens)
            if not cap_tokens:
                continue
            pr = self._align_one(cap_tokens, i, rows, video_hits)
            per_part_results.append(pr)

        if not per_part_results:
            return {
                **r, "sggs_line": None,
                "final_text": clean_text,
                "decision": "unchanged", "shabad_id": None,
                "line_ids": [], "match_score": 0.0, "op_counts": {},
                "retrieval_margin": 0.0,
            }

        if len(per_part_results) == 1:
            pr = per_part_results[0]
        else:
            # Concatenate. Decision = worst of sub-parts (unchanged > review > replaced > matched)
            order = {"matched": 0, "replaced": 1, "review": 2, "unchanged": 3, "simran": 4}
            combined_text = " ".join(p["final_text"] for p in per_part_results)
            combined_sggs = " ".join(
                p["sggs_line"] for p in per_part_results if p.get("sggs_line")
            ) or None
            combined_line_ids = []
            for p in per_part_results:
                combined_line_ids.extend(p.get("line_ids") or [])
            worst = max(per_part_results, key=lambda p: order.get(p["decision"], 99))
            pr = {
                "final_text": combined_text,
                "sggs_line": combined_sggs,
                "decision": worst["decision"],
                "shabad_id": worst.get("shabad_id"),
                "line_ids": combined_line_ids,
                "match_score": worst.get("match_score"),
                "op_counts": worst.get("op_counts", {}),
                "retrieval_margin": worst.get("retrieval_margin"),
            }

        return {
            **r,
            "sggs_line": pr["sggs_line"],
            "final_text": pr["final_text"],
            "decision": pr["decision"],
            "is_simran": False,
            "shabad_id": pr.get("shabad_id"),
            "line_ids": pr.get("line_ids"),
            "match_score": pr.get("match_score"),
            "op_counts": pr.get("op_counts", {}),
            "retrieval_margin": pr.get("retrieval_margin"),
        }

    def _align_one(self, cap_tokens, i, rows, video_hits) -> dict:
        pn, pn2 = self._neighbors(rows, i)
        sid, score, margin, _ = retrieve_shabad(
            cap_tokens, pn, pn2,
            self.shabad_ngrams, self.df,
            video_hits, self.nxt_shabad, self.retrieval_cfg,
        )
        if not sid or score < 2.0:
            return {
                "sggs_line": None,
                "final_text": " ".join(cap_tokens),
                "decision": "unchanged",
                "shabad_id": sid or None,
                "line_ids": [],
                "match_score": 0.0,
                "op_counts": {},
                "retrieval_margin": margin,
            }
        shabad_lines = self.shabad_lines[sid]
        ops = align_nw(cap_tokens, shabad_lines, self.align_cfg)
        ops, used_relaxed = realign_orphan_runs(ops, shabad_lines, self.align_cfg)
        decision = decide(ops, used_relaxed, self.decision_cfg)
        render = render_outputs(ops, cap_tokens, decision)
        op_counts = {
            op: sum(1 for o in ops if o["op"] == op)
            for op in ("match", "fix", "merge", "split", "delete")
        }
        total = sum(op_counts.values())
        score_pct = (
            sum(v for k, v in op_counts.items() if k != "delete") / total
            if total > 0 else 0.0
        )
        return {
            "sggs_line": render["sggs_line"],
            "final_text": render["final_text"],
            "decision": decision,
            "shabad_id": sid,
            "line_ids": render["line_ids"],
            "match_score": score_pct,
            "op_counts": op_counts,
            "retrieval_margin": margin,
        }
