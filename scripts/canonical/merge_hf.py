# scripts/canonical/merge_hf.py
"""Merge Stage-1 + Stage-2 sidecars into a unified dataset ready for HF push."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def merge_sidecars(stage1_parquet: Path | str, llm_parquet: Path | str | None = None) -> pd.DataFrame:
    s1 = pd.read_parquet(stage1_parquet)
    if llm_parquet and Path(llm_parquet).exists():
        llm = pd.read_parquet(llm_parquet)
        merged = s1.merge(llm, on="clip_id", how="left")
    else:
        merged = s1.copy()
        for c in ("final_text_llm", "llm_model", "llm_verified", "llm_reason"):
            merged[c] = None
    return merged
