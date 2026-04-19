# tests/canonical/test_merge_hf.py
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from scripts.canonical.merge_hf import merge_sidecars


def _make_sidecar_frames(tmp_path):
    stage1 = pd.DataFrame([
        {"clip_id": "a", "text": "ਤੇਰੀ", "sggs_line": "ਤੇਰੀ ਸਰਣਿ",
         "final_text": "ਤੇਰੀ ਸਰਣਿ", "decision": "replaced", "is_simran": False},
        {"clip_id": "b", "text": "ਗ", "sggs_line": None,
         "final_text": "ਗ", "decision": "unchanged", "is_simran": False},
    ])
    llm = pd.DataFrame([
        {"clip_id": "b", "final_text_llm": "ਗੁਰੂ",
         "llm_model": "gemini-3.1-pro-preview", "llm_verified": False,
         "llm_reason": "len_drift(1vs1)"},
    ])
    s1p = tmp_path / "s1.parquet"
    llmp = tmp_path / "llm.parquet"
    stage1.to_parquet(s1p)
    llm.to_parquet(llmp)
    return s1p, llmp


class TestMergeSidecars:
    def test_merge_adds_all_columns(self, tmp_path):
        s1, llm = _make_sidecar_frames(tmp_path)
        df = merge_sidecars(s1, llm)
        cols = set(df.columns)
        assert {"sggs_line", "final_text", "decision", "is_simran",
                "final_text_llm", "llm_model", "llm_verified"}.issubset(cols)
        # row a had no LLM → fields are null
        row_a = df[df["clip_id"] == "a"].iloc[0]
        assert row_a["final_text_llm"] is None or pd.isna(row_a["final_text_llm"])
        # row b has LLM output
        row_b = df[df["clip_id"] == "b"].iloc[0]
        assert row_b["final_text_llm"] == "ਗੁਰੂ"
