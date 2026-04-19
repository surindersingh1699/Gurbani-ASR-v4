"""Pre-cleaning: drop rows we can't meaningfully process; strip ASR artifacts."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .gurmukhi_skeleton import tokenize


def strip_unk_artifacts(text: str) -> str:
    """Replace ASR `<unk>` / `<un` / `<k>` etc. with spaces, then collapse."""
    if not text:
        return text
    out = re.sub(r"unk>|un>|<unk?>?|<k>?|<un", " ", text)
    return re.sub(r"\s+", " ", out).strip()


@dataclass
class PreCleanConfig:
    min_duration_s: float = 1.0
    min_gurmukhi_tokens: int = 2
    flag_slow_ratio: float = 4.0
    flag_fast_ratio: float = 0.1


def _gurmukhi_token_count(text: str) -> int:
    stripped = strip_unk_artifacts(text)
    toks = tokenize(stripped)
    gur = [t for t in toks if any("\u0A00" <= ch <= "\u0A7F" for ch in t)]
    return len(gur)


def should_drop_row(row: dict, cfg: PreCleanConfig) -> bool:
    if row.get("duration_s", 0) < cfg.min_duration_s:
        return True
    return _gurmukhi_token_count(row.get("text", "")) < cfg.min_gurmukhi_tokens


def ratio_outlier_flag(row: dict, cfg: PreCleanConfig) -> bool:
    n = _gurmukhi_token_count(row.get("text", ""))
    if n <= 0:
        return True
    r = row.get("duration_s", 0) / n
    return r > cfg.flag_slow_ratio or r < cfg.flag_fast_ratio
