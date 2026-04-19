"""Sirlekh (shabad-header) normalization + mid-row boundary splitting."""
from __future__ import annotations

import re

_SIRLEKH_SUBS = [
    (re.compile(r"ਸੀ\s*ਰਾਗ\s*ਮਹਲਾ\s*ਪੰ"), "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ ੫"),
    (re.compile(r"ਸੀਰਾਗ\s*ਮਹਲਾ"), "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ"),
    (re.compile(r"ਸ੍ਰੀਰਾਗੁ\s*ਮਹਲਾ[>\W]*"), "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ "),
    (re.compile(r"ਸਿ੍ਰ\s*ਰਾਗੁ?\s*ਮਹਲਾ"), "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ"),
    (re.compile(r"ਸਿ੍ਰੀਰਾਗੁ\s*ਮਹਲਾ"), "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ"),
    (re.compile(r"ਸਿਰੀੀ?\s*ਰਾਗੁ?\s*ਮਹਲਾ"), "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ"),
]


def normalize_sirlekh(text: str) -> str:
    """Rewrite common ASR variants of shabad headers to canonical form."""
    if not text:
        return text
    for pat, repl in _SIRLEKH_SUBS:
        text = pat.sub(repl, text)
    return re.sub(r"\s+", " ", text).strip()


_SIRLEKH_DETECT = re.compile(
    r"(ਸ੍ਰੀ\s*ਰਾਗੁ?|ਸੀ\s*ਰਾਗ|ਸਿ੍ਰ\s*ਰਾਗੁ?|ਸੀਰਾਗ|ਸਿਰੀੀ?\s*ਰਾਗੁ?)\s*ਮਹਲਾ"
)


def split_multi_shabad(text: str, min_offset_chars: int = 4) -> list[str]:
    """Split a row where a Sirlekh pattern appears mid-row. If the only match
    is at/near the row start (offset < min_offset_chars), no split."""
    if not text:
        return [text]
    matches = list(_SIRLEKH_DETECT.finditer(text))
    cuts = [m.start() for m in matches if m.start() >= min_offset_chars]
    if not cuts:
        return [text]
    parts: list[str] = []
    prev = 0
    for c in cuts:
        part = text[prev:c].strip()
        if part:
            parts.append(part)
        prev = c
    tail = text[prev:].strip()
    if tail:
        parts.append(tail)
    return parts
