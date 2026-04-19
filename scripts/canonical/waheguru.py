"""Hardcoded ਵਾਹਿਗੁਰੂ normalization.

Any token whose consonant skeleton equals the skeleton of ਵਾਹਿਗੁਰੂ ("ਵਹਗਰ")
is rewritten to the canonical spelling. Catches ASR/transcription variants
like ਵਾਹੇਗੁਰੂ, ਵਾਹਿਗੁਰ, ਵਹਿਗੁਰੂ.
"""
from __future__ import annotations

from .gurmukhi_skeleton import skel

CANONICAL_WAHEGURU = "ਵਾਹਿਗੁਰੂ"
WAHEGURU_SKEL = skel(CANONICAL_WAHEGURU)


def normalize_waheguru_tokens(tokens: list[str]) -> list[str]:
    return [
        CANONICAL_WAHEGURU
        if tok and tok != ">>" and skel(tok) == WAHEGURU_SKEL
        else tok
        for tok in tokens
    ]
