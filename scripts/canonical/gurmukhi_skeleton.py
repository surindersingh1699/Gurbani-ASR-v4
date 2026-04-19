"""Gurmukhi consonant-skeleton extraction + Levenshtein distance.

Skeleton = strip matras, vishraams, digits, bindi/tippi/addak/halant,
ZWJ/ZWNJ, and normalize nukta-modified consonants to their base form.
"""
from __future__ import annotations

import re

_SKELETON_STRIP = re.compile(
    r"[\u0A3C\u0A3E-\u0A4D\u0A51\u0A70\u0A71\u0A75"
    r"\u0A66-\u0A6F"
    r"0-9\s\u200C\u200D।॥॰.,;:!?'\"()\[\]<>]+"
)

_NUKTA_PAIRS = [
    ("ਸ਼", "ਸ"), ("ਖ਼", "ਖ"), ("ਗ਼", "ਗ"),
    ("ਜ਼", "ਜ"), ("ਫ਼", "ਫ"), ("ਲ਼", "ਲ"),
]

_VISHRAAM_TOKEN_RE = re.compile(r"^[॥।੦-੯0-9.,;:!?'\"()\[\]]+$")
VISHRAAM_TOKEN_RE = _VISHRAAM_TOKEN_RE


def _denukta(text: str) -> str:
    for compound, base in _NUKTA_PAIRS:
        text = text.replace(compound, base)
    return text


def skel(text: str) -> str:
    """Return the consonant skeleton of a Gurmukhi string."""
    if not text:
        return ""
    return _SKELETON_STRIP.sub("", _denukta(text))


def lev(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


def clean_token(tok: str) -> str:
    """Strip trailing vishraams/digits from a single token."""
    return re.sub(r"[॥।੦-੯0-9.]+$", "", tok).strip()


def tokenize(text: str) -> list[str]:
    """Whitespace-split; drop `>>` markers and empty tokens."""
    return [t for t in text.split() if t and t != ">>"]
