"""STTM-ASCII skeleton mapping for Gurmukhi.

Skeleton extracts MORE identity than just first letter:
- First letter (current v1/v2/v3 first-letter target)
- Plus the next consonant if present

Example: ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ
- first-letter target: s|n|k|p
- skeleton target:     st|nm|krt|prk

The matcher pipeline:
  1. First-letter head -> 4-gram overlap on first_letters -> top-50 shabads
  2. Skeleton head -> edit-distance rerank against pre-computed skeletons
  3. Top-1 = highest rerank with margin > threshold

This adds disambiguation power for short Gurbani phrases that share first letters.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "canonical"))

from sttm_first_letter_map import (
    UNI_TO_ANCHOR, _COMBINING, _PUNCT_RE, _HYPHEN_WORD_RE,
    _TAIL_DROP_TOKENS, _WHITESPACE_RE, DEFAULT_DELIM, _NONSPOKEN_DB_MARKERS, _DIGITS,
)

# Skeleton extraction: up to N base consonants per word, in order, mapped
# through UNI_TO_ANCHOR. Vowel carriers (a/A/e/E) only count if they ARE the
# first letter (independent vowel words); they don't pad an otherwise-short
# consonant chain.

_MAX_SKELETON_CHARS_PER_WORD = 3
_VOWEL_CARRIERS = {"a", "A", "e", "E"}


def _word_to_skeleton(word: str) -> str:
    """Take a single Gurmukhi word, emit up to N skeleton chars."""
    if not word:
        return ""
    out = []
    for ch in word:
        if ch in _COMBINING:
            continue
        if ch in UNI_TO_ANCHOR:
            mapped = UNI_TO_ANCHOR[ch]
            if not mapped:
                continue
            out.append(mapped)
            if len(out) >= _MAX_SKELETON_CHARS_PER_WORD:
                break
        if ch in _NONSPOKEN_DB_MARKERS or ch in _DIGITS or ch.isspace():
            continue
    if not out:
        return ""
    # Drop trailing vowel-carriers — they're padding, not identity
    while len(out) > 1 and out[-1] in _VOWEL_CARRIERS:
        out.pop()
    return "".join(out)


def _normalize_for_skeleton(text: str) -> list[str]:
    if not text:
        return []
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"[੦-੯0-9]+", " ", text)
    text = text.replace("ੴ", " ")
    text = _HYPHEN_WORD_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    tokens = [t for t in text.split(" ") if t]
    for seq in _TAIL_DROP_TOKENS:
        if len(tokens) >= len(seq) and tuple(tokens[-len(seq):]) == seq:
            tokens = tokens[:-len(seq)]
            break
    return tokens


def gurmukhi_text_to_skeleton_training(text: str, delim: str = DEFAULT_DELIM) -> str:
    """One skeleton token per word, delimiter-separated."""
    out = []
    for tok in _normalize_for_skeleton(text):
        sk = _word_to_skeleton(tok)
        if sk:
            out.append(sk)
    return delim.join(out)


def skeleton_to_search(anchor: str, delim: str = DEFAULT_DELIM) -> str:
    """Strip delimiter for compact DB-comparable form."""
    return anchor.replace(delim, "")


# Vocab inference from training data is the right way; this is a static
# fallback list of all STTM-ASCII letters we expect to see.
SKELETON_VOCAB = sorted(set(c for c in
    "aAbBcCdDeEfFgGhjJkKlmnNpPqQrstTuvVwxX|" if c.strip()))


if __name__ == "__main__":
    # Quick smoke vs the first-letter map
    test_lines = [
        "ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ",
        "ਆਦਿ ਸਚੁ ਜੁਗਾਦਿ ਸਚੁ",
        "ਹਰਿ ਜਨੁ ਅਉਖਧੁ ਸਾਰਿਆ",
    ]
    for line in test_lines:
        print(f"  text:     {line}")
        print(f"  skeleton: {gurmukhi_text_to_skeleton_training(line)}")
        print(f"  search:   {skeleton_to_search(gurmukhi_text_to_skeleton_training(line))}")
        print()
