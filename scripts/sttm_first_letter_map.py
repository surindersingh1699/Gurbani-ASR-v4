"""STTM-ASCII first-letter mapping for Gurmukhi.

PLAN.md M1.1 — Gurmukhi (Unicode) -> STTM-ASCII first-letter token per word.

The output letter set matches `database.sqlite.lines.first_letters`, which is
derived from the AnmolLipi/GurbaniLipi Latin font encoding. So the mapping
codomain is the AnmolLipi base-consonant / vowel-carrier ASCII alphabet, and
the training target for one Gurmukhi word is exactly one such ASCII character.

Two anchor forms:

  training_anchor:  "s|n|k|p"   (delimiter-separated; CTC-friendly)
  search_anchor:    "snkpsg"    (compact; comparable to DB first_letters)

Non-spoken markers ([], <>, digits, vishraams) are stripped before search /
training. The DB round-trip test verifies the codomain matches at >=99%.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

DEFAULT_DELIM = "|"

# ---------------------------------------------------------------------------
# Unicode Gurmukhi -> STTM-ASCII first-letter map
# Codomain matches AnmolLipi single-char base alphabet (see
# scripts/gurmukhi_converter.py SINGLE_CHAR_MAP / MULTI_CHAR_MAP).
# ---------------------------------------------------------------------------

UNI_TO_ANCHOR: dict[str, str] = {
    # Vowel carriers (standalone)
    "ੳ": "a",   # U+0A73 ura
    "ਅ": "A",   # U+0A05 aira
    "ੲ": "e",   # U+0A72 iri
    # Independent vowels
    "ਆ": "A",   # U+0A06  (AnmolLipi "Aw" -> first char 'A')
    "ਇ": "e",   # U+0A07  ("ei" -> 'e')
    "ਈ": "e",   # U+0A08  ("eI" -> 'e')
    "ਉ": "a",   # U+0A09  ("au" -> 'a')
    "ਊ": "a",   # U+0A0A  ("aU" -> 'a')
    "ਏ": "e",   # U+0A0F  ("ey" -> 'e')
    "ਐ": "A",   # U+0A10  ("AY" -> 'A')
    "ਓ": "a",   # U+0A13  — DB convention: Oankar sound uses 'a' vowel carrier,
                #           not the AnmolLipi visual 'E'. Verified on 'EAMkwru' rows.
    "ਔ": "A",   # U+0A14  ("AO" -> 'A')
    # Consonants (row 1 - velars)
    "ਕ": "k", "ਖ": "K", "ਗ": "g", "ਘ": "G", "ਙ": "|",
    # Palatals
    "ਚ": "c", "ਛ": "C", "ਜ": "j", "ਝ": "J", "ਞ": "\\",
    # Retroflex
    "ਟ": "t", "ਠ": "T", "ਡ": "f", "ਢ": "F", "ਣ": "x",
    # Dental
    "ਤ": "q", "ਥ": "Q", "ਦ": "d", "ਧ": "D", "ਨ": "n",
    # Labial
    "ਪ": "p", "ਫ": "P", "ਬ": "b", "ਭ": "B", "ਮ": "m",
    # Semivowel / liquid / fricative
    "ਯ": "X", "ਰ": "r", "ਲ": "l", "ਵ": "v", "ੜ": "V",
    "ਸ": "s", "ਹ": "h",
    # Nukta-modified consonants -> base (DB strips nukta in first_letters)
    "ਸ਼": "s", "ਖ਼": "K", "ਗ਼": "g", "ਜ਼": "j", "ਫ਼": "P", "ਲ਼": "l",
    # Ik Onkar — non-spoken in anchor sequence; map to empty so it's skipped
    "ੴ": "",
}

# Combining marks / matras / virama / nukta / ZWJ / ZWNJ — skip when scanning
# for first base letter.
_COMBINING = set(
    chr(c) for c in range(0x0A01, 0x0A04)  # bindi family
) | set(
    chr(c) for c in range(0x0A3C, 0x0A4E)  # nukta + matras + virama
) | {
    "ੑ", "ੰ", "ੱ", "ੵ",  # udaat, tippi, addak, yakash
    "‌", "‍",                       # ZWNJ, ZWJ
}

# Per-row non-spoken markers / punctuation we strip from BOTH sides when
# comparing to DB first_letters or generating search anchors.
_NONSPOKEN_DB_MARKERS = set("<>[]|.,;:!?\"'()/-\\")
_DIGITS = set("0123456789")
_PUNCT_RE = re.compile(r"[।॥॰\.,;:!?\"'()\[\]/]+")
_WHITESPACE_RE = re.compile(r"\s+")
# Hyphen-joined words count as a single anchor token in the DB. Strip the
# hyphen and any trailing fragment so only the first sub-word contributes.
_HYPHEN_WORD_RE = re.compile(r"-+\S*")

# Section-end / pause markers that are NOT counted in DB.first_letters even
# though they are written words. Strip them from the tail of a line before
# emitting the anchor. Order matters: longer sequences first.
_TAIL_DROP_TOKENS: tuple[tuple[str, ...], ...] = (
    ("ਰਹਾਉ", "ਦੂਜਾ"),
    ("ਰਹਾਉ", "ਤੀਜਾ"),
    ("ਰਹਾਉ",),
    ("ਅਫਜੂੰ",),
    ("ਅਫਜੂ",),
    ("ਸੁਧੁ",),
    ("ਛਕਾ",),
    ("ਛਕੇ",),
    ("ਇਕੁ",),
    ("ਦੁਇ",),
    ("ਤ੍ਰੈ",),
    ("ਚਾਰਿ",),
    ("ਅਠਿ",),
    ("ਬਾਰਾਂ",),
    ("ਤੇਰਾਂ",),
    ("ਚਉਦਾਂ",),
)


def _first_anchor_char(word: str) -> str:
    """Return the STTM-ASCII anchor char for one Gurmukhi word, or '' if none."""
    if not word:
        return ""
    for ch in word:
        if ch in _COMBINING:
            continue
        if ch in UNI_TO_ANCHOR:
            return UNI_TO_ANCHOR[ch]
        # Sometimes the source already has AnmolLipi-style ASCII (e.g. <>).
        # Skip those — they are non-spoken markers, not Gurmukhi.
        if ch in _NONSPOKEN_DB_MARKERS or ch in _DIGITS or ch.isspace():
            continue
        # Unknown character — try to skip but keep scanning so we still find
        # a base consonant later in the word.
        continue
    return ""


def normalize_for_anchor(text: str) -> list[str]:
    """Tokenize a Gurmukhi line into anchor-bearing words.

    - Strips ॥, ।, digits (verse markers).
    - Drops Ik Onkar (ੴ) as a standalone token — it has no anchor char.
    - Returns whitespace-split tokens that contain at least one base letter.
    """
    if not text:
        return []
    # Strip vishraams and punctuation
    text = _PUNCT_RE.sub(" ", text)
    # Strip digits (Gurmukhi or ASCII)
    text = re.sub(r"[੦-੯0-9]+", " ", text)
    # Drop standalone Ik Onkar — sometimes present as ੴ alone
    text = text.replace("ੴ", " ")
    # Collapse hyphen-joined compounds to their first sub-word (matches DB).
    text = _HYPHEN_WORD_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    tokens = [t for t in text.split(" ") if t]
    # Strip known tail markers (rahau, afzun, sudhu, etc.) one pass.
    for seq in _TAIL_DROP_TOKENS:
        if len(tokens) >= len(seq) and tuple(tokens[-len(seq):]) == seq:
            tokens = tokens[:-len(seq)]
            break
    return tokens


def gurmukhi_text_to_training_anchor(text: str, delim: str = DEFAULT_DELIM) -> str:
    """Convert one line of Gurmukhi to delimiter-separated first-letter anchor.

    Empty words (only matras / unknown chars) are silently dropped.
    """
    tokens = normalize_for_anchor(text)
    out = []
    for tok in tokens:
        ch = _first_anchor_char(tok)
        if ch:
            out.append(ch)
    return delim.join(out)


def training_anchor_to_search_anchor(anchor: str, delim: str = DEFAULT_DELIM) -> str:
    """Strip the delimiter to make a compact DB-comparable anchor."""
    return anchor.replace(delim, "")


def gurmukhi_text_to_search_anchor(text: str) -> str:
    return training_anchor_to_search_anchor(gurmukhi_text_to_training_anchor(text))


def db_first_letters_to_search_anchor(first_letters: str) -> str:
    """Normalize DB.first_letters to a compact spoken-only anchor."""
    return "".join(
        ch for ch in (first_letters or "")
        if ch not in _NONSPOKEN_DB_MARKERS and ch not in _DIGITS and not ch.isspace()
    )


# ---------------------------------------------------------------------------
# Round-trip test against database.sqlite.lines
# ---------------------------------------------------------------------------

def _ascii_to_unicode_for_test():
    """Lazy import to avoid hard dep on gurmukhi_converter at module load."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gurmukhi_converter import ascii_to_unicode  # noqa: E402
    return ascii_to_unicode


def round_trip(db_path: str, sample_limit: int | None = None) -> dict:
    """Compare our mapper output to DB.first_letters for every line.

    DB.gurmukhi is AnmolLipi ASCII -> convert to Unicode -> map -> compare to
    DB.first_letters after both are stripped of non-spoken markers.
    """
    ascii_to_unicode = _ascii_to_unicode_for_test()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    q = "SELECT gurmukhi, first_letters FROM lines"
    if sample_limit:
        q += f" LIMIT {sample_limit}"
    cur.execute(q)
    total = exact = 0
    mismatches: list[tuple[str, str, str]] = []
    for ascii_text, db_fl in cur:
        if not ascii_text or db_fl is None:
            continue
        total += 1
        uni = ascii_to_unicode(ascii_text)
        pred = gurmukhi_text_to_search_anchor(uni)
        ref = db_first_letters_to_search_anchor(db_fl)
        if pred == ref:
            exact += 1
        elif len(mismatches) < 20:
            mismatches.append((ascii_text, ref, pred))
    conn.close()
    return {
        "total": total,
        "exact": exact,
        "rate": (exact / total) if total else 0.0,
        "mismatches": mismatches,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(Path(__file__).resolve().parents[1] / "database.sqlite"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional row cap for fast smoke runs.")
    ap.add_argument("--show-mismatches", action="store_true")
    args = ap.parse_args()

    print(f"[round-trip] DB: {args.db}")
    res = round_trip(args.db, args.limit)
    print(f"[round-trip] total={res['total']}  exact={res['exact']}  rate={res['rate']:.4f}")
    if args.show_mismatches and res["mismatches"]:
        print("[round-trip] first 20 mismatches (ascii_text | ref | pred):")
        for a, r, p in res["mismatches"]:
            print(f"  {a!r:40s} | {r!r:30s} | {p!r}")
    return 0 if res["rate"] >= 0.99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
