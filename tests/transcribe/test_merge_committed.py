"""Unit tests for the raw-transcript seam dedup helper.

The live-mic path appends every commit-window's Whisper output to
`StreamState.committed`. Because of the 2 s carry-over between adjacent
commit windows, the same audio is transcribed twice and the duplicated
words show up glued together. `_merge_committed` strips that overlap
at the word level so the visible transcript reads each pangti once.
"""

from __future__ import annotations

import pytest

from apps.transcribe.app import _merge_committed


def test_empty_prev_returns_new():
    assert _merge_committed("", "ਏਹੁ ਨੀਸਾਣੁ") == "ਏਹੁ ਨੀਸਾਣੁ"


def test_empty_new_returns_prev():
    assert _merge_committed("ਏਹੁ ਨੀਸਾਣੁ", "") == "ਏਹੁ ਨੀਸਾਣੁ"


def test_both_empty_returns_empty():
    assert _merge_committed("", "") == ""


def test_no_overlap_joins_with_single_space():
    assert _merge_committed("ਏਹੁ ਨੀਸਾਣੁ", "ਸਚੁ ਨਾਮੁ") == "ਏਹੁ ਨੀਸਾਣੁ ਸਚੁ ਨਾਮੁ"


def test_two_word_overlap_dropped():
    prev = "ਏਹੁ ਨੀਸਾਣੁ ਸੇਤੀ ਘਰਿ"
    new = "ਸੇਤੀ ਘਰਿ ਜਾਈਐ ਸਚੁ ਨਾਮੁ"
    assert _merge_committed(prev, new) == "ਏਹੁ ਨੀਸਾਣੁ ਸੇਤੀ ਘਰਿ ਜਾਈਐ ਸਚੁ ਨਾਮੁ"


def test_new_fully_contained_in_prev_suffix_returns_prev():
    prev = "ਏਹੁ ਨੀਸਾਣੁ ਸੇਤੀ ਘਰਿ"
    new = "ਸੇਤੀ ਘਰਿ"
    assert _merge_committed(prev, new) == "ਏਹੁ ਨੀਸਾਣੁ ਸੇਤੀ ਘਰਿ"


def test_overlap_at_exact_cap_is_deduped():
    # Natural overlap is exactly 12 words; cap is 12. The full overlap
    # should be found and dropped.
    overlap = [f"w{i}" for i in range(12)]
    prev_only = ["p0", "p1"]
    new_only = ["x", "y"]
    prev = " ".join(prev_only + overlap)
    new = " ".join(overlap + new_only)
    expected = " ".join(prev_only + overlap + new_only)
    assert _merge_committed(prev, new) == expected


def test_max_overlap_words_param_bounds_search():
    # With max_overlap_words=2 and a natural 2-word overlap, the cap
    # still lets us find it.
    prev = "a b c d"
    new = "c d e f"
    assert _merge_committed(prev, new, max_overlap_words=2) == "a b c d e f"


def test_max_overlap_words_too_small_misses_overlap():
    # cap=1: only check 1-word overlap. p[-1:]=["d"], n[:1]=["c"] →
    # no match. Result: full append, no dedup. This documents the
    # limit (over-cap overlaps are not deduped).
    prev = "a b c d"
    new = "c d e f"
    assert _merge_committed(prev, new, max_overlap_words=1) == "a b c d c d e f"


def test_full_overlap_within_cap_returns_prev_when_new_is_only_overlap():
    prev = "a b c d"
    new = "a b c d"
    assert _merge_committed(prev, new) == "a b c d"


def test_whitespace_normalised_on_input():
    assert _merge_committed("  a b  ", "  b c  ") == "a b c"


def test_overlap_only_anchored_at_seam_not_anywhere():
    # "a b" appears in the middle of new but not at its prefix —
    # we only dedup prev's suffix vs new's prefix.
    prev = "x y z"
    new = "p q a b"
    assert _merge_committed(prev, new) == "x y z p q a b"


@pytest.mark.parametrize("prev,new,expected", [
    ("a", "a", "a"),
    ("a b", "b", "a b"),
    ("a b", "b c", "a b c"),
    ("a b c", "c d", "a b c d"),
])
def test_short_overlaps(prev, new, expected):
    assert _merge_committed(prev, new) == expected
