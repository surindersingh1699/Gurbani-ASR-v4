from pathlib import Path

import pytest

from scripts.canonical.gurmukhi_skeleton import skel
from scripts.canonical.sttm_index import (
    SggsLine,
    build_shabad_ngram_index,
    load_sggs,
    next_shabad_in_sequence,
)

DB_PATH = Path(__file__).parent.parent.parent / "database.sqlite"


@pytest.mark.skipif(not DB_PATH.exists(), reason="database.sqlite not present")
class TestLoadSggs:
    def test_loads_nonzero_lines(self):
        lines, _ = load_sggs(DB_PATH, include_sirlekh=False)
        assert len(lines) > 50_000

    def test_include_sirlekh_adds_rows(self):
        without, _ = load_sggs(DB_PATH, include_sirlekh=False)
        with_sl, _ = load_sggs(DB_PATH, include_sirlekh=True)
        assert len(with_sl) > len(without)

    def test_lines_have_required_fields(self):
        lines, _ = load_sggs(DB_PATH, include_sirlekh=False)
        sample = lines[0]
        assert isinstance(sample, SggsLine)
        assert sample.shabad_id
        assert sample.ang > 0
        assert len(sample.tokens) > 0
        assert len(sample.tokens) == len(sample.tok_skels)

    def test_global_token_index_has_common_tokens(self):
        _, idx = load_sggs(DB_PATH, include_sirlekh=False)
        key = skel("ਸਤਿਗੁਰੁ")
        assert key in idx
        assert len(idx[key]) > 10


@pytest.mark.skipif(not DB_PATH.exists(), reason="database.sqlite not present")
class TestNgramIndex:
    def test_index_populated(self):
        lines, _ = load_sggs(DB_PATH, include_sirlekh=False)
        ngrams_by_shabad, shabad_lines, df = build_shabad_ngram_index(lines, n=4)
        assert len(ngrams_by_shabad) > 5000
        assert len(shabad_lines) == len(ngrams_by_shabad)
        assert sum(df.values()) > 100_000


@pytest.mark.skipif(not DB_PATH.exists(), reason="database.sqlite not present")
def test_next_shabad_sequence():
    lines, _ = load_sggs(DB_PATH, include_sirlekh=False)
    nxt = next_shabad_in_sequence(lines)
    assert len(nxt) > 5000
    for sid, after in nxt.items():
        assert sid != after
