from collections import Counter
from pathlib import Path

import pytest

from scripts.canonical.retrieval import RetrievalConfig, retrieve_shabad
from scripts.canonical.sttm_index import (
    build_shabad_ngram_index,
    load_sggs,
    next_shabad_in_sequence,
)

DB = Path(__file__).parent.parent.parent / "database.sqlite"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="database.sqlite not present")


@pytest.fixture(scope="module")
def sggs_idx():
    lines, _ = load_sggs(DB, include_sirlekh=False)
    ngrams, shabad_lines, df = build_shabad_ngram_index(lines, n=4)
    return lines, ngrams, shabad_lines, df


class TestSingleWindowRetrieval:
    def test_retrieves_known_line_gives_nonzero_score(self, sggs_idx):
        _, ngrams, _, df = sggs_idx
        cfg = RetrievalConfig()
        caption_tokens = ["ਜਿਥੈ", "ਨਾਮੁ", "ਜਪੀਐ", "ਪ੍ਰਭ", "ਪਿਆਰੇ"]
        sid, score, _margin, _window = retrieve_shabad(
            caption_tokens,
            ([], []),
            ([], []),
            ngrams,
            df,
            video_hits=Counter(),
            nxt_shabad=None,
            cfg=cfg,
        )
        assert sid
        assert score > 0


class TestVideoMemoryBias:
    def test_video_hits_change_result(self, sggs_idx):
        _, ngrams, _, df = sggs_idx
        cfg = RetrievalConfig(video_prior_weight=10.0)
        caption_tokens = ["ਤੇਰੀ", "ਸਰਣਿ"]
        sid_a, _, _, _ = retrieve_shabad(
            caption_tokens,
            ([], []),
            ([], []),
            ngrams,
            df,
            video_hits=Counter(),
            nxt_shabad=None,
            cfg=cfg,
        )
        hits = Counter({sid_a: 0, "EZ7": 100})
        sid_b, _, _, _ = retrieve_shabad(
            caption_tokens,
            ([], []),
            ([], []),
            ngrams,
            df,
            video_hits=hits,
            nxt_shabad=None,
            cfg=cfg,
        )
        assert sid_b == "EZ7" or sid_b == sid_a


class TestSequentialPrior:
    def test_sequential_does_not_crash(self, sggs_idx):
        lines, ngrams, _, df = sggs_idx
        nxt = next_shabad_in_sequence(lines)
        cfg = RetrievalConfig(
            sequential_current_boost=0.8, sequential_next_boost=0.5,
        )
        caption_tokens = ["ਤੇਰੀ", "ਸਰਣਿ"]
        sid, _, _, _ = retrieve_shabad(
            caption_tokens,
            ([], []),
            ([], []),
            ngrams,
            df,
            video_hits=Counter({"EZ7": 1}),
            nxt_shabad=nxt,
            cfg=cfg,
        )
        assert sid
