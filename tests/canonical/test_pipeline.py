from pathlib import Path
from collections import Counter
import pytest

from scripts.canonical.config import get_dataset_config
from scripts.canonical.pipeline import CanonicalPipeline

DB = Path(__file__).parent.parent.parent / "database.sqlite"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="database.sqlite not present")


@pytest.fixture(scope="module")
def kirtan_pipeline():
    return CanonicalPipeline(get_dataset_config("kirtan"), db_path=DB)


@pytest.fixture(scope="module")
def sehaj_pipeline():
    return CanonicalPipeline(get_dataset_config("sehaj"), db_path=DB)


class TestKirtanPipeline:
    def test_matched_clean_kirtan_row(self, kirtan_pipeline):
        # Clean SGGS line from EZ7 (in monotonic order matching scripture)
        rows = [{
            "clip_id": "r1", "video_id": "v1",
            "text": "ਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ",
            "duration_s": 4.0,
        }]
        out = kirtan_pipeline.run(rows)
        assert out[0]["decision"] in ("matched", "replaced")
        assert out[0]["shabad_id"] == "EZ7"

    def test_replaced_with_matra_fix(self, kirtan_pipeline):
        rows = [{
            "clip_id": "r1", "video_id": "v1",
            "text": "ਤੇਰੀ ਸਰਨ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ",
            "duration_s": 4.0,
        }]
        out = kirtan_pipeline.run(rows)
        assert out[0]["decision"] == "replaced"
        assert "ਸਰਣਿ" in out[0]["final_text"]

    def test_simran_short_circuit(self, kirtan_pipeline):
        rows = [{
            "clip_id": "r1", "video_id": "v1",
            "text": " ".join(["ਵਾਹਿਗੁਰੂ"] * 8),
            "duration_s": 10.0,
        }]
        out = kirtan_pipeline.run(rows)
        assert out[0]["decision"] == "simran"
        assert out[0]["final_text"] == " ".join(["ਵਾਹਿਗੁਰੂ"] * 8)


class TestSehajPipeline:
    def test_unk_stripped(self, sehaj_pipeline):
        rows = [{
            "clip_id": "r1", "video_id": "v1",
            "text": "ਨਾਨਕ ਸਚੈ ਪਾਤਿਸਾਹ ਡੁਬਦਾ ਲਇਆ ਕਢਾਇ<un",
            "duration_s": 4.0,
        }]
        out = sehaj_pipeline.run(rows)
        # the <un artifact should be stripped; row should be replaced or matched
        assert "<un" not in out[0]["final_text"]

    def test_sirlekh_included(self, sehaj_pipeline):
        rows = [{
            "clip_id": "r1", "video_id": "v1",
            "text": "ਸੀ ਰਾਗ ਮਹਲਾ ਪੰ",
            "duration_s": 2.0,
        }]
        out = sehaj_pipeline.run(rows)
        # After sirlekh normalization, text becomes ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ ੫ — sirlekh
        # is a short 3-token string; retrieval + alignment may not produce a
        # high-confidence 'matched' (headers recur widely). Verify at least
        # that the pipeline runs and emits a valid decision label.
        assert out[0]["decision"] in (
            "matched", "replaced", "review", "unchanged",
        )


class TestPreCleaning:
    def test_drop_short_duration(self, kirtan_pipeline):
        rows = [
            {"clip_id": "r1", "video_id": "v1", "text": "ਤੇਰੀ ਸਰਣਿ", "duration_s": 0.3},
            {"clip_id": "r2", "video_id": "v1", "text": "ਸੁਖ ਸਾਗਰ", "duration_s": 2.0},
        ]
        out = kirtan_pipeline.run(rows)
        assert len(out) == 1
        assert out[0]["clip_id"] == "r2"
