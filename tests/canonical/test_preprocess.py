from scripts.canonical.preprocess import (
    PreCleanConfig,
    should_drop_row,
    strip_unk_artifacts,
)


class TestStripUnk:
    def test_strips_embedded_unk(self):
        assert strip_unk_artifacts("ਜਾਲunk> ਮੁਕਤੇ") == "ਜਾਲ ਮੁਕਤੇ"

    def test_strips_prefix_unk(self):
        assert strip_unk_artifacts("ਕਢਾਇ<un ਸੀਰਾਗ") == "ਕਢਾਇ ਸੀਰਾਗ"

    def test_strips_short_k_marker(self):
        assert strip_unk_artifacts("ਸਾਲਾਹਿਹਿ<k ਸਭੇ") == "ਸਾਲਾਹਿਹਿ ਸਭੇ"

    def test_full_unk_tag(self):
        result = strip_unk_artifacts("ਪਾਇਆ <unk> ਨਾਮੁ")
        assert result in ("ਪਾਇਆ   ਨਾਮੁ", "ਪਾਇਆ ਨਾਮੁ")

    def test_no_artifacts_unchanged(self):
        assert strip_unk_artifacts("ਤੇਰੀ ਸਰਣਿ") == "ਤੇਰੀ ਸਰਣਿ"

    def test_empty(self):
        assert strip_unk_artifacts("") == ""


class TestShouldDropRow:
    def test_drop_short_duration(self):
        cfg = PreCleanConfig(min_duration_s=1.0)
        assert should_drop_row({"duration_s": 0.5, "text": "ਤੇਰੀ ਸਰਣਿ"}, cfg)

    def test_drop_single_token(self):
        cfg = PreCleanConfig()
        assert should_drop_row({"duration_s": 2.0, "text": "ਜ"}, cfg)
        assert should_drop_row({"duration_s": 2.0, "text": ">> ਜ >>"}, cfg)

    def test_drop_no_gurmukhi(self):
        cfg = PreCleanConfig()
        assert should_drop_row({"duration_s": 2.0, "text": ">> >> 123"}, cfg)

    def test_keep_normal_row(self):
        cfg = PreCleanConfig()
        assert not should_drop_row(
            {"duration_s": 2.5, "text": "ਤੇਰੀ ਸਰਣਿ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ"}, cfg
        )
