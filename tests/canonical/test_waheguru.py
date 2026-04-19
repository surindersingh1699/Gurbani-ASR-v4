from scripts.canonical.waheguru import (
    CANONICAL_WAHEGURU,
    WAHEGURU_SKEL,
    normalize_waheguru_tokens,
)


class TestWaheguruNormalization:
    def test_canonical_unchanged(self):
        assert normalize_waheguru_tokens(["ਵਾਹਿਗੁਰੂ"]) == ["ਵਾਹਿਗੁਰੂ"]

    def test_matra_variant_normalized(self):
        assert normalize_waheguru_tokens(["ਵਾਹੇਗੁਰੂ"]) == ["ਵਾਹਿਗੁਰੂ"]
        assert normalize_waheguru_tokens(["ਵਾਹਿਗੁਰੁ"]) == ["ਵਾਹਿਗੁਰੂ"]
        assert normalize_waheguru_tokens(["ਵਾਹਿਗੁਰ"]) == ["ਵਾਹਿਗੁਰੂ"]
        assert normalize_waheguru_tokens(["ਵਹਿਗੁਰੂ"]) == ["ਵਾਹਿਗੁਰੂ"]

    def test_mixed_row_only_waheguru_normalized(self):
        toks = ["ਮੇਰਾ", "ਵਾਹੇਗੁਰੂ", "ਹੈ"]
        assert normalize_waheguru_tokens(toks) == ["ਮੇਰਾ", "ਵਾਹਿਗੁਰੂ", "ਹੈ"]

    def test_non_waheguru_unchanged(self):
        toks = ["ਤੇਰੀ", "ਸਰਣਿ"]
        assert normalize_waheguru_tokens(toks) == ["ਤੇਰੀ", "ਸਰਣਿ"]

    def test_empty_and_gt_gt_preserved(self):
        assert normalize_waheguru_tokens([">>", "ਵਾਹੇਗੁਰੂ"]) == [">>", "ਵਾਹਿਗੁਰੂ"]

    def test_constants_exist(self):
        assert CANONICAL_WAHEGURU == "ਵਾਹਿਗੁਰੂ"
        assert WAHEGURU_SKEL == "ਵਹਗਰ"
