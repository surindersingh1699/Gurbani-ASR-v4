from scripts.canonical.sirlekh import normalize_sirlekh, split_multi_shabad


class TestNormalizeSirlekh:
    def test_sri_raag_mahala_5(self):
        assert "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ ੫" in normalize_sirlekh("ਸੀ ਰਾਗ ਮਹਲਾ ਪੰ")

    def test_sri_raag_mahala_generic(self):
        assert "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ" in normalize_sirlekh("ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ>")

    def test_siri_raag_variant(self):
        assert "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ" in normalize_sirlekh("ਸਿ੍ਰ ਰਾਗੁ ਮਹਲਾ")

    def test_no_change_needed(self):
        assert normalize_sirlekh("ਤੇਰੀ ਸਰਣਿ") == "ਤੇਰੀ ਸਰਣਿ"


class TestMultiShabadSplit:
    def test_no_split_when_no_header(self):
        out = split_multi_shabad("ਤੇਰੀ ਸਰਣਿ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ")
        assert out == ["ਤੇਰੀ ਸਰਣਿ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ"]

    def test_split_on_midrow_sirlekh(self):
        row = "ਕਢਾਇ ਸੀਰਾਗ ਮਹਲਾ ਘੜੀ ਮੁਹਤ ਕਾ ਪਾਹੁਣਾ"
        parts = split_multi_shabad(row)
        assert len(parts) == 2
        assert "ਕਢਾਇ" in parts[0]
        assert "ਘੜੀ ਮੁਹਤ" in parts[1]

    def test_no_split_at_row_start(self):
        row = "ਸੀਰਾਗ ਮਹਲਾ ਘੜੀ ਮੁਹਤ ਕਾ ਪਾਹੁਣਾ"
        parts = split_multi_shabad(row)
        assert parts == [row]
