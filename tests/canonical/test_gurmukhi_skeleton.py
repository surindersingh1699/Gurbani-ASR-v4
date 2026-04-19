from scripts.canonical.gurmukhi_skeleton import skel, lev, tokenize, clean_token


class TestSkel:
    def test_bare_consonants_preserved(self):
        assert skel("ਗਰ") == "ਗਰ"

    def test_matras_stripped(self):
        assert skel("ਗੁਰੂ") == "ਗਰ"
        assert skel("ਸਰਣਿ") == "ਸਰਣ"

    def test_vishraams_and_digits_stripped(self):
        assert skel("ਗੁਰੂ ॥੧॥") == "ਗਰ"

    def test_nukta_normalized_to_base(self):
        assert skel("ਸ਼ਾਮ") == "ਸਮ"

    def test_bindi_tippi_addak_halant_stripped(self):
        assert skel("ਸੰਤ") == "ਸਤ"
        assert skel("ਅੱਡਾ") == "ਅਡ"

    def test_zwj_zwnj_stripped(self):
        assert skel("ਗ\u200dੁਰੂ") == "ਗਰ"

    def test_empty_and_whitespace(self):
        assert skel("") == ""
        assert skel("   ") == ""


class TestLev:
    def test_equal_strings(self):
        assert lev("abc", "abc") == 0

    def test_one_substitution(self):
        assert lev("ਗਰ", "ਗਣ") == 1

    def test_insertion(self):
        assert lev("ਗਰ", "ਗਰਨ") == 1

    def test_empty(self):
        assert lev("", "ਗਰ") == 2
        assert lev("ਗਰ", "") == 2


class TestTokenize:
    def test_basic_split(self):
        assert tokenize("ਤੇਰੀ ਸਰਣਿ") == ["ਤੇਰੀ", "ਸਰਣਿ"]

    def test_drop_gt_gt(self):
        assert tokenize(">> ਤੇਰੀ >> ਸਰਣਿ") == ["ਤੇਰੀ", "ਸਰਣਿ"]

    def test_collapse_multi_space(self):
        assert tokenize("ਤੇਰੀ    ਸਰਣਿ") == ["ਤੇਰੀ", "ਸਰਣਿ"]

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize(">> >>") == []


class TestCleanToken:
    def test_trailing_vishraam_stripped(self):
        assert clean_token("ਜੀਉ॥") == "ਜੀਉ"

    def test_digit_stripped(self):
        assert clean_token("ਪਿਆਰੇ੧") == "ਪਿਆਰੇ"

    def test_no_change_needed(self):
        assert clean_token("ਤੇਰੀ") == "ਤੇਰੀ"
