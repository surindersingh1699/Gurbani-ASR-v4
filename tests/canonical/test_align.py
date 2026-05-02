from scripts.canonical.align import AlignConfig, align_nw, realign_orphan_runs
from scripts.canonical.gurmukhi_skeleton import skel
from scripts.canonical.sttm_index import SggsLine


def _mk(line_id: str, tokens: tuple[str, ...]) -> SggsLine:
    return SggsLine(
        line_id=line_id,
        shabad_id="TEST",
        ang=1,
        order_id=0,
        type_id=1,
        unicode=" ".join(tokens),
        skel=skel(" ".join(tokens)),
        tokens=tokens,
        tok_skels=tuple(skel(t) for t in tokens),
    )


class TestMatch:
    def test_exact_match_single_line(self):
        shabad = [_mk("L1", ("ਤੇਰੀ", "ਸਰਣਿ", "ਮੇਰੇ"))]
        ops = align_nw(["ਤੇਰੀ", "ਸਰਣਿ", "ਮੇਰੇ"], shabad, AlignConfig())
        assert [op["op"] for op in ops] == ["match", "match", "match"]


class TestFix:
    def test_matra_only_fix(self):
        shabad = [_mk("L1", ("ਸਰਣਿ",))]
        ops = align_nw(["ਸਰਨ"], shabad, AlignConfig())
        assert len(ops) == 1
        assert ops[0]["op"] == "fix"
        assert ops[0]["sggs"] == ["ਸਰਣਿ"]

    def test_single_consonant_fix(self):
        shabad = [_mk("L1", ("ਉਜਾੜੀ",))]
        ops = align_nw(["ਉਦਾੜੀ"], shabad, AlignConfig())
        assert ops[0]["op"] == "fix"
        assert ops[0]["sggs"] == ["ਉਜਾੜੀ"]


class TestMerge:
    def test_merge_two_caption_into_one_sggs(self):
        shabad = [_mk("L1", ("ਬਦਫੈਲੀ",))]
        ops = align_nw(["ਮਦ", "ਫੈਲੀ"], shabad, AlignConfig())
        assert ops[0]["op"] == "merge"
        assert ops[0]["sggs"] == ["ਬਦਫੈਲੀ"]


class TestSplit:
    def test_split_one_caption_into_two_sggs(self):
        shabad = [_mk("L1", ("ਸਾਕਤ", "ਸੰਗਿ"))]
        ops = align_nw(["ਸਾਕਤਸੰਗਿ"], shabad, AlignConfig())
        assert ops[0]["op"] == "split"
        assert ops[0]["sggs"] == ["ਸਾਕਤ", "ਸੰਗਿ"]


class TestMonotonic:
    def test_no_backward_reuse(self):
        shabad = [_mk("L1", ("ਹਰਿ", "ਰੁਖੀ", "ਰੋਟੀ", "ਖਾਇ", "ਸਮਾਲੇ"))]
        ops = align_nw(
            ["ਰੁਖੀ", "ਰੋਤੀ", "ਖਾਇ", "ਸਮਾਲੇ"], shabad, AlignConfig()
        )
        flat = [t for op in ops if op["op"] != "delete" for t in op["sggs"]]
        assert "ਰੋਟੀ" in flat
        assert flat.count("ਰੁਖੀ") <= 1


class TestMinConsLen:
    def test_reject_pathological_1_cons_match(self):
        shabad = [_mk("L1", ("ਨ", "ਮੇਰੇ"))]
        ops = align_nw(["ਗੁਨ", "ਮੇਰੇ"], shabad, AlignConfig())
        types = [op["op"] for op in ops]
        assert "delete" in types


class TestEqualLengthFixEligibility:
    def test_reject_different_length_1_cons_swap(self):
        # ਹਰਿ (skel=ਹਰ, len=2) vs ਕਰੇ (skel=ਕਰ, len=2) — same length, lev=1,
        # accepted. But in the wild we saw ਹਰਿ→ਕਰੇ style bad rewrites where
        # only one of the two consonants matched. Construct a mismatched case:
        # caption ਤੂੰ (skel=ਤ, len=1) aligned against shabad ਤਿਨੑੀ
        # (skel=ਤਨ, len=2). Under old rules abs(1-2)=1 <= max_edit=1 and
        # lev(ਤ,ਤਨ)=1 — would pass. With equal_length=True this is rejected.
        shabad = [_mk("L1", ("ਤਿਨੑੀ", "ਧੁਰਿ"))]
        ops = align_nw(["ਤੂੰ", "ਧੁਰਿ"], shabad, AlignConfig())
        # ਤੂੰ must not be replaced with ਤਿਨੑੀ. Expect delete (or at worst
        # merge/split via a different branch), but NOT a 1-cons 'fix' swap.
        first_op = ops[0]
        if first_op["op"] == "fix":
            assert first_op["sggs"] != ["ਤਿਨੑੀ"], (
                "1-cons swap across different lengths should be rejected"
            )

    def test_equal_length_1_cons_fix_still_works(self):
        # Regression: legitimate same-length 1-cons swap should still pass.
        # ਉਦਾੜੀ (skel=ਦੜ, len=2) vs ਉਜਾੜੀ (skel=ਜੜ, len=2) — same length,
        # lev=1. Must still produce a 'fix'.
        shabad = [_mk("L1", ("ਉਜਾੜੀ",))]
        ops = align_nw(["ਉਦਾੜੀ"], shabad, AlignConfig())
        assert ops[0]["op"] == "fix"
        assert ops[0]["sggs"] == ["ਉਜਾੜੀ"]


class TestOrphanRealign:
    def test_orphan_realign_runs_without_error(self):
        shabad = [_mk("L1", ("ਤੇਰੀ", "ਸਰਣਿ", "ਮੇਰੇ", "ਦੀਨ", "ਦਇਆਲਾ"))]
        ops = align_nw(
            ["ਤੇਰੀ", "ਸਰਣਿ", "ਮੇਰੇ", "ਦੀਨ", "ਦਇਆਲਾ"],
            shabad,
            AlignConfig(),
        )
        new_ops, used_relaxed = realign_orphan_runs(ops, shabad, AlignConfig())
        assert isinstance(new_ops, list)
        assert isinstance(used_relaxed, bool)
