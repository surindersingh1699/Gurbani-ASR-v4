from scripts.canonical.decision import DecisionConfig, decide, render_outputs


def _mk_ops(match=0, fix=0, merge=0, split=0, delete=0, used_relaxed=False):
    ops = []
    ops.extend(
        [{"op": "match", "cap": ["x"], "sggs": ["x"], "line_ids": ["L"]}] * match
    )
    ops.extend(
        [{"op": "fix", "cap": ["a"], "sggs": ["b"], "line_ids": ["L"]}] * fix
    )
    ops.extend(
        [{"op": "merge", "cap": ["a", "b"], "sggs": ["c"], "line_ids": ["L"]}]
        * merge
    )
    ops.extend(
        [{"op": "split", "cap": ["a"], "sggs": ["b", "c"], "line_ids": ["L"]}]
        * split
    )
    ops.extend(
        [{"op": "delete", "cap": ["x"], "sggs": ["x"], "line_ids": []}] * delete
    )
    return ops, used_relaxed


class TestDecide:
    def test_all_match_no_fix(self):
        ops, relaxed = _mk_ops(match=5)
        assert decide(ops, relaxed, DecisionConfig()) == "matched"

    def test_matched_with_fix(self):
        ops, relaxed = _mk_ops(match=4, fix=1)
        assert decide(ops, relaxed, DecisionConfig()) == "replaced"

    def test_review_relaxed_edit(self):
        ops, relaxed = _mk_ops(match=4, fix=1, used_relaxed=True)
        assert decide(ops, relaxed, DecisionConfig()) == "review"

    def test_review_moderate_score(self):
        ops, relaxed = _mk_ops(match=3, fix=1, delete=1)
        assert decide(ops, relaxed, DecisionConfig()) == "review"

    def test_unchanged_low_score(self):
        ops, relaxed = _mk_ops(match=1, delete=3)
        assert decide(ops, relaxed, DecisionConfig()) == "unchanged"

    def test_replaced_requires_accept_threshold_0_95(self):
        # Under the tightened 0.95 accept_threshold a 4-of-5 (score=0.8) no
        # longer lands as 'replaced'. It demotes to 'review' so the LLM pass
        # can double-check.
        ops, relaxed = _mk_ops(match=3, fix=1, delete=1)
        assert decide(ops, relaxed, DecisionConfig()) == "review"

    def test_review_starts_at_0_60(self):
        # Score = 3/5 = 0.60 — at the tightened review_threshold boundary.
        ops, relaxed = _mk_ops(match=2, fix=1, delete=2)
        assert decide(ops, relaxed, DecisionConfig()) == "review"

    def test_below_0_60_is_unchanged(self):
        # Score = 2/5 = 0.40 — below review_threshold, stays unchanged.
        ops, relaxed = _mk_ops(match=1, fix=1, delete=3)
        assert decide(ops, relaxed, DecisionConfig()) == "unchanged"


class TestBoundaryChurnGuard:
    def test_two_leading_diff_skel_fixes_demote_to_review(self):
        # Leading run of 2 diff-skel 1-cons fixes → demote to review even
        # though score=1.0. Catches reorder-snap pathology.
        ops = [
            {"op": "fix", "cap": ["ਜੀਵੈ"], "sggs": ["ਜਾਣੈ"], "line_ids": ["L"]},
            {"op": "fix", "cap": ["ਮਰੀਆ"], "sggs": ["ਜਰੀਆ"], "line_ids": ["L"]},
            {"op": "match", "cap": ["ਮਰਿ"], "sggs": ["ਮਰਿ"], "line_ids": ["L"]},
            {"op": "match", "cap": ["ਜੀਵੈ"], "sggs": ["ਜੀਵੈ"], "line_ids": ["L"]},
            {"op": "match", "cap": ["ਮਰੀਆ"], "sggs": ["ਮਰੀਆ"], "line_ids": ["L"]},
        ]
        assert decide(ops, False, DecisionConfig()) == "review"

    def test_two_trailing_diff_skel_fixes_demote(self):
        ops = [
            {"op": "match", "cap": ["ਮਰਿ"], "sggs": ["ਮਰਿ"], "line_ids": ["L"]},
            {"op": "match", "cap": ["ਜੀਵੈ"], "sggs": ["ਜੀਵੈ"], "line_ids": ["L"]},
            {"op": "match", "cap": ["ਮਰੀਆ"], "sggs": ["ਮਰੀਆ"], "line_ids": ["L"]},
            {"op": "fix", "cap": ["ਹਰ"], "sggs": ["ਹਮ"], "line_ids": ["L"]},
            {"op": "fix", "cap": ["ਵੈ"], "sggs": ["ਵਣ"], "line_ids": ["L"]},
        ]
        assert decide(ops, False, DecisionConfig()) == "review"

    def test_leading_matra_fix_does_not_trigger(self):
        # Matra-only fix (same skel) at boundary does NOT count toward the
        # boundary churn run. This is the common legitimate case.
        ops = [
            {"op": "fix", "cap": ["ਪੰਥ"], "sggs": ["ਪੰਥੁ"], "line_ids": ["L"]},
            {"op": "fix", "cap": ["ਸੁਣਿਓ"], "sggs": ["ਸੁਨਿਓ"], "line_ids": ["L"]},
            {"op": "match", "cap": ["ਗੁਰ"], "sggs": ["ਗੁਰ"], "line_ids": ["L"]},
        ]
        # First op is matra (same skel), second op is 1-cons diff. Lead run
        # of consecutive 1-cons starts fresh at op 2, length 1. Not demoted.
        assert decide(ops, False, DecisionConfig()) == "replaced"

    def test_single_boundary_1cons_fix_does_not_demote(self):
        # One 1-cons fix at boundary flanked by match — legitimate rewrite,
        # must stay 'replaced'.
        ops = [
            {"op": "fix", "cap": ["ਛੇਤ"], "sggs": ["ਖੇਤ"], "line_ids": ["L"]},
            {"op": "match", "cap": ["ਜਿਉ"], "sggs": ["ਜਿਉ"], "line_ids": ["L"]},
            {"op": "match", "cap": ["ਅੰਦਰਿ"], "sggs": ["ਅੰਦਰਿ"], "line_ids": ["L"]},
            {"op": "fix", "cap": ["ਗੁਆੜਿ"], "sggs": ["ਬੂਆੜ"], "line_ids": ["L"]},
        ]
        assert decide(ops, False, DecisionConfig()) == "replaced"


class TestSafeRescue:
    def test_rescue_matches_plus_matra_plus_deletes_to_replaced(self):
        # Real pattern from clip_00018: 2 matches + 1 matra-fix + 1 delete.
        # Score = 3/4 = 0.75 — under 0.95 accept. Previously 'review'; now
        # rescued to 'replaced' because no 1-cons / merge / split.
        ops = [
            {"op": "delete", "cap": ["ਆਪਣੈ"], "sggs": ["ਆਪਣੈ"], "line_ids": []},
            {"op": "fix", "cap": ["ਬਲਹਾਰੀ"], "sggs": ["ਬਲਿਹਾਰੀ"], "line_ids": ["L"]},
            {"op": "match", "cap": ["ਗੁਰ"], "sggs": ["ਗੁਰ"], "line_ids": ["L"]},
        ]
        assert decide(ops, False, DecisionConfig()) == "replaced"

    def test_rescue_matches_plus_deletes_only_to_unchanged(self):
        # Match + delete only — no corrections applied. Caption preserved
        # verbatim. Correct label is 'unchanged', not 'review'.
        ops = [
            {"op": "delete", "cap": ["ਪਲਾ"], "sggs": ["ਪਲਾ"], "line_ids": []},
            {"op": "match", "cap": ["ਬਲਿਹਾਰੀ"], "sggs": ["ਬਲਿਹਾਰੀ"], "line_ids": ["L"]},
            {"op": "match", "cap": ["ਗੁਰ"], "sggs": ["ਗੁਰ"], "line_ids": ["L"]},
        ]
        assert decide(ops, False, DecisionConfig()) == "unchanged"

    def test_no_rescue_when_1cons_fix_present(self):
        # 1-cons swap present (ਮਨ→ਜਨ, skel ਮਨ != ਜਨ). Score 2/3 = 0.67, not
        # rescue-eligible because n_1cons_fix > 0. Stays 'review'.
        ops = [
            {"op": "fix", "cap": ["ਮਨ"], "sggs": ["ਜਨ"], "line_ids": ["L"]},
            {"op": "match", "cap": ["ਗੁਰ"], "sggs": ["ਗੁਰ"], "line_ids": ["L"]},
            {"op": "delete", "cap": ["ਹੈ"], "sggs": ["ਹੈ"], "line_ids": []},
        ]
        assert decide(ops, False, DecisionConfig()) == "review"

    def test_no_rescue_when_merge_present(self):
        # Merges and splits are structural ops — not rescue-eligible even
        # when no 1-cons fix exists.
        ops = [
            {"op": "merge", "cap": ["ਮਦ", "ਫੈਲੀ"], "sggs": ["ਬਦਫੈਲੀ"], "line_ids": ["L"]},
            {"op": "match", "cap": ["ਗੁਰ"], "sggs": ["ਗੁਰ"], "line_ids": ["L"]},
            {"op": "delete", "cap": ["ਹੈ"], "sggs": ["ਹੈ"], "line_ids": []},
        ]
        assert decide(ops, False, DecisionConfig()) == "review"

    def test_no_rescue_when_relaxed(self):
        # used_relaxed=True means orphan realign needed edit-distance=2. That
        # is a softer signal and should still go to 'review' for human/LLM
        # double-check regardless of op mix.
        ops = [
            {"op": "match", "cap": ["ਗੁਰ"], "sggs": ["ਗੁਰ"], "line_ids": ["L"]},
            {"op": "fix", "cap": ["ਪੰਥ"], "sggs": ["ਪੰਥੁ"], "line_ids": ["L"]},
            {"op": "delete", "cap": ["ਹੈ"], "sggs": ["ਹੈ"], "line_ids": []},
        ]
        assert decide(ops, True, DecisionConfig()) == "review"


class TestRenderOutputs:
    def test_safe_fallback_on_unchanged(self):
        ops, _ = _mk_ops(match=1, delete=3)
        cap_tokens = ["a", "b", "c", "d"]
        out = render_outputs(ops, cap_tokens, decision="unchanged")
        assert out["final_text"] == "a b c d"
        assert out["sggs_line"] is None

    def test_replaced_applies_fixes(self):
        ops = [
            {"op": "match", "cap": ["a"], "sggs": ["a"], "line_ids": ["L"]},
            {"op": "fix", "cap": ["b"], "sggs": ["B"], "line_ids": ["L"]},
        ]
        out = render_outputs(ops, ["a", "b"], decision="replaced")
        assert out["final_text"] == "a B"
        assert out["sggs_line"] == "a B"
