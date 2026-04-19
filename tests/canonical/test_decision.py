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
