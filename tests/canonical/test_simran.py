from scripts.canonical.simran import SimranConfig, apply_simran_quota, is_simran


class TestIsSimran:
    def test_all_waheguru_is_simran(self):
        toks = ["ਵਾਹਿਗੁਰੂ"] * 8
        assert is_simran(toks, SimranConfig())

    def test_variant_spellings_detected(self):
        toks = ["ਵਾਹਿਗੁਰੂ", "ਵਾਹੇਗੁਰੂ", "ਵਾਹਿਗੁਰ", "ਵਾਹਿਗੁਰੂ", "ਵਾਹਿਗੁਰੂ"]
        assert is_simran(toks, SimranConfig())

    def test_short_rep_not_simran(self):
        toks = ["ਵਾਹਿਗੁਰੂ", "ਵਾਹਿਗੁਰੂ"]
        assert not is_simran(toks, SimranConfig())

    def test_mixed_content_not_simran(self):
        toks = ["ਤੇਰੀ", "ਸਰਣਿ"] * 3 + ["ਵਾਹਿਗੁਰੂ"] * 2
        assert not is_simran(toks, SimranConfig())

    def test_70_percent_threshold(self):
        toks = ["ਵਾਹਿਗੁਰੂ"] * 7 + ["ਤੇਰੀ", "ਸਰਣਿ", "ਮੇਰੇ"]
        assert is_simran(toks, SimranConfig())

    def test_below_ratio_not_simran(self):
        toks = ["ਵਾਹਿਗੁਰੂ"] * 5 + ["ਤੇਰੀ"] * 5
        assert not is_simran(toks, SimranConfig())


class TestQuota:
    def _make_rows(self, video_counts: dict[str, int]) -> list[dict]:
        rows = []
        for vid, count in video_counts.items():
            for i in range(count):
                rows.append(
                    {"clip_id": f"{vid}_{i:03d}", "video_id": vid, "is_simran": True}
                )
        return rows

    def test_per_video_cap_applied(self):
        rows = self._make_rows({"A": 50, "B": 50, "C": 50})
        cfg = SimranConfig(target_count=100, per_video_min=1, per_video_max=10)
        kept = apply_simran_quota(rows, cfg, seed=42)
        assert len(kept) == 30
        per_video: dict[str, int] = {}
        for r in kept:
            per_video[r["video_id"]] = per_video.get(r["video_id"], 0) + 1
        assert all(c == 10 for c in per_video.values())

    def test_clamp_at_max_when_few_videos(self):
        rows = self._make_rows({"A": 200})
        cfg = SimranConfig(target_count=750, per_video_min=1, per_video_max=10)
        kept = apply_simran_quota(rows, cfg, seed=42)
        assert len(kept) == 10

    def test_global_target_enforced(self):
        rows = self._make_rows({f"V{i:03d}": 20 for i in range(100)})
        cfg = SimranConfig(target_count=750, per_video_min=1, per_video_max=10)
        kept = apply_simran_quota(rows, cfg, seed=42)
        assert len(kept) == 750

    def test_round_robin_preserves_video_diversity(self):
        rows = self._make_rows({f"V{i:03d}": 20 for i in range(100)})
        cfg = SimranConfig(target_count=100, per_video_min=1, per_video_max=10)
        kept = apply_simran_quota(rows, cfg, seed=42)
        per_video: dict[str, int] = {}
        for r in kept:
            per_video[r["video_id"]] = per_video.get(r["video_id"], 0) + 1
        assert len(per_video) == 100
        assert all(c == 1 for c in per_video.values())

    def test_non_simran_preserved(self):
        rows = self._make_rows({"A": 20})
        rows.append({"clip_id": "X_000", "video_id": "X", "is_simran": False})
        cfg = SimranConfig(target_count=5, per_video_min=1, per_video_max=10)
        kept = apply_simran_quota(rows, cfg, seed=42)
        non_simran = [r for r in kept if not r.get("is_simran")]
        assert len(non_simran) == 1
