from scripts.canonical.config import get_dataset_config, DatasetConfig


class TestDatasetConfig:
    def test_kirtan_defaults(self):
        cfg = get_dataset_config("kirtan")
        assert cfg.strip_unk_artifacts is True
        assert cfg.include_sirlekh is True
        assert cfg.split_multi_shabad_rows is True
        assert cfg.sequential_shabad_retrieval is False
        assert cfg.use_llm_fallback is True
        assert cfg.simran_detection is True

    def test_sehaj_defaults(self):
        cfg = get_dataset_config("sehaj")
        assert cfg.strip_unk_artifacts is True
        assert cfg.include_sirlekh is True
        assert cfg.split_multi_shabad_rows is True
        assert cfg.sequential_shabad_retrieval is True
        assert cfg.use_llm_fallback is True
        assert cfg.simran_detection is False

    def test_unknown_dataset_raises(self):
        import pytest
        with pytest.raises(ValueError):
            get_dataset_config("bhangra")

    def test_config_overrides(self):
        cfg = get_dataset_config("kirtan", overrides={"use_llm_fallback": False})
        assert cfg.use_llm_fallback is False
