"""Dataset-specific configuration for the canonical-text pipeline."""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace


@dataclass
class DatasetConfig:
    dataset_name: str
    strip_unk_artifacts: bool = True
    include_sirlekh: bool = True
    split_multi_shabad_rows: bool = True
    sequential_shabad_retrieval: bool = False
    use_llm_fallback: bool = True
    simran_detection: bool = True


_PRESETS = {
    "kirtan": DatasetConfig(
        dataset_name="kirtan",
        sequential_shabad_retrieval=False,
        simran_detection=True,
    ),
    "sehaj": DatasetConfig(
        dataset_name="sehaj",
        sequential_shabad_retrieval=True,
        simran_detection=False,
    ),
}


def get_dataset_config(
    name: str, overrides: dict | None = None,
) -> DatasetConfig:
    if name not in _PRESETS:
        raise ValueError(f"unknown dataset: {name!r} (expected one of {list(_PRESETS)})")
    cfg = _PRESETS[name]
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg
