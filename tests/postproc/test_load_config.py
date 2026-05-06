"""Unit tests for load_postproc_config (corrected superset parser)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from core.postproc import load_postproc_config


def _write_json(tmp: Path, data: dict) -> Path:
    p = tmp / "cfg.json"
    p.write_text(json.dumps(data))
    return p


def test_accepts_v4_canonical_keys(tmp_path: Path):
    """All keys in configs/postproc/v4_canonical.json must be parsed
    (regression vs old parser which silently dropped half of them)."""
    p = _write_json(tmp_path, {
        "confidence_threshold": 0.3,
        "mask_threshold": 0.3,
        "post_conf_threshold": 0.85,
        "min_object_area": 5,
        "max_elongation": 8.0,
        "min_solidity": 0.0,
        "shadow_rgb_thresh": 60,
    })
    out = load_postproc_config(p)
    assert out["mask_threshold"] == 0.3
    assert out["min_solidity"] == 0.0
    assert out["shadow_rgb_thresh"] == 60
    assert out["post_conf_threshold"] == 0.85


def test_accepts_mask_shaping_keys(tmp_path: Path):
    p = _write_json(tmp_path, {
        "merge_mode": "per-detection",
        "vectorize_multi_component": "largest",
        "mask_threshold_area_m2_tiers": [[200, 0.55], [100, 0.45]],
        "mask_hysteresis_high_threshold": 0.7,
        "mask_hysteresis_min_core_area_px": 4,
    })
    out = load_postproc_config(p)
    assert out["merge_mode"] == "per-detection"
    assert out["vectorize_multi_component"] == "largest"
    assert out["mask_threshold_area_m2_tiers"] == [[200, 0.55], [100, 0.45]]
    assert out["mask_hysteresis_high_threshold"] == 0.7
    assert out["mask_hysteresis_min_core_area_px"] == 4


def test_legacy_confidence_threshold_maps_to_pre_vector(tmp_path: Path):
    """V1.4 plan A: legacy confidence_threshold → pre_vector_score_threshold."""
    p = _write_json(tmp_path, {"confidence_threshold": 0.3})
    out = load_postproc_config(p)
    assert out["pre_vector_score_threshold"] == 0.3
    # And the legacy key is still recorded:
    assert out["confidence_threshold"] == 0.3


def test_explicit_pre_vector_overrides_legacy(tmp_path: Path):
    """If both keys present, the explicit pre_vector_score_threshold wins."""
    p = _write_json(tmp_path, {
        "confidence_threshold": 0.3,
        "pre_vector_score_threshold": 0.1,
    })
    out = load_postproc_config(p)
    assert out["pre_vector_score_threshold"] == 0.1


def test_unknown_keys_warn_by_default(tmp_path: Path):
    p = _write_json(tmp_path, {"made_up_key": 99})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_postproc_config(p)
    assert any("made_up_key" in str(x.message) for x in w)


def test_unknown_keys_raise_when_strict(tmp_path: Path):
    p = _write_json(tmp_path, {"made_up_key": 99})
    with pytest.raises(ValueError, match="made_up_key"):
        load_postproc_config(p, strict=True)


def test_meta_key_silently_ignored(tmp_path: Path):
    """`_meta` is reserved for provenance; should not warn."""
    p = _write_json(tmp_path, {"_meta": {"author": "test"}, "min_object_area": 7})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = load_postproc_config(p)
    assert not any("_meta" in str(x.message) for x in w)
    assert out["min_object_area"] == 7


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_postproc_config(tmp_path / "does_not_exist.json")
