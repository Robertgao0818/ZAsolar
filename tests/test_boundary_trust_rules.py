"""Regression guard for the unified boundary-trust rules (Phase 0, 2026-05-29).

Asserts the single-source YAML (data/training_pool/boundary_trust_rules.yaml,
loaded via core.boundary_trust) reproduces the EXACT per-key values that were
previously hardcoded in `export_coco_dataset._MASK_TRUSTED` and train.py's two
maps (`_LABEL_SOURCE_TO_MASK_TRUSTED`, `_LABEL_SOURCE_TO_BOUNDARY_W`) — i.e. the
de-duplication is behavior-preserving (zero behavior change).
"""

from pathlib import Path

import pytest

from core.boundary_trust import (
    boundary_w_map,
    load_boundary_trust_rules,
    mask_trusted_map,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Golden values: the exact hardcoded dicts as they existed BEFORE the YAML
# extraction. Sourced from export_coco_dataset._MASK_TRUSTED and train.py's
# _LABEL_SOURCE_TO_MASK_TRUSTED / _LABEL_SOURCE_TO_BOUNDARY_W on 2026-05-29
# (includes the gemini_reviewed_prediction entries then in the working tree).
GOLDEN_MASK_TRUSTED = {
    "human_manual": True,
    "human_manual_sam_assisted": True,
    "human_manual_qgis_geosam": True,
    "sam_added_browser": True,
    "reviewed_prediction": False,
    "gemini_reviewed_prediction": False,
    "sam_refined_review": False,
    "sam_added_true_fn": False,
    "legacy_weak_supervision": False,
}
GOLDEN_BOUNDARY_W = {
    "human_manual": 1.0,
    "human_manual_sam_assisted": 1.0,
    "human_manual_qgis_geosam": 1.0,
    "sam_added_browser": 1.0,
    "reviewed_prediction": 0.0,
    "gemini_reviewed_prediction": 0.0,
    "sam_refined_review": 0.0,
    "sam_added_true_fn": 0.0,
    "legacy_weak_supervision": 0.0,
}


def test_loader_matches_golden_mask_trusted():
    assert mask_trusted_map() == GOLDEN_MASK_TRUSTED


def test_loader_matches_golden_boundary_w():
    assert boundary_w_map() == GOLDEN_BOUNDARY_W


def test_yaml_schema_and_documented_defaults():
    rules = load_boundary_trust_rules()
    assert rules["schema_version"] == 1
    # documented intent (consumers preserve their own None/unknown edge behavior)
    assert rules["fail_closed_default"] == "untrusted"
    assert rules["legacy_no_source_field"] == "trusted"
    assert set(rules["map"]) == set(GOLDEN_MASK_TRUSTED)
    for src, attrs in rules["map"].items():
        assert {"mask_trusted", "boundary_w"} <= set(attrs), src


def test_export_coco_uses_loader_and_matches_golden():
    import export_coco_dataset as eco

    assert eco._MASK_TRUSTED == GOLDEN_MASK_TRUSTED
    # known source values
    assert eco.mask_trusted_for("human_manual") is True
    assert eco.mask_trusted_for("reviewed_prediction") is False
    # edge behavior preserved verbatim: None and unknown both raise
    with pytest.raises(ValueError):
        eco.mask_trusted_for(None)
    with pytest.raises(ValueError):
        eco.mask_trusted_for("totally_unknown_source")


def test_train_wires_loader_no_hardcoded_dict():
    # train.py asserts torch.cuda.is_available() at import, so we verify wiring
    # at the source level (no import) — proves train consumes the shared loader,
    # not a stale hardcoded dict, and that the old gemini entries are gone.
    src = (PROJECT_ROOT / "train.py").read_text()
    assert "_LABEL_SOURCE_TO_BOUNDARY_W = boundary_w_map()" in src
    assert "_LABEL_SOURCE_TO_MASK_TRUSTED = mask_trusted_map()" in src
    assert '"gemini_reviewed_prediction": 0.0' not in src
    assert '"gemini_reviewed_prediction": False' not in src


def test_trusted_iff_full_edge_weight():
    # internal consistency: the 2-bucket trusted/untrusted view is derived from
    # mask_trusted, and lines up with boundary_w (trusted→1.0, untrusted→0.0).
    mt = mask_trusted_map()
    bw = boundary_w_map()
    for src in mt:
        assert bw[src] == (1.0 if mt[src] else 0.0), f"{src}: mask_trusted={mt[src]} boundary_w={bw[src]}"
