"""CPU tests for the positive-source loader extraction (architecture review
step 8, 2026-06-12).

Locks the behaviour that must survive the move of the CT/JHB positive-source
loaders + label_source derivation out of
``scripts/training/build_unified_reviewall.py`` into
``core.training.positive_sources``:

- ``_ct_source_to_label_source`` mapping table (pure function, no I/O) — the
  exact enum every byte-equivalence claim rests on.
- the DEPRECATED bespoke script re-exports the moved names as the *same
  objects* (single source of truth, no second copy).
- ``_load_jhb_grid_annotations`` now takes an explicit ``review_root``
  parameter (the monkeypatched-global elimination) and defaults to the
  module constant.
- the INTENTIONAL divergence between the detector-train loader's
  ``_ct_source_to_label_source`` (fail-fast: raises on unknown) and
  ``build_training_pool._source_to_label_source`` (fail-closed: returns None /
  legacy_weak_supervision) is preserved — a guard so nobody silently unifies
  the two forks.

All pure-Python; no tiles, no GPU.
"""
from __future__ import annotations

import inspect
import math

import pytest

from core.training import positive_sources as psrc


# ── 1. _ct_source_to_label_source mapping table (byte-equivalence anchor) ──
def test_ct_source_to_label_source_known_values():
    f = psrc._ct_source_to_label_source
    assert f(None) == "reviewed_prediction"
    assert f(float("nan")) == "reviewed_prediction"
    assert f("sam_fn_marker") == "sam_added_true_fn"
    assert f("sam_fn_review") == "sam_added_true_fn"
    assert f("sam2") == "human_manual_sam_assisted"
    assert f("reviewed_prediction") == "reviewed_prediction"
    assert f("human_manual_sam_assisted") == "human_manual_sam_assisted"
    # case/whitespace normalisation
    assert f(" SAM2 ") == "human_manual_sam_assisted"
    assert f("Reviewed_Prediction") == "reviewed_prediction"


def test_ct_source_to_label_source_unknown_raises():
    # Fail-fast on unknown provenance markers (the detector-train contract).
    with pytest.raises(ValueError):
        psrc._ct_source_to_label_source("__never_seen__")
    with pytest.raises(ValueError):
        psrc._ct_source_to_label_source("google_earth")


# ── 2. bespoke script re-exports the same objects (no second copy) ──────────
def test_bespoke_reexports_same_objects():
    bur = pytest.importorskip("scripts.training.build_unified_reviewall")
    for name in (
        "_tiles_for",
        "_assign_intersections",
        "_load_jhb_grid_annotations",
        "_ct_source_to_label_source",
        "_ct_entries",
        "_load_ct_grid_annotations",
        "_per_record_summary",
        "_selected_annotations_from_records",
        "_src_rel",
    ):
        assert getattr(bur, name) is getattr(psrc, name), (
            f"{name} in build_unified_reviewall must be the SAME object as in "
            f"positive_sources (re-import, not a re-definition)"
        )
    # JHB_REVIEW_ROOT default constant resolves to the same path both sides.
    assert bur.JHB_REVIEW_ROOT == psrc.JHB_REVIEW_ROOT


# ── 3. review_root is an explicit parameter (monkeypatch eliminated) ────────
def test_load_jhb_takes_review_root_param():
    sig = inspect.signature(psrc._load_jhb_grid_annotations)
    assert "review_root" in sig.parameters
    # default preserves the bespoke module-level constant
    assert sig.parameters["review_root"].default is None  # resolved to const inside


def test_dataset_builder_does_not_monkeypatch_review_root():
    import pathlib
    src = pathlib.Path("pipeline/dataset_builder.py").read_text()
    assert "JHB_REVIEW_ROOT" not in src, (
        "dataset_builder must not reference/monkeypatch JHB_REVIEW_ROOT anymore"
    )
    assert "build_unified_reviewall as bur" not in src, (
        "dataset_builder must not import the bespoke script's privates"
    )


# ── 4. divergence guard: build_training_pool fork stays divergent ──────────
def test_build_training_pool_fork_is_divergent_by_design():
    tp = pytest.importorskip("scripts.training.build_training_pool")
    tp_f = tp._source_to_label_source
    ct_f = psrc._ct_source_to_label_source

    # google_earth: train-loader raises; pool builder fail-closes to legacy enum.
    with pytest.raises(ValueError):
        ct_f("google_earth")
    assert tp_f("google_earth", "sam2", True) == "legacy_weak_supervision"

    # unknown: train-loader raises; pool builder returns None (fail-closed).
    with pytest.raises(ValueError):
        ct_f("__never_seen__")
    assert tp_f("__never_seen__", "sam2", True) is None

    # column-absent schema default: pool builder is schema-aware (train-loader
    # has no schema concept — it partitions that into _load_ct_grid_annotations).
    assert tp_f(None, "sam2", False) == "human_manual_sam_assisted"
    assert tp_f(None, "v4_reviewed", False) == "reviewed_prediction"
    assert tp_f(None, "other_schema", False) is None

    # shared/known values still agree (the overlap that keeps them recognisable).
    for v in ("sam2", "reviewed_prediction", "human_manual_sam_assisted",
              "sam_fn_marker", "sam_fn_review"):
        assert tp_f(v, "sam2", True) == ct_f(v)


# ── 5. _src_rel repo-relativisation ────────────────────────────────────────
def test_src_rel_repo_relative_and_absolute():
    from pathlib import Path
    inside = psrc.PROJECT_ROOT / "data" / "annotations" / "x.gpkg"
    assert psrc._src_rel(inside) == "data/annotations/x.gpkg"
    outside = Path("/tmp/definitely_outside_repo.gpkg")
    assert psrc._src_rel(outside) == "/tmp/definitely_outside_repo.gpkg"
