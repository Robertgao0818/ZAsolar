#!/usr/bin/env python3
"""Validate cross-references between registry files.

Checks:
  1. annotation_manifest.csv field values are in allowed enumerations
  2. training_sets.yaml holdout grids exist in regions.yaml
  3. model_registry.yaml training_set_id references exist in training_sets.yaml
  4. regions.yaml annotation_source files exist on disk
  5. T1 annotations have semantic_confidence == A1 (if field populated)

Usage:
  python scripts/validate_registry.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Allowed enumerations ---
ALLOWED_LABEL_SOURCE = {
    "human_manual",
    "human_manual_sam_assisted",
    "reviewed_prediction",
    "sam_refined_review",
    "legacy_weak_supervision",
    "",  # empty = not yet backfilled
}
ALLOWED_SEMANTIC_CONFIDENCE = {"A1", "A2", "A3", ""}
ALLOWED_QUALITY_TIER = {"T1", "T2", ""}


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def validate_manifest(manifest_path: Path) -> list[str]:
    """Check annotation_manifest.csv field values."""
    errors: list[str] = []
    warnings: list[str] = []

    if not manifest_path.exists():
        return [f"WARN: {manifest_path} not found, skipping manifest checks"]

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        has_label_source = "label_source" in fields
        has_semantic_confidence = "semantic_confidence" in fields

        for i, row in enumerate(reader, start=2):
            # Check existing fields
            tier = row.get("quality_tier", "")
            if tier and tier not in ALLOWED_QUALITY_TIER:
                errors.append(f"manifest line {i}: invalid quality_tier '{tier}'")

            # Check new provenance fields (if they exist)
            if has_label_source:
                ls = row.get("label_source", "")
                if ls not in ALLOWED_LABEL_SOURCE:
                    errors.append(f"manifest line {i}: invalid label_source '{ls}'")

            if has_semantic_confidence:
                sc = row.get("semantic_confidence", "")
                if sc not in ALLOWED_SEMANTIC_CONFIDENCE:
                    errors.append(f"manifest line {i}: invalid semantic_confidence '{sc}'")

                # T1 with non-A1 semantic_confidence
                if tier == "T1" and sc and sc != "A1":
                    warnings.append(
                        f"manifest line {i}: T1 annotation has "
                        f"semantic_confidence='{sc}' (expected A1)"
                    )

    if not has_label_source:
        warnings.append("manifest: 'label_source' column not yet added")
    if not has_semantic_confidence:
        warnings.append("manifest: 'semantic_confidence' column not yet added")

    return errors + [f"WARN: {w}" for w in warnings]


def validate_training_sets(
    training_sets: dict,
    all_grid_ids: set[str],
    available_regions: set[str],
) -> list[str]:
    """Check training_sets.yaml references."""
    messages: list[str] = []

    valid_source_families = {
        "human_manual", "human_manual_sam_assisted",
        "human_manual_qgis_geosam",
        "reviewed_prediction", "gemini_reviewed_prediction",
        "sam_refined_review", "sam_added_browser", "sam_added_true_fn",
        "legacy_weak_supervision",
    }

    # Known top-level keys for a training_sets entry (unknown-key pass).
    known_entry_keys = {
        "id", "description", "region_scope", "source_family",
        "annotation_source_policy", "tier_policy", "audit_policy",
        "easy_negative_ratio", "hn_policy", "hn_ratio_target",
        "hn_ratio_actual", "hn_actual_composition", "output_dir",
        "val_strategy", "init_weights", "epoch_budget", "builder_script",
        "untrusted_trusted_ratio", "notes",
        # v2 (training-pool normalization, 2026-05-29)
        "spec_schema_version", "selected_annotations",
    }

    for ts_id, ts_data in training_sets.items():
        # Unknown-key pass
        unknown = set(ts_data.keys()) - known_entry_keys
        if unknown:
            messages.append(
                f"training_set '{ts_id}': unknown keys {sorted(unknown)} "
                f"(allowed: {sorted(known_entry_keys)})"
            )

        # Validate source_family values
        source_family = ts_data.get("source_family", []) or []
        for sf in source_family:
            if sf not in valid_source_families:
                messages.append(
                    f"training_set '{ts_id}': unknown source_family '{sf}'"
                )

        # Validate region_scope references
        region_scope = ts_data.get("region_scope", []) or []
        for region in region_scope:
            if region not in available_regions:
                messages.append(
                    f"training_set '{ts_id}': unknown region_scope '{region}'"
                )

    return messages


def validate_model_registry(
    models: dict, training_set_ids: set[str]
) -> list[str]:
    """Check model_registry.yaml references."""
    errors: list[str] = []

    for model_id, model_data in models.items():
        ts_id = model_data.get("training_set_id")
        if ts_id and ts_id not in training_set_ids:
            errors.append(
                f"model '{model_id}': training_set_id '{ts_id}' "
                f"not found in training_sets.yaml"
            )
        postproc = model_data.get("postproc_config")
        if postproc and not (BASE_DIR / postproc).exists():
            errors.append(
                f"model '{model_id}': postproc_config '{postproc}' "
                f"file not found"
            )
    return errors


def validate_annotation_sources(regions: dict) -> list[str]:
    """Check annotation_source files exist on disk."""
    errors: list[str] = []
    warnings: list[str] = []

    for region_key, region_data in regions.items():
        grids = region_data.get("grids", {})
        for grid_id, grid_data in grids.items():
            source = grid_data.get("annotation_source")
            if source:
                full_path = BASE_DIR / source
                if not full_path.exists():
                    warnings.append(
                        f"region '{region_key}', grid '{grid_id}': "
                        f"annotation_source '{source}' not found"
                    )

    return errors + [f"WARN: {w}" for w in warnings]


def main() -> int:
    print("=" * 60)
    print("Registry Validation")
    print("=" * 60)

    all_messages: list[str] = []

    # Load configs
    regions_path = BASE_DIR / "configs" / "datasets" / "regions.yaml"
    training_sets_path = BASE_DIR / "configs" / "datasets" / "training_sets.yaml"
    model_registry_path = BASE_DIR / "configs" / "model_registry.yaml"
    manifest_path = BASE_DIR / "data" / "annotations" / "annotation_manifest.csv"

    regions_data = load_yaml(regions_path).get("regions", {})
    training_sets_data = load_yaml(training_sets_path).get("training_sets", {})
    model_registry_data = load_yaml(model_registry_path).get("models", {})

    # Collect all grid IDs
    all_grid_ids: set[str] = set()
    for region_data in regions_data.values():
        all_grid_ids.update(region_data.get("grids", {}).keys())

    training_set_ids = set(training_sets_data.keys())

    # Run checks
    print(f"\n[1/4] Annotation manifest ({manifest_path.name})")
    msgs = validate_manifest(manifest_path)
    all_messages.extend(msgs)
    for m in msgs:
        print(f"  {m}")
    if not msgs:
        print("  OK")

    print(f"\n[2/4] Training sets ({training_sets_path.name})")
    msgs = validate_training_sets(
        training_sets_data,
        all_grid_ids,
        set(regions_data.keys()),
    )
    all_messages.extend(msgs)
    for m in msgs:
        print(f"  {m}")
    if not msgs:
        print("  OK")

    print(f"\n[3/4] Model registry ({model_registry_path.name})")
    msgs = validate_model_registry(model_registry_data, training_set_ids)
    all_messages.extend(msgs)
    for m in msgs:
        print(f"  {m}")
    if not msgs:
        print("  OK")

    print(f"\n[4/4] Annotation sources (regions.yaml)")
    msgs = validate_annotation_sources(regions_data)
    all_messages.extend(msgs)
    for m in msgs:
        print(f"  {m}")
    if not msgs:
        print("  OK")

    # Summary
    errors = [m for m in all_messages if not m.startswith("WARN:")]
    warnings = [m for m in all_messages if m.startswith("WARN:")]

    print(f"\n{'=' * 60}")
    print(f"Results: {len(errors)} errors, {len(warnings)} warnings")
    if errors:
        print("FAILED")
        return 1
    elif warnings:
        print("PASSED with warnings")
        return 0
    else:
        print("PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
