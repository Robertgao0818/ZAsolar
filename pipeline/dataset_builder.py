"""Declarative dataset builder — thin orchestrator.

Reads a validated dataset spec YAML, calls existing exporter/HN logic
via public APIs, and writes reproducible build manifests.

Usage::

    python -m pipeline.dataset_builder \\
        --spec configs/pipelines/datasets/v4_1_hn.yaml \\
        --output-dir /mnt/d/ZAsolar/coco_v4_1_rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.specs import load_spec, spec_to_dict, resolve_env_vars, DatasetSpec
from pipeline.manifests import (
    generate_build_id,
    compute_string_sha256,
    build_source_inventory,
    write_build_manifest,
    write_dataset_summary,
    DatasetSummary,
)
from core.annotation_loader import discover_annotations


REPO_ROOT = Path(__file__).resolve().parent.parent


def _normalize_path(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_value(value):
    if isinstance(value, str):
        return resolve_env_vars(value)
    if isinstance(value, list):
        return [_resolve_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_value(item) for key, item in value.items()}
    return value


def _selected_annotation_records(entries: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for grid_id, entry in sorted(entries.items()):
        records.append({
            "grid_id": grid_id,
            "region": entry.region_key,
            "path": _normalize_path(entry.path),
            "schema_type": entry.schema_type,
            "registered": entry.registered,
            "annotation_count": entry.annotation_count,
            "annotation_layer": entry.annotation_layer,
        })
    return records


def build_dataset(
    spec_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    check_files: bool = True,
) -> Path:
    """Build a COCO dataset from a declarative spec.

    Steps:
        1. Load + validate spec.
        2. Resolve env vars, compute build ID.
        3. Hash all source annotation files.
        4. Call ``build_base_coco()`` to produce base COCO chips.
        5. Call HN extraction if declared in spec.
        6. Write ``build_manifest.json`` + ``dataset_summary.json``.

    Returns:
        The build output directory.
    """
    # ── 1. Load and validate spec ────────────────────────────────────
    spec = load_spec(spec_path, check_files=check_files)
    print(f"[BUILD] Loaded spec: {spec.name} (regions={spec.regions})")

    # ── 2. Resolve paths and build fingerprint inputs ─────────────────
    resolved_root = resolve_env_vars(spec.output.root)
    build_date = datetime.now(timezone.utc)
    resolved_spec = _resolve_value(spec_to_dict(spec))

    if output_dir is None:
        date_str = build_date.strftime("%Y%m%d")
        dir_name = spec.output.name_template.format(
            name=spec.name, date=date_str,
        )
        build_dir = Path(resolved_root) / dir_name
    else:
        build_dir = Path(output_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    # ── 3. Hash source files ─────────────────────────────────────────
    entries = discover_annotations(
        regions=spec.regions,
        exclude_grids=set(spec.selection.exclude_grids),
    )
    annotation_paths = [e.path for e in entries.values()]

    base_dir = Path(__file__).resolve().parent.parent
    audit_csv_path = (base_dir / spec.selection.audit_csv) if spec.selection.audit_csv else None
    hn_shortlist_paths = []
    for hn in spec.hard_negatives:
        if hn.shortlist_csv:
            hn_shortlist_paths.append(base_dir / hn.shortlist_csv)

    manifest_csv_path = base_dir / "data" / "annotations" / "annotation_manifest.csv"
    source_inventory = build_source_inventory(
        annotation_paths=annotation_paths,
        audit_csv=audit_csv_path,
        manifest_csv=manifest_csv_path if manifest_csv_path.exists() else None,
        hn_shortlist_csvs=hn_shortlist_paths,
    )
    selected_annotations = _selected_annotation_records(entries)
    build_fingerprint = {
        "resolved_spec": resolved_spec,
        "selected_annotations": selected_annotations,
        "source_inventory": source_inventory,
    }
    build_fingerprint_json = json.dumps(build_fingerprint, sort_keys=True)
    build_id = generate_build_id(spec.name, build_fingerprint_json, build_date)
    resolved_spec_hash = compute_string_sha256(build_fingerprint_json)
    print(f"[BUILD] Build ID: {build_id}")
    print(f"[BUILD] Output: {build_dir}")
    print(f"[BUILD] Hashed {len(source_inventory)} source files")

    # ── 4. Build base COCO ───────────────────────────────────────────
    from export_coco_dataset import build_base_coco

    base_coco_dir = build_dir / "_base"
    base_result = build_base_coco({
        "regions": spec.regions,
        "output_dir": str(base_coco_dir),
        "chip_size": spec.chip.size,
        "overlap": spec.chip.overlap,
        "val_fraction": spec.split.val_fraction,
        "seed": spec.split.seed,
        "no_balance": False,
        "neg_ratio": spec.negatives.easy_neg_ratio,
        "exclude_grids": spec.selection.exclude_grids or None,
        "audit_csv": spec.selection.audit_csv,
        "exclude_audit_labels": spec.selection.exclude_audit_labels,
        "tier_filter": spec.selection.tier_filter,
        "manifest": str(manifest_csv_path) if manifest_csv_path.exists() else None,
        "category_name": "solar_panel",
    })

    # ── 5. HN extraction + merge (if declared) ──────────────────────
    hn_results = []
    if spec.hard_negatives:
        from pipeline.hn_ops import (
            extract_reviewed_fp_hn,
            extract_small_fp_hn,
            merge_hn_into_coco,
            HNResult,
        )

        for i, hn_spec in enumerate(spec.hard_negatives):
            if hn_spec.type == "reviewed_fp_hn":
                print(f"\n[HN {i}] Extracting reviewed FP HN "
                      f"({len(hn_spec.grids)} grids)...")
                result = extract_reviewed_fp_hn(
                    grids=hn_spec.grids,
                    output_dir=build_dir,
                    chip_size=spec.chip.size,
                )
                hn_results.append(result)
                print(f"[HN {i}] {result.n_chips} chips extracted")

            elif hn_spec.type == "small_fp_hn":
                csv_path = base_dir / hn_spec.shortlist_csv
                print(f"\n[HN {i}] Extracting small FP HN "
                      f"(shortlist: {csv_path.name})...")
                result = extract_small_fp_hn(
                    shortlist_csv=csv_path,
                    output_dir=build_dir,
                    chip_size=spec.chip.size,
                    sample_rate=hn_spec.sample_rate,
                )
                hn_results.append(result)
                print(f"[HN {i}] {result.n_chips} chips extracted")

        # ── Enforce max_ratio caps ──────────────────────────────────
        base_train_json = base_coco_dir / "train.json"
        if base_train_json.exists():
            with open(base_train_json) as f:
                n_base_train = len(json.load(f)["images"])
        else:
            n_base_train = 0

        for i, (hn_spec, result) in enumerate(zip(spec.hard_negatives, hn_results)):
            if result.n_chips == 0 or hn_spec.max_ratio is None or n_base_train == 0:
                continue
            # max_ratio = n_hn / (n_base + n_hn)
            # => n_hn <= max_ratio * n_base / (1 - max_ratio)
            max_allowed = int(hn_spec.max_ratio / (1 - hn_spec.max_ratio) * n_base_train)
            if result.n_chips > max_allowed:
                print(f"[HN {i}] Capping {result.source_type}: "
                      f"{result.n_chips} → {max_allowed} chips "
                      f"(max_ratio={hn_spec.max_ratio})")
                result.images = result.images[:max_allowed]
                result.provenance = result.provenance[:max_allowed]
                result.n_chips = max_allowed

        if any(r.n_chips > 0 for r in hn_results):
            print(f"\n[BUILD] Merging base + HN into final dataset...")
            merge_result = merge_hn_into_coco(
                base_dir=base_coco_dir,
                hn_results=hn_results,
                output_dir=build_dir,
            )
            print(f"[BUILD] Final: {merge_result.total_train_images} train, "
                  f"HN ratio={merge_result.hn_ratio:.1%}")
        else:
            # No HN chips — just copy base as final
            _copy_base_as_final(base_coco_dir, build_dir)
            merge_result = None
    else:
        _copy_base_as_final(base_coco_dir, build_dir)
        merge_result = None

    # ── 6. Write manifests ───────────────────────────────────────────
    resolved_tile_roots = {}
    from core.grid_utils import resolve_tiles_dir
    for r in spec.regions:
        # Sample one grid to get tile root
        for e in entries.values():
            if e.region_key == r:
                resolved_tile_roots[r] = str(resolve_tiles_dir(e.grid_id, region=r).parent)
                break

    hn_config_list = []
    for hn_spec in spec.hard_negatives:
        hn_config_list.append({
            "type": hn_spec.type,
            "region": hn_spec.region,
            "grids": hn_spec.grids,
            "max_ratio": hn_spec.max_ratio,
            "shortlist_csv": hn_spec.shortlist_csv,
            "sample_rate": hn_spec.sample_rate,
        })

    excluded_reason = ""
    if spec.selection.exclude_grids:
        if spec.evaluation_regime == "historical_holdout":
            excluded_reason = "Historical benchmark holdout reproduction"
        else:
            excluded_reason = "Ad-hoc exclusion"

    write_build_manifest(
        build_dir,
        build_id=build_id,
        spec_path=str(spec_path),
        resolved_spec=resolved_spec,
        resolved_spec_hash=resolved_spec_hash,
        regions=spec.regions,
        evaluation_regime=spec.evaluation_regime,
        exclude_grids=spec.selection.exclude_grids,
        excluded_grids_reason=excluded_reason,
        source_inventory=source_inventory,
        split_strategy=spec.split.strategy,
        split_seed=spec.split.seed,
        easy_neg_ratio=spec.negatives.easy_neg_ratio,
        hard_negatives_config=hn_config_list,
        selected_annotations=selected_annotations,
        resolved_tile_roots=resolved_tile_roots,
        resolved_output_root=str(build_dir),
    )

    # Build summary from base result + HN
    train_json_path = build_dir / "train.json"
    val_json_path = build_dir / "val.json"
    if train_json_path.exists() and val_json_path.exists():
        with open(train_json_path) as f:
            train_data = json.load(f)
        with open(val_json_path) as f:
            val_data = json.load(f)

        n_pos = sum(1 for img in train_data["images"] if img.get("positive", True))
        n_total_train = len(train_data["images"])
        n_hn = sum(r.n_chips for r in hn_results)
        n_easy_neg = n_total_train - n_pos - n_hn

        reviewed_fp_hn = sum(r.n_chips for r in hn_results if r.source_type == "reviewed_fp_hn")
        small_fp_hn = sum(r.n_chips for r in hn_results if r.source_type == "small_fp_hn")

        per_region = {}
        for e in entries.values():
            per_region[e.region_key] = per_region.get(e.region_key, 0) + 1

        summary = DatasetSummary(
            positive_chips=n_pos,
            easy_neg_chips=max(0, n_easy_neg),
            reviewed_fp_hn_chips=reviewed_fp_hn,
            small_fp_hn_chips=small_fp_hn,
            total_train_images=n_total_train,
            total_val_images=len(val_data["images"]),
            train_annotations=len(train_data["annotations"]),
            val_annotations=len(val_data["annotations"]),
            effective_easy_neg_ratio=(n_easy_neg / n_pos) if n_pos else 0,
            effective_hn_ratio=(n_hn / n_total_train) if n_total_train else 0,
            per_region_grid_counts=per_region,
        )
        write_dataset_summary(build_dir, summary)

    print(f"\n{'=' * 60}")
    print(f"Build complete: {build_id}")
    print(f"  Output: {build_dir}")
    print(f"  Manifest: {build_dir / 'build_manifest.json'}")
    print(f"  Summary: {build_dir / 'dataset_summary.json'}")

    return build_dir


def _copy_base_as_final(base_dir: Path, build_dir: Path) -> None:
    """Copy base COCO output as the final build (no HN merge needed)."""
    import shutil
    for name in ("train.json", "val.json", "train_provenance.csv", "val_provenance.csv"):
        src = base_dir / name
        if src.exists():
            shutil.copy2(src, build_dir / name)
    for split in ("train", "val"):
        src_dir = base_dir / split
        dst_dir = build_dir / split
        if src_dir.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                dst = dst_dir / f.name
                if not dst.exists():
                    try:
                        dst.hardlink_to(f)
                    except OSError:
                        shutil.copy2(f, dst)


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Build a COCO dataset from a declarative spec"
    )
    parser.add_argument(
        "--spec", type=Path, required=True,
        help="Path to dataset spec YAML",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (default: derived from spec)",
    )
    parser.add_argument(
        "--no-check-files", action="store_true",
        help="Skip file existence checks during validation",
    )
    args = parser.parse_args()

    build_dataset(
        spec_path=args.spec,
        output_dir=args.output_dir,
        check_files=not args.no_check_files,
    )


if __name__ == "__main__":
    main()
