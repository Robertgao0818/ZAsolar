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
    dry_run: bool = False,
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

    # ── v2 positive-source path ───────────────────────────────────────
    # When a spec declares explicit `positives` (schema_version >= 2) it is
    # using the explicit-source builder, which drives the SAME selection
    # loaders as scripts/training/build_unified_reviewall.py so the emitted
    # build_manifest.json is byte-identical to the bespoke script.
    if spec.positives:
        return _build_dataset_v2(spec, spec_path, output_dir, dry_run=dry_run)

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
# v2 explicit positive-source builder (byte-equivalent to
# scripts/training/build_unified_reviewall.py)
# ════════════════════════════════════════════════════════════════════════
def _build_records_from_positives(spec, *, exclude_imagery_layers: set[str]):
    """Build the per-grid ``records`` list from spec.positives.

    Reuses the bespoke loaders from build_unified_reviewall so the per-record
    annotation ordering (hence ``source_id``) and the ``source_file`` paths
    are byte-identical to the bespoke script.  Each PositiveSourceSpec routes
    to either the CT pool loader or the JHB results-review loader.

    ``exclude_imagery_layers`` drops any positive whose imagery_layer is in
    the set (the aerial_2023 archive enforcement).
    """
    import rasterio
    from scripts.training import build_unified_reviewall as bur

    val_grids = set(spec.val_grids)
    # Group sources by (region, source_kind, imagery_layer) — each such group
    # contributes one record per grid (CT grids come from discover; JHB grids
    # come from the explicit grid list).
    records: list[dict] = []
    seen_record_keys: set[tuple] = set()

    for ps in spec.positives:
        imagery_layers = ps.imagery_layers or []
        for imagery_layer in (imagery_layers or [None]):
            if imagery_layer in exclude_imagery_layers:
                print(f"[EXCLUDE-LAYER] dropping positive source "
                      f"(region={ps.regions}, layer={imagery_layer})")
                continue

            if ps.source_kind == "pool_manifest":
                # CT: discover grids from data/annotations/, skip legacy +
                # spec exclude_grids, tag label_source via bespoke mapping.
                for region in ps.regions:
                    entries = bur._ct_entries() if region == "cape_town" else None
                    if entries is None:
                        raise NotImplementedError(
                            f"pool_manifest source for region {region!r} not "
                            f"wired (only cape_town's _ct_entries loader exists)"
                        )
                    exclude = set(spec.selection.exclude_grids)
                    for grid_id, entry in entries.items():
                        if grid_id in exclude:
                            continue
                        rkey = (region, grid_id, imagery_layer)
                        if rkey in seen_record_keys:
                            continue
                        if entry.schema_type == "legacy_ct":
                            print(f"[SKIP] CT {grid_id}: legacy weak-supervision "
                                  f"(schema={entry.schema_type})")
                            continue
                        tiles = bur._tiles_for(grid_id, region=region,
                                               imagery_layer=imagery_layer)
                        if not tiles:
                            print(f"[SKIP] CT {grid_id}: no tiles")
                            continue
                        with rasterio.open(tiles[0]) as src:
                            tile_crs = src.crs
                        annots = bur._load_ct_grid_annotations(grid_id, tile_crs)
                        if len(annots) == 0:
                            continue
                        seen_record_keys.add(rkey)
                        split = ps.split or (
                            "val" if grid_id in val_grids else "train"
                        )
                        records.append({
                            "split": split,
                            "region": region,
                            "imagery_layer": imagery_layer,
                            "grid_id": grid_id,
                            "tiles": tiles,
                            "tile_map": {t.stem: t for t in tiles},
                            "tile_crs": str(tile_crs),
                            "annots": annots,
                            "tile_to_annots": bur._assign_intersections(annots, tiles),
                            "source_files": {None: Path(entry.path)},
                        })

            elif ps.source_kind == "results_review_dir":
                # JHB: explicit grid list, load reviewed + sam_added gpkgs
                # from results/<region>/<run>/<grid>/review/.
                review_root = (REPO_ROOT / ps.review_root).resolve()
                # Temporarily point the bespoke loader at the spec's root so
                # _load_jhb_grid_annotations + source_files paths match.
                orig_root = bur.JHB_REVIEW_ROOT
                bur.JHB_REVIEW_ROOT = review_root
                try:
                    region = ps.regions[0] if ps.regions else "johannesburg"
                    for grid_id in ps.grids:
                        rkey = (region, grid_id, imagery_layer)
                        if rkey in seen_record_keys:
                            continue
                        tiles = bur._tiles_for(grid_id, region=region,
                                               imagery_layer=imagery_layer)
                        if not tiles:
                            print(f"[SKIP] JHB {grid_id}: no tiles")
                            continue
                        with rasterio.open(tiles[0]) as src:
                            tile_crs = src.crs
                        annots = bur._load_jhb_grid_annotations(grid_id, tile_crs)
                        if len(annots) == 0:
                            print(f"[SKIP] JHB {grid_id}: no annotations after combine")
                            continue
                        seen_record_keys.add(rkey)
                        review_dir = review_root / grid_id / "review"
                        split = ps.split or (
                            "val" if grid_id in val_grids else "train"
                        )
                        records.append({
                            "split": split,
                            "region": region,
                            "imagery_layer": imagery_layer,
                            "grid_id": grid_id,
                            "tiles": tiles,
                            "tile_map": {t.stem: t for t in tiles},
                            "tile_crs": str(tile_crs),
                            "annots": annots,
                            "tile_to_annots": bur._assign_intersections(annots, tiles),
                            "source_files": {
                                "reviewed_prediction": review_dir / f"{grid_id}_reviewed.gpkg",
                                "sam_added_browser": review_dir / f"{grid_id}_sam_added.gpkg",
                            },
                        })
                finally:
                    bur.JHB_REVIEW_ROOT = orig_root
            else:
                raise ValueError(f"unknown source_kind {ps.source_kind!r}")

    return records


def _build_dataset_v2(spec, spec_path, output_dir, *, dry_run: bool = False) -> Path:
    """v2 builder driven by spec.positives. Byte-equivalent selection."""
    from scripts.training import build_unified_reviewall as bur

    resolved_root = resolve_env_vars(spec.output.root)
    build_date = datetime.now(timezone.utc)
    if output_dir is None:
        date_str = build_date.strftime("%Y%m%d")
        dir_name = spec.output.name_template.format(
            name=spec.name, date=date_str,
        )
        build_dir = Path(resolved_root) / dir_name
    else:
        build_dir = Path(output_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    exclude_layers = set(spec.exclude_imagery_layers)
    print(f"[V2] exclude_imagery_layers: {sorted(exclude_layers)}")
    records = _build_records_from_positives(
        spec, exclude_imagery_layers=exclude_layers,
    )

    # Confirm exclude enforcement: 0 records on an excluded layer.
    bad = [r for r in records if r["imagery_layer"] in exclude_layers]
    assert not bad, (
        f"exclude_imagery_layers leak: {[(r['region'], r['grid_id'], r['imagery_layer']) for r in bad]}"
    )

    n_ct = sum(1 for r in records if r["region"] == "cape_town")
    n_jhb_train = sum(1 for r in records
                      if r["region"] == "johannesburg" and r["split"] == "train")
    n_jhb_val = sum(1 for r in records
                    if r["region"] == "johannesburg" and r["split"] == "val")
    print(f"[V2] CT grids={n_ct}  JHB train={n_jhb_train}  val={n_jhb_val}")

    # ── untrusted <= N × trusted assertion (lifted to mask_supervision) ──
    max_x = (spec.mask_supervision.untrusted_max_x_trusted
             if spec.mask_supervision else 4.0)
    train_recs = [r for r in records if r["split"] == "train"]
    train_summary = bur._per_record_summary(train_recs)
    train_trusted = sum(r["n_trusted"] for r in train_summary)
    train_untrusted = sum(r["n_untrusted"] for r in train_summary)
    ratio = train_untrusted / max(1, train_trusted)
    print(f"[V2][ASSERT] train trusted={train_trusted} untrusted={train_untrusted} "
          f"ratio={ratio:.2f} (cap={max_x})")
    assert train_untrusted <= max_x * train_trusted, (
        f"untrusted {train_untrusted} > {max_x} × trusted {train_trusted} "
        f"(mask_supervision.untrusted_max_x_trusted)"
    )

    # ── Build manifest (byte-equivalent shape) ───────────────────────
    selected_annotations = bur._selected_annotations_from_records(records)

    seen: set[str] = set()
    annotation_paths: list[Path] = []
    for rec in records:
        for src in rec.get("source_files", {}).values():
            if src is None:
                continue
            key = str(Path(src).resolve())
            if key not in seen:
                seen.add(key)
                annotation_paths.append(Path(src))
    source_inventory = build_source_inventory(annotation_paths)

    regions = sorted({rec["region"] for rec in records})
    resolved_tile_roots = {
        f"{rec['region']}/{rec['imagery_layer']}": str(rec["tiles"][0].parent)
        for rec in records
    }

    hn_config_list = []
    for hn in spec.hard_negatives:
        hn_config_list.append({
            "type": hn.type,
            "region": hn.region,
            "grids": hn.grids,
            "archetypes": hn.archetypes,
            "min_confidence": hn.min_confidence,
            "max_ratio": hn.max_ratio,
        })

    resolved_spec = _resolve_value(spec_to_dict(spec))

    if not dry_run:
        _scan_and_write_v2_coco(spec, records, build_dir)
        _maybe_add_negative_pool_hn(spec, build_dir)

    write_build_manifest(
        build_dir,
        build_id=generate_build_id(
            spec.name,
            json.dumps({
                "resolved_spec": resolved_spec,
                "source_sha256": sorted(
                    (e["path"], e["sha256"]) for e in source_inventory
                ),
                "selected_annotations": selected_annotations,
            }, sort_keys=True),
            build_date,
        ),
        spec_path=str(spec_path),
        resolved_spec=resolved_spec,
        resolved_spec_hash=compute_string_sha256(
            json.dumps(resolved_spec, sort_keys=True)
        ),
        regions=regions,
        evaluation_regime=spec.evaluation_regime,
        exclude_grids=spec.selection.exclude_grids,
        excluded_grids_reason=(
            "cape_town_independent_26 benchmark holdout + spec exclude_grids"
        ),
        source_inventory=source_inventory,
        split_strategy=spec.split.strategy,
        split_seed=spec.split.seed,
        easy_neg_ratio=spec.negatives.easy_neg_ratio,
        hard_negatives_config=hn_config_list,
        selected_annotations=selected_annotations,
        resolved_tile_roots=resolved_tile_roots,
        resolved_output_root=str(build_dir),
        entrypoint="pipeline.dataset_builder(v2)",
    )
    print(f"[V2][BUILD_MANIFEST] {build_dir / 'build_manifest.json'} "
          f"({len(source_inventory)} sources, "
          f"{len(selected_annotations)} selected_annotations)")
    return build_dir


def _scan_and_write_v2_coco(spec, records, build_dir: Path) -> None:
    """Scan chips + write COCO for the v2 records (mirrors the bespoke
    chip-scan loop; writes mask_trusted per instance via scan_chips_from_tile).
    """
    from export_coco_dataset import (
        scan_chips_from_tile, balance_chips, write_selected_chips,
        build_coco_json,
    )
    for split_name in ("train", "val"):
        all_images, all_annots, all_prov = [], [], []
        img_id, ann_id = 1, 1
        for rec in records:
            if rec["split"] != split_name:
                continue
            for stem, annot_indices in rec["tile_to_annots"].items():
                tile_path = rec["tile_map"][stem]
                imgs, anns, prov = scan_chips_from_tile(
                    tile_path=tile_path,
                    annotations=rec["annots"],
                    annot_indices=annot_indices,
                    chip_size=spec.chip.size,
                    overlap=spec.chip.overlap,
                    split_name=split_name,
                    image_id_start=img_id,
                    annot_id_start=ann_id,
                )
                for img in imgs:
                    img["region"] = rec["region"]
                    img["grid_id"] = rec["grid_id"]
                    img["imagery_layer"] = rec["imagery_layer"]
                img_id += len(imgs)
                ann_id += len(anns)
                all_images.extend(imgs)
                all_annots.extend(anns)
                all_prov.extend(prov)

        if split_name == "train" and spec.negatives.easy_neg_ratio >= 0:
            all_images, all_annots, all_prov = balance_chips(
                all_images, all_annots, all_prov,
                seed=spec.split.seed, neg_ratio=spec.negatives.easy_neg_ratio,
            )

        write_selected_chips(all_images, build_dir, spec.chip.size)
        coco = build_coco_json(all_images, all_annots, split=split_name,
                               category_name="solar_panel")
        (build_dir / f"{split_name}.json").write_text(json.dumps(coco) + "\n")
        print(f"[V2][COCO] wrote {build_dir / f'{split_name}.json'} "
              f"({len(all_images)} chips, {len(all_annots)} instances)")


def _maybe_add_negative_pool_hn(spec, build_dir: Path) -> None:
    """Extract + merge negative_pool HN chips if declared in the spec."""
    np_specs = [hn for hn in spec.hard_negatives if hn.type == "negative_pool"]
    if not np_specs:
        return
    from pipeline.hn_ops import extract_negative_pool_hn, merge_hn_into_coco
    base_train = build_dir / "train.json"
    if not base_train.exists():
        print("[V2][HN] no train.json yet; skipping negative_pool HN")
        return
    with open(base_train) as f:
        n_base_train = len(json.load(f)["images"])
    hn_results = []
    for i, hn in enumerate(np_specs):
        result = extract_negative_pool_hn(
            archetypes=hn.archetypes,
            output_dir=build_dir,
            chip_size=spec.chip.size,
            min_confidence=hn.min_confidence,
            regions=spec.regions,
        )
        # cap to max_ratio
        if hn.max_ratio and result.n_chips and n_base_train:
            max_allowed = int(hn.max_ratio / (1 - hn.max_ratio) * n_base_train)
            if result.n_chips > max_allowed:
                print(f"[V2][HN {i}] capping {result.n_chips} → {max_allowed}")
                result.images = result.images[:max_allowed]
                result.provenance = result.provenance[:max_allowed]
                result.n_chips = max_allowed
        hn_results.append(result)
        print(f"[V2][HN {i}] negative_pool: {result.n_chips} chips")
    if any(r.n_chips > 0 for r in hn_results):
        # merge_hn_into_coco reads base from a dir; build_dir already holds the
        # base train/val + chips, so merge in place.
        merge_hn_into_coco(base_dir=build_dir, hn_results=hn_results,
                           output_dir=build_dir)


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
    parser.add_argument(
        "--dry-run", action="store_true",
        help="v2 path only: compute selection + manifest, no chip writes.",
    )
    args = parser.parse_args()

    build_dataset(
        spec_path=args.spec,
        output_dir=args.output_dir,
        check_files=not args.no_check_files,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
