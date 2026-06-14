#!/usr/bin/env python3
"""Build the unified_reviewall_20260511 COCO dataset.

⚠️ DEPRECATED (2026-05-29, training-pool normalization Phase 4) — kept for
historical reproducibility, DO NOT DELETE. This bespoke builder is now
superseded by the declarative spec
``configs/pipelines/datasets/unified_reviewall_v2.yaml`` driven by
``pipeline.dataset_builder`` (the v2 positive-source path). The v2 builder
reuses the loaders in THIS module, so the two are byte-equivalent: a
dry-run byte-diff (2026-05-29) showed the spec build's
``build_manifest.json::selected_annotations`` SET-EQUAL to this script's on
``(region, grid_id, label_source, source_file, source_id)`` — 9,738 rows
each, 0 added / 0 dropped (also equal including ``split`` + ``imagery_layer``).
New builds should use the v2 spec; the spec adds ``exclude_imagery_layers``
(aerial_2023 archive enforcement) which this script lacks.

Per docs/plans/review-aerial-2023-jhb-opengeoai-v3c-parallel-biscuit.md:
- CT positives  = Batch003/004/002b/EarlySAM2 (69 grids), exclude cape_town_independent_26 (26)
- JHB positives = 20 grids of Vexcel 2024 raw review:
    <grid>_reviewed.gpkg   filtered to review_status=="correct" → label_source=reviewed_prediction (mask_trusted=False)
    <grid>_sam_added.gpkg  all rows                              → label_source=sam_added_browser   (mask_trusted=True)
  with 5 grids held out as val (whole-grid).
- mask_trusted is computed per annotation in export_coco_dataset.scan_chips_from_tile via the
  shared _MASK_TRUSTED dict. Builder verifies Untrusted <= 4 × Trusted (Gerstgrasser 2024 COLM).

HN chips are NOT bundled here — generate separately via export_v4_1_hn.py /
export_targeted_hn.py and pass via downstream merge / --add-hn-coco flag.

Usage:
    python scripts/training/build_unified_reviewall.py \\
        --output-dir /workspace/coco/unified_reviewall_20260511 \\
        --val-jhb-grids G0772 G0816 G0817 G0888 G0925
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box as shapely_box

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from export_coco_dataset import (  # noqa: E402
    balance_chips,
    build_coco_json,
    scan_chips_from_tile,
    write_selected_chips,
)
from pipeline.manifests import (  # noqa: E402
    write_build_manifest,
    generate_build_id,
    build_source_inventory,
    compute_string_sha256,
)

# Positive-source loaders + label_source derivation were extracted to
# core/training/positive_sources.py (2026-06-12, architecture review step 8)
# so the declarative builder can drive them via public functions instead of
# importing these privates + monkeypatching JHB_REVIEW_ROOT. This DEPRECATED
# script re-imports them to keep its CLI behaviour byte-identical.
from core.training.positive_sources import (  # noqa: E402,F401
    JHB_REVIEW_ROOT,
    TRUSTED_SOURCES,
    UNTRUSTED_SOURCES,
    _tiles_for,
    _assign_intersections,
    _load_jhb_grid_annotations,
    _ct_source_to_label_source,
    _ct_entries,
    _load_ct_grid_annotations,
    _per_record_summary,
    _selected_annotations_from_records,
    _src_rel,
)


def _pixel_area_m2(tile_path: Path, cache: dict[str, float]) -> float:
    """Approximate pixel area in m² for any CRS.

    Branches:
    - Geographic CRS (lon/lat): convert deg² → m² via lat-band approximation
      centred on the tile.
    - EPSG:3857 (Web Mercator): linear units are Web Mercator metres, which
      are stretched by 1/cos(φ) relative to ground metres. Ground area =
      WM_area × cos²(φ_centre).
    - Other projected CRS: assume linear unit ≈ ground metre.

    Cached by tile_path to avoid re-opening.

    R0.5 extension for solar_zerov2 W1.a closure — consumed by the
    scale_jitter content-conditional bucket router via chips_metadata.json.
    """
    key = str(tile_path)
    if key in cache:
        return cache[key]
    with rasterio.open(tile_path) as src:
        t = src.transform
        crs = src.crs
        px_x = abs(t.a)
        px_y = abs(t.e)
        center_lat = None
        if crs is None:
            area = px_x * px_y
        elif crs.is_geographic:
            center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
            m_per_deg_lat = 111_320.0
            m_per_deg_lon = 111_320.0 * math.cos(math.radians(center_lat))
            area = (px_x * m_per_deg_lon) * (px_y * m_per_deg_lat)
        elif crs.to_epsg() == 3857:
            # Web Mercator scale factor = 1/cos(φ); need lon/lat of centre.
            from pyproj import Transformer
            xform = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            cx = (src.bounds.left + src.bounds.right) / 2.0
            cy = (src.bounds.top + src.bounds.bottom) / 2.0
            _, center_lat = xform.transform(cx, cy)
            cos_phi = math.cos(math.radians(center_lat))
            area = (px_x * cos_phi) * (px_y * cos_phi)
        elif crs.is_projected:
            area = px_x * px_y
        else:
            area = px_x * px_y
    cache[key] = area
    return area

# ─── cape_town_independent_26 holdout (configs/benchmarks/post_train.yaml) ───
CT_INDEP_26 = {
    "G1240", "G1243", "G1244", "G1245",
    "G1293", "G1294", "G1297", "G1298", "G1299", "G1300",
    "G1349", "G1354",
    "G1410", "G1411",
    "G1466", "G1467",
    "G1516", "G1520", "G1521", "G1522", "G1523", "G1524",
    "G1569", "G1570", "G1571", "G1572",
}

# ─── JHB Vexcel 2024 25-grid clean GT pool ────────────────────────────────
JHB_VEXCEL_25 = (
    "G0772", "G0773", "G0774", "G0775", "G0776",
    "G0814", "G0815", "G0816", "G0817", "G0818",
    "G0853", "G0854", "G0855", "G0856", "G0857",
    "G0888", "G0889", "G0890", "G0891", "G0892",
    "G0922", "G0923", "G0924", "G0925", "G0926",
)

DEFAULT_VAL_JHB = ("G0772", "G0816", "G0817", "G0888", "G0925")

# JHB_REVIEW_ROOT, TRUSTED_SOURCES, UNTRUSTED_SOURCES and the positive-source
# loaders (_tiles_for / _assign_intersections / _load_jhb_grid_annotations /
# _ct_source_to_label_source / _ct_entries / _load_ct_grid_annotations /
# _per_record_summary / _selected_annotations_from_records / _src_rel) now live
# in core/training/positive_sources.py and are re-imported at the top of this
# module (2026-06-12 architecture review step 8). build() below calls them
# directly; _load_jhb_grid_annotations defaults review_root to JHB_REVIEW_ROOT,
# so this script's behaviour is byte-identical to before the extraction.


def _emit_build_manifest(args: argparse.Namespace, out_dir: Path,
                         records: list[dict], val_jhb_grids: set,
                         exclude_ct: set) -> None:
    """Write build_manifest.json (additive provenance, must never break the
    build). Wrapped by caller in try/except."""
    selected_annotations = _selected_annotations_from_records(records)

    # Unique annotation source gpkgs across all records.
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

    resolved_spec = {
        "training_set_id": "unified_reviewall_20260511",
        "regions": regions,
        "ct_excluded_grids": sorted(exclude_ct),
        "jhb_vexcel_grids": sorted(JHB_VEXCEL_25),
        "jhb_val_grids": sorted(val_jhb_grids),
        "ct_imagery_layer": "aerial_2025",
        "jhb_imagery_layer": "vexcel_2024",
        "chip_size": args.chip_size,
        "overlap": args.overlap,
        "neg_ratio": args.neg_ratio,
        "seed": args.seed,
        "skip_ct": bool(args.skip_ct),
        "skip_jhb": bool(args.skip_jhb),
    }

    # Build fingerprint: everything that, if changed, would change the
    # dataset contents — effective config plus the sha256 of every source
    # gpkg. Excludes timestamps so re-running on identical inputs yields the
    # same build_id.
    fingerprint = {
        "resolved_spec": resolved_spec,
        "source_sha256": sorted(
            (e["path"], e["sha256"]) for e in source_inventory
        ),
    }
    build_fingerprint_json = json.dumps(fingerprint, sort_keys=True)
    resolved_spec_hash = compute_string_sha256(
        json.dumps(resolved_spec, sort_keys=True)
    )
    build_id = generate_build_id("unified_reviewall", build_fingerprint_json)

    write_build_manifest(
        out_dir,
        build_id=build_id,
        spec_path="bespoke:scripts/training/build_unified_reviewall.py",
        resolved_spec=resolved_spec,
        resolved_spec_hash=resolved_spec_hash,
        regions=regions,
        evaluation_regime="installation",
        exclude_grids=sorted(exclude_ct),
        excluded_grids_reason=(
            "cape_town_independent_26 benchmark holdout "
            "(configs/benchmarks/post_train.yaml) plus any --exclude-ct-grids"
        ),
        source_inventory=source_inventory,
        split_strategy="whole_grid_jhb_holdout",
        split_seed=args.seed,
        easy_neg_ratio=args.neg_ratio,
        hard_negatives_config=[],
        selected_annotations=selected_annotations,
        resolved_tile_roots=resolved_tile_roots,
        resolved_output_root=str(out_dir),
        entrypoint="scripts/training/build_unified_reviewall.py",
    )
    print(f"[BUILD_MANIFEST] {out_dir / 'build_manifest.json'} "
          f"(build_id={build_id}, "
          f"{len(source_inventory)} sources, "
          f"{len(selected_annotations)} selected_annotations)")


def build(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    val_jhb_grids = set(args.val_jhb_grids) if args.val_jhb_grids else set(DEFAULT_VAL_JHB)
    print(f"[VAL] JHB val grids: {sorted(val_jhb_grids)}")

    exclude_ct = set(args.exclude_ct_grids) if args.exclude_ct_grids else set()
    exclude_ct |= CT_INDEP_26
    print(f"[EXCLUDE-CT] {len(exclude_ct)} grids removed from CT train pool")

    # ── CT train pool ────────────────────────────────────────────────
    records = []
    if not args.skip_ct:
        ct_entries_all = _ct_entries()
        ct_entries = {g: e for g, e in ct_entries_all.items() if g not in exclude_ct}
        for grid_id, entry in ct_entries.items():
            # CT all post-2026-03 batches live on aerial_2025; legacy weak-supervision
            # grids (G0854, G0855, G0910, G1018, G1023, G1134, G909, G964, G967) are
            # skipped via the schema check below — they'd map to aerial_legacy / unknown.
            if entry.schema_type == "legacy_ct":
                print(f"[SKIP] CT {grid_id}: legacy weak-supervision (schema={entry.schema_type})")
                continue
            tiles = _tiles_for(grid_id, region="cape_town",
                               imagery_layer="aerial_2025")
            if not tiles:
                print(f"[SKIP] CT {grid_id}: no tiles")
                continue
            with rasterio.open(tiles[0]) as src:
                tile_crs = src.crs
            annots = _load_ct_grid_annotations(grid_id, tile_crs)
            if len(annots) == 0:
                continue
            records.append({
                "split": "train",
                "region": "cape_town",
                "imagery_layer": "aerial_2025",
                "grid_id": grid_id,
                "tiles": tiles,
                "tile_map": {t.stem: t for t in tiles},
                "tile_crs": str(tile_crs),
                "annots": annots,
                "tile_to_annots": _assign_intersections(annots, tiles),
                # Provenance: each CT grid resolves to a single discovered
                # annotation gpkg (entry.path); label_source maps onto it.
                "source_files": {None: Path(entry.path)},
            })
    print(f"[CT] {sum(1 for r in records if r['region'] == 'cape_town')} grids loaded")

    # ── JHB train + val pool (Vexcel 2024) ───────────────────────────
    if not args.skip_jhb:
        for grid_id in JHB_VEXCEL_25:
            tiles = _tiles_for(grid_id, region="johannesburg",
                               imagery_layer="vexcel_2024")
            if not tiles:
                print(f"[SKIP] JHB {grid_id}: no tiles")
                continue
            with rasterio.open(tiles[0]) as src:
                tile_crs = src.crs
            annots = _load_jhb_grid_annotations(grid_id, tile_crs)
            if len(annots) == 0:
                print(f"[SKIP] JHB {grid_id}: no annotations after combine")
                continue
            jhb_review_dir = JHB_REVIEW_ROOT / grid_id / "review"
            records.append({
                "split": "val" if grid_id in val_jhb_grids else "train",
                "region": "johannesburg",
                "imagery_layer": "vexcel_2024",
                "grid_id": grid_id,
                "tiles": tiles,
                "tile_map": {t.stem: t for t in tiles},
                "tile_crs": str(tile_crs),
                "annots": annots,
                "tile_to_annots": _assign_intersections(annots, tiles),
                # Provenance: label_source identifies which gpkg each row
                # came from (see _load_jhb_grid_annotations).
                "source_files": {
                    "reviewed_prediction": jhb_review_dir / f"{grid_id}_reviewed.gpkg",
                    "sam_added_browser": jhb_review_dir / f"{grid_id}_sam_added.gpkg",
                },
            })
    n_jhb_train = sum(1 for r in records if r["region"] == "johannesburg" and r["split"] == "train")
    n_jhb_val = sum(1 for r in records if r["region"] == "johannesburg" and r["split"] == "val")
    print(f"[JHB] train={n_jhb_train}  val={n_jhb_val} (val grids: {sorted(val_jhb_grids)})")

    # ── Untrusted ≤ 4 × Trusted assertion (Gerstgrasser 2024 COLM) ────
    train_recs = [r for r in records if r["split"] == "train"]
    train_summary = _per_record_summary(train_recs)
    train_trusted = sum(r["n_trusted"] for r in train_summary)
    train_untrusted = sum(r["n_untrusted"] for r in train_summary)
    print(f"[ASSERT] train pool: trusted={train_trusted}  untrusted={train_untrusted}  "
          f"ratio={train_untrusted / max(1, train_trusted):.2f}")
    assert train_untrusted <= 4 * train_trusted, (
        f"untrusted {train_untrusted} > 4 × trusted {train_trusted}; "
        f"violates accumulation principle (Gerstgrasser 2024 COLM). "
        f"Expand trusted pool or shrink untrusted pool."
    )

    # ── Per-grid summary log ─────────────────────────────────────────
    summary_csv = out_dir / "build_summary.csv"
    with summary_csv.open("w", encoding="utf-8") as f:
        f.write("split,region,grid_id,n_polygons,n_trusted,n_untrusted,n_tiles\n")
        for row in _per_record_summary(records):
            f.write(f"{row['split']},{row['region']},{row['grid_id']},"
                    f"{row['n_polygons']},{row['n_trusted']},{row['n_untrusted']},"
                    f"{row['n_tiles']}\n")
    print(f"[SUMMARY] {summary_csv}")

    if args.dry_run:
        # Selection is fully determined here; emit provenance manifest so the
        # dry-run is verifiable. Additive only — never break the build.
        try:
            _emit_build_manifest(args, out_dir, records, val_jhb_grids, exclude_ct)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] build_manifest write failed (non-fatal): {exc}")
        return {"summary": str(summary_csv), "dry_run": True}

    # ── Chip scan + COCO build ───────────────────────────────────────
    # R0.5 (solar_zerov2 Codex v2 W1.a) — collect per-chip metadata for
    # the scale_jitter content-conditional bucket router. See
    # solar_zerov2/core/manifest_validator.py for the schema check.
    pixel_area_cache: dict[str, float] = {}
    chips_metadata: dict[str, dict] = {}
    for split_name in ("train", "val"):
        all_images = []
        all_annots = []
        all_prov = []
        img_id_counter = 1
        ann_id_counter = 1
        for rec in records:
            if rec["split"] != split_name:
                continue
            for stem, annot_indices in rec["tile_to_annots"].items():
                tile_path = rec["tile_map"][stem]
                imgs, anns, prov = scan_chips_from_tile(
                    tile_path=tile_path,
                    annotations=rec["annots"],
                    annot_indices=annot_indices,
                    chip_size=args.chip_size,
                    overlap=args.overlap,
                    split_name=split_name,
                    image_id_start=img_id_counter,
                    annot_id_start=ann_id_counter,
                )
                for img in imgs:
                    img["region"] = rec["region"]
                    img["grid_id"] = rec["grid_id"]
                    img["imagery_layer"] = rec["imagery_layer"]
                for row in prov:
                    row["region"] = rec["region"]
                    row["grid_id"] = rec["grid_id"]
                    row["imagery_layer"] = rec["imagery_layer"]
                img_id_counter += len(imgs)
                ann_id_counter += len(anns)
                all_images.extend(imgs)
                all_annots.extend(anns)
                all_prov.extend(prov)

        n_pos = sum(1 for img in all_images if img["positive"])
        n_neg = len(all_images) - n_pos
        print(f"[SCAN] {split_name}: {len(all_images)} chips "
              f"({n_pos} positive, {n_neg} negative), {len(all_annots)} instances")

        # Negative ratio balancing (only on train; val keeps full empties)
        if split_name == "train" and args.neg_ratio >= 0:
            all_images, all_annots, all_prov = balance_chips(
                all_images, all_annots, all_prov,
                seed=args.seed, neg_ratio=args.neg_ratio,
            )
            n_pos = sum(1 for img in all_images if img["positive"])
            n_neg = len(all_images) - n_pos
            print(f"[BALANCE] train post-balance: {len(all_images)} chips "
                  f"({n_pos} positive, {n_neg} negative)")

        # R0.5 — per-chip metadata for scale_jitter bucket router.
        # Done BEFORE write_selected_chips because _chip_annots is in
        # chip-local pixel coords and we need the source tile's pixel
        # area to convert to m². write_selected_chips() does not mutate
        # the image dicts but we keep the order safe.
        for img in all_images:
            tile_path = Path(img["_tile_path"])
            px_area_m2 = _pixel_area_m2(tile_path, pixel_area_cache)
            chip_annots = img.get("_chip_annots", [])
            if chip_annots:
                max_area_m2 = max(
                    float(geom.area) * px_area_m2 for _, geom in chip_annots
                )
                n_polys = len(chip_annots)
            else:
                max_area_m2 = None
                n_polys = 0
            chips_metadata[img["file_name"]] = {
                "max_polygon_area_m2": (
                    round(max_area_m2, 3) if max_area_m2 is not None else None
                ),
                "n_polygons": n_polys,
                "split": split_name,
            }

        write_selected_chips(all_images, out_dir, args.chip_size)
        coco_json = build_coco_json(
            all_images, all_annots, split=split_name,
            category_name="solar_panel",
        )
        coco_path = out_dir / f"{split_name}.json"
        coco_path.write_text(json.dumps(coco_json) + "\n")
        print(f"[COCO] wrote {coco_path}")

        # provenance CSV
        prov_csv = out_dir / f"{split_name}_provenance.csv"
        if all_prov:
            keys = list(all_prov[0].keys())
            with prov_csv.open("w", encoding="utf-8") as f:
                f.write(",".join(keys) + "\n")
                for row in all_prov:
                    f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    # ── Manifest ─────────────────────────────────────────────────────
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_set_id": "unified_reviewall_20260511",
        "val_jhb_grids": sorted(val_jhb_grids),
        "excluded_ct_grids": sorted(exclude_ct),
        "train_trusted_polygons": train_trusted,
        "train_untrusted_polygons": train_untrusted,
        "untrusted_trusted_ratio": (
            train_untrusted / train_trusted if train_trusted > 0 else None
        ),
        "chip_size": args.chip_size,
        "overlap": args.overlap,
        "neg_ratio": args.neg_ratio,
    }, indent=2) + "\n")
    print(f"[MANIFEST] {manifest_path}")

    # ── Chips metadata (R0.5 — solar_zerov2 W1.a) ────────────────────
    chips_meta_path = out_dir / "chips_metadata.json"
    chips_meta_path.write_text(json.dumps({
        "schema_version": 1,
        "schema_source": "solar_zerov2/core/manifest_validator.py",
        "chips": chips_metadata,
    }, indent=2) + "\n")
    print(f"[CHIPS_META] {chips_meta_path} ({len(chips_metadata)} chips)")

    # ── Build provenance manifest (additive; never breaks the build) ─────
    try:
        _emit_build_manifest(args, out_dir, records, val_jhb_grids, exclude_ct)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] build_manifest write failed (non-fatal): {exc}")

    return {
        "output_dir": str(out_dir),
        "manifest": str(manifest_path),
        "chips_metadata": str(chips_meta_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True,
                        help="Output dir (e.g. /workspace/coco/unified_reviewall_20260511)")
    parser.add_argument("--val-jhb-grids", nargs="+", default=None,
                        help="JHB Vexcel grids held out as val. Default = "
                             f"{','.join(DEFAULT_VAL_JHB)}.")
    parser.add_argument("--exclude-ct-grids", nargs="+", default=None,
                        help="Additional CT grids to exclude beyond cape_town_independent_26.")
    parser.add_argument("--chip-size", type=int, default=400)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--neg-ratio", type=float, default=0.15,
                        help="Negative:positive chip ratio for train (default 0.15).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ct", action="store_true",
                        help="Skip CT side (JHB-only build, debugging).")
    parser.add_argument("--skip-jhb", action="store_true",
                        help="Skip JHB side (CT-only build, debugging).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute summary + assertion only; no chip writes.")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
