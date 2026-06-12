#!/usr/bin/env python3
"""C-3(a) Phase 0 — chip sampler (deliverable A).

Reconstruct the *sampling surface* of the current production training pool
(``unified_reviewall_v2``) without a GPU, stratify by region x imagery_layer,
and draw 150-200 chips for the unlabeled-real-PV-as-background audit.

The sampling surface is rebuilt by replaying the v2 positive-source loaders
(``pipeline.dataset_builder._build_records_from_positives``) — these only read
tile CRS headers + GT GeoPackages, never the model.  For each tile we enumerate
the *exact* chip windows ``export_coco_dataset.scan_chips_from_tile`` would
produce, then keep chips that carry GT (the chips that enter training; these are
where background supervision actually competes with real installations).

Output: ``chip_manifest.csv`` — one row per sampled chip with the source tile
path + pixel window + the existing GT polygon references, so the pod-side runner
(deliverable B) can re-render the chip and overlay GT.  Also writes
``gt_refs.gpkg`` (existing GT footprints in tile CRS, joined by chip_uid) and a
``sample_meta.json`` provenance record.

ZERO GPU.  Strata whose tiles are not present on the current machine are
reported (and skipped); run the sampler where *all* tiles live (the pod, CPU)
to get a fully stratified sample.  See the runbook.

Usage
-----
    python scripts/training/sample_c3a_phase0_chips.py \
        --spec configs/pipelines/datasets/unified_reviewall_v2.yaml \
        --target 180 --seed 42 \
        --out-dir results/analysis/c3a_phase0/<run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import geopandas as gpd  # noqa: E402
import rasterio  # noqa: E402
from shapely.geometry import box as shapely_box  # noqa: E402

from core.training.c3a_phase0 import (  # noqa: E402
    allocate_stratified_quota,
    enumerate_chip_windows,
    make_chip_uid,
    sample_indices,
    stratum_key,
    stratum_sub_seed,
)


def _records(spec):
    """Rebuild the per-grid training records (CPU only)."""
    from pipeline.dataset_builder import _build_records_from_positives

    return _build_records_from_positives(
        spec, exclude_imagery_layers=set(spec.exclude_imagery_layers)
    )


def _spec_expected_strata(spec) -> set[str]:
    """Strata (region:imagery_layer) the spec's positive sources declare.

    Derived from the spec itself (not from realized records) so a machine that
    lacks a region's tiles still reports that stratum as expected-but-missing
    rather than silently dropping it.  Honors ``exclude_imagery_layers``.
    """
    excluded = set(spec.exclude_imagery_layers)
    strata: set[str] = set()
    for ps in spec.positives:
        layers = ps.imagery_layers or [None]
        for region in (ps.regions or []):
            for layer in layers:
                if layer is None or layer in excluded:
                    continue
                strata.add(stratum_key(region, layer))
    return strata


def _enumerate_positive_chips(record, chip_size: int, overlap: float):
    """Yield positive chip descriptors for one record (grid).

    A *positive* chip is one whose window intersects at least one GT polygon —
    exactly the chips that carry annotations in the COCO export.  For each such
    chip we attach the GT polygon pixel/world geometries clipped to the window.
    """
    annots = record["annots"]
    tile_map = record["tile_map"]
    tile_to_annots = record["tile_to_annots"]

    for tile_stem, annot_indices in tile_to_annots.items():
        if not annot_indices:
            continue
        tile_path = tile_map.get(tile_stem)
        if tile_path is None:
            continue
        with rasterio.open(tile_path) as src:
            transform = src.transform
            tile_w, tile_h = src.width, src.height

        # Pre-compute GT pixel geometries once per tile.
        from export_coco_dataset import polygon_to_pixel_coords

        gt_pixel = {}
        for aidx in annot_indices:
            geom = annots.loc[aidx, "geometry"]
            if geom is None or geom.is_empty:
                continue
            pgeom = polygon_to_pixel_coords(geom, transform)
            if not pgeom.is_empty and pgeom.is_valid:
                gt_pixel[aidx] = pgeom

        for x0, y0, w, h in enumerate_chip_windows(
            tile_w, tile_h, chip_size, overlap
        ):
            chip_box = shapely_box(x0, y0, x0 + w, y0 + h)
            hit_aidx = []
            for aidx, pgeom in gt_pixel.items():
                inter = pgeom.intersection(chip_box)
                if inter.is_empty or inter.area < 4:
                    continue
                hit_aidx.append(aidx)
            if not hit_aidx:
                continue
            yield {
                "tile_stem": tile_stem,
                "tile_path": str(tile_path),
                "x0": x0,
                "y0": y0,
                "w": w,
                "h": h,
                "gt_aidx": hit_aidx,
            }


def main() -> int:
    ap = argparse.ArgumentParser(description="C-3(a) Phase 0 chip sampler")
    ap.add_argument("--spec", type=Path,
                    default=REPO_ROOT / "configs/pipelines/datasets/unified_reviewall_v2.yaml",
                    help="Read-only provenance spec of the production train pool")
    ap.add_argument("--target", type=int, default=180,
                    help="Total chips to sample (plan: 150-200)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--include-splits", nargs="+", default=["train"],
                    help="Which record splits to sample from (train pool by default; "
                         "val grids are whole-grid JHB holdout)")
    args = ap.parse_args()

    if not (1 <= args.target <= 1000):
        ap.error("--target out of sane range [1, 1000]")

    from pipeline.specs import load_spec

    spec = load_spec(str(args.spec), check_files=False)
    print(f"[sample] spec={spec.name} chip={spec.chip.size} overlap={spec.chip.overlap}")

    records = _records(spec)
    records = [r for r in records if r["split"] in set(args.include_splits)]

    # ── Build the per-stratum positive-chip universe ─────────────────────
    # chip_pool[stratum] = list of chip descriptors (deterministic order).
    chip_pool: dict[str, list[dict]] = {}
    # Track which (region, imagery_layer) strata the spec expected but whose
    # tiles were not on disk here (reported, never faked).  Seed the expected
    # set from the SPEC's positive sources so a CT-only local machine still
    # flags the JHB vexcel stratum as missing (records for grids whose tiles
    # are absent never make it into `records`, so realized < spec-expected).
    expected_strata: set[str] = _spec_expected_strata(spec)
    realized_strata: set[str] = set()

    for r in records:
        skey = stratum_key(r["region"], r["imagery_layer"])
        expected_strata.add(skey)  # idempotent (already seeded from spec)
        chips = list(
            _enumerate_positive_chips(r, spec.chip.size, spec.chip.overlap)
        )
        if not chips:
            continue
        realized_strata.add(skey)
        for c in chips:
            c["region"] = r["region"]
            c["imagery_layer"] = r["imagery_layer"]
            c["grid_id"] = r["grid_id"]
            c["tile_crs"] = r["tile_crs"]
            c["annots_ref"] = r["annots"]  # GeoDataFrame, kept in-mem for GT export
        chip_pool.setdefault(skey, []).extend(chips)

    missing_strata = sorted(expected_strata - realized_strata)
    if missing_strata:
        print(f"[sample][WARN] strata with NO local tiles (skipped): {missing_strata}")
        print("[sample][WARN] re-run on a machine/pod where these tiles exist "
              "for a fully stratified sample.")

    stratum_counts = {k: len(v) for k, v in chip_pool.items()}
    print(f"[sample] positive-chip universe by stratum: {stratum_counts}")
    if not stratum_counts:
        print("[sample][ERROR] no positive chips found in any stratum; nothing to sample.")
        return 2

    # ── Stratified quota + deterministic sample ──────────────────────────
    quota = allocate_stratified_quota(stratum_counts, args.target)
    print(f"[sample] stratified quota: {quota}")

    sampled_rows: list[dict] = []
    gt_records: list[dict] = []
    for skey, chips in sorted(chip_pool.items()):
        q = quota.get(skey, 0)
        if q <= 0:
            continue
        # Per-stratum seed offset keeps strata independent but reproducible.
        sub_seed = stratum_sub_seed(args.seed, skey)
        idxs = sample_indices(len(chips), q, sub_seed)
        for i in idxs:
            c = chips[i]
            chip_uid = make_chip_uid(
                c["region"], c["imagery_layer"], c["grid_id"],
                c["tile_stem"], c["x0"], c["y0"],
            )
            sampled_rows.append({
                "chip_uid": chip_uid,
                "region": c["region"],
                "imagery_layer": c["imagery_layer"],
                "grid_id": c["grid_id"],
                "tile_stem": c["tile_stem"],
                "tile_path": c["tile_path"],
                "tile_crs": c["tile_crs"],
                "x0": c["x0"],
                "y0": c["y0"],
                "w": c["w"],
                "h": c["h"],
                "chip_size": spec.chip.size,
                "n_gt": len(c["gt_aidx"]),
            })
            # Export existing GT footprints (world coords) for this chip.
            annots = c["annots_ref"]
            for aidx in c["gt_aidx"]:
                geom = annots.loc[aidx, "geometry"]
                ls = annots.loc[aidx, "label_source"] if "label_source" in annots.columns else ""
                gt_records.append({
                    "chip_uid": chip_uid,
                    "grid_id": c["grid_id"],
                    "source_aidx": int(aidx),
                    "label_source": str(ls),
                    "geometry": geom,
                })

    print(f"[sample] sampled {len(sampled_rows)} chips "
          f"({len(gt_records)} GT refs)")

    # ── Write outputs ────────────────────────────────────────────────────
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import csv as _csv
    manifest_cols = [
        "chip_uid", "region", "imagery_layer", "grid_id", "tile_stem",
        "tile_path", "tile_crs", "x0", "y0", "w", "h", "chip_size", "n_gt",
    ]
    manifest_path = out_dir / "chip_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=manifest_cols)
        w.writeheader()
        for row in sorted(sampled_rows, key=lambda r: r["chip_uid"]):
            w.writerow(row)
    print(f"[sample] wrote {manifest_path}")

    # GT refs as GeoPackage (one CRS per stratum; write per-grid CRS groups).
    if gt_records:
        # Group by tile_crs via the sampled rows; GT geoms are already in tile CRS.
        crs_by_chip = {r["chip_uid"]: r["tile_crs"] for r in sampled_rows}
        by_crs: dict[str, list[dict]] = {}
        for rec in gt_records:
            crs = crs_by_chip.get(rec["chip_uid"], "EPSG:4326")
            by_crs.setdefault(crs, []).append(rec)
        # Write each CRS group to a layer-suffixed gpkg path.
        for crs, recs in by_crs.items():
            gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs=crs)
            safe = crs.replace(":", "_")
            gpkg_path = out_dir / f"gt_refs__{safe}.gpkg"
            gdf.to_file(gpkg_path, driver="GPKG")
            print(f"[sample] wrote {gpkg_path} ({len(recs)} GT refs, crs={crs})")

    meta = {
        "spec": str(args.spec),
        "spec_name": spec.name,
        "chip_size": spec.chip.size,
        "overlap": spec.chip.overlap,
        "target": args.target,
        "seed": args.seed,
        "include_splits": args.include_splits,
        "n_sampled": len(sampled_rows),
        "stratum_counts_universe": stratum_counts,
        "stratum_quota": quota,
        "expected_strata": sorted(expected_strata),
        "realized_strata": sorted(realized_strata),
        "missing_strata_no_local_tiles": missing_strata,
    }
    meta_path = out_dir / "sample_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[sample] wrote {meta_path}")

    if missing_strata:
        print("[sample] NOTE: sample is NOT fully stratified — missing strata "
              "above. Re-run where their tiles exist before drawing gate "
              "conclusions per stratum.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
