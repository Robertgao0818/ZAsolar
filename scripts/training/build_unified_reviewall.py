#!/usr/bin/env python3
"""Build the unified_reviewall_20260511 COCO dataset.

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
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box as shapely_box

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.grid_utils import resolve_tiles_dir  # noqa: E402
from core.annotation_loader import (  # noqa: E402
    discover_annotations, load_annotation_gdf,
)
from export_coco_dataset import (  # noqa: E402
    balance_chips,
    build_coco_json,
    scan_chips_from_tile,
    split_tiles,
    write_selected_chips,
    mask_trusted_for,
    _MASK_TRUSTED,
)

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

JHB_REVIEW_ROOT = (
    PROJECT_ROOT / "results" / "johannesburg" / "v3c_vexcel_2024_ch1_sample"
)

UNTRUSTED_SOURCES = {k for k, v in _MASK_TRUSTED.items() if not v}
TRUSTED_SOURCES = {k for k, v in _MASK_TRUSTED.items() if v}


def _tiles_for(grid_id: str, region: str, imagery_layer: str | None = None) -> list[Path]:
    tiles_dir = resolve_tiles_dir(grid_id, region=region, imagery_layer=imagery_layer)
    if tiles_dir.is_file():
        return [tiles_dir]
    tiles = sorted(tiles_dir.glob(f"{grid_id}_*_*_geo.tif"))
    if not tiles:
        tiles = sorted(p for p in tiles_dir.glob(f"{grid_id}_*.tif")
                       if "mosaic" not in p.stem)
    return tiles


def _assign_intersections(annotations: gpd.GeoDataFrame,
                          tiles: list[Path]) -> dict[str, list[int]]:
    tile_bounds = {}
    for tile in tiles:
        with rasterio.open(tile) as src:
            b = src.bounds
            tile_bounds[tile.stem] = shapely_box(b.left, b.bottom, b.right, b.top)
    out: dict[str, list[int]] = {stem: [] for stem in tile_bounds}
    for idx, row in annotations.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        for stem, bbox in tile_bounds.items():
            if geom.intersects(bbox):
                out[stem].append(idx)
    return out


def _load_jhb_grid_annotations(grid_id: str, tile_crs) -> gpd.GeoDataFrame:
    """Combine V3C-correct reviewed predictions + browser SAM_added FN into
    one GDF with label_source tagged per row."""
    review_dir = JHB_REVIEW_ROOT / grid_id / "review"
    reviewed_path = review_dir / f"{grid_id}_reviewed.gpkg"
    sam_added_path = review_dir / f"{grid_id}_sam_added.gpkg"

    parts = []
    if reviewed_path.exists():
        g_rev = gpd.read_file(reviewed_path)
        if "review_status" in g_rev.columns:
            g_rev = g_rev[g_rev["review_status"] == "correct"].copy()
        else:
            print(f"[WARN] {reviewed_path.name} missing review_status column; keeping all rows")
        g_rev["label_source"] = "reviewed_prediction"
        parts.append(g_rev)
    else:
        print(f"[WARN] missing {reviewed_path}")

    if sam_added_path.exists():
        g_sam = gpd.read_file(sam_added_path)
        g_sam["label_source"] = "sam_added_browser"
        parts.append(g_sam)
    else:
        print(f"[WARN] missing {sam_added_path}")

    if not parts:
        return gpd.GeoDataFrame(columns=["geometry", "label_source"], crs="EPSG:4326")

    common_cols = set.intersection(*(set(p.columns) for p in parts))
    common_cols = list(common_cols | {"label_source", "geometry"})
    parts = [p[[c for c in common_cols if c in p.columns]].copy() for p in parts]
    gdf = pd.concat(parts, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=parts[0].crs)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(tile_crs).reset_index(drop=True)


def _ct_source_to_label_source(src):
    """Map CT-batch GeoPackage 'source' value → label_source enum.

    CT batch003/004 schemas use NaN for V3C-reviewed-accepted and
    'sam_fn_marker' for non-interactive FN catches.  CT early SAM2
    (G1189/G1190/G1238) uses 'sam2'.  CT batch001/002/002b have no
    'source' column at all (default to human_manual_sam_assisted).
    """
    if src is None or (isinstance(src, float) and pd.isna(src)):
        return "reviewed_prediction"     # V3-C accepted; halo-prone → untrusted
    s = str(src).lower().strip()
    # CT FN-补切 family: all non-interactive batch SAM cut (pre-browser-tool
    # 2026-04-13). marker = clicked marker that triggers a SAM cut at the
    # marker location; review = sam_fn_review.py CLI batch tool. Both produce
    # boundary noise without per-instance human refine → untrusted.
    if s in ("sam_fn_marker", "sam_fn_review"):
        return "sam_added_true_fn"
    if s == "sam2":
        return "human_manual_sam_assisted"
    if s == "reviewed_prediction":
        return "reviewed_prediction"
    if s == "human_manual_sam_assisted":
        return "human_manual_sam_assisted"
    # Unknown provenance marker → fail fast. Conservative default to
    # human_manual_sam_assisted (trusted) was wrong: it silently marked
    # halo-prone batch-SAM outputs as trusted, defeating the mask_trusted
    # gate. Add new sources here explicitly with their correct
    # trusted/untrusted classification.
    raise ValueError(
        f"unknown CT 'source' value {s!r}; map it explicitly above with "
        f"a trusted/untrusted classification (see export_coco_dataset."
        f"_MASK_TRUSTED for the enum)"
    )


_CT_ENTRIES_CACHE = None


def _ct_entries():
    global _CT_ENTRIES_CACHE
    if _CT_ENTRIES_CACHE is None:
        _CT_ENTRIES_CACHE = discover_annotations(regions=["cape_town"])
    return _CT_ENTRIES_CACHE


def _load_ct_grid_annotations(grid_id: str, tile_crs) -> gpd.GeoDataFrame:
    """Load CT annotations + tag label_source per row."""
    entries = _ct_entries()
    if grid_id not in entries:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    gdf = load_annotation_gdf(entries[grid_id])
    if "label_source" in gdf.columns:
        pass  # already tagged
    elif "source" in gdf.columns:
        gdf["label_source"] = gdf["source"].apply(_ct_source_to_label_source)
    else:
        # batch001/002/002b SAM2-QGIS manual schema → all human_manual_sam_assisted
        gdf["label_source"] = "human_manual_sam_assisted"
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(tile_crs).reset_index(drop=True)


def _per_record_summary(records, kind="train"):
    rows = []
    for rec in records:
        n_trusted = (
            rec["annots"]["label_source"].isin(TRUSTED_SOURCES).sum()
            if "label_source" in rec["annots"].columns else 0
        )
        n_untrusted = (
            rec["annots"]["label_source"].isin(UNTRUSTED_SOURCES).sum()
            if "label_source" in rec["annots"].columns else 0
        )
        rows.append({
            "split": rec["split"],
            "region": rec["region"],
            "grid_id": rec["grid_id"],
            "n_polygons": len(rec["annots"]),
            "n_trusted": int(n_trusted),
            "n_untrusted": int(n_untrusted),
            "n_tiles": len(rec["tiles"]),
        })
    return rows


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
        return {"summary": str(summary_csv), "dry_run": True}

    # ── Chip scan + COCO build ───────────────────────────────────────
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

    return {"output_dir": str(out_dir), "manifest": str(manifest_path)}


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
