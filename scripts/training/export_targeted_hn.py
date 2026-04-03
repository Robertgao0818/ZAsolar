"""
Export targeted hard-negative chips from reviewed FP predictions.

Instead of random empty tiles, this extracts 400×400 chips centered on
false-positive detections (review_status == "delete") from batch 003.
These are locations where the model incorrectly predicted solar panels —
the hardest negatives available.

The output merges into an existing COCO dataset (coco_v3_no_hn) to create
a third experiment variant: coco_v3_targeted_hn.

Usage:
    python scripts/training/export_targeted_hn.py \
        --base-coco /mnt/d/ZAsolar/coco_v3_no_hn \
        --output-dir /mnt/d/ZAsolar/coco_v3_targeted_hn \
        --chip-size 400
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.grid_utils import TILES_ROOT

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# Batch 003 grids with reviewed predictions
BATCH_003_GRIDS = [
    "G1682", "G1683", "G1685", "G1686", "G1687", "G1688",
    "G1689", "G1690", "G1691", "G1692", "G1693",
    "G1743", "G1744", "G1747", "G1749", "G1750",
    "G1798", "G1800", "G1806", "G1807",
]


def load_fp_locations(grid_ids: list[str]) -> dict[str, gpd.GeoDataFrame]:
    """Load FP polygons (review_status == 'delete') per grid.

    Excludes FPs that overlap with ground-truth annotations — those are
    segmentation errors (wrong boundary), not true false positives.
    Including them as negatives would teach the model to suppress real panels.

    Returns dict: grid_id -> GeoDataFrame of pure FP polygons in EPSG:4326.
    """
    import glob

    fp_by_grid: dict[str, gpd.GeoDataFrame] = {}

    for gid in grid_ids:
        reviewed_path = RESULTS_DIR / gid / "review" / f"{gid}_reviewed.gpkg"
        if not reviewed_path.exists():
            continue

        gdf = gpd.read_file(reviewed_path)

        # Only original model predictions (exclude SAM FN additions)
        if "source" in gdf.columns:
            gdf = gdf[gdf["source"] != "sam_fn_marker"]

        fp = gdf[gdf["review_status"] == "delete"].copy()
        if len(fp) == 0:
            continue

        # Ensure EPSG:4326 — handle missing CRS with UTM coordinate detection
        if fp.crs is None:
            sample_x = fp.iloc[0].geometry.bounds[0]
            if sample_x > 1000:  # UTM-like coordinates
                fp = fp.set_crs(epsg=32734)
                print(f"  {gid}: assigned EPSG:32734 (detected UTM coords, CRS was None)")
        if fp.crs and fp.crs.to_epsg() != 4326:
            fp = fp.to_crs(epsg=4326)

        # Load ground-truth annotations for this grid
        ann_files = glob.glob(
            str(RESULTS_DIR.parent / f"data/annotations/cleaned/{gid}_SAM2_*.gpkg")
        )
        if ann_files:
            gt = gpd.read_file(ann_files[0])
            if gt.crs and gt.crs.to_epsg() != 4326:
                gt = gt.to_crs(epsg=4326)

            # Build spatial index for GT annotations
            gt_sindex = gt.sindex

            # Filter: keep only FPs with NO overlap with any GT annotation
            pure_fp_mask = []
            for _, fp_row in fp.iterrows():
                fp_geom = fp_row.geometry
                # Query spatial index for candidate overlaps
                candidates = list(gt_sindex.intersection(fp_geom.bounds))
                has_overlap = False
                for cidx in candidates:
                    gt_geom = gt.iloc[cidx].geometry
                    if fp_geom.intersects(gt_geom):
                        # Check meaningful overlap (IoU > 0.05 or intersection > 10% of FP)
                        inter_area = fp_geom.intersection(gt_geom).area
                        if inter_area > fp_geom.area * 0.05:
                            has_overlap = True
                            break
                pure_fp_mask.append(not has_overlap)

            n_before = len(fp)
            fp = fp[pure_fp_mask].copy()
            n_removed = n_before - len(fp)
            if n_removed > 0:
                print(f"  {gid}: filtered {n_removed} FPs overlapping GT "
                      f"({n_before} -> {len(fp)})")

        if len(fp) > 0:
            fp_by_grid[gid] = fp

    return fp_by_grid


def find_tile_for_point(lon: float, lat: float, grid_id: str,
                        tiles_root: Path) -> Path | None:
    """Find the tile GeoTIFF that contains a given lon/lat point."""
    grid_dir = tiles_root / grid_id
    if not grid_dir.exists():
        return None

    for tif in grid_dir.glob(f"{grid_id}_*_*_geo.tif"):
        with rasterio.open(tif) as src:
            left, bottom, right, top = src.bounds
            if left <= lon <= right and bottom <= lat <= top:
                return tif
    return None


def extract_fp_chips(
    fp_by_grid: dict[str, gpd.GeoDataFrame],
    output_dir: Path,
    chip_size: int = 400,
    tiles_root: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Extract chips centered on FP centroids.

    Returns (images_list, provenance_list) for COCO integration.
    """
    if tiles_root is None:
        tiles_root = TILES_ROOT

    chip_dir = output_dir / "train"
    chip_dir.mkdir(parents=True, exist_ok=True)

    images = []
    provenance = []
    img_id = 900000  # High offset to avoid ID collision with base dataset

    # Cache tile handles per grid to avoid repeated opens
    for grid_id, fp_gdf in sorted(fp_by_grid.items()):
        tile_cache: dict[str, rasterio.DatasetReader] = {}
        tile_handles: list[rasterio.DatasetReader] = []

        try:
            for _, fp_row in fp_gdf.iterrows():
                centroid = fp_row.geometry.centroid
                lon, lat = centroid.x, centroid.y

                tile_path = find_tile_for_point(lon, lat, grid_id, tiles_root)
                if tile_path is None:
                    continue

                tile_key = tile_path.stem
                if tile_key not in tile_cache:
                    handle = rasterio.open(tile_path)
                    tile_cache[tile_key] = handle
                    tile_handles.append(handle)

                src = tile_cache[tile_key]

                # Convert FP centroid to pixel coordinates
                py, px = src.index(lon, lat)

                # Center chip on FP centroid
                x0 = max(0, int(px - chip_size // 2))
                y0 = max(0, int(py - chip_size // 2))

                # Clamp to tile bounds
                x0 = min(x0, max(0, src.width - chip_size))
                y0 = min(y0, max(0, src.height - chip_size))

                w = min(chip_size, src.width - x0)
                h = min(chip_size, src.height - y0)

                if w < chip_size * 0.5 or h < chip_size * 0.5:
                    continue  # Skip tiny edge chips

                window = Window(x0, y0, w, h)
                data = src.read(window=window)

                # Pad if needed
                if w < chip_size or h < chip_size:
                    padded = np.zeros(
                        (data.shape[0], chip_size, chip_size), dtype=data.dtype
                    )
                    padded[:, :h, :w] = data
                    data = padded

                # Skip blank chips
                if np.all(data >= 245):
                    continue

                chip_name = f"fp_{grid_id}_{tile_key}__{x0}_{y0}.tif"
                chip_path = chip_dir / chip_name

                profile = src.profile.copy()
                for key in ("photometric", "compress", "jpeg_quality",
                            "jpegtablesmode"):
                    profile.pop(key, None)
                profile.update(
                    driver="GTiff",
                    width=chip_size,
                    height=chip_size,
                    transform=src.window_transform(window),
                    compress="lzw",
                )
                with rasterio.open(str(chip_path), "w", **profile) as dst:
                    dst.write(data)

                images.append({
                    "id": img_id,
                    "file_name": f"train/{chip_name}",
                    "width": chip_size,
                    "height": chip_size,
                    "positive": False,
                    "fp_source": grid_id,
                })
                provenance.append({
                    "image_id": img_id,
                    "chip_file": chip_name,
                    "source_tile": tile_key,
                    "x0": x0,
                    "y0": y0,
                    "width": w,
                    "height": h,
                    "n_annotations": 0,
                    "split": "train",
                    "source_type": "targeted_fp",
                })
                img_id += 1

        finally:
            for h in tile_handles:
                h.close()

        n_chips = sum(1 for p in provenance if p["source_tile"].startswith(grid_id))
        print(f"  {grid_id}: {len(fp_gdf)} FPs -> {n_chips} chips")

    return images, provenance


def merge_with_base_coco(
    base_dir: Path,
    fp_images: list[dict],
    fp_provenance: list[dict],
    output_dir: Path,
) -> None:
    """Merge targeted HN chips into base COCO (no-HN) dataset."""
    # Load base JSONs
    with open(base_dir / "train.json") as f:
        base_train = json.load(f)
    with open(base_dir / "val.json") as f:
        base_val = json.load(f)

    # Hard-link base image files to output dir
    for split in ("train", "val"):
        src_split = base_dir / split
        dst_split = output_dir / split
        dst_split.mkdir(parents=True, exist_ok=True)

        if src_split.exists():
            for img_file in src_split.iterdir():
                dst_file = dst_split / img_file.name
                if not dst_file.exists():
                    try:
                        dst_file.hardlink_to(img_file)
                    except OSError:
                        import shutil
                        shutil.copy2(img_file, dst_file)

    # Merge FP images into train split
    merged_train_images = base_train["images"] + fp_images
    # No new annotations (FP chips are negatives)
    merged_train_annots = base_train["annotations"]

    # Write merged train JSON
    merged_train = {
        "info": {
            **base_train["info"],
            "description": base_train["info"]["description"] + " + targeted FP hard negatives",
        },
        "licenses": base_train.get("licenses", []),
        "categories": base_train["categories"],
        "images": merged_train_images,
        "annotations": merged_train_annots,
    }

    with open(output_dir / "train.json", "w") as f:
        json.dump(merged_train, f)

    # Val stays the same
    with open(output_dir / "val.json", "w") as f:
        json.dump(base_val, f)

    # Write provenance
    if fp_provenance:
        import csv
        prov_path = output_dir / "targeted_hn_provenance.csv"
        with open(prov_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fp_provenance[0].keys())
            writer.writeheader()
            writer.writerows(fp_provenance)

    n_base_pos = sum(1 for img in base_train["images"] if img.get("positive", True))
    n_base_neg = len(base_train["images"]) - n_base_pos
    n_fp = len(fp_images)

    print(f"\n=== Merged Dataset Summary ===")
    print(f"  Base (no-HN) train: {len(base_train['images'])} images "
          f"({n_base_pos} positive, {n_base_neg} existing negative)")
    print(f"  + Targeted FP chips: {n_fp}")
    print(f"  = Total train: {len(merged_train_images)} images")
    print(f"  Annotations: {len(merged_train_annots)} (unchanged)")
    print(f"  Val: {len(base_val['images'])} images (unchanged)")


def main():
    parser = argparse.ArgumentParser(
        description="Export targeted hard-negative chips from reviewed FP predictions"
    )
    parser.add_argument(
        "--base-coco", type=Path, default=Path("/mnt/d/ZAsolar/coco_v3_no_hn"),
        help="Base COCO dataset (no hard negatives) to merge into",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/mnt/d/ZAsolar/coco_v3_targeted_hn"),
        help="Output directory for merged dataset",
    )
    parser.add_argument("--chip-size", type=int, default=400)
    parser.add_argument(
        "--tiles-root", type=Path, default=None,
        help="Override tiles root (default: SOLAR_TILES_ROOT or ./tiles)",
    )
    parser.add_argument(
        "--grid-ids", nargs="+", default=BATCH_003_GRIDS,
        help="Grid IDs to extract FPs from",
    )
    args = parser.parse_args()

    print("[1/3] Loading reviewed FP predictions...")
    fp_by_grid = load_fp_locations(args.grid_ids)
    total_fp = sum(len(gdf) for gdf in fp_by_grid.values())
    print(f"  Found {total_fp} FPs across {len(fp_by_grid)} grids")

    if total_fp == 0:
        print("No FP predictions found. Nothing to do.")
        return

    print(f"\n[2/3] Extracting {args.chip_size}×{args.chip_size} chips at FP locations...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp_images, fp_provenance = extract_fp_chips(
        fp_by_grid, args.output_dir,
        chip_size=args.chip_size,
        tiles_root=args.tiles_root,
    )
    print(f"  Extracted {len(fp_images)} targeted HN chips")

    print(f"\n[3/3] Merging with base dataset ({args.base_coco})...")
    merge_with_base_coco(args.base_coco, fp_images, fp_provenance, args.output_dir)

    print(f"\nOutput: {args.output_dir}")


if __name__ == "__main__":
    main()
