"""
Export V4 hard-negative chips from curated small-FP shortlist.

Reads hn_small_fp_shortlist.csv (safe_true_fp from batch 004 analysis),
samples 50% stratified by grid, extracts 400×400 chips from prediction
locations, and merges into a base COCO dataset.

Usage:
    python scripts/training/export_v4_hn.py \
        --base-coco /workspace/coco_v4_no_hn \
        --output-dir /workspace/coco_v4_hn \
        --shortlist results/analysis/small_fp/taxonomy_run/hn_small_fp_shortlist.csv \
        --sample-rate 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.grid_utils import TILES_ROOT

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Predictions that were manually corrected back to "correct" during taxonomy
# labeling — exclude from HN pool
EXCLUDE_CORRECTIONS = {
    ("G1975", 58),
    ("G1919", 41),
    ("G1971", 217),
}


def load_shortlist(shortlist_path: Path) -> pd.DataFrame:
    """Load HN shortlist CSV and filter out corrected predictions."""
    df = pd.read_csv(shortlist_path)
    # Exclude manually corrected items
    mask = ~df.apply(
        lambda r: (r["grid_id"], r["pred_id"]) in EXCLUDE_CORRECTIONS, axis=1
    )
    n_excluded = (~mask).sum()
    if n_excluded > 0:
        print(f"  Excluded {n_excluded} corrected predictions")
    return df[mask].reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, sample_rate: float,
                      seed: int = 42) -> pd.DataFrame:
    """Sample `sample_rate` fraction per grid, ensuring geographic diversity."""
    rng = random.Random(seed)
    sampled = []
    for grid_id, group in df.groupby("grid_id"):
        n = max(1, int(len(group) * sample_rate))
        indices = list(group.index)
        rng.shuffle(indices)
        sampled.extend(indices[:n])

    result = df.loc[sampled].reset_index(drop=True)
    print(f"  Sampled {len(result)} / {len(df)} "
          f"({len(result)/len(df)*100:.0f}%) across "
          f"{result['grid_id'].nunique()} grids")
    return result


def load_fp_geometries(
    sampled: pd.DataFrame,
) -> dict[str, gpd.GeoDataFrame]:
    """Load FP polygon geometries from predictions_metric.gpkg for sampled FPs.

    Returns dict: grid_id -> GeoDataFrame with geometry in EPSG:4326.
    """
    fp_by_grid: dict[str, gpd.GeoDataFrame] = {}

    for grid_id, group in sampled.groupby("grid_id"):
        pred_path = RESULTS_DIR / grid_id / "predictions_metric.gpkg"
        if not pred_path.exists():
            print(f"  WARN: {pred_path} not found, skipping {grid_id}")
            continue

        preds = gpd.read_file(pred_path)

        # predictions_metric.gpkg uses row index as pred_id (0-based)
        pred_ids = set(group["pred_id"].tolist())
        fp = preds.iloc[list(pred_ids)].copy()

        # Ensure EPSG:4326
        if fp.crs and fp.crs.to_epsg() != 4326:
            fp = fp.to_crs(epsg=4326)

        fp_by_grid[grid_id] = fp
        print(f"  {grid_id}: loaded {len(fp)} FP geometries")

    return fp_by_grid


def extract_hn_chips(
    fp_by_grid: dict[str, gpd.GeoDataFrame],
    output_dir: Path,
    chip_size: int = 400,
    tiles_root: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Extract chips centered on FP centroids. Returns (images, provenance)."""
    if tiles_root is None:
        tiles_root = TILES_ROOT

    chip_dir = output_dir / "train"
    chip_dir.mkdir(parents=True, exist_ok=True)

    images = []
    provenance = []
    img_id = 900000  # High offset to avoid ID collision

    for grid_id, fp_gdf in sorted(fp_by_grid.items()):
        grid_chips = 0
        tile_cache: dict[str, rasterio.DatasetReader] = {}
        tile_handles: list[rasterio.DatasetReader] = []

        try:
            for idx, fp_row in fp_gdf.iterrows():
                centroid = fp_row.geometry.centroid
                lon, lat = centroid.x, centroid.y

                # Find tile containing this point
                tile_path = _find_tile(lon, lat, grid_id, tiles_root)
                if tile_path is None:
                    continue

                tile_key = tile_path.stem
                if tile_key not in tile_cache:
                    handle = rasterio.open(tile_path)
                    tile_cache[tile_key] = handle
                    tile_handles.append(handle)

                src = tile_cache[tile_key]
                py, px = src.index(lon, lat)

                # Center chip on FP centroid
                x0 = max(0, int(px - chip_size // 2))
                y0 = max(0, int(py - chip_size // 2))
                x0 = min(x0, max(0, src.width - chip_size))
                y0 = min(y0, max(0, src.height - chip_size))

                w = min(chip_size, src.width - x0)
                h = min(chip_size, src.height - y0)

                if w < chip_size * 0.5 or h < chip_size * 0.5:
                    continue

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

                chip_name = f"hn_v4_{grid_id}_{tile_key}__{x0}_{y0}.tif"
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
                    "hn_source": grid_id,
                })
                provenance.append({
                    "image_id": img_id,
                    "chip_file": chip_name,
                    "source_tile": tile_key,
                    "x0": x0, "y0": y0,
                    "width": w, "height": h,
                    "n_annotations": 0,
                    "split": "train",
                    "source_type": "v4_small_fp_hn",
                })
                img_id += 1
                grid_chips += 1

        finally:
            for h in tile_handles:
                h.close()

        print(f"  {grid_id}: {len(fp_gdf)} FPs -> {grid_chips} chips")

    return images, provenance


def _find_tile(lon: float, lat: float, grid_id: str,
               tiles_root: Path) -> Path | None:
    """Find tile GeoTIFF containing a lon/lat point."""
    grid_dir = tiles_root / grid_id
    if not grid_dir.exists():
        return None
    for tif in grid_dir.glob(f"{grid_id}_*_*_geo.tif"):
        with rasterio.open(tif) as src:
            left, bottom, right, top = src.bounds
            if left <= lon <= right and bottom <= lat <= top:
                return tif
    return None


def merge_with_base(
    base_dir: Path,
    hn_images: list[dict],
    hn_provenance: list[dict],
    output_dir: Path,
) -> None:
    """Merge V4 HN chips into base COCO dataset."""
    import shutil

    with open(base_dir / "train.json") as f:
        base_train = json.load(f)
    with open(base_dir / "val.json") as f:
        base_val = json.load(f)

    # Hard-link base images to output
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
                        shutil.copy2(img_file, dst_file)

    # Merge
    merged_images = base_train["images"] + hn_images
    merged_annots = base_train["annotations"]

    merged = {
        "info": {
            **base_train["info"],
            "description": base_train["info"]["description"]
            + " + V4 small-FP hard negatives",
        },
        "licenses": base_train.get("licenses", []),
        "categories": base_train["categories"],
        "images": merged_images,
        "annotations": merged_annots,
    }

    with open(output_dir / "train.json", "w") as f:
        json.dump(merged, f)
    with open(output_dir / "val.json", "w") as f:
        json.dump(base_val, f)

    if hn_provenance:
        prov_path = output_dir / "v4_hn_provenance.csv"
        with open(prov_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=hn_provenance[0].keys())
            writer.writeheader()
            writer.writerows(hn_provenance)

    # Summary
    n_base_pos = sum(1 for img in base_train["images"] if img.get("positive", True))
    n_base_neg = len(base_train["images"]) - n_base_pos
    n_hn = len(hn_images)
    total = len(merged_images)
    hn_pct = n_hn / total * 100 if total else 0

    print(f"\n=== V4 Merged Dataset ===")
    print(f"  Base train: {len(base_train['images'])} images "
          f"({n_base_pos} positive, {n_base_neg} negative)")
    print(f"  + V4 HN chips: {n_hn}")
    print(f"  = Total train: {total} images (HN = {hn_pct:.1f}%)")
    print(f"  Annotations: {len(merged_annots)} (unchanged)")
    print(f"  Val: {len(base_val['images'])} images (unchanged)")

    if hn_pct > 15:
        print(f"  ⚠ WARNING: HN ratio {hn_pct:.1f}% exceeds recommended 15% cap")


def main():
    parser = argparse.ArgumentParser(
        description="Export V4 hard-negative chips from curated small-FP shortlist"
    )
    parser.add_argument(
        "--base-coco", type=Path, required=True,
        help="Base COCO dataset (no HN) to merge into",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for merged V4 dataset",
    )
    parser.add_argument(
        "--shortlist", type=Path,
        default=PROJECT_ROOT / "results/analysis/small_fp/taxonomy_run/hn_small_fp_shortlist.csv",
        help="Path to hn_small_fp_shortlist.csv",
    )
    parser.add_argument("--sample-rate", type=float, default=0.5,
                        help="Fraction of shortlist to sample (default: 0.5)")
    parser.add_argument("--chip-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiles-root", type=Path, default=None)
    args = parser.parse_args()

    print("[1/4] Loading HN shortlist...")
    shortlist = load_shortlist(args.shortlist)
    print(f"  {len(shortlist)} candidates from {shortlist['grid_id'].nunique()} grids")

    print(f"\n[2/4] Stratified sampling ({args.sample_rate*100:.0f}%)...")
    sampled = stratified_sample(shortlist, args.sample_rate, seed=args.seed)

    print(f"\n[3/4] Loading FP geometries and extracting chips...")
    fp_by_grid = load_fp_geometries(sampled)
    total_fp = sum(len(gdf) for gdf in fp_by_grid.values())
    print(f"  {total_fp} FP geometries loaded")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hn_images, hn_provenance = extract_hn_chips(
        fp_by_grid, args.output_dir,
        chip_size=args.chip_size,
        tiles_root=args.tiles_root,
    )
    print(f"  Extracted {len(hn_images)} HN chips")

    print(f"\n[4/4] Merging with base dataset...")
    merge_with_base(args.base_coco, hn_images, hn_provenance, args.output_dir)

    print(f"\nOutput: {args.output_dir}")


if __name__ == "__main__":
    main()
