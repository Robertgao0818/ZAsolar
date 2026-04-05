"""
Build PV vs non-PV classification dataset from reviewed predictions.

Extracts 224x224 chips centered on detection centroids from tiles, using
human review decisions (correct → PV, delete → non-PV) as labels.
Uses whole-grid holdout split to prevent geographic leakage.

Usage:
    python scripts/classifier/build_cls_dataset.py \
        --output-dir data/cls_pv_thermal \
        --area-cutoff 30 \
        --val-fraction 0.2

    # Include taxonomy chips as auxiliary data
    python scripts/classifier/build_cls_dataset.py \
        --output-dir data/cls_pv_thermal \
        --taxonomy-csv results/analysis/small_fp/taxonomy_run/small_fp_taxonomy_labeled.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.grid_utils import TILES_ROOT

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

CHIP_SIZE = 400       # extraction chip size (pixels)
IMG_SIZE = 224        # output image size (pixels)
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # for reference only; applied at training time


def discover_reviewed_grids() -> list[str]:
    """Find all grids that have review decision files."""
    grids = []
    for review_csv in sorted(RESULTS_DIR.glob("*/review/detection_review_decisions.csv")):
        grids.append(review_csv.parent.parent.name)
    return grids


def load_reviewed_predictions(
    grid_ids: list[str],
    area_cutoff: float = 30.0,
) -> pd.DataFrame:
    """Load reviewed predictions across grids, filter by area, assign labels.

    Returns DataFrame with columns:
        grid_id, pred_id, label (pv/non_pv), area_m2, confidence,
        source_tile, centroid_lon, centroid_lat
    """
    records = []

    for grid_id in grid_ids:
        review_path = RESULTS_DIR / grid_id / "review" / "detection_review_decisions.csv"
        pred_path = RESULTS_DIR / grid_id / "predictions_metric.gpkg"

        if not review_path.exists() or not pred_path.exists():
            print(f"  WARN: missing files for {grid_id}, skipping")
            continue

        # Load review decisions
        with open(review_path) as f:
            decisions = {int(r["pred_id"]): r["status"] for r in csv.DictReader(f)}

        # Load predictions (EPSG:32734) and reproject to EPSG:4326 for tile lookup
        preds = gpd.read_file(pred_path)
        if preds.crs and preds.crs.to_epsg() != 4326:
            preds_4326 = preds.to_crs(epsg=4326)
        else:
            preds_4326 = preds

        for pred_id, status in decisions.items():
            # correct = PV, edit = PV (real panel, polygon needs fix), delete = non-PV
            if status not in ("correct", "edit", "delete"):
                continue
            if pred_id >= len(preds):
                continue

            row = preds.iloc[pred_id]
            row_4326 = preds_4326.iloc[pred_id]

            area_m2 = row.get("area_m2", 0)
            if area_m2 >= area_cutoff:
                continue

            centroid = row_4326.geometry.centroid
            source_tile = row.get("source_tile", "")

            records.append({
                "grid_id": grid_id,
                "pred_id": pred_id,
                "label": "pv" if status in ("correct", "edit") else "non_pv",
                "area_m2": area_m2,
                "confidence": row.get("confidence", 0),
                "source_tile": source_tile,
                "centroid_lon": centroid.x,
                "centroid_lat": centroid.y,
            })

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df)} reviewed predictions from {df['grid_id'].nunique()} grids")
    print(f"  PV: {(df['label'] == 'pv').sum()}, non-PV: {(df['label'] == 'non_pv').sum()}")
    return df


def load_taxonomy_chips(
    taxonomy_csv: Path,
    area_cutoff: float = 30.0,
) -> pd.DataFrame:
    """Load taxonomy-labeled chips, mapping correct_detection → PV, rest → non-PV.

    Returns DataFrame with same columns as load_reviewed_predictions,
    plus chip_path for pre-existing chips.
    """
    df = pd.read_csv(taxonomy_csv)

    # Map labels
    label_map = {"correct_detection": "pv"}
    # Everything else is non-PV
    df["label"] = df["human_label"].map(
        lambda x: label_map.get(x, "non_pv")
    )

    # Filter by area
    df = df[df["area_m2"] < area_cutoff].copy()

    result = df[["grid_id", "pred_id", "label", "area_m2", "confidence", "chip_path"]].copy()
    result["source"] = "taxonomy"

    print(f"  Taxonomy: {len(result)} chips "
          f"(PV: {(result['label'] == 'pv').sum()}, "
          f"non-PV: {(result['label'] == 'non_pv').sum()})")
    return result


def whole_grid_split(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by whole grid — all crops from one grid go to train or val, never both."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["grid_id"]))

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    train_grids = set(train_df["grid_id"].unique())
    val_grids = set(val_df["grid_id"].unique())
    overlap = train_grids & val_grids
    assert len(overlap) == 0, f"Grid leakage detected: {overlap}"

    print(f"  Train: {len(train_df)} chips from {len(train_grids)} grids: "
          f"{sorted(train_grids)}")
    print(f"  Val: {len(val_df)} chips from {len(val_grids)} grids: "
          f"{sorted(val_grids)}")
    return train_df, val_df


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


def extract_chip(
    lon: float, lat: float, grid_id: str,
    tiles_root: Path, tile_cache: dict,
    chip_size: int = CHIP_SIZE,
) -> np.ndarray | None:
    """Extract a chip centered on (lon, lat). Returns HWC uint8 array or None."""
    tile_path = _find_tile(lon, lat, grid_id, tiles_root)
    if tile_path is None:
        return None

    tile_key = str(tile_path)
    if tile_key not in tile_cache:
        tile_cache[tile_key] = rasterio.open(tile_path)

    src = tile_cache[tile_key]
    py, px = src.index(lon, lat)

    x0 = max(0, int(px - chip_size // 2))
    y0 = max(0, int(py - chip_size // 2))
    x0 = min(x0, max(0, src.width - chip_size))
    y0 = min(y0, max(0, src.height - chip_size))

    w = min(chip_size, src.width - x0)
    h = min(chip_size, src.height - y0)

    if w < chip_size * 0.5 or h < chip_size * 0.5:
        return None

    window = Window(x0, y0, w, h)
    data = src.read(window=window)  # (C, H, W)

    if w < chip_size or h < chip_size:
        padded = np.zeros((data.shape[0], chip_size, chip_size), dtype=data.dtype)
        padded[:, :h, :w] = data
        data = padded

    # Skip blank/overexposed chips
    if np.all(data >= 245):
        return None

    # CHW -> HWC, keep only first 3 bands (RGB)
    img = data[:3].transpose(1, 2, 0)
    return img


def extract_and_save_chips(
    df: pd.DataFrame,
    output_dir: Path,
    split: str,
    tiles_root: Path,
    img_size: int = IMG_SIZE,
) -> int:
    """Extract chips for a dataframe split, save as PNG. Returns count of saved chips."""
    saved = 0
    skipped = 0
    tile_cache: dict[str, rasterio.DatasetReader] = {}

    for label in ("pv", "non_pv"):
        label_dir = output_dir / split / label
        label_dir.mkdir(parents=True, exist_ok=True)

    try:
        for idx, row in df.iterrows():
            grid_id = row["grid_id"]
            pred_id = row["pred_id"]
            label = row["label"]

            chip = extract_chip(
                row["centroid_lon"], row["centroid_lat"],
                grid_id, tiles_root, tile_cache,
            )
            if chip is None:
                skipped += 1
                continue

            # Resize to target size
            resized = cv2.resize(chip, (img_size, img_size), interpolation=cv2.INTER_AREA)

            # Save as PNG
            fname = f"{grid_id}_pred{pred_id}.png"
            out_path = output_dir / split / label / fname
            cv2.imwrite(str(out_path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            saved += 1

    finally:
        for handle in tile_cache.values():
            handle.close()

    if skipped > 0:
        print(f"  {split}: saved {saved}, skipped {skipped} (no tile / blank)")
    else:
        print(f"  {split}: saved {saved}")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Build PV vs non-PV classification dataset"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "cls_pv_thermal",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--area-cutoff", type=float, default=30.0,
        help="Max area_m2 for classifier training (default: 30)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="Fraction of grids held out for validation (default: 0.2)",
    )
    parser.add_argument(
        "--taxonomy-csv", type=Path, default=None,
        help="Optional: taxonomy_labeled.csv for auxiliary labeled data",
    )
    parser.add_argument(
        "--tiles-root", type=Path, default=None,
        help="Override tile root directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--img-size", type=int, default=IMG_SIZE,
        help="Output image size (default: 224)",
    )
    args = parser.parse_args()

    tiles_root = args.tiles_root or TILES_ROOT

    # --- Step 1: Load reviewed predictions ---
    print("[1/4] Discovering reviewed grids...")
    grid_ids = discover_reviewed_grids()
    print(f"  Found {len(grid_ids)} reviewed grids")

    print("\n[2/4] Loading reviewed predictions (area < {:.0f} m²)...".format(args.area_cutoff))
    reviewed_df = load_reviewed_predictions(grid_ids, area_cutoff=args.area_cutoff)

    # --- Step 2: Whole-grid split ---
    print(f"\n[3/4] Splitting by whole grid (val_fraction={args.val_fraction})...")
    train_df, val_df = whole_grid_split(reviewed_df, args.val_fraction, args.seed)

    # Record grid assignments
    train_grids = sorted(train_df["grid_id"].unique().tolist())
    val_grids = sorted(val_df["grid_id"].unique().tolist())

    # --- Step 3: Add taxonomy chips (train-only, from train grids or ungrouped) ---
    taxonomy_count = 0
    if args.taxonomy_csv and args.taxonomy_csv.exists():
        print(f"\n  Loading taxonomy chips...")
        tax_df = load_taxonomy_chips(args.taxonomy_csv, args.area_cutoff)

        # Taxonomy chips go to train only, never val
        # We need to extract their centroids from predictions_metric.gpkg
        tax_records = []
        for _, row in tax_df.iterrows():
            grid_id = row["grid_id"]
            pred_id = int(row["pred_id"])
            pred_path = RESULTS_DIR / grid_id / "predictions_metric.gpkg"
            if not pred_path.exists():
                continue
            preds = gpd.read_file(pred_path)
            if pred_id >= len(preds):
                continue
            if preds.crs and preds.crs.to_epsg() != 4326:
                preds = preds.to_crs(epsg=4326)
            centroid = preds.iloc[pred_id].geometry.centroid
            tax_records.append({
                "grid_id": grid_id,
                "pred_id": pred_id,
                "label": row["label"],
                "area_m2": row["area_m2"],
                "confidence": row["confidence"],
                "source_tile": preds.iloc[pred_id].get("source_tile", ""),
                "centroid_lon": centroid.x,
                "centroid_lat": centroid.y,
            })

        if tax_records:
            tax_enriched = pd.DataFrame(tax_records)
            # Only keep taxonomy chips from train grids (prevent val grid leakage)
            train_grid_set = set(train_grids)
            tax_enriched = tax_enriched[
                tax_enriched["grid_id"].isin(train_grid_set)
            ]
            # Remove duplicates already in reviewed set
            existing_keys = set(
                zip(train_df["grid_id"], train_df["pred_id"])
            )
            tax_enriched = tax_enriched[
                ~tax_enriched.apply(
                    lambda r: (r["grid_id"], r["pred_id"]) in existing_keys, axis=1
                )
            ]
            taxonomy_count = len(tax_enriched)
            if taxonomy_count > 0:
                train_df = pd.concat([train_df, tax_enriched], ignore_index=True)
                print(f"  Added {taxonomy_count} taxonomy chips to train set "
                      f"(restricted to train grids)")

    # --- Step 4: Extract and save chips ---
    print(f"\n[4/4] Extracting chips (tiles_root={tiles_root})...")
    # Clean output directories for idempotency (avoid stale PNGs from prior runs)
    for split in ("train", "val"):
        for label in ("pv", "non_pv"):
            label_dir = args.output_dir / split / label
            if label_dir.exists():
                shutil.rmtree(label_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_saved = extract_and_save_chips(
        train_df, args.output_dir, "train", tiles_root, args.img_size
    )
    val_saved = extract_and_save_chips(
        val_df, args.output_dir, "val", tiles_root, args.img_size
    )

    # --- Write metadata ---
    train_pv = len(list((args.output_dir / "train" / "pv").glob("*.png")))
    train_nonpv = len(list((args.output_dir / "train" / "non_pv").glob("*.png")))
    val_pv = len(list((args.output_dir / "val" / "pv").glob("*.png")))
    val_nonpv = len(list((args.output_dir / "val" / "non_pv").glob("*.png")))

    meta = {
        "description": "PV vs non-PV binary classification dataset",
        "area_cutoff_m2": args.area_cutoff,
        "img_size": args.img_size,
        "chip_extraction_size": CHIP_SIZE,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_grids": train_grids,
        "val_grids": val_grids,
        "taxonomy_chips_added": taxonomy_count,
        "counts": {
            "train_pv": train_pv,
            "train_non_pv": train_nonpv,
            "val_pv": val_pv,
            "val_non_pv": val_nonpv,
            "total": train_pv + train_nonpv + val_pv + val_nonpv,
        },
        "label_mapping": {
            "pv": "Solar PV panel (correct detection)",
            "non_pv": "Non-PV (thermal water heater, shadow, skylight, etc.)",
        },
    }
    meta_path = args.output_dir / "dataset_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n=== Dataset Summary ===")
    print(f"  Train: {train_pv} PV + {train_nonpv} non-PV = {train_pv + train_nonpv}")
    print(f"  Val:   {val_pv} PV + {val_nonpv} non-PV = {val_pv + val_nonpv}")
    print(f"  Total: {train_pv + train_nonpv + val_pv + val_nonpv}")
    print(f"  Train grids ({len(train_grids)}): {train_grids}")
    print(f"  Val grids ({len(val_grids)}): {val_grids}")
    print(f"  Metadata: {meta_path}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
