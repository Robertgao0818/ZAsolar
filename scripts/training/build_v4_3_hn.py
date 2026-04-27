"""
Build V4.3 HN COCO dataset by extending v4_2_jhb_ft with hard-negative chips.

HN sources (see configs/datasets/training_sets.yaml v4_3_hn):
  1. JHB CBD train-20 V4 FPs (244 chips, region=jhb, imagery=aerial_2023)
  2. JHB Sandton train-20 V4 FPs (444 chips, same region/imagery)
  3. CT residual V4.1 small-FP shortlist (455 chips, region=cape_town)

Audit: any HN chip whose 400x400 window overlaps a reviewed TP polygon
(IoU >= 0.1 on the window itself) is dropped — prevents teaching the
detector to suppress real PV.

Usage:
    python scripts/training/build_v4_3_hn.py \
        --base-coco /home/gaosh/zasolar_data/coco/coco_v4_2_jhb_ft \
        --output-dir /home/gaosh/zasolar_data/coco/coco_v4_3_hn \
        --jhb-fp-pool results/analysis/v4_3_hn/jhb_fp_pool.gpkg \
        --ct-shortlist results/analysis/small_fp/taxonomy_run/hn_small_fp_shortlist.csv

    # Guardrail 10% variant:
    python scripts/training/build_v4_3_hn.py ... \
        --output-dir /home/gaosh/zasolar_data/coco/coco_v4_3_hn_10pct \
        --subsample-frac 0.71  # 0.71 * 1143 -> ~810 chips -> ~10% ratio
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.grid_utils import resolve_tiles_dir
from core import region_registry

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ID offsets per source — prevents collision with base chip IDs (< 900000)
ID_OFFSET_CBD = 910000
ID_OFFSET_SANDTON = 920000
ID_OFFSET_CT = 930000

JHB_REVIEWED_DIR = PROJECT_ROOT / "data/annotations/Joburg"


def _find_tile(lon: float, lat: float, grid_id: str,
               region: str, imagery_layer: str | None) -> Path | None:
    grid_dir = resolve_tiles_dir(grid_id, region=region, imagery_layer=imagery_layer)
    if grid_dir is None or not grid_dir.exists():
        return None
    for tif in grid_dir.glob(f"{grid_id}_*_*_geo.tif"):
        with rasterio.open(tif) as src:
            left, bottom, right, top = src.bounds
            if left <= lon <= right and bottom <= lat <= top:
                return tif
    return None


def load_jhb_reviewed_tp(grid_id: str) -> gpd.GeoDataFrame | None:
    paths = sorted(JHB_REVIEWED_DIR.glob(f"{grid_id}_V4_*.gpkg"))
    if not paths:
        return None
    rev = gpd.read_file(paths[0])
    return rev if len(rev) else None


def chip_overlaps_tp(window_bounds, tp_gdf_4326) -> bool:
    """Return True if the chip window (EPSG:4326 bounds) intersects any TP polygon."""
    if tp_gdf_4326 is None or len(tp_gdf_4326) == 0:
        return False
    wleft, wbottom, wright, wtop = window_bounds
    chip_poly = box(wleft, wbottom, wright, wtop)
    if tp_gdf_4326.crs is None:
        return False
    if tp_gdf_4326.crs.to_epsg() != 4326:
        tp_gdf_4326 = tp_gdf_4326.to_crs(epsg=4326)
    return bool(tp_gdf_4326.intersects(chip_poly).any())


def extract_chip(src: rasterio.DatasetReader, lon: float, lat: float,
                 chip_size: int) -> tuple[np.ndarray, Window, tuple] | None:
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
    data = src.read(window=window)
    if w < chip_size or h < chip_size:
        padded = np.zeros((data.shape[0], chip_size, chip_size), dtype=data.dtype)
        padded[:, :h, :w] = data
        data = padded
    if np.all(data >= 245):
        return None
    # Window bounds in source CRS (tile is EPSG:4326 for both CT + JHB aerial)
    win_bounds = rasterio.windows.bounds(window, src.transform)
    return data, window, win_bounds


def extract_hn_for_grid(grid_id: str, fp_points_4326: list[tuple[float, float]],
                        region: str, imagery_layer: str | None,
                        chip_dir: Path, prefix: str, id_start: int,
                        tp_filter_gdf: gpd.GeoDataFrame | None,
                        chip_size: int) -> tuple[list[dict], list[dict], int]:
    images, provenance = [], []
    tile_cache: dict[str, rasterio.DatasetReader] = {}
    tile_handles: list[rasterio.DatasetReader] = []
    seen_windows: set[tuple[str, int, int]] = set()  # dedup (tile_key, x0, y0)
    cur_id = id_start
    try:
        for (lon, lat) in fp_points_4326:
            tile_path = _find_tile(lon, lat, grid_id, region=region,
                                   imagery_layer=imagery_layer)
            if tile_path is None:
                continue
            tile_key = tile_path.stem
            if tile_key not in tile_cache:
                h = rasterio.open(tile_path)
                tile_cache[tile_key] = h
                tile_handles.append(h)
            src = tile_cache[tile_key]
            result = extract_chip(src, lon, lat, chip_size)
            if result is None:
                continue
            data, window, win_bounds = result
            # Audit: drop if chip overlaps any reviewed TP
            if chip_overlaps_tp(win_bounds, tp_filter_gdf):
                continue

            x0, y0 = int(window.col_off), int(window.row_off)
            if (tile_key, x0, y0) in seen_windows:
                continue  # same chip window already extracted for another FP
            seen_windows.add((tile_key, x0, y0))
            chip_name = f"{prefix}_{grid_id}_{tile_key}__{x0}_{y0}.tif"
            chip_path = chip_dir / chip_name

            profile = src.profile.copy()
            for key in ("photometric", "compress", "jpeg_quality", "jpegtablesmode"):
                profile.pop(key, None)
            profile.update(
                driver="GTiff", width=chip_size, height=chip_size,
                transform=src.window_transform(window), compress="lzw",
            )
            with rasterio.open(str(chip_path), "w", **profile) as dst:
                dst.write(data)
            images.append({
                "id": cur_id,
                "file_name": f"train/{chip_name}",
                "width": chip_size,
                "height": chip_size,
                "positive": False,
                "hn_source": prefix,
                "grid_id": grid_id,
                "region": region,
            })
            provenance.append({
                "image_id": cur_id,
                "chip_file": chip_name,
                "grid_id": grid_id,
                "region": region,
                "imagery_layer": imagery_layer or "",
                "source_tile": tile_key,
                "x0": x0, "y0": y0,
                "width": chip_size, "height": chip_size,
                "source_type": prefix,
                "split": "train",
            })
            cur_id += 1
    finally:
        for h in tile_handles:
            h.close()
    return images, provenance, cur_id


def build_jhb_hn(fp_pool_path: Path, chip_dir: Path, chip_size: int,
                 subsample_frac: float, seed: int
                 ) -> tuple[list[dict], list[dict], dict]:
    fp = gpd.read_file(fp_pool_path)
    # Convert to 4326 for tile lookup
    if fp.crs.to_epsg() != 4326:
        fp = fp.to_crs(epsg=4326)

    all_images, all_prov = [], []
    stats = {"CBD": {"n_fp": 0, "n_chip": 0}, "Sandton": {"n_fp": 0, "n_chip": 0}}
    rng = random.Random(seed)

    for group, id_offset, prefix in [
        ("CBD", ID_OFFSET_CBD, "v43_jhb_cbd_hn"),
        ("Sandton", ID_OFFSET_SANDTON, "v43_jhb_sandton_hn"),
    ]:
        sub = fp[fp["group"] == group].reset_index(drop=True)
        # Subsample (stratified by grid) if requested
        if subsample_frac < 1.0:
            kept_indices = []
            for gid, grp in sub.groupby("grid_id"):
                n = max(1, int(round(len(grp) * subsample_frac)))
                idxs = list(grp.index)
                rng.shuffle(idxs)
                kept_indices.extend(idxs[:n])
            sub = sub.loc[kept_indices].reset_index(drop=True)
        print(f"  {group}: {len(sub)} FPs after subsample {subsample_frac}")
        stats[group]["n_fp"] = len(sub)

        cur_id = id_offset
        for grid_id, grp in sub.groupby("grid_id"):
            pts = [(geom.centroid.x, geom.centroid.y) for geom in grp.geometry]
            tp_filter = load_jhb_reviewed_tp(grid_id)
            imgs, provs, cur_id = extract_hn_for_grid(
                grid_id=grid_id,
                fp_points_4326=pts,
                region="jhb",
                imagery_layer="aerial_2023",
                chip_dir=chip_dir,
                prefix=prefix,
                id_start=cur_id,
                tp_filter_gdf=tp_filter,
                chip_size=chip_size,
            )
            all_images.extend(imgs)
            all_prov.extend(provs)
            stats[group]["n_chip"] += len(imgs)
            print(f"    {grid_id}: {len(pts)} FPs -> {len(imgs)} chips")
    return all_images, all_prov, stats


def build_ct_hn(shortlist_path: Path, chip_dir: Path, chip_size: int,
                subsample_frac: float, seed: int,
                ct_model_run: str = "v3c_targeted_hn_aerial_2025"
                ) -> tuple[list[dict], list[dict], dict]:
    df = pd.read_csv(shortlist_path)
    # Exclude manual corrections (same set export_v4_hn.py uses)
    EXCLUDE = {("G1975", 58), ("G1919", 41), ("G1971", 217)}
    df = df[~df.apply(lambda r: (r["grid_id"], r["pred_id"]) in EXCLUDE, axis=1)].reset_index(drop=True)

    if subsample_frac < 1.0:
        rng = random.Random(seed)
        kept = []
        for gid, grp in df.groupby("grid_id"):
            n = max(1, int(round(len(grp) * subsample_frac)))
            idxs = list(grp.index)
            rng.shuffle(idxs)
            kept.extend(idxs[:n])
        df = df.loc[kept].reset_index(drop=True)
    print(f"  CT residual: {len(df)} FPs after subsample {subsample_frac}")

    # Load CT prediction geometries per grid to get centroids
    all_images, all_prov = [], []
    cur_id = ID_OFFSET_CT
    n_chip_total = 0
    ct_run_root = region_registry.get_model_run_path("cape_town", ct_model_run)
    for grid_id, grp in df.groupby("grid_id"):
        if ct_run_root is None:
            continue
        pred_path = ct_run_root / grid_id / "predictions_metric.gpkg"
        if not pred_path.exists():
            print(f"    {grid_id}: predictions not found at {pred_path}")
            continue
        preds = gpd.read_file(pred_path)
        if preds.crs and preds.crs.to_epsg() != 4326:
            preds = preds.to_crs(epsg=4326)
        pred_ids = [int(i) for i in grp["pred_id"].tolist()]
        valid_ids = [i for i in pred_ids if 0 <= i < len(preds)]
        fp_rows = preds.iloc[valid_ids]
        pts = [(g.centroid.x, g.centroid.y) for g in fp_rows.geometry]

        imgs, provs, cur_id = extract_hn_for_grid(
            grid_id=grid_id,
            fp_points_4326=pts,
            region="cape_town",
            imagery_layer=None,  # CT has a single canonical layer
            chip_dir=chip_dir,
            prefix="v43_ct_residual_hn",
            id_start=cur_id,
            tp_filter_gdf=None,  # CT HN was already audited in V4.1 pipeline
            chip_size=chip_size,
        )
        all_images.extend(imgs)
        all_prov.extend(provs)
        n_chip_total += len(imgs)
        print(f"    {grid_id}: {len(pts)} FPs -> {len(imgs)} chips")
    stats = {"CT": {"n_fp": len(df), "n_chip": n_chip_total}}
    return all_images, all_prov, stats


def merge_into_base(base_dir: Path, output_dir: Path,
                    hn_images: list[dict], hn_prov: list[dict]) -> dict:
    with open(base_dir / "train.json") as f:
        base_train = json.load(f)
    with open(base_dir / "val.json") as f:
        base_val = json.load(f)

    # Hard-link base chips into output dir
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

    merged_images = base_train["images"] + hn_images
    merged = {
        "info": {
            **base_train["info"],
            "description": base_train["info"].get("description", "") + " + V4.3 HN",
        },
        "licenses": base_train.get("licenses", []),
        "categories": base_train["categories"],
        "images": merged_images,
        "annotations": base_train["annotations"],
    }
    with open(output_dir / "train.json", "w") as f:
        json.dump(merged, f)
    with open(output_dir / "val.json", "w") as f:
        json.dump(base_val, f)

    if hn_prov:
        with open(output_dir / "v4_3_hn_provenance.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hn_prov[0].keys())
            w.writeheader()
            w.writerows(hn_prov)

    n_total = len(merged_images)
    n_hn = len(hn_images)
    hn_pct = n_hn / n_total * 100 if n_total else 0.0
    summary = {
        "n_base_train": len(base_train["images"]),
        "n_hn_added": n_hn,
        "n_total_train": n_total,
        "hn_ratio_pct": round(hn_pct, 2),
        "n_annotations": len(base_train["annotations"]),
        "n_val": len(base_val["images"]),
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-coco", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--jhb-fp-pool", type=Path,
                    default=PROJECT_ROOT / "results/analysis/v4_3_hn/jhb_fp_pool.gpkg")
    ap.add_argument("--ct-shortlist", type=Path,
                    default=PROJECT_ROOT / "results/analysis/small_fp/taxonomy_run/hn_small_fp_shortlist.csv")
    ap.add_argument("--ct-model-run", type=str, default="v3c_targeted_hn_aerial_2025",
                    help="CT model_run from regions.yaml whose predictions_metric.gpkg hosts HN pred_ids")
    ap.add_argument("--subsample-frac", type=float, default=1.0,
                    help="Stratified subsample fraction across all sources (for guardrail variant)")
    ap.add_argument("--chip-size", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chip_dir = args.output_dir / "train"
    chip_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Building JHB CBD + Sandton HN chips...")
    jhb_imgs, jhb_prov, jhb_stats = build_jhb_hn(
        args.jhb_fp_pool, chip_dir, args.chip_size, args.subsample_frac, args.seed
    )

    print("\n[2/3] Building CT residual HN chips...")
    ct_imgs, ct_prov, ct_stats = build_ct_hn(
        args.ct_shortlist, chip_dir, args.chip_size, args.subsample_frac, args.seed,
        ct_model_run=args.ct_model_run,
    )

    hn_images = jhb_imgs + ct_imgs
    hn_prov = jhb_prov + ct_prov
    print(f"\n  Total HN chips: {len(hn_images)}")
    print(f"    CBD: {jhb_stats['CBD']['n_chip']}")
    print(f"    Sandton: {jhb_stats['Sandton']['n_chip']}")
    print(f"    CT residual: {ct_stats['CT']['n_chip']}")

    print("\n[3/3] Merging with base COCO...")
    summary = merge_into_base(args.base_coco, args.output_dir, hn_images, hn_prov)
    summary["hn_composition"] = {
        "cbd": jhb_stats["CBD"]["n_chip"],
        "sandton": jhb_stats["Sandton"]["n_chip"],
        "ct_residual": ct_stats["CT"]["n_chip"],
    }
    with open(args.output_dir / "v4_3_hn_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
