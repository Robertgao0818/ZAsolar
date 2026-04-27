"""
Build PV vs non-PV classifier dataset from the V3-C + V4.2 SAM mask+box cascade
on the V1.4 main benchmark (25 JHB CBD grids × Li hand-labeled GT × GEID 2024-02).

Pool sources:
- V3-C + SAM mask+box predictions  → results/johannesburg/v3c_sam_mask_geid_2024_02
- V4.2 + SAM mask+box predictions  → results/johannesburg/v4_2_sam_mask_geid_2024_02
- Li GT                            → /mnt/d/ZAsolar/annotations_inbox/Joburg_CBD_Li/{GRID}*.gpkg
- Imagery                          → ~/zasolar_data/tiles/johannesburg/geid_2024_02/{GRID}_mosaic.tif (EPSG:4326)

Labeling:
- For each prediction, max IoU to any Li GT polygon in the same grid is computed.
- label = "pv"  if iou_to_gt >= --tp-iou (default 0.1, matching cluster_level_eval default)
        = "nonpv" otherwise

Cross-detector dedup (per grid):
- Pair V3-C and V4.2 polygons via IoU >= --pair-iou (default 0.3)
        OR  inter/area_a >= 0.5  OR  inter/area_b >= 0.5
- Tag source_detector ∈ {v3c, v4_2, both} on every row.
  A V3-C row paired with any V4.2 row (with same label) → "both";
  a V4.2 row paired with any V3-C row (with same label) → "both";
  unpaired rows keep their own detector key.

Output:
- {out}/manifest.csv      one row per polygon in the pool
- {out}/manifest.gpkg     same rows + geometry, for QGIS audit
- {out}/chips/{label}/{chip_id}.png
- {out}/build_meta.json   counts, thresholds, paths

Usage:
    python scripts/classifier/build_cls_dataset_cascade.py \
        --output-dir data/cls_pv_nonpv_v3c_v42_cascade

The script does not split train/val — that's done by a separate splitter so
the same pool can be reused under different split policies.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import box as shapely_box

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_GT_DIR = Path("/mnt/d/ZAsolar/annotations_inbox/Joburg_CBD_Li")
DEFAULT_V3C_DIR = PROJECT_ROOT / "results/johannesburg/v3c_sam_mask_geid_2024_02"
DEFAULT_V42_DIR = PROJECT_ROOT / "results/johannesburg/v4_2_sam_mask_geid_2024_02"
DEFAULT_IMAGERY_ROOT = Path.home() / "zasolar_data/tiles/johannesburg/geid_2024_02"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/cls_pv_nonpv_v3c_v42_cascade"

GRID_IDS_25 = [
    "G0772", "G0773", "G0774", "G0775", "G0776",
    "G0814", "G0815", "G0816", "G0817", "G0818",
    "G0853", "G0854", "G0855", "G0856", "G0857",
    "G0888", "G0889", "G0890", "G0891", "G0892",
    "G0922", "G0923", "G0924", "G0925", "G0926",
]


@dataclass
class PoolRow:
    chip_id: str
    grid_id: str
    detector: str
    pred_idx: int
    label: str
    iou_to_gt: float
    area_m2: float
    geometry: object


def load_li_gt(gt_dir: Path, grid_id: str, target_crs: str) -> gpd.GeoDataFrame:
    """Li GT files are named like 'G0772(28).gpkg'. Read whichever matches."""
    candidates = sorted(gt_dir.glob(f"{grid_id}*.gpkg"))
    if not candidates:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    g = gpd.read_file(candidates[0])
    if g.crs is None:
        raise RuntimeError(f"GT file {candidates[0]} has no CRS")
    if str(g.crs) != target_crs:
        g = g.to_crs(target_crs)
    return g


def load_predictions(results_dir: Path, grid_id: str) -> gpd.GeoDataFrame:
    p = results_dir / grid_id / "predictions_metric.gpkg"
    if not p.exists():
        return gpd.GeoDataFrame(geometry=[])
    return gpd.read_file(p)


def max_iou_to_gt(geom, gt_geoms: list, gt_areas: list) -> float:
    """Max IoU between a single prediction and the GT polygon list (already pre-prepared)."""
    if not gt_geoms:
        return 0.0
    best = 0.0
    a_area = geom.area
    for gg, garea in zip(gt_geoms, gt_areas):
        if not geom.intersects(gg):
            continue
        inter = geom.intersection(gg).area
        if inter <= 0:
            continue
        union = a_area + garea - inter
        if union > 0:
            iou = inter / union
            if iou > best:
                best = iou
    return best


def build_grid_rows(
    grid_id: str,
    gt_geoms: list,
    gt_areas: list,
    detector: str,
    preds: gpd.GeoDataFrame,
    tp_iou: float,
) -> list[PoolRow]:
    rows = []
    for idx in range(len(preds)):
        geom = preds.geometry.iloc[idx]
        if geom is None or geom.is_empty:
            continue
        iou = max_iou_to_gt(geom, gt_geoms, gt_areas)
        label = "pv" if iou >= tp_iou else "nonpv"
        chip_id = f"{detector}_{grid_id}_p{idx:04d}"
        rows.append(
            PoolRow(
                chip_id=chip_id,
                grid_id=grid_id,
                detector=detector,
                pred_idx=idx,
                label=label,
                iou_to_gt=float(iou),
                area_m2=float(geom.area),
                geometry=geom,
            )
        )
    return rows


def tag_cross_detector(
    rows_v3c: list[PoolRow],
    rows_v42: list[PoolRow],
    pair_iou: float,
    contain_thresh: float,
) -> dict:
    """Per-grid pairwise overlap → source_detector tag map keyed by chip_id.

    Two rows are 'paired' if they share a grid, share a label, and their
    polygons overlap by IoU >= pair_iou OR by containment >= contain_thresh
    on either side.
    """
    tag = {r.chip_id: "v3c" for r in rows_v3c}
    tag.update({r.chip_id: "v4_2" for r in rows_v42})

    by_grid_label_v3c: dict[tuple, list[PoolRow]] = {}
    by_grid_label_v42: dict[tuple, list[PoolRow]] = {}
    for r in rows_v3c:
        by_grid_label_v3c.setdefault((r.grid_id, r.label), []).append(r)
    for r in rows_v42:
        by_grid_label_v42.setdefault((r.grid_id, r.label), []).append(r)

    for key, a_rows in by_grid_label_v3c.items():
        b_rows = by_grid_label_v42.get(key, [])
        if not b_rows:
            continue
        # build a tree on b
        b_geoms = [r.geometry for r in b_rows]
        for ar in a_rows:
            ag = ar.geometry
            ag_area = ag.area
            paired = False
            for br, bg in zip(b_rows, b_geoms):
                if not ag.intersects(bg):
                    continue
                inter = ag.intersection(bg).area
                if inter <= 0:
                    continue
                bg_area = bg.area
                iou = inter / (ag_area + bg_area - inter) if (ag_area + bg_area - inter) > 0 else 0.0
                if (
                    iou >= pair_iou
                    or inter / ag_area >= contain_thresh
                    or inter / bg_area >= contain_thresh
                ):
                    tag[ar.chip_id] = "both"
                    tag[br.chip_id] = "both"
                    paired = True
                    # don't break — let one A row tag multiple B rows
            _ = paired
    return tag


def crop_chip(
    src: rasterio.io.DatasetReader,
    geom_metric,
    metric_crs: str,
    chip_size: int,
    pad_ratio: float,
) -> np.ndarray | None:
    """Crop a square chip centered on geom; resize to chip_size×chip_size RGB."""
    minx, miny, maxx, maxy = geom_metric.bounds
    span = max(maxx - minx, maxy - miny) * (1.0 + pad_ratio)
    span = max(span, 4.0)  # min 4 m on a side to avoid degenerate windows
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = span / 2

    # Re-project the metric bbox into the raster CRS (EPSG:4326) for windowing.
    raster_crs = str(src.crs)
    bbox_metric = (cx - half, cy - half, cx + half, cy + half)
    if raster_crs != metric_crs:
        left, bottom, right, top = transform_bounds(
            metric_crs, raster_crs, *bbox_metric, densify_pts=21
        )
    else:
        left, bottom, right, top = bbox_metric

    raster_bounds = src.bounds
    if (
        right < raster_bounds.left
        or left > raster_bounds.right
        or top < raster_bounds.bottom
        or bottom > raster_bounds.top
    ):
        return None

    win = from_bounds(left, bottom, right, top, transform=src.transform)
    win = win.round_offsets().round_lengths()
    if win.width <= 0 or win.height <= 0:
        return None

    arr = src.read(
        indexes=[1, 2, 3] if src.count >= 3 else None,
        window=win,
        boundless=True,
        fill_value=0,
    )
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))  # H,W,C
    elif arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return None

    arr = cv2.resize(arr, (chip_size, chip_size), interpolation=cv2.INTER_AREA)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--v3c-results", type=Path, default=DEFAULT_V3C_DIR)
    parser.add_argument("--v42-results", type=Path, default=DEFAULT_V42_DIR)
    parser.add_argument("--imagery-root", type=Path, default=DEFAULT_IMAGERY_ROOT)
    parser.add_argument("--grids", nargs="+", default=GRID_IDS_25)
    parser.add_argument("--metric-crs", default="EPSG:32735")
    parser.add_argument("--tp-iou", type=float, default=0.1, help="IoU >= this → label=pv")
    parser.add_argument("--pair-iou", type=float, default=0.3)
    parser.add_argument("--contain-thresh", type=float, default=0.5)
    parser.add_argument("--chip-size", type=int, default=224)
    parser.add_argument("--pad-ratio", type=float, default=0.5, help="Extra context around polygon bbox")
    parser.add_argument("--skip-chips", action="store_true", help="Manifest only, skip chip extraction")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    chips_dir = out / "chips"
    if not args.skip_chips:
        (chips_dir / "pv").mkdir(parents=True, exist_ok=True)
        (chips_dir / "nonpv").mkdir(parents=True, exist_ok=True)

    print(f"Building cls pool from {len(args.grids)} grids", flush=True)

    all_v3c_rows: list[PoolRow] = []
    all_v42_rows: list[PoolRow] = []
    grids_with_gt = 0

    for grid_id in args.grids:
        gt = load_li_gt(args.gt_dir, grid_id, args.metric_crs)
        gt_geoms = [g for g in gt.geometry if g is not None and not g.is_empty]
        gt_areas = [g.area for g in gt_geoms]
        if gt_geoms:
            grids_with_gt += 1

        v3c_preds = load_predictions(args.v3c_results, grid_id)
        if str(v3c_preds.crs) != args.metric_crs and len(v3c_preds):
            v3c_preds = v3c_preds.to_crs(args.metric_crs)
        v42_preds = load_predictions(args.v42_results, grid_id)
        if str(v42_preds.crs) != args.metric_crs and len(v42_preds):
            v42_preds = v42_preds.to_crs(args.metric_crs)

        v3c_rows = build_grid_rows(grid_id, gt_geoms, gt_areas, "v3c", v3c_preds, args.tp_iou)
        v42_rows = build_grid_rows(grid_id, gt_geoms, gt_areas, "v4_2", v42_preds, args.tp_iou)

        all_v3c_rows.extend(v3c_rows)
        all_v42_rows.extend(v42_rows)

        n_v3c_pv = sum(1 for r in v3c_rows if r.label == "pv")
        n_v3c_nonpv = sum(1 for r in v3c_rows if r.label == "nonpv")
        n_v42_pv = sum(1 for r in v42_rows if r.label == "pv")
        n_v42_nonpv = sum(1 for r in v42_rows if r.label == "nonpv")
        print(
            f"  {grid_id}: GT={len(gt_geoms)} | "
            f"V3-C {len(v3c_rows)} ({n_v3c_pv}pv/{n_v3c_nonpv}nonpv) | "
            f"V4.2 {len(v42_rows)} ({n_v42_pv}pv/{n_v42_nonpv}nonpv)",
            flush=True,
        )

    print(f"\n{grids_with_gt}/{len(args.grids)} grids had GT loaded.", flush=True)

    cross = tag_cross_detector(
        all_v3c_rows, all_v42_rows, args.pair_iou, args.contain_thresh
    )

    rows: list[dict] = []
    for r in [*all_v3c_rows, *all_v42_rows]:
        rows.append(
            {
                "chip_id": r.chip_id,
                "grid_id": r.grid_id,
                "detector": r.detector,
                "pred_idx": r.pred_idx,
                "label": r.label,
                "iou_to_gt": round(r.iou_to_gt, 4),
                "area_m2": round(r.area_m2, 3),
                "source_detector": cross[r.chip_id],
                "geometry": r.geometry,
            }
        )

    manifest_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=args.metric_crs)

    # Chip extraction
    if not args.skip_chips:
        print("\nExtracting chips...", flush=True)
        chip_paths: list[str] = []
        n_skipped = 0
        for grid_id in args.grids:
            mosaic = args.imagery_root / f"{grid_id}_mosaic.tif"
            if not mosaic.exists():
                print(f"  [WARN] {mosaic} missing — skipping {grid_id}", flush=True)
                grid_mask = manifest_gdf["grid_id"] == grid_id
                for cid in manifest_gdf.loc[grid_mask, "chip_id"]:
                    chip_paths.append("")
                continue
            with rasterio.open(mosaic) as src:
                grid_rows = manifest_gdf[manifest_gdf["grid_id"] == grid_id]
                for _, row in grid_rows.iterrows():
                    arr = crop_chip(
                        src,
                        row.geometry,
                        args.metric_crs,
                        args.chip_size,
                        args.pad_ratio,
                    )
                    if arr is None:
                        chip_paths.append("")
                        n_skipped += 1
                        continue
                    rel = Path("chips") / row["label"] / f"{row['chip_id']}.png"
                    abs_path = out / rel
                    cv2.imwrite(str(abs_path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                    chip_paths.append(str(rel))
        manifest_gdf["chip_path"] = chip_paths
        if n_skipped:
            print(f"  [WARN] skipped {n_skipped} chips (out-of-bounds / empty)", flush=True)
    else:
        manifest_gdf["chip_path"] = ""

    manifest_gdf["split"] = ""

    csv_path = out / "manifest.csv"
    gpkg_path = out / "manifest.gpkg"
    manifest_gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)
    manifest_gdf.to_file(gpkg_path, driver="GPKG", layer="cls_pool")

    counts = manifest_gdf.groupby(["detector", "label", "source_detector"]).size().reset_index(name="n")
    print("\nFinal counts (detector × label × source_detector):")
    print(counts.to_string(index=False))

    meta = {
        "build_script": "scripts/classifier/build_cls_dataset_cascade.py",
        "grids": list(args.grids),
        "n_grids": len(args.grids),
        "tp_iou": args.tp_iou,
        "pair_iou": args.pair_iou,
        "contain_thresh": args.contain_thresh,
        "chip_size": args.chip_size,
        "pad_ratio": args.pad_ratio,
        "v3c_results": str(args.v3c_results),
        "v42_results": str(args.v42_results),
        "gt_dir": str(args.gt_dir),
        "imagery_root": str(args.imagery_root),
        "metric_crs": args.metric_crs,
        "n_total": int(len(manifest_gdf)),
        "n_pv": int((manifest_gdf["label"] == "pv").sum()),
        "n_nonpv": int((manifest_gdf["label"] == "nonpv").sum()),
        "by_detector_label_source": (
            counts.assign(n=counts["n"].astype(int))
            .to_dict(orient="records")
        ),
    }
    (out / "build_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nWrote {csv_path}")
    print(f"      {gpkg_path}")
    print(f"      {out/'build_meta.json'}")
    if not args.skip_chips:
        print(f"      {chips_dir}/{{pv,nonpv}}/<chip_id>.png")


if __name__ == "__main__":
    main()
