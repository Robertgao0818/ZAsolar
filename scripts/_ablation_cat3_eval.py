#!/usr/bin/env python3
"""Ad-hoc cat-3 postproc ablation eval for a single grid.

Compares predictions_metric.gpkg from each finalize variant against
clean_gt at pixel level (rasterize @ 0.1 m, CRS = grid metric CRS).

Reports: pixel_P, pixel_R, pixel_F1, pixel_IoU, bulk_ratio = pred_area / gt_area.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


def _rasterize(gdf: gpd.GeoDataFrame, transform, height: int, width: int) -> np.ndarray:
    if gdf.empty:
        return np.zeros((height, width), dtype=bool)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros((height, width), dtype=bool)
    return rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype="uint8").astype(bool)


def pixel_metrics(pred_gpkg: Path, gt_gpkg: Path, crs: str, resolution_m: float = 0.1) -> dict:
    pred = gpd.read_file(pred_gpkg)
    gt = gpd.read_file(gt_gpkg)
    if pred.crs is None:
        pred = pred.set_crs(crs)
    pred = pred.to_crs(crs)
    if gt.crs is None:
        gt = gt.set_crs(crs)
    gt = gt.to_crs(crs)

    if pred.empty and gt.empty:
        return {"pred_n": 0, "gt_n": 0, "pred_area": 0, "gt_area": 0,
                "inter": 0, "union": 0, "pixel_P": 0, "pixel_R": 0,
                "pixel_F1": 0, "pixel_IoU": 0, "bulk_ratio": 0}

    bounds_list = []
    if not pred.empty:
        bounds_list.append(pred.total_bounds)
    if not gt.empty:
        bounds_list.append(gt.total_bounds)
    bs = np.array(bounds_list)
    minx, miny = bs[:, 0].min(), bs[:, 1].min()
    maxx, maxy = bs[:, 2].max(), bs[:, 3].max()
    minx -= 1; miny -= 1; maxx += 1; maxy += 1

    width = int(np.ceil((maxx - minx) / resolution_m))
    height = int(np.ceil((maxy - miny) / resolution_m))
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    pred_mask = _rasterize(pred, transform, height, width)
    gt_mask = _rasterize(gt, transform, height, width)

    px_area = resolution_m ** 2
    inter = float((pred_mask & gt_mask).sum() * px_area)
    union = float((pred_mask | gt_mask).sum() * px_area)
    pred_area = float(pred_mask.sum() * px_area)
    gt_area = float(gt_mask.sum() * px_area)

    p = inter / pred_area if pred_area else 0.0
    r = inter / gt_area if gt_area else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    iou = inter / union if union else 0.0
    bulk = pred_area / gt_area if gt_area else 0.0

    return {"pred_n": len(pred), "gt_n": len(gt),
            "pred_area": pred_area, "gt_area": gt_area,
            "inter": inter, "union": union,
            "pixel_P": p, "pixel_R": r, "pixel_F1": f1,
            "pixel_IoU": iou, "bulk_ratio": bulk}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variants-dir", type=Path, required=True,
                   help="Dir with subdirs <variant>/predictions_metric.gpkg")
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--crs", default="EPSG:32735")
    p.add_argument("--resolution-m", type=float, default=0.1)
    args = p.parse_args()

    rows = []
    for sub in sorted(args.variants_dir.iterdir()):
        if not sub.is_dir():
            continue
        gpkg = sub / "predictions_metric.gpkg"
        if not gpkg.exists():
            print(f"[skip] {sub.name}: no predictions_metric.gpkg", file=sys.stderr)
            continue
        m = pixel_metrics(gpkg, args.gt, args.crs, args.resolution_m)
        m["variant"] = sub.name
        rows.append(m)

    if not rows:
        print("no variants found", file=sys.stderr)
        return 1

    cols = ["variant", "pred_n", "gt_n", "pred_area", "gt_area",
            "pixel_P", "pixel_R", "pixel_F1", "pixel_IoU", "bulk_ratio"]
    print(f"{'variant':<32} {'pred_n':>6} {'gt_n':>5} {'pred_m2':>10} {'gt_m2':>9} "
          f"{'P':>6} {'R':>6} {'F1':>6} {'IoU':>6} {'bulk':>6}")
    for r in sorted(rows, key=lambda x: -x["pixel_F1"]):
        print(f"{r['variant']:<32} {r['pred_n']:>6} {r['gt_n']:>5} "
              f"{r['pred_area']:>10.1f} {r['gt_area']:>9.1f} "
              f"{r['pixel_P']:>6.3f} {r['pixel_R']:>6.3f} "
              f"{r['pixel_F1']:>6.3f} {r['pixel_IoU']:>6.3f} "
              f"{r['bulk_ratio']:>6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
