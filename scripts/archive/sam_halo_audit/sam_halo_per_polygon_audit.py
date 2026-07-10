#!/usr/bin/env python3
"""Per-polygon SAM halo suppression audit (25-grid Vexcel JHB CBD).

For each V3-C+SAM(mask+box) prediction matched to a clean_gt polygon at
IoU >= match_thresh, dump area_ratio before vs after SAM, the per-polygon
SAM shrink fraction, and the roof-swallow tail.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[2]
PRED_LAYER = "v3c_sam_maskbox_vexcel_2024"
GT_DIR = ROOT / "data" / "annotations_channel2_clean"
GRIDS = [
    "G0772", "G0773", "G0774", "G0775", "G0776",
    "G0814", "G0815", "G0816", "G0817", "G0818",
    "G0853", "G0854", "G0855", "G0856", "G0857",
    "G0888", "G0889", "G0890", "G0891", "G0892",
    "G0922", "G0923", "G0924", "G0925", "G0926",
]


def iou_pair(a, b) -> float:
    if not a.intersects(b):
        return 0.0
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    return inter / (a.area + b.area - inter)


def match_grid(grid: str, iou_thresh: float):
    pred_path = ROOT / "results" / "johannesburg" / PRED_LAYER / grid / "predictions_metric.gpkg"
    gt_path = GT_DIR / grid / f"{grid}_clean_gt.gpkg"
    if not pred_path.exists() or not gt_path.exists():
        return []
    preds = gpd.read_file(pred_path)
    gts = gpd.read_file(gt_path)
    if preds.crs != gts.crs:
        gts = gts.to_crs(preds.crs)
    preds = preds[~preds.geometry.is_empty & preds.geometry.notna()].reset_index(drop=True)
    gts = gts[~gts.geometry.is_empty & gts.geometry.notna()].reset_index(drop=True)
    if len(preds) == 0 or len(gts) == 0:
        return []
    gt_geoms = list(gts.geometry.values)
    tree = STRtree(gt_geoms)

    rows = []
    for pi, prow in preds.iterrows():
        pg = prow.geometry
        cand_idx = tree.query(pg)
        best_iou, best_gi = 0.0, -1
        for gi in cand_idx:
            iou = iou_pair(pg, gt_geoms[int(gi)])
            if iou > best_iou:
                best_iou, best_gi = iou, int(gi)
        if best_gi < 0 or best_iou < iou_thresh:
            continue
        gg = gt_geoms[best_gi]
        gt_area = gg.area
        sam_area = float(prow["sam_area_m2"]) if prow["sam_area_m2"] is not None else pg.area
        orig_area = float(prow["orig_area_m2"]) if prow["orig_area_m2"] is not None else sam_area
        rows.append({
            "grid": grid,
            "pred_idx": int(pi),
            "gt_idx": best_gi,
            "iou": best_iou,
            "gt_area_m2": gt_area,
            "orig_area_m2": orig_area,
            "sam_area_m2": sam_area,
            "ratio_before": orig_area / gt_area if gt_area > 0 else float("nan"),
            "ratio_after": sam_area / gt_area if gt_area > 0 else float("nan"),
            "sam_shrink_pct": (1.0 - sam_area / orig_area) * 100 if orig_area > 0 else float("nan"),
            "confidence": float(prow["confidence"]) if prow["confidence"] is not None else float("nan"),
        })
    return rows


def summarize(rows, label):
    arr = np.array([r["ratio_after"] for r in rows])
    before = np.array([r["ratio_before"] for r in rows])
    shrink = np.array([r["sam_shrink_pct"] for r in rows])
    deciles = np.percentile(arr, [10, 25, 50, 75, 90, 95, 99])
    before_dec = np.percentile(before, [10, 25, 50, 75, 90, 95, 99])
    shrink_dec = np.percentile(shrink, [10, 25, 50, 75, 90])
    n_total = len(arr)

    def frac(mask):
        return mask.sum() / n_total if n_total else 0.0

    print(f"\n=== {label} (n={n_total} matched TP @ IoU>=thresh) ===")
    print("ratio_before (V3-C raw / GT):")
    print(f"  p10/25/50/75/90/95/99 = "
          f"{before_dec[0]:.2f} / {before_dec[1]:.2f} / {before_dec[2]:.2f} / "
          f"{before_dec[3]:.2f} / {before_dec[4]:.2f} / {before_dec[5]:.2f} / {before_dec[6]:.2f}")
    print(f"  mean = {before.mean():.3f}  median = {np.median(before):.3f}")
    print("ratio_after  (V3-C+SAM / GT):")
    print(f"  p10/25/50/75/90/95/99 = "
          f"{deciles[0]:.2f} / {deciles[1]:.2f} / {deciles[2]:.2f} / "
          f"{deciles[3]:.2f} / {deciles[4]:.2f} / {deciles[5]:.2f} / {deciles[6]:.2f}")
    print(f"  mean = {arr.mean():.3f}  median = {np.median(arr):.3f}")
    print("sam_shrink_pct (per-polygon SAM tighten, positive=tighter):")
    print(f"  p10/25/50/75/90 = "
          f"{shrink_dec[0]:.1f}% / {shrink_dec[1]:.1f}% / {shrink_dec[2]:.1f}% / "
          f"{shrink_dec[3]:.1f}% / {shrink_dec[4]:.1f}%")
    print(f"  mean = {shrink.mean():.2f}%  median = {np.median(shrink):.2f}%")
    print("Roof-swallow tail (ratio_after > X):")
    for thr in (1.25, 1.5, 2.0, 3.0):
        print(f"  ratio_after > {thr}: {int((arr > thr).sum())} / {n_total} "
              f"({frac(arr > thr) * 100:.1f}%)")
    print("Halo-suppress success (ratio_before > 1.25 AND ratio_after <= 1.10):")
    success = (before > 1.25) & (arr <= 1.10)
    over_before = before > 1.25
    print(f"  {int(success.sum())} / {int(over_before.sum())} over-painted preds "
          f"({success.sum() / max(over_before.sum(), 1) * 100:.1f}%) had halo cleaned to within ±10% GT")
    print("Worse-after-SAM cases (ratio_after > ratio_before, both > 1.0):")
    worse = (arr > before) & (before > 1.0)
    print(f"  {int(worse.sum())} / {n_total} ({worse.sum() / n_total * 100:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou-thresh", type=float, default=0.3,
                    help="IoU threshold to declare TP match for per-polygon comparison.")
    ap.add_argument("--out-dir", default="results/analysis/sam_halo_per_polygon_2026-05-11")
    args = ap.parse_args()

    all_rows = []
    for grid in GRIDS:
        rows = match_grid(grid, args.iou_thresh)
        print(f"[{grid}] matched={len(rows)}")
        all_rows.extend(rows)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"per_polygon_iou{int(args.iou_thresh*100)}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[WRITE] {csv_path}  rows={len(all_rows)}")

    if not all_rows:
        sys.exit("no matched polygons")

    summarize(all_rows, f"V3-C+SAM(mask+box) vs clean_gt  IoU>={args.iou_thresh}")

    print("\n=== Per-grid medians ===")
    by_grid = {}
    for r in all_rows:
        by_grid.setdefault(r["grid"], []).append(r)
    print(f"{'grid':6s} {'n':>4s} {'med_before':>10s} {'med_after':>10s} {'med_shrink%':>12s} {'tail>1.5':>10s}")
    for grid in GRIDS:
        rows = by_grid.get(grid, [])
        if not rows:
            print(f"{grid:6s} {0:>4d}    (no matches)")
            continue
        before = np.array([r["ratio_before"] for r in rows])
        after = np.array([r["ratio_after"] for r in rows])
        shrink = np.array([r["sam_shrink_pct"] for r in rows])
        tail = (after > 1.5).sum()
        print(f"{grid:6s} {len(rows):>4d} {np.median(before):>10.3f} {np.median(after):>10.3f} "
              f"{np.median(shrink):>11.1f}% {tail:>10d}")


if __name__ == "__main__":
    main()
