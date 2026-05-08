#!/usr/bin/env python3
"""Compute Channel 2 exhaustive recall against clean GT.

Inputs
------
--clean-gt-root   data/annotations_channel2_clean
                   (per-grid `<G>/<G>_clean_gt.gpkg`, EPSG:32735)
--pred-root       results/johannesburg/<model_run>
                   (per-grid `<G>/predictions_metric.gpkg`, EPSG:32735)
--output-dir      where to drop CSV outputs

Per grid, runs installation-profile matching (pred-side many-to-one merge
via `iou_matching(merge_preds=True)` from detect_and_evaluate.py) at IoU
thresholds {0.1, 0.3, 0.5} and reports:

  - per-grid recall + counts
  - aggregate recall (pooled across grids)
  - aggregate recall stratified by GT `source` (V3C_TP / SAM_supp / Li / sam_added)
  - aggregate recall stratified by GT area_m2 bucket (xs/s/m/l/xl)

Confidence intervals: Wilson on the GT-count denominator.

Caveat
------
clean GT was built from the same V3-C predictions used here for V3-C recall.
The V3C_TP source subset is therefore self-recall (~100% expected). The
informative numbers are paired V4.1 recall on the same GT and per-source
breakdown (SAM_supp + Li_include reflect what V3-C originally missed).
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detect_and_evaluate import iou_matching  # noqa: E402

IOU_THRESHOLDS = (0.1, 0.3, 0.5)

SIZE_BUCKETS = [
    ("xs", 0.0, 10.0),
    ("s",  10.0, 30.0),
    ("m",  30.0, 100.0),
    ("l",  100.0, 300.0),
    ("xl", 300.0, float("inf")),
]


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def size_bucket(area: float) -> str:
    for label, lo, hi in SIZE_BUCKETS:
        if lo <= area < hi:
            return label
    return "xl"


def load_grid(
    grid: str,
    clean_gt_root: Path,
    pred_root: Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | None:
    gt_path = clean_gt_root / grid / f"{grid}_clean_gt.gpkg"
    pred_path = pred_root / grid / "predictions_metric.gpkg"
    if not gt_path.exists():
        print(f"[SKIP] {grid}: clean GT missing ({gt_path})")
        return None
    if not pred_path.exists():
        print(f"[SKIP] {grid}: predictions missing ({pred_path})")
        return None
    gt = gpd.read_file(gt_path)
    pred = gpd.read_file(pred_path)
    if gt.crs != pred.crs:
        pred = pred.to_crs(gt.crs)
    # Ensure area_m2 / source columns exist on GT.
    if "area_m2" not in gt.columns:
        gt = gt.copy()
        gt["area_m2"] = gt.geometry.area
    if "source" not in gt.columns:
        gt = gt.copy()
        gt["source"] = "unknown"
    gt = gt.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    return gt, pred


def match_grid(
    gt: gpd.GeoDataFrame,
    pred: gpd.GeoDataFrame,
    iou_threshold: float,
) -> set[int]:
    """Return matched GT indices at this IoU threshold (installation profile)."""
    if len(gt) == 0 or len(pred) == 0:
        return set()
    res = iou_matching(gt, pred, iou_threshold=iou_threshold, merge_preds=True)
    return set(res["matched_gt_indices"])


def per_gt_intersection_area(
    gt: gpd.GeoDataFrame,
    pred: gpd.GeoDataFrame,
) -> np.ndarray:
    """For each GT polygon, return the area of its intersection with the
    union of all predictions touching it.

    This is **threshold-free area coverage** — independent of IoU matching
    decisions, it measures how much GT area is actually covered by any
    prediction overlap. Used for area-weighted recall that exposes
    partial-detection (large GT panels whose predictions cover only part).
    """
    if len(gt) == 0:
        return np.zeros(0)
    if len(pred) == 0:
        return np.zeros(len(gt))
    from shapely.ops import unary_union
    psindex = pred.sindex
    out = np.zeros(len(gt))
    for i, gt_geom in enumerate(gt.geometry):
        cand = list(psindex.intersection(gt_geom.bounds))
        if not cand:
            continue
        try:
            merged = unary_union([pred.iloc[pidx].geometry for pidx in cand])
            out[i] = gt_geom.intersection(merged).area
        except Exception:
            continue
    return out


def recall_block(matched: int, total: int) -> dict:
    p = matched / total if total else 0.0
    lo, hi = wilson(p, total)
    return {
        "matched": matched,
        "total": total,
        "recall": p,
        "ci_lo": lo,
        "ci_hi": hi,
    }


def area_recall_block(matched_area: float, total_area: float) -> dict:
    p = matched_area / total_area if total_area else 0.0
    return {
        "matched_area_m2": matched_area,
        "total_area_m2": total_area,
        "area_recall": p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-gt-root", type=Path,
                        default=PROJECT_ROOT / "data" / "annotations_channel2_clean")
    parser.add_argument("--pred-root", type=Path, required=True,
                        help="e.g. results/johannesburg/v3c_vexcel_2024")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grids", nargs="*", default=None,
                        help="optional grid filter; default = all grids found in clean-gt-root")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = args.pred_root.name

    if args.grids:
        grids = args.grids
    else:
        grids = sorted(p.name for p in args.clean_gt_root.iterdir()
                       if p.is_dir() and p.name.startswith("G"))

    per_grid_rows: list[dict] = []
    per_gt_rows: list[dict] = []  # for source/size aggregation

    for grid in grids:
        loaded = load_grid(grid, args.clean_gt_root, args.pred_root)
        if loaded is None:
            continue
        gt, pred = loaded
        n_gt = len(gt)
        n_pred = len(pred)

        gt_records = gt[["source", "area_m2"]].copy()
        gt_records["grid"] = grid
        gt_records["size_bucket"] = gt_records["area_m2"].apply(size_bucket)

        # Threshold-free per-GT covered area (from union of all overlapping preds).
        covered_area = per_gt_intersection_area(gt, pred)
        gt_records["covered_area_m2"] = covered_area
        total_area = float(gt["area_m2"].sum())
        total_covered = float(covered_area.sum())

        for iou in IOU_THRESHOLDS:
            matched_idx = match_grid(gt, pred, iou)
            # area-conditional-on-match: how much GT area belongs to GTs that
            # crossed the IoU threshold (this is whole-polygon, not coverage).
            matched_full_area = float(gt.loc[list(matched_idx), "area_m2"].sum()) if matched_idx else 0.0
            row = {"grid": grid, "iou": iou, "n_gt": n_gt, "n_pred": n_pred,
                   **recall_block(len(matched_idx), n_gt),
                   "matched_full_area_m2": matched_full_area,
                   "total_area_m2": total_area,
                   "matched_full_area_recall": matched_full_area / total_area if total_area else 0.0,
                   "covered_area_m2": total_covered,
                   "area_coverage_recall": total_covered / total_area if total_area else 0.0}
            per_grid_rows.append(row)

            tag = f"matched_iou_{iou}"
            gt_records[tag] = gt_records.index.isin(matched_idx)

        per_gt_rows.append(gt_records)

        # console line
        line = f"{grid}: GT={n_gt:4d}  pred={n_pred:4d}  "
        for iou in IOU_THRESHOLDS:
            r = next(r for r in per_grid_rows[-3:] if r["iou"] == iou)
            line += f"R@{iou}={r['recall']:.3f} ({r['matched']}/{r['total']})  "
        print(line)

    if not per_grid_rows:
        print("No grids processed")
        return

    per_grid_df = pd.DataFrame(per_grid_rows)
    per_gt_df = pd.concat(per_gt_rows, ignore_index=True)

    per_grid_path = args.output_dir / f"ch2_recall_per_grid_{label}.csv"
    per_grid_df.to_csv(per_grid_path, index=False)
    print(f"\n[WRITE] {per_grid_path}")

    # ── Pooled overall ────────────────────────────────────────────────
    pooled_rows = []
    for iou in IOU_THRESHOLDS:
        sub = per_grid_df[per_grid_df["iou"] == iou]
        m, t = int(sub["matched"].sum()), int(sub["total"].sum())
        mfa = float(sub["matched_full_area_m2"].sum())
        cov = float(sub["covered_area_m2"].sum())
        ta = float(sub["total_area_m2"].sum())
        pooled_rows.append({"iou": iou, "scope": "all", "key": "all",
                            **recall_block(m, t),
                            "matched_full_area_m2": mfa,
                            "covered_area_m2": cov,
                            "total_area_m2": ta,
                            "matched_full_area_recall": mfa / ta if ta else 0.0,
                            "area_coverage_recall": cov / ta if ta else 0.0})

    # ── By source ────────────────────────────────────────────────────
    by_src_rows = []
    for iou in IOU_THRESHOLDS:
        tag = f"matched_iou_{iou}"
        for src, grp in per_gt_df.groupby("source"):
            m = int(grp[tag].sum())
            t = len(grp)
            mfa = float(grp.loc[grp[tag], "area_m2"].sum())
            cov = float(grp["covered_area_m2"].sum())
            ta = float(grp["area_m2"].sum())
            by_src_rows.append({"iou": iou, "scope": "by_source", "key": src,
                                **recall_block(m, t),
                                "matched_full_area_m2": mfa,
                                "covered_area_m2": cov,
                                "total_area_m2": ta,
                                "matched_full_area_recall": mfa / ta if ta else 0.0,
                                "area_coverage_recall": cov / ta if ta else 0.0})

    # ── By size bucket ───────────────────────────────────────────────
    by_size_rows = []
    for iou in IOU_THRESHOLDS:
        tag = f"matched_iou_{iou}"
        for label_, lo, hi in SIZE_BUCKETS:
            grp = per_gt_df[per_gt_df["size_bucket"] == label_]
            m = int(grp[tag].sum())
            t = len(grp)
            mfa = float(grp.loc[grp[tag], "area_m2"].sum())
            cov = float(grp["covered_area_m2"].sum())
            ta = float(grp["area_m2"].sum())
            by_size_rows.append({"iou": iou, "scope": "by_size",
                                 "key": f"{label_}({lo:g}-{hi:g}m²)",
                                 **recall_block(m, t),
                                 "matched_full_area_m2": mfa,
                                 "covered_area_m2": cov,
                                 "total_area_m2": ta,
                                 "matched_full_area_recall": mfa / ta if ta else 0.0,
                                 "area_coverage_recall": cov / ta if ta else 0.0})

    agg_df = pd.DataFrame(pooled_rows + by_src_rows + by_size_rows)
    agg_path = args.output_dir / f"ch2_recall_aggregate_{label}.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"[WRITE] {agg_path}")

    cols_count = ["iou", "key", "matched", "total", "recall"]
    cols_area = ["covered_area_m2", "total_area_m2", "area_coverage_recall",
                 "matched_full_area_recall"]
    print("\n── Pooled overall ──")
    print(agg_df[agg_df["scope"] == "all"]
          [cols_count + ["ci_lo", "ci_hi"] + cols_area].to_string(index=False))
    print("\n── By GT source ──")
    print(agg_df[agg_df["scope"] == "by_source"]
          [cols_count + cols_area].to_string(index=False))
    print("\n── By GT size bucket ──")
    print(agg_df[agg_df["scope"] == "by_size"]
          [cols_count + cols_area].to_string(index=False))
    print("\nLegend: count_recall=GT polygons matched (presence@IoU); "
          "area_coverage_recall=∑(GT∩pred_union)/∑GT_area (threshold-free, "
          "exposes partial detection on large panels); "
          "matched_full_area_recall=∑(area of GTs matched at IoU)/∑GT_area "
          "(weights small wins by their full size — risk of overstating).")


if __name__ == "__main__":
    main()
