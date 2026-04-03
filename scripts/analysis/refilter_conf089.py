"""
Re-filter existing predictions at confidence >= 0.89 and recompute P/R/F1.
Uses IoU > 0.1 matching (pred is TP if it overlaps any GT polygon with IoU > 0.1).
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

BASE_DIR = Path("/workspace/ZAsolar")
RESULTS_DIR = BASE_DIR / "results"
GT_DIR = BASE_DIR / "data" / "annotations" / "cleaned"
CONF_THRESHOLD = 0.89
IOU_THRESHOLD = 0.1

GRIDS = [
    "G1682", "G1683", "G1685", "G1686", "G1687", "G1688", "G1689", "G1690",
    "G1691", "G1692", "G1693", "G1743", "G1744", "G1747", "G1749", "G1750",
    "G1798", "G1800", "G1806", "G1807",
]


def compute_iou_matrix(preds, gts):
    """Compute IoU between each pred and each GT polygon."""
    n_pred = len(preds)
    n_gt = len(gts)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt))

    iou_matrix = np.zeros((n_pred, n_gt))
    for i, p_geom in enumerate(preds.geometry):
        for j, g_geom in enumerate(gts.geometry):
            if p_geom.is_empty or g_geom.is_empty:
                continue
            if not p_geom.intersects(g_geom):
                continue
            inter = p_geom.intersection(g_geom).area
            union = p_geom.area + g_geom.area - inter
            if union > 0:
                iou_matrix[i, j] = inter / union
    return iou_matrix


def evaluate_grid(grid_id):
    """Evaluate a single grid. Returns dict with TP/FP/FN counts."""
    pred_path = RESULTS_DIR / grid_id / "predictions_metric.gpkg"
    if not pred_path.exists():
        print(f"  SKIP {grid_id}: no predictions_metric.gpkg")
        return None

    preds = gpd.read_file(str(pred_path))

    # Filter by confidence
    if "confidence" not in preds.columns:
        # Try 'score' as alternative column name
        if "score" in preds.columns:
            preds = preds.rename(columns={"score": "confidence"})
        else:
            print(f"  SKIP {grid_id}: no confidence/score column. Columns: {list(preds.columns)}")
            return None

    n_before = len(preds)
    preds = preds[preds["confidence"] >= CONF_THRESHOLD].copy()
    n_after = len(preds)

    # Find GT file
    gt_files = list(GT_DIR.glob(f"{grid_id}*.gpkg"))
    if not gt_files:
        # Try without prefix
        gt_files = list(GT_DIR.glob(f"*{grid_id}*.gpkg"))
    if not gt_files:
        print(f"  SKIP {grid_id}: no GT file found in {GT_DIR}")
        return None

    gt = gpd.read_file(str(gt_files[0]))

    # Reproject GT to match pred CRS if needed
    if gt.crs is None:
        # Assume GT is in EPSG:4326 (common for annotation files)
        gt = gt.set_crs("EPSG:4326")
    if gt.crs != preds.crs:
        gt = gt.to_crs(preds.crs)

    # Compute IoU matrix
    iou_mat = compute_iou_matrix(preds, gt)

    # Greedy matching: for each pred, check if max IoU > threshold
    tp = 0
    matched_gt = set()

    if len(preds) > 0 and len(gt) > 0:
        # Sort preds by confidence descending for greedy matching
        pred_order = preds["confidence"].values.argsort()[::-1]

        for pred_idx in pred_order:
            best_gt = iou_mat[pred_idx].argmax()
            if iou_mat[pred_idx, best_gt] > IOU_THRESHOLD and best_gt not in matched_gt:
                tp += 1
                matched_gt.add(best_gt)

    fp = len(preds) - tp
    fn = len(gt) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "grid": grid_id,
        "n_pred_orig": n_before,
        "n_pred_filtered": n_after,
        "n_gt": len(gt),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def main():
    print(f"Re-filtering predictions at confidence >= {CONF_THRESHOLD}")
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print(f"Grids: {len(GRIDS)}")
    print("=" * 100)

    results = []
    for grid_id in GRIDS:
        print(f"Processing {grid_id}...")
        r = evaluate_grid(grid_id)
        if r is not None:
            results.append(r)
            print(f"  preds: {r['n_pred_orig']} -> {r['n_pred_filtered']} | "
                  f"GT: {r['n_gt']} | TP={r['TP']} FP={r['FP']} FN={r['FN']} | "
                  f"P={r['Precision']:.3f} R={r['Recall']:.3f} F1={r['F1']:.3f}")

    if not results:
        print("No results!")
        return

    df = pd.DataFrame(results)

    # Summary
    total_tp = df["TP"].sum()
    total_fp = df["FP"].sum()
    total_fn = df["FN"].sum()
    total_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_p * total_r / (total_p + total_r) if (total_p + total_r) > 0 else 0

    print("\n" + "=" * 100)
    print(f"{'Grid':<8} {'Pred(orig)':>10} {'Pred(0.89)':>10} {'GT':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 100)
    for _, row in df.iterrows():
        print(f"{row['grid']:<8} {row['n_pred_orig']:>10} {row['n_pred_filtered']:>10} {row['n_gt']:>6} "
              f"{row['TP']:>5} {row['FP']:>5} {row['FN']:>5} "
              f"{row['Precision']:>7.3f} {row['Recall']:>7.3f} {row['F1']:>7.3f}")
    print("-" * 100)
    print(f"{'TOTAL':<8} {df['n_pred_orig'].sum():>10} {df['n_pred_filtered'].sum():>10} {df['n_gt'].sum():>6} "
          f"{total_tp:>5} {total_fp:>5} {total_fn:>5} "
          f"{total_p:>7.3f} {total_r:>7.3f} {total_f1:>7.3f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
