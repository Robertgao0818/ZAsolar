"""V4 Joburg confidence-threshold sweep.

Uses the 50 reviewed grids under results/johannesburg/v4_aerial_2023/ where
every original prediction carries a review_status label (correct/edit/delete)
and missed positives are re-annotated with source='sam_fn_review'.

For each threshold t in the sweep, retain preds with confidence >= t and
compute precision / recall / F1 per group (CBD = G0xxx, Sandton = G1xxx)
and combined.

Outputs:
  results/analysis/v4_joburg_conf_sweep/sweep.csv
  results/analysis/v4_joburg_conf_sweep/sweep_by_grid.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

TP_STATUSES = {"correct", "edit"}
FP_STATUSES = {"delete"}


def load_reviewed(grid_dir: Path) -> pd.DataFrame:
    gpkg = grid_dir / "review" / f"{grid_dir.name}_reviewed.gpkg"
    gdf = gpd.read_file(gpkg)
    if "source" not in gdf.columns:
        gdf["source"] = None
    is_orig = gdf["source"].isna()
    preds = gdf[is_orig][["confidence", "review_status"]].copy()
    preds["grid"] = grid_dir.name
    fn_count = int((~is_orig).sum())
    return preds, fn_count


def sweep(df: pd.DataFrame, fn_totals: dict[str, int], thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        kept = df[df["confidence"] >= t]
        dropped = df[df["confidence"] < t]
        for group, group_df in kept.groupby("group"):
            tp = group_df["review_status"].isin(TP_STATUSES).sum()
            fp = group_df["review_status"].isin(FP_STATUSES).sum()
            # FN = original FN (added) + TP preds dropped below threshold
            dropped_group = dropped[dropped["group"] == group]
            tp_dropped = dropped_group["review_status"].isin(TP_STATUSES).sum()
            fn = fn_totals[group] + int(tp_dropped)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            rows.append({
                "threshold": round(float(t), 3),
                "group": group,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "kept_preds": int(len(group_df)),
            })
    return pd.DataFrame(rows)


def per_grid_sweep(df: pd.DataFrame, fn_per_grid: dict[str, int], thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        kept = df[df["confidence"] >= t]
        dropped = df[df["confidence"] < t]
        for grid, grid_df in kept.groupby("grid"):
            tp = grid_df["review_status"].isin(TP_STATUSES).sum()
            fp = grid_df["review_status"].isin(FP_STATUSES).sum()
            tp_dropped = dropped[dropped["grid"] == grid]["review_status"].isin(TP_STATUSES).sum()
            fn = fn_per_grid.get(grid, 0) + int(tp_dropped)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            rows.append({
                "threshold": round(float(t), 3),
                "grid": grid,
                "group": "CBD" if grid < "G1000" else "Sandton",
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-root",
        default="results/johannesburg/v4_aerial_2023",
        help="V4 JHB model run directory (relative to repo root)",
    )
    ap.add_argument("--output-dir", default="results/analysis/v4_joburg_conf_sweep")
    ap.add_argument("--thresh-min", type=float, default=0.60)
    ap.add_argument("--thresh-max", type=float, default=0.98)
    ap.add_argument("--thresh-step", type=float, default=0.02)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_root = repo_root / args.results_root
    out_dir = repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_dirs = sorted(
        [p for p in results_root.iterdir()
         if p.is_dir() and (p / "review" / f"{p.name}_reviewed.gpkg").exists()]
    )
    print(f"[info] found {len(grid_dirs)} reviewed grids")

    all_preds = []
    fn_per_grid: dict[str, int] = {}
    for gd in grid_dirs:
        preds, fn_count = load_reviewed(gd)
        preds["group"] = "CBD" if gd.name < "G1000" else "Sandton"
        fn_per_grid[gd.name] = fn_count
        all_preds.append(preds)
    df = pd.concat(all_preds, ignore_index=True)

    fn_totals: dict[str, int] = {
        "CBD": sum(v for g, v in fn_per_grid.items() if g < "G1000"),
        "Sandton": sum(v for g, v in fn_per_grid.items() if g >= "G1000"),
    }
    print(f"[info] total preds: {len(df)}, FN added: {sum(fn_per_grid.values())}")
    print(f"[info] FN by group: {fn_totals}")

    thresholds = np.arange(args.thresh_min, args.thresh_max + 1e-9, args.thresh_step)

    sweep_df = sweep(df, fn_totals, thresholds)
    # Combined group
    combined_rows = []
    for t in thresholds:
        sub = sweep_df[sweep_df["threshold"] == round(float(t), 3)]
        tp = sub["tp"].sum(); fp = sub["fp"].sum(); fn = sub["fn"].sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        combined_rows.append({
            "threshold": round(float(t), 3),
            "group": "ALL",
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "kept_preds": int(sub["kept_preds"].sum()),
        })
    sweep_df = pd.concat([sweep_df, pd.DataFrame(combined_rows)], ignore_index=True)

    sweep_csv = out_dir / "sweep.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"[write] {sweep_csv}")

    per_grid_df = per_grid_sweep(df, fn_per_grid, thresholds)
    per_grid_csv = out_dir / "sweep_by_grid.csv"
    per_grid_df.to_csv(per_grid_csv, index=False)
    print(f"[write] {per_grid_csv}")

    # Summary table
    print("\n=== Group sweep (precision / recall / F1) ===")
    for group in ["CBD", "Sandton", "ALL"]:
        sub = sweep_df[sweep_df["group"] == group].sort_values("threshold")
        best = sub.loc[sub["f1"].idxmax()]
        print(f"\n[{group}] best F1 = {best['f1']:.4f} at threshold={best['threshold']} "
              f"(P={best['precision']:.3f}, R={best['recall']:.3f}, "
              f"TP={int(best['tp'])}, FP={int(best['fp'])}, FN={int(best['fn'])})")
        show_thresh = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        rows = sub[sub["threshold"].isin([round(v, 3) for v in show_thresh])]
        print(rows[["threshold", "tp", "fp", "fn", "precision", "recall", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
