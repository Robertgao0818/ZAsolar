"""
Compare original vs classifier-filtered detection results.

Computes detection count changes, removal stats, and reviewed-precision
proxy on grids that have human review data.

Note: recall/F1 CANNOT be computed from review decisions alone (no FN
inventory). For full metrics, use detect_and_evaluate.py's GT matcher.

Usage:
    python scripts/classifier/compare_results.py --grid-id G1238
    python scripts/classifier/compare_results.py --grid-ids G1238 G1689 G1690
    python scripts/classifier/compare_results.py --all-classified
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def compare_grid(grid_id: str, results_dir: Path) -> dict | None:
    """Compare original vs classifier-filtered predictions for one grid."""
    grid_dir = results_dir / grid_id
    orig_path = grid_dir / "predictions_metric.gpkg"
    cls_path = grid_dir / "predictions_metric_cls.gpkg"
    filtered_path = grid_dir / "predictions_metric_cls_filtered.gpkg"
    review_path = grid_dir / "review" / "detection_review_decisions.csv"

    if not orig_path.exists():
        return None
    if not cls_path.exists():
        print(f"  {grid_id}: no classifier results found, skipping")
        return None

    orig = gpd.read_file(orig_path)
    cls_full = gpd.read_file(cls_path)
    filtered = gpd.read_file(filtered_path) if filtered_path.exists() else cls_full[cls_full["cls_label"] == "pv"]

    n_orig = len(orig)
    n_filtered = len(filtered)
    n_removed = n_orig - n_filtered

    # Removal breakdown by cls_score range
    removed = cls_full[cls_full["cls_label"] == "non_pv"]
    score_bins = {
        "0.0-0.1": ((removed["cls_score"] >= 0.0) & (removed["cls_score"] < 0.1)).sum(),
        "0.1-0.2": ((removed["cls_score"] >= 0.1) & (removed["cls_score"] < 0.2)).sum(),
        "0.2-0.3": ((removed["cls_score"] >= 0.2) & (removed["cls_score"] < 0.3)).sum(),
        "0.3-0.4": ((removed["cls_score"] >= 0.3) & (removed["cls_score"] < 0.4)).sum(),
        "0.4-0.5": ((removed["cls_score"] >= 0.4) & (removed["cls_score"] < 0.5)).sum(),
    }

    # Size distribution of removed
    if len(removed) > 0 and "area_m2" in removed.columns:
        removed_area_stats = {
            "mean": float(removed["area_m2"].mean()),
            "median": float(removed["area_m2"].median()),
            "min": float(removed["area_m2"].min()),
            "max": float(removed["area_m2"].max()),
        }
    else:
        removed_area_stats = {}

    result = {
        "grid_id": grid_id,
        "original_count": n_orig,
        "filtered_count": n_filtered,
        "removed_count": n_removed,
        "removal_rate": n_removed / n_orig if n_orig > 0 else 0,
        "score_distribution": {k: int(v) for k, v in score_bins.items()},
        "removed_area_stats": removed_area_stats,
    }

    # --- Reviewed-grid metrics (precision proxy) ---
    if review_path.exists():
        with open(review_path) as f:
            decisions = {int(r["pred_id"]): r["status"] for r in csv.DictReader(f)}

        # "correct" and "edit" are both real PV; "delete" is non-PV
        pv_statuses = ("correct", "edit")

        # Original reviewed precision
        orig_reviewed = sum(1 for pid, s in decisions.items()
                           if s in pv_statuses and pid < n_orig)
        orig_reviewed_total = sum(1 for pid, s in decisions.items()
                                 if s in (*pv_statuses, "delete") and pid < n_orig)
        orig_precision = orig_reviewed / orig_reviewed_total if orig_reviewed_total > 0 else 0

        # Filtered reviewed precision
        filtered_indices = set(filtered.index)
        filt_reviewed = sum(1 for pid, s in decisions.items()
                           if s in pv_statuses and pid in filtered_indices)
        filt_reviewed_total = sum(1 for pid, s in decisions.items()
                                 if s in (*pv_statuses, "delete") and pid in filtered_indices)
        filt_precision = filt_reviewed / filt_reviewed_total if filt_reviewed_total > 0 else 0

        # Removal accuracy: fraction of removed that were actually "delete"
        removed_indices = set(cls_full.index) - filtered_indices
        removal_correct = sum(1 for pid in removed_indices
                             if decisions.get(pid) == "delete")
        removal_incorrect = sum(1 for pid in removed_indices
                               if decisions.get(pid) in pv_statuses)
        removal_total = removal_correct + removal_incorrect
        removal_accuracy = removal_correct / removal_total if removal_total > 0 else 0

        result["reviewed"] = {
            "original_precision": orig_precision,
            "filtered_precision": filt_precision,
            "precision_delta": filt_precision - orig_precision,
            "removal_accuracy": removal_accuracy,
            "removal_correct": removal_correct,
            "removal_incorrect": removal_incorrect,
            "note": "Precision proxy from review decisions only; recall/F1 not computable",
        }

    return result


def print_comparison(results: list[dict]) -> None:
    """Print formatted comparison table."""
    print(f"\n{'Grid':<8} {'Orig':>6} {'Filt':>6} {'Rmvd':>6} {'Rate':>7}", end="")

    has_reviewed = any("reviewed" in r for r in results)
    if has_reviewed:
        print(f" {'OrgPrec':>8} {'FltPrec':>8} {'Delta':>7} {'RmAcc':>7}", end="")
    print()
    print("-" * (35 + (32 if has_reviewed else 0)))

    for r in results:
        print(f"{r['grid_id']:<8} "
              f"{r['original_count']:>6} "
              f"{r['filtered_count']:>6} "
              f"{r['removed_count']:>6} "
              f"{r['removal_rate']:>6.1%}", end="")

        if has_reviewed:
            rev = r.get("reviewed", {})
            if rev:
                print(f" {rev['original_precision']:>7.1%} "
                      f"{rev['filtered_precision']:>7.1%} "
                      f"{rev['precision_delta']:>+6.1%} "
                      f"{rev['removal_accuracy']:>6.1%}", end="")
            else:
                print(f" {'—':>8} {'—':>8} {'—':>7} {'—':>7}", end="")
        print()

    # Totals
    if len(results) > 1:
        tot_orig = sum(r["original_count"] for r in results)
        tot_filt = sum(r["filtered_count"] for r in results)
        tot_rmvd = sum(r["removed_count"] for r in results)
        print("-" * (35 + (32 if has_reviewed else 0)))
        print(f"{'TOTAL':<8} "
              f"{tot_orig:>6} "
              f"{tot_filt:>6} "
              f"{tot_rmvd:>6} "
              f"{tot_rmvd/tot_orig if tot_orig else 0:>6.1%}", end="")

        if has_reviewed:
            reviewed_results = [r for r in results if "reviewed" in r]
            if reviewed_results:
                avg_delta = sum(r["reviewed"]["precision_delta"] for r in reviewed_results) / len(reviewed_results)
                tot_rm_correct = sum(r["reviewed"]["removal_correct"] for r in reviewed_results)
                tot_rm_incorrect = sum(r["reviewed"]["removal_incorrect"] for r in reviewed_results)
                tot_rm = tot_rm_correct + tot_rm_incorrect
                avg_rm_acc = tot_rm_correct / tot_rm if tot_rm > 0 else 0
                print(f" {'':>8} {'':>8} {avg_delta:>+6.1%} {avg_rm_acc:>6.1%}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare original vs classifier-filtered predictions"
    )
    parser.add_argument("--grid-id", type=str, default=None)
    parser.add_argument("--grid-ids", nargs="+", default=None)
    parser.add_argument("--all-classified", action="store_true",
                        help="Process all grids that have cls_summary.json")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Save comparison to CSV")
    args = parser.parse_args()

    results_dir = args.results_dir or RESULTS_DIR

    # Determine grid list
    if args.grid_id:
        grid_ids = [args.grid_id]
    elif args.grid_ids:
        grid_ids = args.grid_ids
    elif args.all_classified:
        grid_ids = sorted([
            d.name for d in results_dir.iterdir()
            if d.is_dir() and (d / "cls_summary.json").exists()
        ])
    else:
        print("ERROR: Specify --grid-id, --grid-ids, or --all-classified")
        sys.exit(1)

    print(f"Comparing {len(grid_ids)} grid(s)...")

    results = []
    for gid in grid_ids:
        r = compare_grid(gid, results_dir)
        if r:
            results.append(r)

    if not results:
        print("No results to compare.")
        return

    print_comparison(results)

    # Save CSV
    output_csv = args.output_csv
    if output_csv is None and len(grid_ids) > 1:
        output_csv = results_dir / "cls_comparison.csv"

    if output_csv:
        rows = []
        for r in results:
            row = {
                "grid_id": r["grid_id"],
                "original_count": r["original_count"],
                "filtered_count": r["filtered_count"],
                "removed_count": r["removed_count"],
                "removal_rate": r["removal_rate"],
            }
            rev = r.get("reviewed", {})
            if rev:
                row["original_precision"] = rev["original_precision"]
                row["filtered_precision"] = rev["filtered_precision"]
                row["precision_delta"] = rev["precision_delta"]
                row["removal_accuracy"] = rev["removal_accuracy"]
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()
