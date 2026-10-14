#!/usr/bin/env python3
"""Compute Horvitz-Thompson stratified precision from Ch1 review decisions.

Reads `<sample_root>/<grid>/predictions_metric.gpkg` (the sampled subset, with
`stratum_pop` / `sample_weight` per row) and the matching review decisions CSV
written by `review_detections.py` at `<sample_root>/<grid>/review/detection_review_decisions.csv`.

Review CSV schema (from review_detections.py):
    pred_id,status,updated_at
where `pred_id` is the row index in the sampled gpkg and `status` is one of
`correct` / `edit` (TP) or `delete` (FP). Empty / missing rows = unreviewed.

SAM-added polygons get pred_ids beyond the sample's row count and are dropped
here: they affect recall, not the model's raw precision.

Outputs:
  - Per-stratum HT-pooled precision (Wilson CI on raw n)
  - Per-grid HT precision
  - Overall pooled HT precision
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "results" / "johannesburg" / "v3c_vexcel_2024_ch1_sample"

TP_STATUSES = {"correct", "edit"}
FP_STATUSES = {"delete"}


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_grid(gdir: Path) -> pd.DataFrame | None:
    sample = gdir / "predictions_metric.gpkg"
    review = gdir / "review" / "detection_review_decisions.csv"
    if not sample.exists():
        return None
    if not review.exists():
        print(f"[SKIP] {gdir.name}: no review CSV")
        return None

    s = gpd.read_file(sample)
    s = s.reset_index(drop=False).rename(columns={"index": "pred_id"})
    s["pred_id"] = s["pred_id"].astype(int)

    r = pd.read_csv(review)
    if len(r) == 0 or "pred_id" not in r.columns:
        print(f"[SKIP] {gdir.name}: empty review CSV")
        return None
    r["pred_id"] = pd.to_numeric(r["pred_id"], errors="coerce").astype("Int64")
    r = r.dropna(subset=["pred_id"])
    r["pred_id"] = r["pred_id"].astype(int)
    # Drop SAM-added rows (pred_id outside sample range).
    r = r[r["pred_id"] < len(s)]
    # Drop empty status (unreviewed entries that were touched but not labelled).
    r = r[r["status"].isin(TP_STATUSES | FP_STATUSES)]
    if len(r) == 0:
        print(f"[SKIP] {gdir.name}: 0 valid decisions")
        return None

    merged = s.merge(r[["pred_id", "status"]], on="pred_id", how="inner")
    merged["is_tp"] = merged["status"].isin(TP_STATUSES).astype(int)
    merged["grid"] = gdir.name
    return merged


def ht_precision(df: pd.DataFrame) -> tuple[float, float, int]:
    """Return (precision, ci_lo, ci_hi) using HT weights and Wilson CI on raw n."""
    if len(df) == 0:
        return (0.0, 0.0, 0)
    w_tp = (df["is_tp"] * df["sample_weight"]).sum()
    w_n = df["sample_weight"].sum()
    p = w_tp / w_n if w_n else 0.0
    return p, *wilson(p, len(df))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Optional path to dump per-grid + per-stratum precision table")
    args = ap.parse_args()

    parts = []
    for gdir in sorted(args.root.iterdir()):
        if not gdir.is_dir():
            continue
        m = load_grid(gdir)
        if m is not None:
            parts.append(m)

    if not parts:
        print("No reviewed grids found.")
        return
    df = pd.concat(parts, ignore_index=True)
    n_grids = df["grid"].nunique()
    print(f"Total reviewed: {len(df)} predictions across {n_grids} grids\n")

    print("=== Pooled HT-weighted precision per stratum (size × conf) ===")
    stratum_rows = []
    for (sb, cb), g in df.groupby(["size_bucket", "conf_bucket"], observed=True):
        p, lo, hi = ht_precision(g)
        n = len(g)
        print(f"  {str(sb):>5s} × {str(cb):>5s}  n={n:>4d}  P={p:.3f}  CI95=[{lo:.2f},{hi:.2f}]")
        stratum_rows.append({"size_bucket": sb, "conf_bucket": cb, "n_reviewed": n,
                             "precision": p, "ci_lo": lo, "ci_hi": hi})

    print("\n=== Per-grid HT-weighted precision ===")
    grid_rows = []
    for grid, g in df.groupby("grid"):
        p, lo, hi = ht_precision(g)
        n = len(g)
        print(f"  {grid}  n={n:>3d}  P={p:.3f}  CI95=[{lo:.2f},{hi:.2f}]")
        grid_rows.append({"grid": grid, "n_reviewed": n,
                          "precision": p, "ci_lo": lo, "ci_hi": hi})

    p, lo, hi = ht_precision(df)
    print(f"\n=== Overall HT-weighted precision: P={p:.3f}  "
          f"CI95=[{lo:.2f},{hi:.2f}]  (n_reviewed={len(df)}, grids={n_grids}) ===")

    if args.out_csv:
        out = pd.concat([
            pd.DataFrame(stratum_rows).assign(level="stratum"),
            pd.DataFrame(grid_rows).assign(level="grid"),
            pd.DataFrame([{"level": "overall", "n_reviewed": len(df),
                           "precision": p, "ci_lo": lo, "ci_hi": hi}]),
        ], ignore_index=True)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
