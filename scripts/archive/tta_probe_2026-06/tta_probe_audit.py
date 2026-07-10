#!/usr/bin/env python3
"""B1 TTA falsification pilot — step 3: measure scale-view conversion.

For every missed clean_gt polygon (anchored by tta_probe_baseline.py at the
production operating point), compare raw detector proposals from each view
(1.0x baseline, 1.5x, 2.0x upscale) and measure:

  (i)  existence: any raw proposal at conf >= 0.3 intersecting the GT
       polygon in any magnified view (secondary metric — existence is
       already 97-100% at 1.0x per exp_finalizer raw-hint audit);
  (ii) PRIMARY: per-missed-polygon best-proposal mask IoU, reported at
       mask binarization 0.3 and 0.5, with the kill-bar statistic =
       fraction of missed polygons whose best magnified-view proposal
       reaches IoU >= 0.5 at conf >= 0.3.

Pre-registered kill bar (do not move): < 10-15% conversion => abandon TTA.

Methodology reuses raw_hint_audit._raw_sets_from_artifact (exp_finalizer
lineage): boxes + per-detection vectorized masks from raw_detections.pkl,
reprojected into the clean_gt metric CRS. No merging, no inverse-transform
cascade — raw proposals only.

Inputs:
  --baseline-dir   output of tta_probe_baseline.py (<grid>_missed_gt.gpkg)
  --view-root      dir with <view>/<grid>/raw_detections.pkl, views named
                   x10, x15, x20

Outputs (to --output-dir):
  per_polygon.csv  one row per missed GT x view x mask-threshold
  summary.csv      per-grid + pooled conversion fractions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.raw_hint_audit import _raw_sets_from_artifact  # noqa: E402

VIEWS = ["x10", "x15", "x20"]
MAG_VIEWS = ["x15", "x20"]
SCORE_FLOOR_PRIMARY = 0.3
SCORE_FLOOR_DIAG = 0.05
MASK_THRESHOLDS = [0.3, 0.5]


def best_iou_per_gt(gt_geoms, geom_set, score_floor: float) -> np.ndarray:
    """Best proposal IoU per GT geometry, proposals filtered by score."""
    keep = [i for i, s in enumerate(geom_set.scores) if s >= score_floor]
    geoms = [geom_set.geoms[i] for i in keep]
    out = np.zeros(len(gt_geoms), dtype=float)
    if not geoms:
        return out
    tree = STRtree(geoms)
    for gi, gt_geom in enumerate(gt_geoms):
        best = 0.0
        for idx in tree.query(gt_geom):
            g = geoms[int(idx)]
            inter = g.intersection(gt_geom).area
            if inter <= 0:
                continue
            union = g.area + gt_geom.area - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best:
                best = iou
        out[gi] = best
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-dir", type=Path,
                    default=PROJECT_ROOT / "results/analysis/tta_scale_probe/baseline")
    ap.add_argument("--view-root", type=Path, required=True,
                    help="dir containing x10/x15/x20/<grid>/raw_detections.pkl")
    ap.add_argument("--grids", nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path,
                    default=PROJECT_ROOT / "results/analysis/tta_scale_probe/audit")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for grid in args.grids:
        missed_path = args.baseline_dir / f"{grid}_missed_gt.gpkg"
        if not missed_path.exists():
            print(f"[SKIP] {grid}: no missed gpkg")
            continue
        missed = gpd.read_file(missed_path)
        gt_geoms = list(missed.geometry.values)
        dst_crs = missed.crs
        print(f"[{grid}] {len(gt_geoms)} missed polygons")

        for view in VIEWS:
            raw_path = args.view_root / view / grid / "raw_detections.pkl"
            if not raw_path.exists():
                print(f"  [WARN] missing artifact {raw_path}")
                continue
            for mt in MASK_THRESHOLDS:
                box_set, mask_set, _thr = _raw_sets_from_artifact(
                    raw_path, dst_crs, mask_threshold=mt)
                for floor in (SCORE_FLOOR_PRIMARY, SCORE_FLOOR_DIAG):
                    box_iou = best_iou_per_gt(gt_geoms, box_set, floor)
                    mask_iou = best_iou_per_gt(gt_geoms, mask_set, floor)
                    for gi in range(len(gt_geoms)):
                        rows.append({
                            "grid": grid,
                            "gt_index": int(missed.iloc[gi].get("gt_index", gi)),
                            "gt_area_m2": float(missed.iloc[gi].get(
                                "area_m2", gt_geoms[gi].area)),
                            "gt_source": missed.iloc[gi].get("source", ""),
                            "view": view,
                            "mask_threshold": mt,
                            "score_floor": floor,
                            "best_box_iou": round(float(box_iou[gi]), 4),
                            "best_mask_iou": round(float(mask_iou[gi]), 4),
                        })
                print(f"  [{view}] mt={mt}: boxes={len(box_set.geoms)} "
                      f"masks={len(mask_set.geoms)}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "per_polygon.csv", index=False)

    # --- summary: kill-bar statistics ---
    summaries = []
    prim = df[(df.score_floor == SCORE_FLOOR_PRIMARY)]
    for mt in MASK_THRESHOLDS:
        d = prim[prim.mask_threshold == mt]
        if d.empty:
            continue
        piv_mask = d.pivot_table(index=["grid", "gt_index"], columns="view",
                                 values="best_mask_iou", aggfunc="max")
        piv_box = d.pivot_table(index=["grid", "gt_index"], columns="view",
                                values="best_box_iou", aggfunc="max")
        have_mag = [v for v in MAG_VIEWS if v in piv_mask.columns]
        mag_best = piv_mask[have_mag].max(axis=1) if have_mag else pd.Series(0, index=piv_mask.index)
        base = piv_mask["x10"] if "x10" in piv_mask.columns else pd.Series(0.0, index=piv_mask.index)
        mag_any_floor = (
            piv_box[have_mag].max(axis=1).combine(mag_best, max)
            if have_mag else mag_best
        )

        def frac(s):
            return round(float(s.mean()), 4) if len(s) else 0.0

        grids = sorted({g for g, _ in piv_mask.index})
        for scope, sel in [("ALL", piv_mask.index)] + [
                (g, [ix for ix in piv_mask.index if ix[0] == g]) for g in grids]:
            mb = mag_best.loc[sel]
            bb = base.loc[sel]
            summaries.append({
                "scope": scope,
                "mask_threshold": mt,
                "n_missed": len(mb),
                "existence_mag_conf03": frac(mag_any_floor.loc[sel] > 0),
                "conv_iou05_mag": frac(mb >= 0.5),          # KILL-BAR STAT
                "conv_iou03_mag": frac(mb >= 0.3),
                "conv_iou05_x10": frac(bb >= 0.5),
                "conv_iou05_mag_incremental": frac((mb >= 0.5) & (bb < 0.5)),
                "median_best_mag_iou": round(float(mb.median()), 4) if len(mb) else 0.0,
                "p90_best_mag_iou": round(float(mb.quantile(0.9)), 4) if len(mb) else 0.0,
            })
    sdf = pd.DataFrame(summaries)
    sdf.to_csv(args.output_dir / "summary.csv", index=False)
    print("\n=== KILL-BAR SUMMARY (score>=0.3) ===")
    print(sdf[sdf.scope == "ALL"].to_string(index=False))
    print(f"\n[done] {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
