#!/usr/bin/env python3
"""B1 TTA falsification pilot — step 1: anchor the 1.0x baseline missed set.

For each pilot grid, take the production-chain predictions
(unified_reviewall_A per-detection + SAM mask+box) filtered at the
production operating point (polygon confidence >= 0.925,
`project_jhb_production_model_2026-05-14`), match against locked clean_gt
with the same installation-profile semantics as Channel 2
(`iou_matching(merge_preds=True)` @ IoU 0.5), and export the UNMATCHED GT
polygons. These frozen missed sets are the denominator of the TTA pilot
kill bar (docs/handoffs/2026-06-10-f1-gap-tierB-agent-prompt.md, B1).

Outputs per grid under --output-dir:
  <grid>_missed_gt.gpkg     unmatched clean_gt polygons (all columns kept)
  baseline_summary.csv      per-grid n_gt / n_pred@op / matched / missed

Zero GPU. Run from repo root with the project venv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detect_and_evaluate import iou_matching  # noqa: E402

DEFAULT_GRIDS = ["G0925", "G0817", "G0816", "G0924", "G0889"]
DEFAULT_PRED_ROOT = (
    PROJECT_ROOT
    / "results/analysis/jhb_cbd25_3model_20260514/unified_A_perdet_sam_maskbox"
)
DEFAULT_GT_ROOT = PROJECT_ROOT / "data/annotations_channel2_clean"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grids", nargs="*", default=DEFAULT_GRIDS)
    ap.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    ap.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    ap.add_argument("--conf", type=float, default=0.925,
                    help="production polygon-confidence operating point")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--output-dir", type=Path,
                    default=PROJECT_ROOT / "results/analysis/tta_scale_probe/baseline")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for grid in args.grids:
        gt_path = args.gt_root / grid / f"{grid}_clean_gt.gpkg"
        pred_path = args.pred_root / grid / "predictions_metric.gpkg"
        if not gt_path.exists() or not pred_path.exists():
            print(f"[SKIP] {grid}: missing {gt_path if not gt_path.exists() else pred_path}")
            continue
        gt = gpd.read_file(gt_path).reset_index(drop=True)
        pred = gpd.read_file(pred_path)
        if gt.crs != pred.crs:
            pred = pred.to_crs(gt.crs)
        if "confidence" not in pred.columns:
            raise SystemExit(f"{pred_path} has no confidence column")
        pred_op = pred[pred["confidence"] >= args.conf].reset_index(drop=True)

        if len(pred_op) == 0:
            matched: set[int] = set()
        else:
            res = iou_matching(gt, pred_op, iou_threshold=args.iou, merge_preds=True)
            matched = set(res["matched_gt_indices"])
        missed = gt.loc[[i for i in range(len(gt)) if i not in matched]].copy()
        missed["gt_index"] = missed.index
        out = args.output_dir / f"{grid}_missed_gt.gpkg"
        if len(missed):
            missed.to_file(out, driver="GPKG")
        else:
            print(f"[WARN] {grid}: no missed polygons — nothing to probe")
        rows.append({
            "grid": grid,
            "n_gt": len(gt),
            "n_pred_total": len(pred),
            "n_pred_at_op": len(pred_op),
            "matched": len(matched),
            "missed": len(gt) - len(matched),
            "recall_at_op": round(len(matched) / len(gt), 4) if len(gt) else 0.0,
            "conf_op": args.conf,
            "iou": args.iou,
            "missed_gpkg": str(out),
        })
        print(f"[{grid}] gt={len(gt)} pred@{args.conf}={len(pred_op)} "
              f"matched={len(matched)} missed={len(gt) - len(matched)}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "baseline_summary.csv", index=False)
    print(f"\n[done] {args.output_dir}/baseline_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
