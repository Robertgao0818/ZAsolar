#!/usr/bin/env python3
"""Eval-only adapter: predictions_metric.gpkg → presence_metrics.csv etc.

Decouples evaluation from detection. Reuses the matching/scoring helpers
from `detect_and_evaluate.py` directly (no port; that file is frozen as the
authoritative reference for evaluator behavior). Only the path-resolution +
GT loading + per-tile glue is re-done so we don't need a fresh inference run.

Outputs match the eval-only subset of `detect_and_evaluate.py`:
  - presence_metrics.csv
  - evaluation_per_tile.csv
  - footprint_metrics.csv
  - area_error_metrics.csv
  - iou_threshold_metrics.csv (multi-IoU sweep)

Used by `scripts/analysis/run_benchmark.py --predictions-source direct`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# We import directly from the (frozen) script. This is intentional — see
# plan v1.4 decision #22: "Reuses the existing evaluator helpers from
# detect_and_evaluate.py ... import directly without moving."
import detect_and_evaluate as dae


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate_predictions.py",
        description="Run evaluation on an existing predictions_metric.gpkg.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--predictions-gpkg", required=True, type=Path,
                   help="path to predictions_metric.gpkg")
    p.add_argument("--region", required=True)
    p.add_argument("--grid-id", required=True)
    p.add_argument("--imagery-layer", default=None)
    p.add_argument("--model-run", default=None)
    p.add_argument("--output-dir", required=True, type=Path,
                   help="where presence_metrics.csv etc. land")
    p.add_argument("--evaluation-profile", choices=["installation", "legacy_instance"],
                   default="installation",
                   help="installation = pred-side many-to-one merge (V1.3); "
                        "legacy_instance = strict 1:1")
    p.add_argument("--iou-threshold", type=float, default=0.3,
                   help="IoU threshold for primary metrics")
    return p


def _merge_mode_label(predictions_gpkg: Path) -> str | None:
    """从预测同目录的 config.json 读 finalize 层 merge_mode(provenance)。

    finalize.py 把 merge_mode 写进 config.json;legacy 管线没有该键 → None。
    """
    cfg_path = Path(predictions_gpkg).parent / "config.json"
    if not cfg_path.exists():
        return None
    try:
        import json
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for container in (payload, payload.get("config") or {},
                      payload.get("postproc") or {}):
        if isinstance(container, dict) and container.get("merge_mode"):
            return str(container["merge_mode"])
    return None


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Set the globals detect_and_evaluate expects ──────────────────
    dae.set_grid_context(
        grid_id=args.grid_id,
        region=args.region,
        imagery_layer=args.imagery_layer,
        model_run=args.model_run,
    )

    # ── Load GT ──────────────────────────────────────────────────────
    gt = dae.load_ground_truth()
    print(f"[eval] GT: {len(gt)} polygons")

    # ── Load predictions ─────────────────────────────────────────────
    pred = gpd.read_file(args.predictions_gpkg)
    pred = dae.to_metric_crs(pred, assumed_crs=str(pred.crs), label="predictions")
    print(f"[eval] predictions: {len(pred)} polygons (CRS={pred.crs})")

    if len(pred) == 0:
        # Write empty presence + tile metrics so downstream parsers don't crash.
        pd.DataFrame([{
            "grid_id": args.grid_id, "gt_count": len(gt),
            "pred_count": 0, "tp": 0, "fp": 0, "fn": len(gt),
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }]).to_csv(args.output_dir / "presence_metrics.csv", index=False)
        print(f"[eval] empty predictions → wrote zero-row metrics")
        return 0

    merge_preds = (args.evaluation_profile == "installation")

    # ── Primary IoU matching at the chosen threshold ─────────────────
    matching = dae.iou_matching(
        gt, pred,
        iou_threshold=args.iou_threshold,
        merge_preds=merge_preds,
        return_match_details=True,
    )

    # ── Presence (single-row 主口径 + dual-caliber 副表) ─────────────
    # 副口径 = {0.1, 0.3} 中非主口径者(evaluation_protocol.md §1.1)。
    secondary = []
    for cal in (0.1, 0.3):
        if abs(cal - args.iou_threshold) > 1e-9:
            secondary.append((cal, dae.iou_matching(
                gt, pred, iou_threshold=cal, merge_preds=merge_preds)))
    dae.evaluate_presence(
        matching, args.grid_id, args.output_dir,
        iou_caliber=args.iou_threshold,
        eval_profile=args.evaluation_profile,
        merge_mode=_merge_mode_label(args.predictions_gpkg),
        secondary=secondary,
    )

    # ── Footprint (per-match IoU/Dice summary) ───────────────────────
    dae.evaluate_footprint(matching, args.output_dir)

    # ── Area error bucketed by GT size ───────────────────────────────
    dae.evaluate_area_error(matching, gt, args.output_dir)

    # ── Multi-IoU sweep ──────────────────────────────────────────────
    iou_df = dae.evaluate_at_multiple_thresholds(gt, pred, merge_preds=merge_preds)
    iou_df.to_csv(args.output_dir / "iou_threshold_metrics.csv", index=False)
    print(f"[eval] iou_threshold_metrics.csv ({len(iou_df)} rows)")

    # ── Per-tile breakdown (uses TILES_DIR set by set_grid_context) ──
    try:
        tile_df = dae.evaluate_per_tile(gt, pred)
        tile_df.to_csv(args.output_dir / "evaluation_per_tile.csv", index=False)
        print(f"[eval] evaluation_per_tile.csv ({len(tile_df)} rows)")
    except Exception as e:
        print(f"[WARN] evaluate_per_tile failed (likely no tile dir on this machine): {e}")

    # ── Size-stratified ──────────────────────────────────────────────
    try:
        size_df = dae.evaluate_by_size(gt, pred)
        size_df.to_csv(args.output_dir / "size_stratified_metrics.csv", index=False)
        print(f"[eval] size_stratified_metrics.csv ({len(size_df)} rows)")
    except Exception as e:
        print(f"[WARN] evaluate_by_size failed: {e}")

    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
