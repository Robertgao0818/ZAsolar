#!/usr/bin/env python3
"""
Model-assisted annotation: export detection bboxes as QGIS hint layers.

Usage:
    python scripts/annotations/export_hints.py --grid-id G1238
    python scripts/annotations/export_hints.py --grid-id G1238 --model-path checkpoints/v2_sam2_260320/best_model.pth
    python scripts/annotations/export_hints.py --grid-id G1238 --skip-detect  # reuse existing predictions

Outputs (in results/<GridID>/hints/):
    hints_high.geojson    — confidence >= 0.95, bbox outlines (green in QGIS)
    hints_medium.geojson  — confidence 0.70-0.95, bbox outlines (yellow in QGIS)
    hints_all.geojson     — all predictions with confidence + tier fields

Load in QGIS → set style to "No Fill" + colored outline → see through to imagery.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def bbox_from_geometry(geom):
    """Convert a geometry to its bounding box polygon."""
    return box(*geom.bounds)


def export_hints(
    grid_id: str,
    model_path: str | None = None,
    skip_detect: bool = False,
    high_thresh: float = 0.95,
    low_thresh: float = 0.70,
) -> Path:
    from core.grid_utils import get_grid_paths

    paths = get_grid_paths(grid_id)
    hints_dir = paths.output_dir / "hints"
    hints_dir.mkdir(parents=True, exist_ok=True)

    # ── Get predictions ──────────────────────────────────────────────
    predictions_path = paths.output_dir / "predictions_metric.gpkg"

    if skip_detect and predictions_path.exists():
        print(f"[SKIP] Reusing existing predictions: {predictions_path}")
        pred = gpd.read_file(str(predictions_path))
    else:
        print(f"[DETECT] Running detection on {grid_id}...")
        # Import and run detection
        import detect_and_evaluate as det
        det.set_grid_context(grid_id)
        pred = det.detect_solar_panels(model_path=model_path)

    if "confidence" not in pred.columns:
        print("[ERROR] No confidence column in predictions")
        sys.exit(1)

    print(f"[INFO] {len(pred)} predictions, confidence range: "
          f"{pred['confidence'].min():.3f} - {pred['confidence'].max():.3f}")

    # ── Convert to bboxes ────────────────────────────────────────────
    pred_bbox = pred.copy()
    pred_bbox["geometry"] = pred_bbox.geometry.apply(bbox_from_geometry)

    # Add tier field
    pred_bbox["tier"] = "low"
    pred_bbox.loc[pred_bbox["confidence"] >= low_thresh, "tier"] = "medium"
    pred_bbox.loc[pred_bbox["confidence"] >= high_thresh, "tier"] = "high"

    # Keep useful columns only
    keep_cols = ["geometry", "confidence", "tier"]
    if "area_m2" in pred_bbox.columns:
        keep_cols.append("area_m2")
    if "source_tile" in pred_bbox.columns:
        keep_cols.append("source_tile")
    pred_bbox = pred_bbox[[c for c in keep_cols if c in pred_bbox.columns]]

    # ── Convert to WGS84 for QGIS ───────────────────────────────────
    pred_bbox = pred_bbox.to_crs("EPSG:4326")

    # ── Export layers ────────────────────────────────────────────────
    high = pred_bbox[pred_bbox["tier"] == "high"]
    medium = pred_bbox[pred_bbox["tier"] == "medium"]

    out_all = hints_dir / "hints_all.geojson"
    out_high = hints_dir / "hints_high.geojson"
    out_medium = hints_dir / "hints_medium.geojson"

    pred_bbox.to_file(str(out_all), driver="GeoJSON")
    if len(high) > 0:
        high.to_file(str(out_high), driver="GeoJSON")
    if len(medium) > 0:
        medium.to_file(str(out_medium), driver="GeoJSON")

    print(f"\n[DONE] Hint layers exported to {hints_dir}/")
    print(f"  hints_all.geojson    : {len(pred_bbox)} bboxes")
    print(f"  hints_high.geojson   : {len(high)} bboxes (conf >= {high_thresh})")
    print(f"  hints_medium.geojson : {len(medium)} bboxes (conf {low_thresh}-{high_thresh})")
    print(f"\nQGIS tips:")
    print(f"  1. Drag geojson into QGIS")
    print(f"  2. Layer Properties → Symbology → Simple Fill → Fill: No Brush")
    print(f"  3. Stroke color: green (high) / yellow (medium)")
    print(f"  4. Use SAM2 plugin to click inside each bbox")

    return hints_dir


def main():
    parser = argparse.ArgumentParser(description="Export detection hints for QGIS annotation")
    parser.add_argument("--grid-id", required=True, help="Grid ID (e.g. G1238)")
    parser.add_argument("--model-path", default=None, help="Custom model checkpoint path")
    parser.add_argument("--skip-detect", action="store_true",
                        help="Reuse existing predictions instead of re-running detection")
    parser.add_argument("--high-thresh", type=float, default=0.95,
                        help="Confidence threshold for 'high' tier (default: 0.95)")
    parser.add_argument("--low-thresh", type=float, default=0.70,
                        help="Confidence threshold for 'medium' tier (default: 0.70)")
    args = parser.parse_args()

    export_hints(
        grid_id=args.grid_id,
        model_path=args.model_path,
        skip_detect=args.skip_detect,
        high_thresh=args.high_thresh,
        low_thresh=args.low_thresh,
    )


if __name__ == "__main__":
    main()
