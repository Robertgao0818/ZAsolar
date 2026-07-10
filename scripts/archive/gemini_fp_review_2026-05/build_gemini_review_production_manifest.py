#!/usr/bin/env python3
"""Build a PRODUCTION Gemini-review candidate manifest from RAW predictions.

This is the production sibling of ``build_gemini_review_calibration_manifest.py``.
The calibration builder reads reviewed gpkgs (``<grid>_reviewed.gpkg``) to attach
an ``ra_label`` from ``review_status`` and stratified-samples for evaluation.
Production has no RA labels yet -- Gemini *is* the reviewer -- so this builder
scans the raw detector output (``predictions_metric.gpkg``) directly, filters to
a detector-confidence band, and emits one renderable candidate per surviving
prediction. No ``ra_label`` / ``review_status`` columns.

``pred_id`` is the prediction's POSITIONAL row index in the gpkg
(``reset_index(drop=True)``), matching the iloc contract that
``build_gemini_detection_review_chips.py`` (renderer) and
``apply_two_stage_decisions.py`` (applier) both rely on. ``candidate_id`` is
``{grid_id}_pred{pred_id:06d}``.

The renderer only *requires* ``grid_id`` + ``pred_id``; the remaining columns
(predictions_path, tile_path, source_tile, confidence, area_m2, imagery_layer,
model_run, results_root) are carried for provenance and to let the renderer skip
its own path resolution.

Confidence routing note: stage-2 skylight pass is only net-positive at
conf >= 0.95 (prelaunch 2026-05-31). Build the >=0.95 band for the two-stage
path; build a separate <0.95 band (``--min-conf 0.5 --max-conf 0.95``) for the
stage-1-only path. Keep the two bands in separate manifests / runs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from core import region_registry  # noqa: E402
from core.grid_utils import resolve_tiles_dir  # noqa: E402

MANIFEST_FIELDS = [
    "candidate_id",
    "grid_id",
    "pred_id",
    "region_key",
    "region",
    "predictions_path",
    "tile_path",
    "source_tile",
    "confidence",
    "area_m2",
    "imagery_layer",
    "model_run",
    "results_root",
]


def _grid_id_from_pred_path(path: Path) -> str:
    # .../<model_run>/<grid>/predictions_metric.gpkg
    return path.parent.name.upper()


def _resolve_tile_path(
    grid_id: str, region: str, imagery_layer: str | None, source_tile: str
) -> Path | None:
    try:
        tiles = resolve_tiles_dir(grid_id, region=region, imagery_layer=imagery_layer)
    except Exception:
        return None
    if tiles.is_file():  # mosaic layout
        return tiles
    if not source_tile:
        return None
    return tiles / f"{source_tile}.tif"


def collect_rows(
    gpkgs: list[Path],
    *,
    region: str,
    imagery_layer: str | None,
    min_conf: float,
    max_conf: float,
    require_tile: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = {
        "gpkgs_scanned": 0,
        "gpkgs_no_confidence": 0,
        "rows_total": 0,
        "rows_in_band": 0,
        "skipped_no_conf": 0,
        "skipped_tile_missing": 0,
    }
    for path in gpkgs:
        try:
            gdf = gpd.read_file(path)
        except Exception:
            continue
        if "confidence" not in gdf.columns:
            stats["gpkgs_no_confidence"] += 1
            continue
        stats["gpkgs_scanned"] += 1
        grid_id = _grid_id_from_pred_path(path)
        model_run = path.parent.parent.name
        results_root = str(path.parent.parent.resolve())
        gdf = gdf.reset_index(drop=True)  # positional index == pred_id for iloc
        stats["rows_total"] += len(gdf)
        for pred_id, row in gdf.iterrows():
            conf = row.get("confidence")
            if conf is None or (isinstance(conf, float) and math.isnan(conf)):
                stats["skipped_no_conf"] += 1
                continue
            conf = float(conf)
            if not (min_conf <= conf < max_conf):
                continue
            source_tile = str(row.get("source_tile") or "").strip()
            tile_path = _resolve_tile_path(grid_id, region, imagery_layer, source_tile)
            if require_tile and (tile_path is None or not tile_path.exists()):
                stats["skipped_tile_missing"] += 1
                continue
            stats["rows_in_band"] += 1
            area = row.get("area_m2")
            rows.append(
                {
                    "candidate_id": f"{grid_id}_pred{int(pred_id):06d}",
                    "grid_id": grid_id,
                    "pred_id": int(pred_id),
                    "region_key": region,
                    "region": region,
                    "predictions_path": str(path.resolve()),
                    "tile_path": "" if tile_path is None else str(tile_path.resolve()),
                    "source_tile": source_tile,
                    "confidence": f"{conf:.6f}",
                    "area_m2": ""
                    if area is None or (isinstance(area, float) and math.isnan(area))
                    else f"{float(area):.4f}",
                    "imagery_layer": imagery_layer or "",
                    "model_run": model_run,
                    "results_root": results_root,
                }
            )
    return rows, stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--predictions-glob",
        required=True,
        help="Glob for raw predictions gpkgs, e.g. "
        "'results/johannesburg/v3c_vexcel_2024_ch1_sample/G*/predictions_metric.gpkg'.",
    )
    ap.add_argument("--region", default="johannesburg")
    ap.add_argument("--imagery-layer", default=None, help="Imagery layer id for tile resolution (e.g. vexcel_2024).")
    ap.add_argument("--min-conf", type=float, default=0.95)
    ap.add_argument("--max-conf", type=float, default=1.0001)
    ap.add_argument(
        "--grids",
        nargs="+",
        default=None,
        help="Optional whitelist of grid ids; restrict the glob hits to these.",
    )
    ap.add_argument(
        "--allow-missing-tiles",
        action="store_true",
        help="Keep candidates whose tile cannot be resolved (renderer will fail on them). "
        "Default drops them so the manifest is fully renderable.",
    )
    ap.add_argument("--out-csv", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    region = region_registry.normalize_region_key(args.region) or args.region
    gpkgs = [Path(p) for p in sorted(glob.glob(args.predictions_glob))]
    if args.grids:
        keep = {g.upper() for g in args.grids}
        gpkgs = [p for p in gpkgs if _grid_id_from_pred_path(p) in keep]
    if not gpkgs:
        raise SystemExit(f"No predictions gpkgs matched: {args.predictions_glob}")

    rows, stats = collect_rows(
        gpkgs,
        region=region,
        imagery_layer=args.imagery_layer,
        min_conf=args.min_conf,
        max_conf=args.max_conf,
        require_tile=not args.allow_missing_tiles,
    )
    rows.sort(key=lambda r: (r["grid_id"], r["pred_id"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_grids = len({r["grid_id"] for r in rows})
    print(
        f"[OK] {len(rows)} candidates across {n_grids} grid(s) "
        f"(conf in [{args.min_conf}, {args.max_conf})) -> {args.out_csv}"
    )
    print(f"     stats: {stats}")


if __name__ == "__main__":
    main()
