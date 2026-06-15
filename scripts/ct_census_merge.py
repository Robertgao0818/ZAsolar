#!/usr/bin/env python3
"""Merge per-grid cls-filtered predictions into one CT census inventory.

Census is addressed by CPT id end-to-end, so results live under CPT-keyed dirs
(``<results-dir>/<CPT>/predictions_metric_cls_filtered.gpkg``) and the merged
inventory is already CPT-native — no G<->CPT relabel. Each row is tagged with
its source ``gridcell_id`` (the CPT cell) + provenance. Output CRS = the region
metric CRS the predictions are already in (EPSG:32734).

Tolerates grids with zero detections (missing/empty filtered gpkg).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import geopandas as gpd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--glist", required=True, type=Path)
    ap.add_argument("--state", required=True, type=Path)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--layer-name",
        default="predictions_metric_cls_filtered.gpkg",
        help="per-grid file to merge (default: cls-filtered)",
    )
    args = ap.parse_args()

    grids = [g.strip() for g in args.glist.read_text().splitlines() if g.strip()]

    frames: list[gpd.GeoDataFrame] = []
    n_with_dets = n_empty = n_missing = 0
    crs = None
    for g in grids:
        if not (args.state / f"infer_{g}.ok").exists():
            continue
        fp = args.results_dir / g / args.layer_name
        if not fp.exists():
            n_missing += 1
            continue
        try:
            gdf = gpd.read_file(fp)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] read failed {g}: {e}")
            n_missing += 1
            continue
        if len(gdf) == 0:
            n_empty += 1
            continue
        crs = crs or gdf.crs
        gdf = gdf.to_crs(crs)
        gdf["gridcell_id"] = g           # CPT census cell
        gdf["region"] = "cape_town"
        gdf["model_run"] = args.run
        gdf["imagery_layer"] = "aerial_2025"
        frames.append(gdf)
        n_with_dets += 1

    if not frames:
        print("[ERROR] no detections merged — wrote nothing")
        return

    merged = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(str(args.out), driver="GPKG")

    print(
        f"[DONE] {len(merged)} polygons from {n_with_dets} grids "
        f"(empty={n_empty}, missing={n_missing}) -> {args.out}  CRS={crs}"
    )


if __name__ == "__main__":
    main()
