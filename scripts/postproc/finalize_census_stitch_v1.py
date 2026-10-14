#!/usr/bin/env python3
"""Write a finalized census layer with the stitching rule decided 2026-10-09 (葉).

Rule (same as stitch_census_v1.py): pieces that overlap or are within GAP metres
(default 0.25) are one block; block geometry = union, then +/-CLOSE (default 0.15 m)
mitre closing. Unit: one contiguous panel patch.

Attributes: every input column is carried from the highest-scoring member (the
same convention the delivered IoU>0.1 merge used), area_m2 is recomputed on the
stitched geometry, and stitch fields are added: block_id, n_pieces, n_merged
(=n_pieces, for schema parity with *_merged_iou010), score_max,
score_mean_area_weighted, cell_ids, n_cells, raw_area_sum_m2, max_side_m,
n_parts, stitch_rule.

The input is never modified; the output path must not exist.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stitch_census_v1 as sc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--cell-col", default="gridcell_id")
    ap.add_argument("--tile-col", default=None)
    ap.add_argument("--gap", type=float, default=0.25)
    ap.add_argument("--close-m", type=float, default=0.15)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if args.out.resolve().parent == args.input.resolve().parent and args.out.name == args.input.name:
        raise SystemExit("output equals input")

    gdf = gpd.read_file(args.input)
    if gdf.crs is None or not gdf.crs.is_projected:
        raise SystemExit(f"need a metric CRS, got {gdf.crs}")
    sc.CRS = gdf.crs  # module constant defaults to Durban EPSG:32736
    gdf = gdf.reset_index(drop=True)
    geoms = sc.repair_geometries(sc.as_geom_array(gdf.geometry.values))
    areas = np.asarray(gpd.GeoSeries(geoms, crs=gdf.crs).area, dtype=np.float64)
    scores = np.nan_to_num(pd.to_numeric(gdf[args.score_col], errors="coerce").to_numpy(np.float64), nan=0.0)
    cells = gdf[args.cell_col].fillna("").astype(str).to_numpy()
    tiles = gdf[args.tile_col].fillna("").astype(str).to_numpy() if args.tile_col else cells

    in_overlap, in_contained = sc.overlap_stats(geoms, areas)
    in_stats = {"n": int(len(gdf)), "area_m2": float(areas.sum()),
                "pairwise_overlap_m2": in_overlap, "n_gt50pct_contained": in_contained}
    sc.log(f"input: {in_stats}")

    blocks, stats, row_ids = sc.stitch(geoms, areas, scores, cells, tiles, rule=f"gap_{args.gap}",
                                       gap=args.gap, close_m=args.close_m, repair=True)
    # carry attributes from best-scoring member
    tmp = pd.DataFrame({"block_id": row_ids, "s": scores, "i": np.arange(len(gdf))})
    best = tmp.sort_values(["block_id", "s", "i"], ascending=[True, False, True]).drop_duplicates("block_id")
    attrs = gdf.drop(columns=gdf.geometry.name).iloc[best.i.to_numpy()].copy()
    attrs["block_id"] = best.block_id.to_numpy()
    attrs = attrs.drop(columns=[c for c in ("area_m2", "n_merged") if c in attrs.columns])
    out = blocks.rename(columns={"gridcell_ids": "cell_ids"}).merge(attrs, on="block_id", how="left", validate="1:1")
    out["n_merged"] = out["n_pieces"]
    out["stitch_rule"] = f"overlap_or_gap<={args.gap}m_close{args.close_m}m_v1"
    out = out.drop(columns=["n_tiles"] if not args.tile_col else [])
    lead = [c for c in gdf.columns if c != gdf.geometry.name and c != "n_merged"]
    extra = ["n_merged", "block_id", "n_pieces", "score_max", "score_mean_area_weighted", "cell_ids", "n_cells",
             "raw_area_sum_m2", "max_side_m", "n_parts", "stitch_rule"]
    cols = [c for c in lead if c in out.columns] + [c for c in extra if c in out.columns]
    out = gpd.GeoDataFrame(out[cols], geometry=blocks.geometry.values, crs=gdf.crs)
    if "area_m2" not in out.columns:
        out.insert(0, "area_m2", np.nan)
    out["area_m2"] = out.geometry.area
    if len(out) != stats["n_blocks"]:
        raise SystemExit("row count mismatch after attribute join")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(args.out, driver="GPKG", layer=args.out.stem)
    summary = {"input": str(args.input), "output": str(args.out), "input_stats": in_stats,
               "stitched_stats": stats, "gap_m": args.gap, "close_m": args.close_m}
    args.out.with_suffix(".stats.json").write_text(json.dumps(summary, indent=2, default=float))
    sc.log(f"wrote {args.out}: blocks={stats['n_blocks']} area={stats['total_area_m2']:.1f} "
           f"overlap={stats['residual_overlap_m2']:.1f} contained={stats['n_blocks_gt50pct_contained']}")


if __name__ == "__main__":
    main()
