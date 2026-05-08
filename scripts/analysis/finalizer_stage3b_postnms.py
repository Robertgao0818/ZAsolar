#!/usr/bin/env python3
"""Post-process Stage 3B hybrid outputs with spatial NMS to dedup
near-duplicate SAM polygons produced by clusters sharing an envelope.

The hybrid SAM call (envelope bbox + cluster mask) recovers V3C_TP recall
but ignores the cluster mask when the bbox is wide, so multiple clusters
in the same envelope output near-identical envelope-sized polygons. This
post-processor reads the existing ``stage3b_hybrid_*`` gpkgs from a
Stage 3B sweep directory, runs IoU-based NMS keyed on the SAM IoU score
recorded in ``cluster_assignments.csv``, recomputes overall + source
metrics, and **appends** new variants to the existing CSVs.

Usage:
    python scripts/analysis/finalizer_stage3b_postnms.py \\
        --grid-id G0817 --region jhb \\
        --stage3b-dir results/analysis/finalizer_mask_shaping_ablation/G0817/stage3b_sam \\
        --clean-gt data/annotations_channel2_clean/G0817/G0817_clean_gt.gpkg \\
        --nms-iou 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.grid_utils import get_metric_crs, normalize_grid_id  # noqa: E402

from scripts.analysis.finalizer_envelope_group_sweep import (  # noqa: E402
    _safe_iou,
    compute_overall,
    compute_source_metrics,
)


def nms_indices(
    geoms: list[BaseGeometry],
    scores: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Greedy NMS. Returns indices into `geoms` (kept in score-descending
    order). Drops any candidate whose IoU with an already-kept geom is
    >= iou_threshold."""
    n = len(geoms)
    if n == 0:
        return []
    order = np.argsort(-scores)  # descending
    kept_idx: list[int] = []
    kept_geoms: list[BaseGeometry] = []
    # Lazily index kept geoms with STRtree for quick rejection.
    for i in order:
        i = int(i)
        g = geoms[i]
        if g is None or g.is_empty:
            continue
        suppressed = False
        if kept_geoms:
            tree = STRtree(kept_geoms)
            for k in tree.query(g):
                if _safe_iou(g, kept_geoms[int(k)]) >= iou_threshold:
                    suppressed = True
                    break
        if not suppressed:
            kept_idx.append(i)
            kept_geoms.append(g)
    return kept_idx


def main() -> int:
    ap = argparse.ArgumentParser(__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-id", required=True)
    ap.add_argument("--region", default="jhb")
    ap.add_argument("--stage3b-dir", type=Path, required=True,
                    help="existing finalizer_stage3b_sam.py output directory")
    ap.add_argument("--clean-gt", type=Path, required=True)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    args = ap.parse_args()

    grid_id = normalize_grid_id(args.grid_id)
    metric_crs = str(get_metric_crs(grid_id, region=args.region))
    print(f"[postnms] grid={grid_id} crs={metric_crs} nms_iou={args.nms_iou}")

    cluster_csv = args.stage3b_dir / "cluster_assignments.csv"
    overall_csv = args.stage3b_dir / "overall_metrics.csv"
    source_csv = args.stage3b_dir / "source_metrics.csv"
    if not cluster_csv.exists():
        print(f"ERROR: {cluster_csv} not found", file=sys.stderr)
        return 1
    df_cluster = pd.read_csv(cluster_csv)
    if "sam_hy_area_m2" not in df_cluster.columns:
        print("ERROR: hybrid columns missing - run finalizer_stage3b_sam.py "
              "with --enable-hybrid first", file=sys.stderr)
        return 1

    clean_gt = gpd.read_file(args.clean_gt)
    if str(clean_gt.crs) != metric_crs:
        clean_gt = clean_gt.to_crs(metric_crs)
    sources = list(clean_gt["source"].unique()) if "source" in clean_gt.columns else []

    # Resolve geometries from the per-variant gpkgs. Order in the gpkg matches
    # cluster_id (variant_all/variant_multi iterates clusters in order).
    new_overall_rows = []
    new_source_rows = []

    for vname in ("stage3b_hybrid_all", "stage3b_hybrid_multi"):
        gpkg_path = args.stage3b_dir / vname / "predictions_metric.gpkg"
        if not gpkg_path.exists():
            print(f"[postnms]  skip {vname}: {gpkg_path} missing")
            continue
        gdf = gpd.read_file(gpkg_path)
        if str(gdf.crs) != metric_crs:
            gdf = gdf.to_crs(metric_crs)
        gdf = gdf.reset_index(drop=True)
        # NMS score: prefer SAM IoU score; for solo clusters in `multi` the
        # geom is the per-detection cluster.geom, not SAM output, so its
        # sam_hy_iou_score is still recorded but typically high (those rarely
        # need to suppress anything since per-det geoms are small).
        geoms = list(gdf.geometry.values)
        if vname == "stage3b_hybrid_multi":
            # In multi mode, solo clusters keep cluster.geom (no SAM). We use
            # cluster score as the NMS score for solo entries.
            score_col = np.where(
                df_cluster["n_members"].values >= 2,
                df_cluster["sam_hy_iou_score"].values,
                df_cluster["score"].values,
            ).astype(float)
        else:
            score_col = df_cluster["sam_hy_iou_score"].values.astype(float)
        if len(geoms) != len(score_col):
            print(f"[postnms]  WARN {vname}: geom count {len(geoms)} != "
                  f"cluster count {len(score_col)} — slicing to min")
            n = min(len(geoms), len(score_col))
            geoms = geoms[:n]
            score_col = score_col[:n]

        kept = nms_indices(geoms, score_col, args.nms_iou)
        kept_geoms = [geoms[i] for i in kept]
        out_name = f"{vname}_nms{args.nms_iou:.2f}".rstrip("0").rstrip(".")
        # write nms gpkg
        out_dir = args.stage3b_dir / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        if kept_geoms:
            out_gdf = gpd.GeoDataFrame({"geometry": kept_geoms},
                                        geometry="geometry", crs=metric_crs)
        else:
            out_gdf = gpd.GeoDataFrame({"geometry": []},
                                        geometry="geometry", crs=metric_crs)
        out_gdf.to_file(out_dir / "predictions_metric.gpkg", driver="GPKG")
        print(f"[postnms]  {vname}: {len(geoms)} -> {len(kept_geoms)} after NMS "
              f"(iou>={args.nms_iou})")

        # metrics
        ovr = compute_overall(kept_geoms, clean_gt)
        ovr["variant"] = out_name
        new_overall_rows.append(ovr)
        for src in sources:
            sub = clean_gt[clean_gt["source"] == src].reset_index(drop=True)
            srm = compute_source_metrics(kept_geoms, sub)
            srm["variant"] = out_name
            srm["source"] = src
            new_source_rows.append(srm)

    if not new_overall_rows:
        print("[postnms] no variants produced — exiting")
        return 0

    # append to existing CSVs (drop any pre-existing rows with the same variant
    # so the script is rerunnable)
    new_overall_df = pd.DataFrame(new_overall_rows)
    new_source_df = pd.DataFrame(new_source_rows)

    if overall_csv.exists():
        cur_overall = pd.read_csv(overall_csv)
        cur_overall = cur_overall[~cur_overall["variant"].isin(new_overall_df["variant"])]
        merged = pd.concat([cur_overall, new_overall_df[cur_overall.columns]],
                            ignore_index=True)
    else:
        merged = new_overall_df
    merged.to_csv(overall_csv, index=False)

    if source_csv.exists():
        cur_source = pd.read_csv(source_csv)
        cur_source = cur_source[~cur_source["variant"].isin(new_source_df["variant"])]
        merged_src = pd.concat([cur_source, new_source_df[cur_source.columns]],
                                ignore_index=True)
    else:
        merged_src = new_source_df
    merged_src.to_csv(source_csv, index=False)

    print(f"[postnms] appended {len(new_overall_rows)} variants to "
          f"{overall_csv.name} and {len(new_source_rows)} rows to {source_csv.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
