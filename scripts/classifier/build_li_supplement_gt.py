"""
Build supplemented Li GT for the 25 JHB CBD grids.

Reads `nonpv_subtype_labeled.csv` (audit output of the V3-C ∩ V4.2
shared FP core), extracts every polygon labeled `actually_pv_mislabeled`,
joins it onto the per-grid Li GT, and writes the union to a new
annotation dir for cluster_level_eval reruns.

Each supplement polygon is added to its `grid_id` source GT file; a
polygon at the boundary between two grids will therefore appear in
both grids' GT (mirroring how V3-C/V4.2 detects it twice). This is the
correct behaviour for cluster_level_eval which evaluates per-grid.

Usage:
    python scripts/classifier/build_li_supplement_gt.py \\
        --labeled-csv data/cls_pv_nonpv_v3c_v42_cascade/labeler/v3c__both/nonpv_subtype_labeled.csv \\
        --li-gt-dir /mnt/d/ZAsolar/annotations_inbox/Joburg_CBD_Li \\
        --v3c-results results/johannesburg/v3c_sam_mask_geid_2024_02 \\
        --output-dir data/annotations/Joburg_CBD_Li_supp_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

LI_GT_NAME_PATTERN = ["G*.gpkg"]


def find_li_gt(li_dir: Path, grid_id: str) -> Path | None:
    candidates = sorted(li_dir.glob(f"{grid_id}*.gpkg"))
    return candidates[0] if candidates else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labeled-csv", type=Path, required=True)
    p.add_argument("--li-gt-dir", type=Path, required=True)
    p.add_argument(
        "--v3c-results", type=Path,
        default=Path("results/johannesburg/v3c_sam_mask_geid_2024_02"),
        help="V3-C SAM mask+box results dir (used to look up polygons by pred_idx)",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--metric-crs", default="EPSG:32735")
    p.add_argument(
        "--mislabel-tag", default="actually_pv_mislabeled",
        help="human_label value to extract as supplement",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Read {args.labeled_csv}")
    labels = pd.read_csv(args.labeled_csv)
    supp = labels[labels["human_label"] == args.mislabel_tag].copy()
    print(f"  {len(supp)} polygons tagged '{args.mislabel_tag}'")
    print(f"  spans {supp['grid_id'].nunique()} grids")

    print(f"\n[2/3] Pull polygons from {args.v3c_results}")
    rows_by_grid: dict[str, list] = {}
    for grid_id, sub in supp.groupby("grid_id"):
        gpkg = args.v3c_results / grid_id / "predictions_metric.gpkg"
        if not gpkg.exists():
            print(f"  [WARN] {gpkg} missing — skipping {grid_id}")
            continue
        preds = gpd.read_file(gpkg)
        if str(preds.crs) != args.metric_crs:
            preds = preds.to_crs(args.metric_crs)
        idxs = sub["pred_idx"].astype(int).tolist()
        keep = preds.iloc[idxs].copy()
        keep["supp_chip_id"] = sub["chip_id"].values
        keep["supp_grid_id"] = grid_id
        keep["supp_source"] = "v3c_sam_mask_audit_2026-04-27"
        rows_by_grid[grid_id] = keep

    print(f"\n[3/3] Merge with Li GT and write to {args.output_dir}")
    merge_summary = []
    for grid_id, supp_gdf in rows_by_grid.items():
        li_path = find_li_gt(args.li_gt_dir, grid_id)
        if li_path is None:
            print(f"  [WARN] no Li GT for {grid_id} — writing supp-only")
            li_gdf = gpd.GeoDataFrame(geometry=[], crs=args.metric_crs)
        else:
            li_gdf = gpd.read_file(li_path)
            if str(li_gdf.crs) != args.metric_crs:
                li_gdf = li_gdf.to_crs(args.metric_crs)
            li_gdf["supp_source"] = "li_2026-04-12"

        # Keep only geometry + provenance columns to avoid schema clashes
        li_keep = li_gdf[["geometry"]].copy()
        li_keep["supp_source"] = "li_2026-04-12"
        supp_keep = supp_gdf[["geometry"]].copy()
        supp_keep["supp_source"] = "v3c_sam_mask_audit_2026-04-27"

        merged = gpd.GeoDataFrame(
            pd.concat([li_keep, supp_keep], ignore_index=True),
            geometry="geometry", crs=args.metric_crs,
        )
        out_path = args.output_dir / f"{grid_id}.gpkg"
        merged.to_file(out_path, driver="GPKG", layer="annotations")
        merge_summary.append({
            "grid_id": grid_id,
            "li_n": len(li_keep),
            "supp_n": len(supp_keep),
            "merged_n": len(merged),
            "path": str(out_path.relative_to(Path.cwd())) if out_path.is_relative_to(Path.cwd()) else str(out_path),
        })
        print(f"  {grid_id}: Li {len(li_keep):>3} + supp {len(supp_keep):>2} = {len(merged):>3} → {out_path.name}")

    # Also copy unaltered grids (those with no supplements) so cluster_eval
    # picks up all 25 grids from a single annotation dir
    grids_with_supp = set(rows_by_grid.keys())
    for li_path in sorted(args.li_gt_dir.glob("G*.gpkg")):
        import re
        m = re.match(r"(G\d{4}|JHB\d{2})", li_path.name)
        if not m:
            continue
        gid = m.group(1)
        if gid in grids_with_supp:
            continue
        li_gdf = gpd.read_file(li_path)
        if str(li_gdf.crs) != args.metric_crs:
            li_gdf = li_gdf.to_crs(args.metric_crs)
        out = args.output_dir / f"{gid}.gpkg"
        li_keep = li_gdf[["geometry"]].copy()
        li_keep["supp_source"] = "li_2026-04-12"
        gpd.GeoDataFrame(li_keep, geometry="geometry", crs=args.metric_crs).to_file(
            out, driver="GPKG", layer="annotations"
        )
        merge_summary.append({
            "grid_id": gid,
            "li_n": len(li_gdf),
            "supp_n": 0,
            "merged_n": len(li_gdf),
            "path": str(out),
        })

    summary_path = args.output_dir / "_build_summary.json"
    summary_path.write_text(json.dumps({
        "labeled_csv": str(args.labeled_csv),
        "li_gt_dir": str(args.li_gt_dir),
        "v3c_results": str(args.v3c_results),
        "mislabel_tag": args.mislabel_tag,
        "n_supplement_polygons": int(len(supp)),
        "grids": merge_summary,
    }, indent=2))
    print(f"\nSummary: {summary_path}")
    print(f"Total supplement polygons added: {len(supp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
