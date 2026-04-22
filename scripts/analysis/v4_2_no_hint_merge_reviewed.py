"""Merge reviewed no-hint predictions into Sandton annotation sources.

After review_detections.py produces <grid>_reviewed.gpkg in
  results/analysis/v4_2_no_hint_staged/<grid>/review/<grid>_reviewed.gpkg

this script:
  1. Filters rows to keep (review_status='correct' AND source IS NULL)
     OR (source='sam_fn_review')  — same rule as export_jhb_sandton_annotations.py
  2. Normalizes to the 25-column annotation source schema
  3. Merges with existing data/annotations/Joburg/<grid>_V4_260421.gpkg
  4. Writes data/annotations/Joburg/<grid>_V4_260421_plus_v4_2_augment.gpkg
     (does NOT overwrite the original; user promotes manually after sanity check)

Also prints a summary CSV at
  results/analysis/v4_2_no_hint_merge_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
STAGE_DIR = PROJECT / "results/analysis/v4_2_no_hint_staged"
ANNOT_DIR = PROJECT / "data/annotations/Joburg"
SUMMARY_CSV = PROJECT / "results/analysis/v4_2_no_hint_merge_summary.csv"

SANDTON = [
    "G1110", "G1111", "G1112", "G1113", "G1114",
    "G1144", "G1145", "G1146", "G1147", "G1148",
    "G1179", "G1180", "G1181", "G1182", "G1183",
    "G1214", "G1215", "G1216", "G1217", "G1218",
    "G1250", "G1251", "G1252", "G1253", "G1254",
]


def filter_reviewed(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    status = gdf["review_status"].astype(str).str.strip() if "review_status" in gdf.columns else pd.Series([""] * len(gdf))
    source = gdf["source"].astype(str).str.strip() if "source" in gdf.columns else pd.Series([""] * len(gdf))
    keep_orig = (status == "correct") & (source.isin(["", "nan", "None"]))
    keep_sam = source == "sam_fn_review"
    return gdf[keep_orig | keep_sam].reset_index(drop=True)


def normalize_to_source_schema(new_rows: gpd.GeoDataFrame, template: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cols = list(template.columns)
    out = gpd.GeoDataFrame(geometry=new_rows.geometry.values, crs=template.crs)
    for c in cols:
        if c == "geometry":
            continue
        if c in new_rows.columns:
            out[c] = new_rows[c].values
        else:
            out[c] = None
    if "review_status" in cols:
        out["review_status"] = "correct"
    return out[cols]


def merge_grid(grid: str) -> dict:
    reviewed_path = STAGE_DIR / grid / "review" / f"{grid}_reviewed.gpkg"
    source_path = ANNOT_DIR / f"{grid}_V4_260421.gpkg"
    out_path = ANNOT_DIR / f"{grid}_V4_260421_plus_v4_2_augment.gpkg"

    row = {"grid": grid, "reviewed_exists": reviewed_path.exists(),
           "source_exists": source_path.exists(),
           "n_reviewed": 0, "n_kept": 0, "n_existing": 0, "n_final": 0}

    if not reviewed_path.exists() or not source_path.exists():
        return row

    reviewed = gpd.read_file(reviewed_path)
    source = gpd.read_file(source_path)
    row["n_reviewed"] = len(reviewed)
    row["n_existing"] = len(source)

    kept = filter_reviewed(reviewed)
    if reviewed.crs is not None and source.crs is not None and reviewed.crs != source.crs:
        kept = kept.to_crs(source.crs)
    row["n_kept"] = len(kept)
    if len(kept) == 0:
        source.to_file(out_path, driver="GPKG")
        row["n_final"] = len(source)
        return row

    new_rows = normalize_to_source_schema(kept, source)
    merged = gpd.GeoDataFrame(pd.concat([source, new_rows], ignore_index=True), crs=source.crs)
    merged.to_file(out_path, driver="GPKG")
    row["n_final"] = len(merged)
    return row


def main() -> None:
    rows = [merge_grid(g) for g in SANDTON]
    df = pd.DataFrame(rows)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)

    print("=== Merge summary (25 Sandton grids) ===")
    print(df.to_string(index=False))
    print()
    tot_kept = df["n_kept"].sum()
    tot_existing = df["n_existing"].sum()
    tot_final = df["n_final"].sum()
    print(f"TOTAL new correct rows kept: {tot_kept}")
    print(f"TOTAL existing GT rows:      {tot_existing}")
    print(f"TOTAL final rows:            {tot_final}  (delta +{tot_final - tot_existing})")
    print(f"[write] {SUMMARY_CSV}")
    print(f"[write] {ANNOT_DIR}/<grid>_V4_260421_plus_v4_2_augment.gpkg")
    print()
    print("To promote: manually replace <grid>_V4_260421.gpkg with the _plus_v4_2_augment.gpkg")
    print("then bump annotation_count in configs/datasets/regions.yaml.")


if __name__ == "__main__":
    main()
