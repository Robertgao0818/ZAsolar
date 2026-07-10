"""Convert 25 Sandton V4 reviewed.gpkg files to annotation source files.

Conversion rule (matches existing CBD batch1 sources at
data/annotations/Joburg/G0xxx_V4_260407.gpkg):

  Keep:
    - review_status == 'correct' AND source is null  (original V4 pred accepted)
    - source == 'sam_fn_review'                       (SAM-recut FN additions)
  Drop:
    - review_status in {edit, delete, unreviewed}
  Rewrite:
    - All kept rows get review_status = 'correct'

Output: data/annotations/Joburg/G1xxx_V4_260421.gpkg (25 files)
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "johannesburg" / "v4_aerial_2023"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations" / "Joburg"
EXPORT_DATE = "260421"

SANDTON_GRIDS = [
    "G1110", "G1111", "G1112", "G1113", "G1114",
    "G1144", "G1145", "G1146", "G1147", "G1148",
    "G1179", "G1180", "G1181", "G1182", "G1183",
    "G1214", "G1215", "G1216", "G1217", "G1218",
    "G1250", "G1251", "G1252", "G1253", "G1254",
]


def convert(grid_id: str) -> tuple[Path, int, int, int]:
    """Convert one reviewed.gpkg → annotation source. Returns (path, kept, orig_correct, fn_added)."""
    src = RESULTS_ROOT / grid_id / "review" / f"{grid_id}_reviewed.gpkg"
    dst = ANNOTATIONS_DIR / f"{grid_id}_V4_{EXPORT_DATE}.gpkg"

    gdf = gpd.read_file(src)
    if "source" not in gdf.columns:
        gdf["source"] = None

    is_added = gdf["source"] == "sam_fn_review"
    is_orig_correct = (gdf["review_status"] == "correct") & gdf["source"].isna()

    kept_mask = is_orig_correct | is_added
    kept = gdf[kept_mask].copy()
    kept["review_status"] = "correct"

    kept.to_file(dst, driver="GPKG")
    return dst, len(kept), int(is_orig_correct.sum()), int(is_added.sum())


def main() -> None:
    if not RESULTS_ROOT.exists():
        raise SystemExit(f"results root not found: {RESULTS_ROOT}")
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    total_kept = total_orig = total_added = 0
    print(f"Exporting {len(SANDTON_GRIDS)} Sandton grids → {ANNOTATIONS_DIR}/G1xxx_V4_{EXPORT_DATE}.gpkg")
    print()
    for grid in SANDTON_GRIDS:
        dst, kept, orig, added = convert(grid)
        summary.append((grid, kept, orig, added))
        total_kept += kept
        total_orig += orig
        total_added += added
        print(f"  {grid}: kept={kept:4d} (orig_correct={orig:4d} + sam_fn_review={added:4d}) → {dst.name}")

    print()
    print(f"TOTAL: kept={total_kept}  orig_correct={total_orig}  sam_fn_review={total_added}")
    print()
    print("YAML entries (paste into configs/datasets/regions.yaml under regions.johannesburg.grids):")
    for grid, kept, _, _ in summary:
        print(f'      {grid}: {{ annotation_source: "data/annotations/Joburg/{grid}_V4_{EXPORT_DATE}.gpkg", annotation_count: {kept}, notes: "Sandton batch2 V4-reviewed (aerial_2023)" }}')


if __name__ == "__main__":
    main()
