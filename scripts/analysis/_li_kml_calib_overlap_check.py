"""Physical-overlap / leakage audit for the Li KML batch as a calibration set.

Context (evaluation_protocol.md §2.5, option (b), 2026-06-11): the 57 KML
L-grids (L0208..L1841) are candidates for the unified_A × cape_town/aerial_2025
operating-point calibration set.  The namespace argument (memory
project_li_grid_namespace) only establishes that Li G-numbers must not be
*ID-matched* against Gao grids; it does NOT establish physical disjointness.
This script provides that evidence:

  For each staged KML-batch L<NNNN>.gpkg, compute the panel-union bbox in
  EPSG:32734, expand it by one full Li cell size in every direction (panels
  lie inside their cell, so the true cell is contained in the expanded box —
  a conservative superset), then measure intersection / distance against:

    a) every Gao-annotated CT grid cell (annotated = *.gpkg under
       data/annotations/Capetown — superset of unified_A's CT training grids
       AND of the G-side reporting suites independent_26 / t1_smoke), and
    b) the Li held-out reporting cells (task_grid_li.gpkg, true KML geometry).

  A candidate is leakage-clean iff its expanded box intersects neither (a)
  nor (b).  Distances are reported so near-misses are visible.

Output: results/analysis/operating_point_lock/li_kml_calib_split/overlap_audit.csv
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parents[2]
LI_DIR = REPO_ROOT / "data" / "annotations" / "Capetown_Li"
QA_CSV = LI_DIR / "_kml_batch_qa.csv"
OUT_DIR = (
    REPO_ROOT / "results" / "analysis" / "operating_point_lock" / "li_kml_calib_split"
)
METRIC_CRS = "EPSG:32734"


def main() -> None:
    batch = pd.read_csv(QA_CSV)["Lid"].tolist()

    gao = gpd.read_file(REPO_ROOT / "data" / "task_grid.gpkg").to_crs(METRIC_CRS)
    annotated_ids = sorted(
        {
            m.group(1)
            for p in (REPO_ROOT / "data" / "annotations" / "Capetown").glob("*.gpkg")
            if (m := re.match(r"(G\d{4})", p.name))
        }
    )
    gao_annot = gao[gao["gridcell_id"].isin(annotated_ids)]
    missing = set(annotated_ids) - set(gao_annot["gridcell_id"])
    if missing:
        raise SystemExit(f"annotated grids missing from task_grid.gpkg: {missing}")
    gao_union = gao_annot.union_all()
    gao_sidx = gao_annot.sindex

    li_cells = gpd.read_file(REPO_ROOT / "data" / "task_grid_li.gpkg").to_crs(METRIC_CRS)
    li_union = li_cells.union_all()
    # Li cell size from the registered true-geometry cells (regular lattice).
    cw = float((li_cells.bounds.maxx - li_cells.bounds.minx).median())
    ch = float((li_cells.bounds.maxy - li_cells.bounds.miny).median())
    print(f"Li cell size ~ {cw:.0f} x {ch:.0f} m; expanding panel bboxes by that margin")

    rows = []
    for lid in batch:
        panels = gpd.read_file(LI_DIR / f"{lid}.gpkg").to_crs(METRIC_CRS)
        bb = panels.total_bounds
        cell_sup = box(bb[0] - cw, bb[1] - ch, bb[2] + cw, bb[3] + ch)

        hit_idx = list(gao_sidx.query(cell_sup, predicate="intersects"))
        gao_hits = sorted(gao_annot.iloc[hit_idx]["gridcell_id"]) if hit_idx else []
        rows.append(
            {
                "Lid": lid,
                "n_panels": len(panels),
                "gao_annot_hits": ";".join(gao_hits),
                "dist_gao_annot_m": round(cell_sup.distance(gao_union), 1),
                "li_heldout_hit": bool(cell_sup.intersects(li_union)),
                "dist_li_heldout_m": round(cell_sup.distance(li_union), 1),
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "overlap_audit.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_gao = sum(1 for r in rows if r["gao_annot_hits"])
    n_li = sum(1 for r in rows if r["li_heldout_hit"])
    print(f"{len(rows)} candidates | {n_gao} touch Gao-annotated cells | "
          f"{n_li} touch Li held-out cells")
    print(f"min dist to Gao-annotated: "
          f"{min(r['dist_gao_annot_m'] for r in rows):.0f} m | "
          f"min dist to Li held-out: "
          f"{min(r['dist_li_heldout_m'] for r in rows):.0f} m")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
