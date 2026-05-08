#!/usr/bin/env python3
"""Quick verification of dissolve_hairline_gaps on the cat-1 case observed
in G0922 V3C predictions (1121 m² installation split across TIF seam).

Loads the existing predictions_metric.gpkg, applies dissolve at several
tolerance levels, reports polygon counts + writes the dissolved gpkg for
visual diff against the original.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from core.postproc import dissolve_hairline_gaps

REPO = Path("/home/gaosh/projects/ZAsolar")
GRIDS_TO_TEST = ["G0922", "G0856", "G0816", "G0817"]
TOLERANCES_M = [0.0, 0.3, 0.5, 1.0, 2.0]


def main() -> None:
    out_dir = REPO / "results/diag/cross_tif_dissolve_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    for grid in GRIDS_TO_TEST:
        src = REPO / f"results/johannesburg/v3c_vexcel_2024/{grid}/predictions_metric.gpkg"
        if not src.exists():
            print(f"[skip] {grid}: no predictions_metric.gpkg")
            continue
        gdf = gpd.read_file(src)
        print(f"\n=== {grid}: {len(gdf)} input polygons ===")
        for tol in TOLERANCES_M:
            out = dissolve_hairline_gaps(gdf, tolerance_m=tol)
            n_in = len(gdf)
            n_out = len(out)
            delta = n_in - n_out
            print(f"  tolerance={tol:>3.1f} m  →  {n_out:4d} polygons  (merged {delta} into others)")
            if tol == 0.5:  # write 0.5m output for visual diff
                save_path = out_dir / f"{grid}_dissolve_0p5m.gpkg"
                out.to_file(save_path, driver="GPKG")

    print(f"\n0.5 m dissolved outputs → {out_dir}")


if __name__ == "__main__":
    main()
