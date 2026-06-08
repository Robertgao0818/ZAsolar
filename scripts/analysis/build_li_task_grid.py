"""Build data/task_grid_li.gpkg from Li's Cape Town KML (Phase 0.5).

Li uses an independent grid scheme. The KML
(cape_town_grid_Li_G0029_G1841.kml) — despite its filename — actually carries
the full base-grid cell polygons (G0029..G4429) in two layers, including all 17
Li-annotated cells (G1842..G1954). We extract those 17 cells, remap the Gao-named
IDs to L-prefix IDs, and write a task grid with column ``gridcell_id`` (matching
the schema core/grid_utils expects). Cell geometry is the AUTHORITATIVE WMS
download AOI; cross-checked against Li GT bbox + on-disk aerial_2025/G1895 tiles.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

KML = Path("/mnt/c/Users/gaosh/Dropbox/RA_Solar/Li/capetown/cape_town_grid_Li_G0029_G1841.kml")
OUT = Path(__file__).resolve().parents[2] / "data" / "task_grid_li.gpkg"
GT_DIR = Path(__file__).resolve().parents[2] / "data" / "annotations" / "Capetown_Li"

# Gao source id -> Li L-prefix id
LI_MAP = {
    "G1842": "L1842", "G1843": "L1843", "G1844": "L1844", "G1846": "L1846",
    "G1895": "L1895", "G1896": "L1896", "G1897": "L1897", "G1898": "L1898",
    "G1899": "L1899", "G1900": "L1900", "G1901": "L1901", "G1902": "L1902",
    "G1950": "L1950", "G1951": "L1951", "G1952": "L1952", "G1953": "L1953",
    "G1954": "L1954",
}

# Layers in the KML that contain bare base-grid cell polygons.
BASE_GRID_LAYERS = ["G0436_G1114", "G1115_G1506"]


def main() -> None:
    found: dict[str, object] = {}
    for lyr in BASE_GRID_LAYERS:
        g = gpd.read_file(KML, layer=lyr)
        name_col = [c for c in g.columns if c.lower() == "name"][0]
        for gao, lid in LI_MAP.items():
            if lid in found:
                continue
            sub = g[g[name_col].astype(str) == gao]
            if len(sub) >= 1:
                found[lid] = sub.geometry.iloc[0]

    missing = set(LI_MAP.values()) - set(found.keys())
    if missing:
        raise SystemExit(f"KML missing cells for: {sorted(missing)}")

    rows = []
    for lid in sorted(found.keys()):
        geom = found[lid]
        gt = gpd.read_file(GT_DIR / f"{lid}.gpkg")
        b = gt.total_bounds  # gt covered by cell?
        cb = geom.bounds
        gt_inside = (cb[0] - 1e-4 <= b[0] and cb[1] - 1e-4 <= b[1]
                     and cb[2] + 1e-4 >= b[2] and cb[3] + 1e-4 >= b[3])
        rows.append({
            "gridcell_id": lid,
            "source_gao_id": [k for k, v in LI_MAP.items() if v == lid][0],
            "scheme": "li",
            "geom_source": "kml_cell",  # real KML cell geometry (not GT-extent approx)
            "gt_bbox_inside_cell": bool(gt_inside),
            "geometry": geom,
        })

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.to_file(OUT, layer="task_grid_li", driver="GPKG")
    print(f"=== Wrote {len(gdf)} Li cells -> {OUT}")
    for _, r in gdf.iterrows():
        cb = r.geometry.bounds
        print(f"{r['gridcell_id']} <- {r['source_gao_id']} | geom={r['geom_source']} | "
              f"lon[{cb[0]:.4f},{cb[2]:.4f}] lat[{cb[1]:.4f},{cb[3]:.4f}] | "
              f"gt_inside={r['gt_bbox_inside_cell']}")


if __name__ == "__main__":
    main()
