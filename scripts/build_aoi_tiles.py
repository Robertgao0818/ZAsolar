#!/usr/bin/env python3
"""Enumerate the expected AOI units for the coverage gate (scripts/check_coverage.py)
straight from AOI geometry, independent of any downloader manifest or tile directory.

Modes (tiling rules mirror the production tilers exactly):
  ct-native  CT 0.05 m WMS tiles: grid CSV with grid,minx,miny,maxx,maxy (EPSG:4326),
             100 m tiles E{x}_N{y}.tif in EPSG:32734 with 50 m buffer (dl_native_v2.tiles_for).
             Tiles shared by neighbouring grids are listed once.
  jhb-vexcel JHB Vexcel tiles: task-grid GPKG, 100 m tiles {grid}/{grid}_E{x}_N{y}.tif over each
             grid's EPSG:32735 envelope (jhb tile_city.py).
  grid       one unit per task-grid cell (grid-level pipelines, e.g. Durban pregate).
Output CSV: key column (tile or grid), grid, minx,miny,maxx,maxy (in the stated metric CRS).
"""
from __future__ import annotations

import argparse
import math

import geopandas as gpd
import pandas as pd
from pyproj import Transformer

STEP = 100


def ct_tiles_for(r, tf, buf=50.0):
    xs, ys = tf.transform([r.minx, r.maxx, r.minx, r.maxx], [r.miny, r.miny, r.maxy, r.maxy])
    x0 = math.floor((min(xs) - buf) / STEP) * STEP; x1 = math.ceil((max(xs) + buf) / STEP) * STEP
    y0 = math.floor((min(ys) - buf) / STEP) * STEP; y1 = math.ceil((max(ys) + buf) / STEP) * STEP
    return [(int(x), int(y)) for x in range(int(x0), int(x1), STEP) for y in range(int(y0), int(y1), STEP)]


def ct_native(grid_csv):
    tf = Transformer.from_crs(4326, 32734, always_xy=True)
    rows = {}
    for r in pd.read_csv(grid_csv).itertuples():
        for x, y in ct_tiles_for(r, tf):
            rows.setdefault(f"E{x}_N{y}.tif", (r.grid, x, y))
    return pd.DataFrame([(k, g, x, y, x + STEP, y + STEP) for k, (g, x, y) in rows.items()],
                        columns=["tile", "grid", "minx", "miny", "maxx", "maxy"])


def jhb_vexcel(gpkg, id_col="gridcell_id"):
    G = gpd.read_file(gpkg).to_crs("EPSG:32735")
    out = []
    for g, geom in zip(G[id_col], G.geometry):
        x0, y0, x1, y1 = [round(v) for v in geom.envelope.bounds]
        x0 -= x0 % 100
        for tx in range(x0, x1, STEP):
            for ty in range(y0 - y0 % 100, y1, STEP):
                out.append((f"{g}/{g}_E{tx}_N{ty}.tif", g, tx, ty, tx + STEP, ty + STEP))
    return pd.DataFrame(out, columns=["tile", "grid", "minx", "miny", "maxx", "maxy"])


def grid_units(gpkg, id_col, crs):
    G = gpd.read_file(gpkg).to_crs(crs)
    b = G.geometry.bounds
    return pd.DataFrame({"grid": G[id_col].astype(str), "minx": b.minx, "miny": b.miny, "maxx": b.maxx, "maxy": b.maxy})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["ct-native", "jhb-vexcel", "grid"])
    ap.add_argument("--grids", required=True, help="grid CSV (ct-native) or task-grid GPKG")
    ap.add_argument("--id-col", default="gridcell_id")
    ap.add_argument("--crs", default="EPSG:32736", help="metric CRS for mode=grid")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    df = {"ct-native": lambda: ct_native(a.grids), "jhb-vexcel": lambda: jhb_vexcel(a.grids, a.id_col),
          "grid": lambda: grid_units(a.grids, a.id_col, a.crs)}[a.mode]()
    df.to_csv(a.out, index=False)
    print(f"{a.mode}: {len(df)} units -> {a.out}")


if __name__ == "__main__":
    main()
