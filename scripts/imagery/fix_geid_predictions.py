#!/usr/bin/env python3
"""Fix lat/lon of predictions produced from a wrongly-georeferenced GEID
mosaic.

Background: stitch_geid_python.py originally used the manifest's
bbox_lon_lat as the geotransform, but GEID tiles are addressed in an
equirectangular scheme (PROTOCOL.md §"Coordinate system"): each tile is
1/(2^(z-1)/360) degrees square. The wrong transform stretched pixels
~5% in x and shifted everything by up to ~30 m. The model predictions
were produced on the correct pixels but written out with the wrong
transform, so they live at wrong lon/lat. This script applies the
per-grid affine that maps wrong-frame coords back through pixel space
into the correct frame, in place on results/<GRID>/predictions.geojson
and predictions_metric.gpkg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
from shapely.affinity import affine_transform

TILE_SIZE = 256


def fix_one(grid_id: str, manifest_path: Path, results_dir: Path) -> bool:
    manifest = json.loads(manifest_path.read_text())
    z = int(manifest["zoom"])
    tr = manifest["tile_range"]
    cols = tr["x"][1] - tr["x"][0] + 1
    rows = tr["y"][1] - tr["y"][0] + 1
    width = cols * TILE_SIZE
    height = rows * TILE_SIZE

    # OLD (wrong) bounds: manifest bbox spread uniformly across width × height
    bbox = manifest["bbox_lon_lat"]  # [W, S, E, N]
    W_old, S_old, E_old, N_old = bbox
    px_old = (E_old - W_old) / width
    py_old = (N_old - S_old) / height

    # NEW (correct) bounds: equirectangular per-tile addressing
    factor = (1 << (z - 1)) / 360.0
    W_new = tr["x"][0] / factor - 180.0
    E_new = (tr["x"][0] + cols) / factor - 180.0
    S_new = tr["y"][0] / factor - 180.0
    N_new = (tr["y"][0] + rows) / factor - 180.0
    px_new = (E_new - W_new) / width
    py_new = (N_new - S_new) / height

    # Affine on (lon, lat):
    #   lon_new = lon_old * (px_new/px_old) + (W_new - W_old * px_new/px_old)
    #   lat_new = lat_old * (py_new/py_old) + (N_new - N_old * py_new/py_old)
    # (lat uses the top edge because y-flip cancels: row counts down from N
    # in both old and new transforms.)
    a = px_new / px_old
    e = py_new / py_old
    b_ = W_new - W_old * a
    f_ = N_new - N_old * e
    # shapely.affine: [a, b, d, e, xoff, yoff] for x' = a*x + b*y + xoff
    affine = [a, 0.0, 0.0, e, b_, f_]

    geo_path = results_dir / "predictions.geojson"
    if not geo_path.exists():
        print(f"  [skip] {grid_id}: no predictions.geojson")
        return False

    gdf = gpd.read_file(geo_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    n = len(gdf)
    if n == 0:
        print(f"  [skip] {grid_id}: empty")
        return False

    gdf["geometry"] = gdf.geometry.apply(lambda g: affine_transform(g, affine))

    # Write back: geojson in 4326, gpkg reprojected to metric (EPSG:32735)
    geo_path.unlink()
    gdf.to_file(geo_path, driver="GeoJSON")

    metric_path = results_dir / "predictions_metric.gpkg"
    if metric_path.exists():
        metric_path.unlink()
    gdf.to_crs(32735).to_file(metric_path, driver="GPKG")

    dx_m = (b_) * 111000 * 0.9  # rough deg→m at -26°
    dy_m = (f_) * 111000
    print(f"  [ok]   {grid_id}: n={n}  scale=({a:.5f},{e:.5f})  shift≈({dx_m:+.1f}m,{dy_m:+.1f}m)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-root", type=Path,
                    default=Path("/mnt/d/ZAsolar/geid_raw/joburg_geid_python"))
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--grids", nargs="*")
    args = ap.parse_args()

    manifests = sorted(args.manifest_root.glob("*/manifest.json"))
    if args.grids:
        wanted = set(args.grids)
        manifests = [m for m in manifests if m.parent.name in wanted]

    n_ok = 0
    for m in manifests:
        gid = m.parent.name
        rdir = args.results_root / gid
        if not rdir.exists():
            continue
        try:
            if fix_one(gid, m, rdir):
                n_ok += 1
        except Exception as e:
            print(f"  [err]  {gid}: {e}")
    print(f"Done: {n_ok} grids fixed")


if __name__ == "__main__":
    main()
