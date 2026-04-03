"""
Download tiles for grids from the Li grid file (cape_town_grid_Li.gpkg).

These grids are NOT in the main task_grid.gpkg, so we read bounds directly
from the Li file and output to a separate folder.

Usage:
  python scripts/imagery/download_li_grids.py \
    --grid-ids G1841 G1842 G1843 G1844 G1845 G1846 \
    --output-root Li
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.grid_utils import GridSpec, DEFAULT_TILE_SIZE_DEG, DEFAULT_PIXEL_SIZE
from scripts.imagery.download_tiles import download_tile
from scripts.imagery.build_grid_vrts import (
    iter_tile_paths, build_vrt_for_grid, verify_vrt,
    GDAL_DTYPE, format_float,
)

LI_GRID_GPKG = Path("data/cape_town_grid_Li.gpkg")


def get_li_grid_spec(grid_id: str, li_gdf: gpd.GeoDataFrame) -> GridSpec:
    row = li_gdf[li_gdf["Name"] == grid_id]
    if len(row) == 0:
        raise KeyError(f"grid_id not found in Li grid file: {grid_id}")
    geom = row.iloc[0].geometry
    xmin, ymin, xmax, ymax = geom.bounds
    n_cols = math.ceil((xmax - xmin) / DEFAULT_TILE_SIZE_DEG)
    n_rows = math.ceil((ymax - ymin) / DEFAULT_TILE_SIZE_DEG)
    return GridSpec(
        grid_id=grid_id,
        xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
        tile_size_deg=DEFAULT_TILE_SIZE_DEG,
        pixel_size=DEFAULT_PIXEL_SIZE,
        n_cols=n_cols, n_rows=n_rows,
    )


def _is_blank_tile(path: Path, white_threshold: int = 245) -> bool:
    """Check if a tile is mostly white/blank (>95% white pixels)."""
    import numpy as np
    import rasterio
    with rasterio.open(path) as src:
        data = src.read()  # (bands, H, W)
        white_mask = np.all(data >= white_threshold, axis=0)
        return float(white_mask.mean()) > 0.95


def download_li_grid(spec: GridSpec, output_root: Path, skip_blank: bool = False):
    import time
    grid_dir = output_root / spec.grid_id
    grid_dir.mkdir(parents=True, exist_ok=True)

    total = spec.n_cols * spec.n_rows
    print(f"Grid: {spec.grid_id}")
    print(f"  Bounds: ({spec.xmin:.6f}, {spec.ymin:.6f}) -> ({spec.xmax:.6f}, {spec.ymax:.6f})")
    print(f"  Tiles: {spec.n_cols} cols x {spec.n_rows} rows = {total}")
    print(f"  Output: {grid_dir}")
    if skip_blank:
        print(f"  Skip blank tiles: ON")

    downloaded = 0
    skipped = 0
    blank_removed = 0
    errors = 0

    for col in range(spec.n_cols):
        for row in range(spec.n_rows):
            tile_name = f"{spec.grid_id}_{col}_{row}_geo.tif"
            out_path = grid_dir / tile_name
            if out_path.exists():
                skipped += 1
                continue
            try:
                download_tile(spec, col, row, out_path)
                if skip_blank and _is_blank_tile(out_path):
                    out_path.unlink()
                    blank_removed += 1
                    print(f"  [{downloaded + skipped + blank_removed}/{total}] {tile_name} [BLANK, removed]")
                    continue
                downloaded += 1
                print(f"  [{downloaded + skipped}/{total}] {tile_name}")
            except Exception as e:
                errors += 1
                print(f"  [ERROR] {tile_name}: {e}")
                time.sleep(3)
                try:
                    download_tile(spec, col, row, out_path, timeout=600)
                    if skip_blank and _is_blank_tile(out_path):
                        out_path.unlink()
                        blank_removed += 1
                        errors -= 1
                        print(f"  [RETRY] {tile_name} [BLANK, removed]")
                        continue
                    downloaded += 1
                    errors -= 1
                    print(f"  [RETRY OK] {tile_name}")
                except Exception as e2:
                    print(f"  [RETRY FAIL] {tile_name}: {e2}")

    print(f"[DONE] {spec.grid_id}: downloaded={downloaded}, skipped={skipped}, "
          f"blank_removed={blank_removed}, errors={errors}")
    return grid_dir


def build_vrt_custom(grid_id: str, grid_dir: Path):
    """Build VRT for tiles in a custom directory."""
    import rasterio
    from xml.etree import ElementTree as ET

    tile_paths = sorted(p for p in grid_dir.glob("*_geo.tif") if p.is_file())
    if not tile_paths:
        print(f"[SKIP] {grid_id}: no tiles")
        return None

    vrt_path = grid_dir / f"{grid_id}_mosaic.vrt"

    tile_bounds = {}
    tile_sizes = {}
    lefts, tops = set(), set()
    for path in tile_paths:
        with rasterio.open(path) as src:
            left, bottom, right, top = src.bounds
            tile_bounds[path] = (left, bottom, right, top)
            tile_sizes[path] = (src.width, src.height)
            lefts.add(left)
            tops.add(top)

    sample_w = next(iter(tile_sizes.values()))[0]
    sample_h = next(iter(tile_sizes.values()))[1]
    raster_x = len(lefts) * sample_w
    raster_y = len(tops) * sample_h

    with rasterio.open(tile_paths[0]) as sample:
        crs_wkt = sample.crs.to_wkt()
        band_count = sample.count
        data_types = list(sample.dtypes)
        colorinterp = [i.name.title() for i in sample.colorinterp]
        block_shapes = list(sample.block_shapes)

    xmin = min(b[0] for b in tile_bounds.values())
    ymin = min(b[1] for b in tile_bounds.values())
    xmax = max(b[2] for b in tile_bounds.values())
    ymax = max(b[3] for b in tile_bounds.values())
    x_res = (xmax - xmin) / raster_x
    y_res = (ymax - ymin) / raster_y

    root = ET.Element("VRTDataset", rasterXSize=str(raster_x), rasterYSize=str(raster_y))
    srs = ET.SubElement(root, "SRS")
    srs.text = crs_wkt
    gt = ET.SubElement(root, "GeoTransform")
    gt.text = f"  {format_float(xmin)},  {format_float(x_res)},  0.0,  {format_float(ymax)},  0.0,  {format_float(-y_res)}"

    for bi in range(1, band_count + 1):
        bd = GDAL_DTYPE[data_types[bi - 1]]
        vb = ET.SubElement(root, "VRTRasterBand", dataType=bd, band=str(bi))
        if bi - 1 < len(colorinterp):
            c = ET.SubElement(vb, "ColorInterp")
            c.text = colorinterp[bi - 1]
        block_y, block_x = block_shapes[bi - 1]
        for tp in tile_paths:
            left, bottom, right, top = tile_bounds[tp]
            sw, sh = tile_sizes[tp]
            xo = (left - xmin) / x_res
            yo = (ymax - top) / y_res
            xs = (right - left) / x_res
            ys = (top - bottom) / y_res
            src = ET.SubElement(vb, "SimpleSource")
            fn = ET.SubElement(src, "SourceFilename", relativeToVRT="1")
            fn.text = tp.name
            sb = ET.SubElement(src, "SourceBand")
            sb.text = str(bi)
            ET.SubElement(src, "SourceProperties", RasterXSize=str(sw), RasterYSize=str(sh),
                          DataType=bd, BlockXSize=str(block_x), BlockYSize=str(block_y))
            ET.SubElement(src, "SrcRect", xOff="0", yOff="0", xSize=str(sw), ySize=str(sh))
            ET.SubElement(src, "DstRect", xOff=format_float(xo), yOff=format_float(yo),
                          xSize=format_float(xs), ySize=format_float(ys))

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(vrt_path, encoding="utf-8", xml_declaration=False)

    with rasterio.open(vrt_path) as src:
        print(f"[OK] {vrt_path} ({src.width}x{src.height}, bands={src.count})")
    return vrt_path


def main():
    parser = argparse.ArgumentParser(description="Download tiles for Li grid file grids")
    parser.add_argument("--grid-ids", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("Li"))
    parser.add_argument("--grid-file", type=Path, default=LI_GRID_GPKG)
    parser.add_argument("--no-vrt", action="store_true")
    parser.add_argument("--skip-blank", action="store_true",
                        help="Remove downloaded tiles that are >95%% white")
    args = parser.parse_args()

    li_gdf = gpd.read_file(args.grid_file)
    print(f"Loaded {len(li_gdf)} grids from {args.grid_file}")

    for gid in args.grid_ids:
        gid = gid.strip().upper()
        spec = get_li_grid_spec(gid, li_gdf)
        grid_dir = download_li_grid(spec, args.output_root,
                                    skip_blank=args.skip_blank)
        if not args.no_vrt:
            build_vrt_custom(gid, grid_dir)


if __name__ == "__main__":
    main()
