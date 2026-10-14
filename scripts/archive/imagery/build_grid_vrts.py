"""
Build per-grid VRT mosaics from downloaded *_geo.tif tiles.

Examples:
  python scripts/imagery/build_grid_vrts.py --grid-ids G1240 G1241

  python scripts/imagery/build_grid_vrts.py \
    --batch-dir results/grid_previews/batch_001
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from xml.etree import ElementTree as ET

import rasterio

DEFAULT_DECISIONS = ("keep",)
GDAL_DTYPE = {
    "uint8": "Byte",
    "int8": "Int8",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "float32": "Float32",
    "float64": "Float64",
}


def load_grid_ids_from_batch(batch_dir: Path, decisions: tuple[str, ...]) -> list[str]:
    decisions_path = batch_dir / "grid_review_decisions.csv"
    if not decisions_path.exists():
        raise FileNotFoundError(f"review decisions not found: {decisions_path}")

    wanted = {decision.strip().lower() for decision in decisions if decision.strip()}
    grid_ids: list[str] = []
    seen: set[str] = set()
    with decisions_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            decision = str(row.get("decision", "")).strip().lower()
            grid_id = str(row.get("grid_id", "")).strip().upper()
            if decision not in wanted or not grid_id or grid_id in seen:
                continue
            seen.add(grid_id)
            grid_ids.append(grid_id)
    return grid_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build VRT mosaics for downloaded grid tiles")
    parser.add_argument("--grid-ids", nargs="+", help="Explicit grid IDs to process")
    parser.add_argument("--batch-dir", type=Path, help="Preview batch dir with grid_review_decisions.csv")
    parser.add_argument(
        "--decision",
        nargs="+",
        default=list(DEFAULT_DECISIONS),
        help="Decision labels to include when --batch-dir is used",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing VRT files")
    return parser


def get_grid_ids(args: argparse.Namespace) -> list[str]:
    if args.grid_ids:
        return [str(grid_id).strip().upper() for grid_id in args.grid_ids if str(grid_id).strip()]
    if args.batch_dir:
        return load_grid_ids_from_batch(args.batch_dir, tuple(args.decision))
    raise ValueError("either --grid-ids or --batch-dir is required")


def iter_tile_paths(grid_dir: Path) -> list[Path]:
    return sorted(
        path for path in grid_dir.glob("*_geo.tif") if path.is_file()
    )


def format_float(value: float) -> str:
    return f"{value:.15g}"


def infer_mosaic_dimensions(tile_paths: list[Path]) -> tuple[int, int, dict[Path, tuple[float, float, float, float]], dict[Path, tuple[int, int]]]:
    tile_bounds: dict[Path, tuple[float, float, float, float]] = {}
    tile_sizes: dict[Path, tuple[int, int]] = {}
    lefts: set[float] = set()
    tops: set[float] = set()

    for path in tile_paths:
        with rasterio.open(path) as src:
            left, bottom, right, top = src.bounds
            tile_bounds[path] = (left, bottom, right, top)
            tile_sizes[path] = (src.width, src.height)
            lefts.add(left)
            tops.add(top)

    sample_width = next(iter(tile_sizes.values()))[0]
    sample_height = next(iter(tile_sizes.values()))[1]
    raster_x_size = len(lefts) * sample_width
    raster_y_size = len(tops) * sample_height
    return raster_x_size, raster_y_size, tile_bounds, tile_sizes


def build_vrt_for_grid(grid_id: str, force: bool = False) -> Path:
    grid_dir = Path("tiles") / grid_id
    if not grid_dir.exists():
        raise FileNotFoundError(f"grid tiles dir not found: {grid_dir}")

    tile_paths = iter_tile_paths(grid_dir)
    if not tile_paths:
        raise FileNotFoundError(f"no *_geo.tif tiles found in {grid_dir}")

    vrt_path = grid_dir / f"{grid_id}_mosaic.vrt"
    if vrt_path.exists() and not force:
        return vrt_path

    raster_x_size, raster_y_size, tile_bounds, tile_sizes = infer_mosaic_dimensions(tile_paths)

    with rasterio.open(tile_paths[0]) as sample:
        crs_wkt = sample.crs.to_wkt()
        band_count = sample.count
        data_types = list(sample.dtypes)
        colorinterp = [interp.name.title() for interp in sample.colorinterp]
        block_shapes = list(sample.block_shapes)

    xmin = min(bounds[0] for bounds in tile_bounds.values())
    ymin = min(bounds[1] for bounds in tile_bounds.values())
    xmax = max(bounds[2] for bounds in tile_bounds.values())
    ymax = max(bounds[3] for bounds in tile_bounds.values())
    x_res = (xmax - xmin) / raster_x_size
    y_res = (ymax - ymin) / raster_y_size

    root = ET.Element(
        "VRTDataset",
        rasterXSize=str(raster_x_size),
        rasterYSize=str(raster_y_size),
    )
    srs = ET.SubElement(root, "SRS")
    srs.text = crs_wkt
    geotransform = ET.SubElement(root, "GeoTransform")
    geotransform.text = (
        f"  {format_float(xmin)},  {format_float(x_res)},  0.0,  "
        f"{format_float(ymax)},  0.0,  {format_float(-y_res)}"
    )

    for band_idx in range(1, band_count + 1):
        band_dtype = GDAL_DTYPE[data_types[band_idx - 1]]
        vrt_band = ET.SubElement(
            root,
            "VRTRasterBand",
            dataType=band_dtype,
            band=str(band_idx),
        )
        if band_idx - 1 < len(colorinterp):
            color = ET.SubElement(vrt_band, "ColorInterp")
            color.text = colorinterp[band_idx - 1]

        block_y, block_x = block_shapes[band_idx - 1]
        for tile_path in tile_paths:
            left, bottom, right, top = tile_bounds[tile_path]
            src_width, src_height = tile_sizes[tile_path]
            x_off = (left - xmin) / x_res
            y_off = (ymax - top) / y_res
            x_size = (right - left) / x_res
            y_size = (top - bottom) / y_res

            source = ET.SubElement(vrt_band, "SimpleSource")
            filename = ET.SubElement(source, "SourceFilename", relativeToVRT="1")
            filename.text = tile_path.name
            source_band = ET.SubElement(source, "SourceBand")
            source_band.text = str(band_idx)
            ET.SubElement(
                source,
                "SourceProperties",
                RasterXSize=str(src_width),
                RasterYSize=str(src_height),
                DataType=band_dtype,
                BlockXSize=str(block_x),
                BlockYSize=str(block_y),
            )
            ET.SubElement(
                source,
                "SrcRect",
                xOff="0",
                yOff="0",
                xSize=str(src_width),
                ySize=str(src_height),
            )
            ET.SubElement(
                source,
                "DstRect",
                xOff=format_float(x_off),
                yOff=format_float(y_off),
                xSize=format_float(x_size),
                ySize=format_float(y_size),
            )

    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    tree.write(vrt_path, encoding="utf-8", xml_declaration=False)
    return vrt_path


def verify_vrt(vrt_path: Path) -> tuple[int, int, int]:
    with rasterio.open(vrt_path) as src:
        return src.width, src.height, src.count


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    grid_ids = get_grid_ids(args)
    if not grid_ids:
        print("[INFO] no grids selected")
        return

    print(f"[INFO] selected_grids={len(grid_ids)}")
    for grid_id in grid_ids:
        grid_dir = Path("tiles") / grid_id
        if not grid_dir.exists():
            print(f"[SKIP] {grid_id}: no tiles directory")
            continue
        tile_paths = iter_tile_paths(grid_dir)
        if not tile_paths:
            print(f"[SKIP] {grid_id}: no tiles found")
            continue
        vrt_path = build_vrt_for_grid(grid_id, force=args.force)
        width, height, count = verify_vrt(vrt_path)
        print(f"[OK] {grid_id}: {vrt_path} ({width}x{height}, bands={count})")


if __name__ == "__main__":
    main()
