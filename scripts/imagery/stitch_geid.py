#!/usr/bin/env python3
"""Stitch a Google Earth Images Downloader task without using the Combiner UI."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


FILENAME_RE = re.compile(r"^ges_(?P<x>\d+)_(?P<y>\d+)_(?P<zoom>\d+)\.jpg$", re.IGNORECASE)
LIST_LINE_RE = re.compile(
    r"^(?P<filename>ges_\d+_\d+_\d+\.jpg):\s+"
    r"(?P<left>-?\d+(?:\.\d+)?)\s+"
    r"(?P<right>-?\d+(?:\.\d+)?)\s+"
    r"(?P<top>-?\d+(?:\.\d+)?)\s+"
    r"(?P<bottom>-?\d+(?:\.\d+)?)$"
)


@dataclass(frozen=True)
class TileRecord:
    filename: str
    x: int
    y: int
    zoom: int
    left: float
    right: float
    top: float
    bottom: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a VRT and optionally a GeoTIFF from a GEID task folder."
    )
    parser.add_argument("task_dir", type=Path, help="GEID task directory, e.g. /mnt/d/ZAsolar/geid_raw/joburg_cbd_geid/G0854")
    parser.add_argument("--zoom", type=int, default=None, help="Zoom level to stitch. Default: infer from list1")
    parser.add_argument(
        "--gdal-translate",
        default=r"C:\allmapsoft\geid\bin64\gdal_translate.exe",
        help="Windows path to gdal_translate.exe used for the final GeoTIFF step.",
    )
    parser.add_argument(
        "--gdal-prj",
        default=r"C:\allmapsoft\geid\geotiff\4326.prj",
        help="Windows path to the WGS84 .prj file used by gdal_translate.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Final TIFF basename. Default: <task_name>_zoom_<zoom>.tif",
    )
    parser.add_argument("--vrt-only", action="store_true", help="Only write VRT files; skip GeoTIFF creation.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing TIFF/VRT.")
    return parser.parse_args()


def load_records(list_path: Path) -> list[TileRecord]:
    records: list[TileRecord] = []
    with list_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("ImageFileName"):
                continue
            match_line = LIST_LINE_RE.match(line)
            if not match_line:
                continue
            filename = match_line.group("filename")
            match = FILENAME_RE.match(filename)
            if not match:
                continue
            records.append(
                TileRecord(
                    filename=filename,
                    x=int(match.group("x")),
                    y=int(match.group("y")),
                    zoom=int(match.group("zoom")),
                    left=float(match_line.group("left")),
                    right=float(match_line.group("right")),
                    top=float(match_line.group("top")),
                    bottom=float(match_line.group("bottom")),
                )
            )
    if not records:
        raise ValueError(f"No tile rows found in {list_path}")
    return records


def infer_task_name(task_dir: Path) -> str:
    geid_files = sorted(task_dir.glob("*.geid"))
    if geid_files:
        return geid_files[0].stem
    return task_dir.name


def find_existing_tifs(task_dir: Path, task_name: str) -> dict[int, Path]:
    combined_dir = task_dir / f"{task_name}_combined"
    outputs: dict[int, Path] = {}
    if not combined_dir.exists():
        return outputs
    pattern = re.compile(rf"^{re.escape(task_name)}_zoom_(\d+)\.tif$", re.IGNORECASE)
    for tif_path in sorted(combined_dir.glob("*.tif")):
        match = pattern.match(tif_path.name)
        if match:
            outputs[int(match.group(1))] = tif_path
    return outputs


def relative_windows_path(path: Path) -> str:
    return str(path).replace("/", "\\")


def to_windows_path(path: Path) -> str:
    parts = path.resolve().parts
    if len(parts) >= 3 and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper()
        tail = "\\".join(parts[3:])
        return f"{drive}:\\{tail}" if tail else f"{drive}:\\"
    raise ValueError(f"Path is not on a Windows-mounted drive: {path}")


def build_source_element(source_name: str, band: int, dst_x: int, dst_y: int, tile_size: int) -> str:
    src = escape(source_name)
    return (
        "<SimpleSource>\n"
        f"<SourceFilename relativeToVRT=\"1\">{src}</SourceFilename>\n"
        f"<SourceBand>{band}</SourceBand>\n"
        f"<SourceProperties RasterXSize=\"{tile_size}\" RasterYSize=\"{tile_size}\" DataType=\"Byte\" BlockXSize=\"{tile_size}\" BlockYSize=\"{tile_size}\"/>\n"
        f"<SrcRect xOff=\"0\" yOff=\"0\" xSize=\"{tile_size}\" ySize=\"{tile_size}\"/>\n"
        f"<DstRect xOff=\"{dst_x}\" yOff=\"{dst_y}\" xSize=\"{tile_size}\" ySize=\"{tile_size}\"/>\n"
        "</SimpleSource>\n"
    )


def write_column_vrt(
    out_path: Path,
    records: Iterable[TileRecord],
    *,
    min_y: int,
    max_y: int,
    overall_top: float,
    overall_left: float,
    pixel_width: float,
    pixel_height: float,
    tile_size: int,
) -> None:
    column = sorted(records, key=lambda item: item.y)
    height = (max_y - min_y + 1) * tile_size
    geotransform = f"{overall_left},{pixel_width},0,{overall_top},0,{-pixel_height}"
    bands: list[str] = []
    for band, interp in enumerate(("Red", "Green", "Blue"), start=1):
        parts = [f"<VRTRasterBand dataType=\"Byte\" band=\"{band}\">", f"<ColorInterp>{interp}</ColorInterp>"]
        for rec in column:
            y_index = max_y - rec.y
            dst_y = y_index * tile_size
            src_rel = relative_windows_path(Path(str(rec.zoom)) / str(rec.x) / rec.filename)
            parts.append(build_source_element(src_rel, band, 0, dst_y, tile_size).rstrip())
        parts.append("</VRTRasterBand>")
        bands.append("\n".join(parts))
    xml = (
        f"<VRTDataset rasterXSize=\"{tile_size}\" rasterYSize=\"{height}\">\n"
        "<SRS>GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"Degree\",0.017453292519943295]]</SRS>\n"
        f"<GeoTransform>{geotransform}</GeoTransform>\n"
        f"{chr(10).join(bands)}\n"
        "</VRTDataset>\n"
    )
    out_path.write_text(xml, encoding="utf-8")


def write_main_vrt(
    out_path: Path,
    x_values: list[int],
    *,
    rows: int,
    overall_top: float,
    overall_left: float,
    pixel_width: float,
    pixel_height: float,
    tile_size: int,
    zoom: int,
) -> None:
    width = len(x_values) * tile_size
    height = rows * tile_size
    geotransform = f"{overall_left},{pixel_width},0,{overall_top},0,{-pixel_height}"
    bands: list[str] = []
    for band, interp in enumerate(("Red", "Green", "Blue"), start=1):
        parts = [f"<VRTRasterBand dataType=\"Byte\" band=\"{band}\">", f"<ColorInterp>{interp}</ColorInterp>"]
        for idx, x_value in enumerate(x_values):
            dst_x = idx * tile_size
            src_rel = f"{zoom}_{x_value}.vrt"
            parts.append(build_source_element(src_rel, band, dst_x, 0, tile_size).rstrip().replace(
                f"RasterYSize=\"{tile_size}\"", f"RasterYSize=\"{height}\"", 1
            ).replace(
                f"BlockYSize=\"{tile_size}\"", f"BlockYSize=\"{height}\"", 1
            ).replace(
                f"<SrcRect xOff=\"0\" yOff=\"0\" xSize=\"{tile_size}\" ySize=\"{tile_size}\"/>",
                f"<SrcRect xOff=\"0\" yOff=\"0\" xSize=\"{tile_size}\" ySize=\"{height}\"/>",
                1,
            ).replace(
                f"<DstRect xOff=\"{dst_x}\" yOff=\"0\" xSize=\"{tile_size}\" ySize=\"{tile_size}\"/>",
                f"<DstRect xOff=\"{dst_x}\" yOff=\"0\" xSize=\"{tile_size}\" ySize=\"{height}\"/>",
                1,
            ))
        parts.append("</VRTRasterBand>")
        bands.append("\n".join(parts))
    xml = (
        f"<VRTDataset rasterXSize=\"{width}\" rasterYSize=\"{height}\">\n"
        "<SRS>GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"Degree\",0.017453292519943295]]</SRS>\n"
        f"<GeoTransform>{geotransform}</GeoTransform>\n"
        f"{chr(10).join(bands)}\n"
        "</VRTDataset>\n"
    )
    out_path.write_text(xml, encoding="utf-8")


def run_gdal_translate(
    *,
    gdal_translate: str,
    gdal_prj: str,
    vrt_path: Path,
    tif_path: Path,
    bounds: tuple[float, float, float, float],
) -> None:
    left, top, right, bottom = bounds
    command = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        (
            f"& '{gdal_translate}' -of GTiff "
            "-co COMPRESS=JPEG -co TILED=YES -co BLOCKXSIZE=256 -co BLOCKYSIZE=256 "
            "-co PHOTOMETRIC=YCBCR -co BIGTIFF=YES -co TFW=YES "
            f"-a_srs '{gdal_prj}' "
            f"-a_ullr {left} {top} {right} {bottom} "
            f"'{to_windows_path(vrt_path)}' "
            f"'{to_windows_path(tif_path)}'"
        ),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"gdal_translate failed with code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)


def main() -> None:
    args = parse_args()
    task_dir = args.task_dir.resolve()
    task_name = infer_task_name(task_dir)
    existing_tifs = find_existing_tifs(task_dir, task_name)
    if args.zoom is None and len(existing_tifs) == 1 and not args.force:
        only_zoom, only_path = next(iter(existing_tifs.items()))
        print(f"Existing TIFF found for zoom {only_zoom}: {only_path}")
        return

    list_path = task_dir / f"{task_name}_list1.txt"
    if not list_path.exists():
        raise SystemExit(f"list file not found: {list_path}")

    records = load_records(list_path)
    zoom_values = sorted({record.zoom for record in records})
    zoom = args.zoom or zoom_values[0]
    if zoom in existing_tifs and not args.force:
        print(f"Existing TIFF found for zoom {zoom}: {existing_tifs[zoom]}")
        return
    selected = [record for record in records if record.zoom == zoom]
    if not selected:
        raise SystemExit(f"No records found for zoom {zoom}")

    combined_dir = task_dir / f"{task_name}_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{task_name}_zoom_{zoom}.tif"
    tif_path = combined_dir / output_name

    if tif_path.exists() and not args.force:
        print(f"Existing TIFF found: {tif_path}")
        return

    x_values = sorted({record.x for record in selected})
    y_values = sorted({record.y for record in selected})
    min_x, max_x = x_values[0], x_values[-1]
    min_y, max_y = y_values[0], y_values[-1]
    tile_size = 256
    cols = max_x - min_x + 1
    rows = max_y - min_y + 1

    overall_left = min(record.left for record in selected)
    overall_right = max(record.right for record in selected)
    overall_top = max(record.top for record in selected)
    overall_bottom = min(record.bottom for record in selected)
    pixel_width = (overall_right - overall_left) / (cols * tile_size)
    pixel_height = (overall_top - overall_bottom) / (rows * tile_size)

    tile_root = task_dir / task_name / str(zoom)
    missing_tiles = [
        record.filename
        for record in selected
        if not (tile_root / str(record.x) / record.filename).exists()
    ]
    if missing_tiles:
        sample = ", ".join(missing_tiles[:5])
        raise SystemExit(
            f"JPEG tiles are missing under {tile_root}. Example missing files: {sample}. "
            "GEID may already have deleted temp tiles after combining."
        )

    groups: dict[int, list[TileRecord]] = defaultdict(list)
    for record in selected:
        groups[record.x].append(record)

    vrt_root = task_dir / task_name
    vrt_root.mkdir(parents=True, exist_ok=True)
    for x_value in x_values:
        column_path = vrt_root / f"{zoom}_{x_value}.vrt"
        if not column_path.exists() or args.force:
            write_column_vrt(
                column_path,
                groups[x_value],
                min_y=min_y,
                max_y=max_y,
                overall_top=overall_top,
                overall_left=min(record.left for record in groups[x_value]),
                pixel_width=pixel_width,
                pixel_height=pixel_height,
                tile_size=tile_size,
            )
            print(f"Wrote {column_path}")

    main_vrt = vrt_root / f"{zoom}.vrt"
    if not main_vrt.exists() or args.force:
        write_main_vrt(
            main_vrt,
            x_values,
            rows=rows,
            overall_top=overall_top,
            overall_left=overall_left,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            tile_size=tile_size,
            zoom=zoom,
        )
        print(f"Wrote {main_vrt}")

    if args.vrt_only:
        return

    run_gdal_translate(
        gdal_translate=args.gdal_translate,
        gdal_prj=args.gdal_prj,
        vrt_path=main_vrt,
        tif_path=tif_path,
        bounds=(overall_left, overall_top, overall_right, overall_bottom),
    )
    print(f"Wrote {tif_path}")


if __name__ == "__main__":
    main()
