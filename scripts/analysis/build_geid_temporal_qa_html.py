#!/usr/bin/env python3
"""Side-by-side QA HTML for GEID temporal anchor chips.

Renders, per random anchor, the aerial reference chip (with the source PV
polygon outlined) next to the historical GEID chip stack (one mosaic per
captured year, with embedded JPEG capture-date label).

This is a one-off visual sanity check before Phase-0 manual labelling: are
the anchors hitting actual rooftops where PV was detected, and do the GEID
historical mosaics span the expected location?
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import random
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import geopandas as gpd
import rasterio
from PIL import Image, ImageDraw
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely.geometry import box, mapping
from shapely.ops import transform

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TILE_SIZE = 256

DEFAULT_GPKG = PROJECT_ROOT / "data" / "annotations" / "Joburg" / "G0922_V4_260407.gpkg"
DEFAULT_GEID_ROOT = Path("/mnt/d/ZAsolar/geid_raw/temporal_pv_grid_G0922_full_probe/johannesburg/G0922")
DEFAULT_REFERENCE_DIR = Path("/home/gaosh/zasolar_data/tiles/johannesburg/vexcel_2024/G0922")
REFERENCE_LABEL = "vexcel_2024 (6.7cm)"
DEFAULT_OUTPUT_ROOT = Path("/tmp")

JPEG_COMMENT_RE = re.compile(rb"\*AD\*(\d{4}):(\d{2}):(\d{2})\*")
TASK_DATE_RE = re.compile(r"_(?P<date>\d{8})$")


def extract_capture_date(jpg_bytes: bytes) -> str:
    m = JPEG_COMMENT_RE.search(jpg_bytes[:4096])
    if not m:
        return ""
    return f"{m.group(1).decode()}-{m.group(2).decode()}-{m.group(3).decode()}"


def _draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, *, color=(255, 225, 0), r: int = 12) -> None:
    draw.line([(x - r, y), (x + r, y)], fill=color, width=3)
    draw.line([(x, y - r), (x, y + r)], fill=color, width=3)


def _geid_mosaic_bounds(xs: list[int], ys: list[int], zoom: int) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) for GEID's equirectangular tile grid."""
    factor = (1 << (zoom - 1)) / 360.0
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    west = min_x / factor - 180.0
    east = (max_x + 1) / factor - 180.0
    south = min_y / factor - 180.0
    north = (max_y + 1) / factor - 180.0
    return west, south, east, north


def _lonlat_to_pixel(
    lon: float,
    lat: float,
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float]:
    west, south, east, north = bounds
    x = (lon - west) / (east - west) * width
    y = (north - lat) / (north - south) * height
    return x, y


def stitch_geid_tiles(task_dir: Path) -> tuple[Image.Image | None, str, tuple[float, float, float, float] | None]:
    """Stitch the 2x2 (or NxM) z21 tiles inside a task dir into a single PIL image.

    GEID's tile y index grows northward, so the largest y tile belongs at the
    top of the rendered mosaic.

    Returns (image, capture_date_str, bounds). Capture date is read from the
    first JPEG. Bounds are (west, south, east, north) in EPSG:4326.
    """
    jpgs = sorted(task_dir.rglob("*.jpg"))
    if not jpgs:
        return None, "", None

    coords: list[tuple[int, int, int, Path]] = []
    for jpg in jpgs:
        m = re.match(r"gesh_(\d+)_(\d+)_(\d+)\.jpg$", jpg.name)
        if not m:
            continue
        x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
        coords.append((x, y, z, jpg))
    if not coords:
        return None, "", None

    xs = sorted({c[0] for c in coords})
    ys = sorted({c[1] for c in coords})
    zooms = sorted({c[2] for c in coords})
    if len(zooms) != 1:
        raise ValueError(f"mixed zoom levels in {task_dir}: {zooms}")
    zoom = zooms[0]
    min_x, max_y = min(xs), max(ys)
    out = Image.new("RGB", (TILE_SIZE * len(xs), TILE_SIZE * len(ys)))
    capture_date = ""
    for x, y, _z, jpg in coords:
        data = jpg.read_bytes()
        if not capture_date:
            capture_date = extract_capture_date(data)
        try:
            tile = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            continue
        col = x - min_x
        row = max_y - y
        out.paste(tile.resize((TILE_SIZE, TILE_SIZE)), (col * TILE_SIZE, row * TILE_SIZE))
    return out, capture_date, _geid_mosaic_bounds(xs, ys, zoom)


def crop_geid_chip(
    img: Image.Image,
    bounds: tuple[float, float, float, float],
    crop_bounds: tuple[float, float, float, float],
    centroid_lon: float,
    centroid_lat: float,
    out_path: Path,
    target_size: int = 512,
) -> bool:
    lon_min, lat_min, lon_max, lat_max = crop_bounds
    west, south, east, north = min(lon_min, lon_max), min(lat_min, lat_max), max(lon_min, lon_max), max(lat_min, lat_max)
    x0, y0 = _lonlat_to_pixel(west, north, bounds, img.width, img.height)
    x1, y1 = _lonlat_to_pixel(east, south, bounds, img.width, img.height)
    left, top, right, bottom = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    if right <= left or bottom <= top:
        return False
    crop = img.crop((left, top, right, bottom)).resize((target_size, target_size))
    cx, cy = _lonlat_to_pixel(centroid_lon, centroid_lat, bounds, img.width, img.height)
    scale_x = target_size / max(1, right - left)
    scale_y = target_size / max(1, bottom - top)
    draw = ImageDraw.Draw(crop)
    _draw_cross(draw, (cx - left) * scale_x, (cy - top) * scale_y)
    crop.save(out_path, "JPEG", quality=88)
    return True


def _wgs_to_src(src_crs) -> Transformer:
    return Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)


def find_aerial_chunk(reference_dir: Path, lon: float, lat: float) -> Path | None:
    for tif in sorted(reference_dir.glob("*_geo.tif")):
        with rasterio.open(tif) as src:
            if str(src.crs).upper() in ("EPSG:4326", ""):
                x, y = lon, lat
            else:
                x, y = _wgs_to_src(src.crs).transform(lon, lat)
            l, b, r, t = src.bounds
            if l <= x <= r and b <= y <= t:
                return tif
    return None


def crop_aerial_chip(
    aerial_path: Path,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    polygon_lonlat,
    out_path: Path,
    target_size: int = 512,
) -> bool:
    with rasterio.open(aerial_path) as src:
        if str(src.crs).upper() in ("EPSG:4326", ""):
            x_min, y_min, x_max, y_max = lon_min, lat_min, lon_max, lat_max
            poly_src = polygon_lonlat
        else:
            wgs2src = _wgs_to_src(src.crs)
            x_min, y_min = wgs2src.transform(lon_min, lat_min)
            x_max, y_max = wgs2src.transform(lon_max, lat_max)
            if polygon_lonlat is not None:
                poly_src = transform(lambda xs, ys, zs=None: wgs2src.transform(xs, ys), polygon_lonlat)
            else:
                poly_src = None
        try:
            window = from_bounds(x_min, y_min, x_max, y_max, src.transform)
        except Exception:
            return False
        window = window.round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            return False
        data = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
        win_transform = src.window_transform(window)

    img = Image.fromarray(data.transpose(1, 2, 0)).convert("RGB")
    if img.width == 0 or img.height == 0:
        return False
    scale_x = target_size / img.width
    scale_y = target_size / img.height
    img = img.resize((target_size, target_size))
    draw = ImageDraw.Draw(img)
    if poly_src is not None:
        coords_px = []
        for x, y in poly_src.exterior.coords:
            col, row = ~win_transform * (x, y)
            coords_px.append((col * scale_x, row * scale_y))
        draw.line(coords_px + coords_px[:1], fill=(255, 60, 60), width=3)
    _draw_cross(draw, target_size / 2, target_size / 2)
    img.save(out_path, "JPEG", quality=88)
    return True


def list_year_tasks(anchor_root: Path) -> list[tuple[str, Path]]:
    """Return [(task_name, task_dir), ...] sorted by embedded date."""
    out: list[tuple[str, Path]] = []
    for year_dir in sorted(p for p in anchor_root.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
            m = TASK_DATE_RE.search(task_dir.name)
            if not m:
                continue
            out.append((task_dir.name, task_dir))
    out.sort(key=lambda kv: kv[0])
    return out


def read_anchor_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_anchor_row(
    *,
    anchor_id: str,
    feature_idx: int,
    polygon_4326,
    centroid_lon: float,
    centroid_lat: float,
    chip_half_m: float,
    reference_dir: Path,
    geid_anchor_root: Path,
    output_dir: Path,
    metric_to_wgs: Transformer,
    wgs_to_metric: Transformer,
) -> dict:
    cx_m, cy_m = wgs_to_metric.transform(centroid_lon, centroid_lat)
    lon_min, lat_min = metric_to_wgs.transform(cx_m - chip_half_m, cy_m - chip_half_m)
    lon_max, lat_max = metric_to_wgs.transform(cx_m + chip_half_m, cy_m + chip_half_m)
    crop_bounds = (lon_min, lat_min, lon_max, lat_max)

    aerial_chunk = find_aerial_chunk(reference_dir, centroid_lon, centroid_lat)
    aerial_rel = ""
    if aerial_chunk is not None:
        aerial_out = output_dir / f"{anchor_id}_aerial.jpg"
        if crop_aerial_chip(
            aerial_chunk,
            lon_min,
            lat_min,
            lon_max,
            lat_max,
            polygon_4326,
            aerial_out,
        ):
            aerial_rel = aerial_out.name

    geid_panels: list[dict] = []
    for task_name, task_dir in list_year_tasks(geid_anchor_root):
        img, capture_date, bounds = stitch_geid_tiles(task_dir)
        m = TASK_DATE_RE.search(task_name)
        requested_date = ""
        if m:
            d = m.group("date")
            requested_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        rel = ""
        if img is not None and bounds is not None:
            geid_out = output_dir / f"{anchor_id}_{task_name}.jpg"
            if crop_geid_chip(
                img,
                bounds,
                crop_bounds,
                centroid_lon,
                centroid_lat,
                geid_out,
            ):
                rel = geid_out.name
        geid_panels.append(
            {
                "task_name": task_name,
                "requested_date": requested_date,
                "capture_date": capture_date,
                "image": rel,
            }
        )

    return {
        "anchor_id": anchor_id,
        "feature_idx": feature_idx,
        "centroid_lon": centroid_lon,
        "centroid_lat": centroid_lat,
        "chip_half_m": chip_half_m,
        "aerial_image": aerial_rel,
        "geid_panels": geid_panels,
    }


def render_html(rows: list[dict], output_dir: Path) -> Path:
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>GEID temporal anchor QA</title>",
        "<style>",
        "body{font-family:-apple-system,Segoe UI,sans-serif;margin:16px;background:#111;color:#eee}",
        "h1{font-size:20px;margin:0 0 12px} h2{font-size:14px;margin:24px 0 8px}",
        ".row{display:flex;flex-wrap:nowrap;overflow-x:auto;gap:8px;border-bottom:1px solid #333;padding:12px 0;align-items:flex-start}",
        ".cell{flex:0 0 auto;width:240px;text-align:center;font-size:11px;line-height:1.3}",
        ".cell.aerial{border:2px solid #4af;border-radius:4px;padding:4px}",
        ".cell.geid{border:1px solid #444;border-radius:4px;padding:4px}",
        ".cell img{width:100%;height:auto;display:block;border-radius:2px}",
        ".meta{margin-top:4px;color:#aaa}",
        ".cap{font-weight:bold;color:#fc8}",
        ".bad{color:#f66}",
        "</style></head><body>",
        "<h1>GEID temporal anchor QA — visual sanity check</h1>",
        f"<p>{len(rows)} anchors sampled. Blue border = aerial reference (where main-repo annotated PV polygon, drawn red). Right of it = historical GEID chip mosaics by capture year.</p>",
    ]
    for row in rows:
        parts.append(
            f"<h2>{html.escape(row['anchor_id'])} &middot; "
            f"feature_idx={row['feature_idx']} &middot; "
            f"centroid=({row['centroid_lon']:.6f}, {row['centroid_lat']:.6f}) &middot; "
            f"chip_half={row['chip_half_m']:.1f}m</h2>"
        )
        parts.append("<div class='row'>")
        if row["aerial_image"]:
            parts.append(
                f"<div class='cell aerial'><img src='{html.escape(row['aerial_image'])}'>"
                f"<div class='meta'><b>{html.escape(REFERENCE_LABEL)}</b><br>(red = source polygon)</div></div>"
            )
        else:
            parts.append("<div class='cell aerial bad'>no aerial chunk found</div>")
        for panel in row["geid_panels"]:
            cap = panel["capture_date"] or "<span class='bad'>NO_AD</span>"
            req = panel["requested_date"]
            note = ""
            if panel["capture_date"] and req and panel["capture_date"][:4] != req[:4]:
                note = "<div class='bad'>date drift</div>"
            if not panel["image"]:
                parts.append(
                    f"<div class='cell geid bad'>req {html.escape(req)}<br>no jpg</div>"
                )
                continue
            parts.append(
                f"<div class='cell geid'><img src='{html.escape(panel['image'])}'>"
                f"<div class='meta'>req {html.escape(req)}<br>"
                f"capture <span class='cap'>{cap}</span>{note}</div></div>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    out = output_dir / "index.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotation-gpkg", type=Path, default=DEFAULT_GPKG)
    ap.add_argument("--geid-root", type=Path, default=DEFAULT_GEID_ROOT)
    ap.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR,
                    help=f"Reference imagery dir (chunked, EPSG:4326). Default: vexcel_2024 G0922.")
    ap.add_argument("--region-key", default="johannesburg")
    ap.add_argument("--grid-id", default="G0922")
    ap.add_argument(
        "--anchors-csv",
        type=Path,
        help="Optional anchor manifest. When set, render these anchor_id rows in order instead of random GPKG features.",
    )
    ap.add_argument("--metric-crs", default="EPSG:32735")
    ap.add_argument("--n-anchors", type=int, default=8)
    ap.add_argument("--chip-half-m", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--port", type=int, default=8765)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.annotation_gpkg.exists():
        raise SystemExit(f"Annotation GPKG missing: {args.annotation_gpkg}")
    if not args.geid_root.exists():
        raise SystemExit(f"GEID root missing: {args.geid_root}")
    if not args.reference_dir.exists():
        raise SystemExit(f"Aerial dir missing: {args.reference_dir}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"geid_temporal_qa_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(args.annotation_gpkg).to_crs("EPSG:4326")
    if gdf.empty:
        raise SystemExit("Annotation GPKG has no rows.")
    rng = random.Random(args.seed)
    indices = list(range(len(gdf)))
    rng.shuffle(indices)

    metric_to_wgs = Transformer.from_crs(args.metric_crs, "EPSG:4326", always_xy=True)
    wgs_to_metric = Transformer.from_crs("EPSG:4326", args.metric_crs, always_xy=True)

    rows = []
    if args.anchors_csv:
        if not args.anchors_csv.exists():
            raise SystemExit(f"Anchors CSV missing: {args.anchors_csv}")
        render_specs = []
        for anchor in read_anchor_rows(args.anchors_csv):
            anchor_id = str(anchor.get("anchor_id", "")).strip()
            feature_idx = int(anchor.get("source_feature_id", ""))
            render_specs.append(
                {
                    "anchor_id": anchor_id,
                    "feature_idx": feature_idx,
                    "centroid_lon": float(anchor["centroid_lon"]),
                    "centroid_lat": float(anchor["centroid_lat"]),
                    "chip_half_m": float(anchor.get("chip_half_m") or args.chip_half_m),
                }
            )
    else:
        render_specs = []
        for feature_idx in indices:
            local_idx = feature_idx + 1
            render_specs.append(
                {
                    "anchor_id": f"{args.region_key}_{args.grid_id}_a{local_idx:06d}",
                    "feature_idx": feature_idx,
                    "centroid_lon": None,
                    "centroid_lat": None,
                    "chip_half_m": args.chip_half_m,
                }
            )

    used = 0
    for spec in render_specs:
        if used >= args.n_anchors:
            break
        anchor_id = spec["anchor_id"]
        feature_idx = int(spec["feature_idx"])
        geid_anchor_root = args.geid_root / anchor_id
        if not geid_anchor_root.exists():
            print(f"[SKIP] no GEID dir for {anchor_id}", file=sys.stderr)
            continue
        geom = gdf.geometry.iloc[feature_idx]
        if geom is None or geom.is_empty:
            continue
        if spec["centroid_lon"] is None or spec["centroid_lat"] is None:
            centroid = geom.centroid
            centroid_lon = float(centroid.x)
            centroid_lat = float(centroid.y)
        else:
            centroid_lon = float(spec["centroid_lon"])
            centroid_lat = float(spec["centroid_lat"])
        try:
            row = build_anchor_row(
                anchor_id=anchor_id,
                feature_idx=feature_idx,
                polygon_4326=geom,
                centroid_lon=centroid_lon,
                centroid_lat=centroid_lat,
                chip_half_m=float(spec["chip_half_m"]),
                reference_dir=args.reference_dir,
                geid_anchor_root=geid_anchor_root,
                output_dir=output_dir,
                metric_to_wgs=metric_to_wgs,
                wgs_to_metric=wgs_to_metric,
            )
        except Exception as exc:
            print(f"[ERR] {anchor_id}: {exc}", file=sys.stderr)
            continue
        rows.append(row)
        used += 1

    if not rows:
        raise SystemExit("No anchors rendered.")
    index_path = render_html(rows, output_dir)
    print(f"Wrote {len(rows)} anchors -> {index_path}")
    print(f"Output dir: {output_dir}")
    print(f"To serve: python3 -m http.server {args.port} --directory {output_dir}")
    print(f"Then open: http://localhost:{args.port}/")


if __name__ == "__main__":
    main()
