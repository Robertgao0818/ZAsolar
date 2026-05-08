#!/usr/bin/env python3
"""Chip-boundary diagnostic: overlay V3C predictions + chip stride grid +
TIF boundary on each ≥600 m² polygon thumbnail.

Goal: confirm or refute the hypothesis that V3C clean-cutoffs on large panel
arrays align with chip stride boundaries (chip_size=400, stride=300 px →
26.8 m / 20.1 m at Vexcel 6.7 cm GSD).

Layers (drawn from back to front):
  - Vexcel imagery (Web Mercator)
  - White DASHED lines: chip stride grid (every 300 px from each TIF origin)
  - Orange SOLID lines: TIF boundary (physical chunk seam)
  - Cyan filled polygons: V3C predictions intersecting the view
  - Red/Blue thick outline: GT polygon (red=SAM_added, blue=V3C_correct)

Output:
  results/analysis/sam_supp_audit/ge600_diag_thumbs/*.png
  results/analysis/sam_supp_audit/ge600_diagnostic.html
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from rasterio.merge import merge as rio_merge
from shapely.geometry import box

REPO = Path("/home/gaosh/projects/ZAsolar")
DATA = Path("/home/gaosh/zasolar_data")
TILES_ROOT = DATA / "tiles/johannesburg/vexcel_2024"
V3C_ROOT = REPO / "results/johannesburg/v3c_vexcel_2024"
INPUT_GPKG = REPO / "results/analysis/sam_supp_audit/ge600_polygons.gpkg"
OUT_DIR = REPO / "results/analysis/sam_supp_audit/ge600_diag_thumbs"
OUT_HTML = REPO / "results/analysis/sam_supp_audit/ge600_diagnostic.html"

TILE_CRS = "EPSG:3857"
PAD_M = 25.0  # bigger pad: see V3C cutoff context outside GT polygon
THUMB_LONG_EDGE = 720
CHIP_SIZE_PX = 400
CHIP_STRIDE_PX = 300

POOL_COLOR = {"SAM_added": (220, 50, 50), "V3C_correct": (40, 110, 220)}
V3C_FILL = (0, 200, 220, 70)
V3C_EDGE = (0, 220, 240, 220)
V3C_BBOX_COLOR = (255, 230, 0, 230)  # yellow envelope = detection box footprint
CHIP_GRID_COLOR = (255, 255, 255, 220)
TIF_BOUNDARY_COLOR = (255, 150, 0, 240)
CHIP_GRID_WIDTH = 2
CHIP_GRID_DASH = 14


def grid_tile_files(grid: str) -> list[Path]:
    return sorted((TILES_ROOT / grid).glob(f"{grid}_*_geo.tif"))


def covering_tifs(grid: str, bbox: tuple[float, float, float, float]) -> list[Path]:
    bb = box(*bbox)
    out = []
    for p in grid_tile_files(grid):
        with rasterio.open(p) as src:
            if box(*src.bounds).intersects(bb):
                out.append(p)
    return out


def crop_polygon(tifs: list[Path], pad_bbox: tuple[float, float, float, float]):
    if not tifs:
        return None
    sources = [rasterio.open(p) for p in tifs]
    try:
        mosaic, _ = rio_merge(sources, bounds=pad_bbox)
    finally:
        for s in sources:
            s.close()
    arr = mosaic[:3]
    return np.transpose(arr, (1, 2, 0))


def load_v3c_predictions(grid: str) -> gpd.GeoDataFrame:
    p = V3C_ROOT / grid / "predictions_metric.gpkg"
    if not p.exists():
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:32735")
    g = gpd.read_file(p)
    if str(g.crs) != "EPSG:32735":
        g = g.to_crs("EPSG:32735")
    return g


def render(idx: int, grid: str, gt_geom_3857, pool: str, area_m2: float,
           v3c_in_view_3857: gpd.GeoSeries, tifs: list[Path]) -> Path | None:
    minx, miny, maxx, maxy = gt_geom_3857.bounds
    minx -= PAD_M; miny -= PAD_M; maxx += PAD_M; maxy += PAD_M
    arr = crop_polygon(tifs, (minx, miny, maxx, maxy))
    if arr is None:
        return None

    h, w = arr.shape[:2]
    img = Image.fromarray(arr)
    scale = THUMB_LONG_EDGE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    draw = ImageDraw.Draw(img, "RGBA")

    def proj(x, y):
        return ((x - minx) / (maxx - minx) * new_w,
                (maxy - y) / (maxy - miny) * new_h)

    # --- Layer 1: chip stride grid per TIF (white dashed) ---
    for tif in tifs:
        with rasterio.open(tif) as src:
            tt = src.transform
            tif_origin_x, tif_origin_y = tt.c, tt.f  # top-left in TIF CRS
            px_w, px_h = tt.a, abs(tt.e)
            tif_w_m = src.width * px_w
            tif_h_m = src.height * px_h
        # Chip stride (300 px) along x and y, anchored at TIF origin.
        # Vertical lines at x = origin_x + k*stride*px_w
        stride_m_x = CHIP_STRIDE_PX * px_w
        stride_m_y = CHIP_STRIDE_PX * px_h
        # Iterate stride positions inside (and slightly beyond) view bbox
        kx0 = int(np.floor((minx - tif_origin_x) / stride_m_x))
        kx1 = int(np.ceil((maxx - tif_origin_x) / stride_m_x))
        ky0 = int(np.floor((tif_origin_y - maxy) / stride_m_y))
        ky1 = int(np.ceil((tif_origin_y - miny) / stride_m_y))
        for k in range(kx0, kx1 + 1):
            x = tif_origin_x + k * stride_m_x
            if x < minx or x > maxx:
                continue
            x1, y1 = proj(x, miny)
            x2, y2 = proj(x, maxy)
            _draw_dashed_line(draw, (x1, y1), (x2, y2), CHIP_GRID_COLOR,
                              dash=CHIP_GRID_DASH, width=CHIP_GRID_WIDTH)
        for k in range(ky0, ky1 + 1):
            y = tif_origin_y - k * stride_m_y
            if y < miny or y > maxy:
                continue
            x1, y1 = proj(minx, y)
            x2, y2 = proj(maxx, y)
            _draw_dashed_line(draw, (x1, y1), (x2, y2), CHIP_GRID_COLOR,
                              dash=CHIP_GRID_DASH, width=CHIP_GRID_WIDTH)

        # --- Layer 2: TIF boundary (orange solid) ---
        bx0 = tif_origin_x
        bx1 = tif_origin_x + tif_w_m
        by1 = tif_origin_y
        by0 = tif_origin_y - tif_h_m
        for (xa, ya, xb, yb) in [(bx0, by0, bx1, by0), (bx0, by1, bx1, by1),
                                  (bx0, by0, bx0, by1), (bx1, by0, bx1, by1)]:
            # Clip line to view
            if xa == xb and (xa < minx or xa > maxx):
                continue
            if ya == yb and (ya < miny or ya > maxy):
                continue
            xa_, ya_ = proj(max(minx, min(maxx, xa)), max(miny, min(maxy, ya)))
            xb_, yb_ = proj(max(minx, min(maxx, xb)), max(miny, min(maxy, yb)))
            draw.line([(xa_, ya_), (xb_, yb_)], fill=TIF_BOUNDARY_COLOR, width=2)

    # --- Layer 3a: V3C prediction polygons (cyan filled) ---
    for v3c_geom in v3c_in_view_3857:
        polys = [v3c_geom] if v3c_geom.geom_type == "Polygon" else list(v3c_geom.geoms)
        for poly in polys:
            coords = [proj(x, y) for x, y in poly.exterior.coords]
            draw.polygon(coords, fill=V3C_FILL, outline=V3C_EDGE)

    # --- Layer 3b: V3C envelope (axis-aligned bbox of polygon, yellow dashed) ---
    # Polygon envelope ≈ detection box footprint; if cyan ends INSIDE its own
    # envelope, that's mask-head/threshold cutoff (not RoI box clip).
    for v3c_geom in v3c_in_view_3857:
        env = v3c_geom.envelope
        if env.is_empty:
            continue
        ex0, ey0, ex1, ey1 = env.bounds
        # Draw 4 edges as dashed lines
        corners_proj = [proj(ex0, ey0), proj(ex1, ey0), proj(ex1, ey1), proj(ex0, ey1)]
        for i in range(4):
            p0 = corners_proj[i]
            p1 = corners_proj[(i + 1) % 4]
            _draw_dashed_line(draw, p0, p1, V3C_BBOX_COLOR, dash=10, width=2)

    # --- Layer 4: GT polygon outline ---
    color = POOL_COLOR[pool] + (255,)
    polys = [gt_geom_3857] if gt_geom_3857.geom_type == "Polygon" else list(gt_geom_3857.geoms)
    for poly in polys:
        coords = [proj(x, y) for x, y in poly.exterior.coords]
        draw.line(coords, fill=color, width=4)
        for interior in poly.interiors:
            ic = [proj(x, y) for x, y in interior.coords]
            draw.line(ic, fill=color, width=2)

    label = f"{grid} | {pool} | {area_m2:.0f} m²"
    draw.rectangle([(0, 0), (len(label) * 7 + 12, 20)], fill=(0, 0, 0, 200))
    draw.text((6, 3), label, fill=(255, 255, 255))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{idx:02d}_{grid}_{pool}_{int(area_m2)}.png"
    img.save(out_path, format="PNG", optimize=True)
    return out_path


def _draw_dashed_line(draw, p0, p1, color, dash=8, width=1):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return
    n = max(1, int(length / dash))
    for i in range(0, n, 2):
        t0 = i / n
        t1 = min(1, (i + 1) / n)
        a = (x0 + dx * t0, y0 + dy * t0)
        b = (x0 + dx * t1, y0 + dy * t1)
        draw.line([a, b], fill=color, width=width)


def build_html(records: list[dict]) -> None:
    sam = sorted([r for r in records if r["pool"] == "SAM_added"], key=lambda x: -x["area_m2"])
    v3c = sorted([r for r in records if r["pool"] == "V3C_correct"], key=lambda x: -x["area_m2"])

    def card(r):
        return (
            f'<figure class="card">'
            f'<img src="ge600_diag_thumbs/{r["png"]}" alt="{r["grid"]}">'
            f'<figcaption>{r["grid"]} · {r["area_m2"]:.0f} m² · MRR {r["mrr_fill"]:.3f}</figcaption>'
            f'</figure>'
        )

    sam_html = "\n".join(card(r) for r in sam)
    v3c_html = "\n".join(card(r) for r in v3c)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>≥600 m² chip-boundary diagnostic</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #1a1a1a; color: #eee; margin: 16px; }}
h1 {{ margin-bottom: 4px; }}
h2 {{ margin-top: 28px; padding-bottom: 4px; border-bottom: 2px solid #444; }}
h2.v3c {{ border-color: #2870dc; }}
h2.sam {{ border-color: #dc3232; }}
.summary {{ color: #aaa; font-size: 13px; margin-bottom: 8px; max-width: 980px; }}
.legend {{ background: #2a2a2a; padding: 8px 12px; border-radius: 4px; margin: 8px 0 16px; font-size: 13px; }}
.legend span {{ display: inline-block; margin-right: 18px; }}
.swatch {{ display: inline-block; width: 28px; height: 12px; vertical-align: middle; margin-right: 5px; border: 1px solid #444; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(540px, 1fr)); gap: 14px; }}
.card {{ margin: 0; background: #2a2a2a; border-radius: 6px; overflow: hidden; }}
.card img {{ width: 100%; display: block; }}
figcaption {{ padding: 6px 8px; font-size: 13px; color: #ccc; }}
</style></head>
<body>
<h1>≥600 m² chip-boundary diagnostic — JHB CBD 25 grid</h1>
<p class="summary">
Hypothesis under test: V3C clean-cutoffs on large panel installations align with
inference chip stride (chip 400 px = 26.8 m, stride 300 px = 20.1 m at Vexcel 6.7 cm GSD).
If V3C cyan polygons end abruptly along a white dashed line, hypothesis confirmed —
fix is to raise <code>--overlap</code> or <code>--chip-size</code> at inference.
</p>
<div class="legend">
  <span><span class="swatch" style="background:#dc3232"></span>GT (SAM_added)</span>
  <span><span class="swatch" style="background:#2870dc"></span>GT (V3C_correct)</span>
  <span><span class="swatch" style="background:#00c8dc;opacity:0.5"></span>V3C prediction polygon</span>
  <span><span class="swatch" style="background:#ffe600;border:1px dashed #888"></span>V3C polygon envelope (axis-aligned bbox)</span>
  <span><span class="swatch" style="background:#fff;border:1px dashed #888"></span>Chip stride seam (every 20.1 m)</span>
  <span><span class="swatch" style="background:#ff9600"></span>TIF (chunk) boundary</span>
</div>

<h2 class="v3c">V3C_correct (n={len(v3c)}) — focus pool, look for cutoff alignment</h2>
<div class="grid">
{v3c_html}
</div>

<h2 class="sam">SAM_added (n={len(sam)}) — comparison</h2>
<div class="grid">
{sam_html}
</div>
</body></html>
"""
    OUT_HTML.write_text(html)


def main() -> None:
    gdf = gpd.read_file(INPUT_GPKG)
    if str(gdf.crs) != "EPSG:32735":
        gdf = gdf.to_crs("EPSG:32735")
    mrr_area = gdf.geometry.minimum_rotated_rectangle().area.replace(0, np.nan)
    gdf["mrr_fill"] = (gdf.geometry.area / mrr_area).clip(0, 1).fillna(0)

    # Pre-load V3C predictions per grid (so we don't re-read for each polygon)
    grids_needed = sorted(gdf["grid"].unique())
    v3c_by_grid_3857 = {g: load_v3c_predictions(g).to_crs(TILE_CRS) for g in grids_needed}

    gdf_tile = gdf.to_crs(TILE_CRS)
    records = []
    for idx, (i, row) in enumerate(gdf_tile.iterrows()):
        gt_geom = row.geometry
        bbox = gt_geom.bounds
        pad_bbox = (bbox[0] - PAD_M, bbox[1] - PAD_M, bbox[2] + PAD_M, bbox[3] + PAD_M)
        tifs = covering_tifs(row["grid"], pad_bbox)
        if not tifs:
            print(f"[skip] {row['grid']} — no covering tiles")
            continue

        # V3C predictions intersecting the padded bbox
        v3c_g = v3c_by_grid_3857[row["grid"]]
        view_box = box(*pad_bbox)
        v3c_view = v3c_g[v3c_g.geometry.intersects(view_box)].geometry

        png = render(idx, row["grid"], gt_geom, row["pool"], row["area_m2_calc"],
                     v3c_view, tifs)
        if png is None:
            continue
        records.append({
            "grid": row["grid"], "pool": row["pool"],
            "area_m2": row["area_m2_calc"], "mrr_fill": gdf.iloc[i]["mrr_fill"],
            "png": png.name,
        })
        print(f"[{idx:02d}] {row['grid']} {row['pool']:<12} area={row['area_m2_calc']:7.0f} v3c_in_view={len(v3c_view)}")

    build_html(records)
    print(f"\nWrote HTML → {OUT_HTML}")


if __name__ == "__main__":
    main()
