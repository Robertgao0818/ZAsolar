#!/usr/bin/env python3
"""Render PNG thumbnails + HTML grid for the 26 polygons ≥600 m² in the
JHB CBD 25-grid SAM-supp / V3C-reviewed reliability audit.

For each polygon:
  - Pad bbox by 8 m
  - Find covering Vexcel tiles, mosaic, crop to padded bbox
  - Draw polygon outline (red for SAM_added, blue for V3C_correct)
  - Save PNG
Then build a single HTML page with all 26 thumbnails grouped by pool.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from rasterio.windows import from_bounds
from shapely.geometry import box

REPO = Path("/home/gaosh/projects/ZAsolar")
DATA = Path("/home/gaosh/zasolar_data")
TILES_ROOT = DATA / "tiles/johannesburg/vexcel_2024"
INPUT_GPKG = REPO / "results/analysis/sam_supp_audit/ge600_polygons.gpkg"
OUT_DIR = REPO / "results/analysis/sam_supp_audit/ge600_thumbs"
OUT_HTML = REPO / "results/analysis/sam_supp_audit/ge600_review.html"
TILE_CRS = "EPSG:3857"  # Vexcel tiles are stored in Web Mercator
PAD_M = 10.0  # ~9m in UTM after 3857 scale distortion at JHB latitude
THUMB_LONG_EDGE = 480

POOL_COLOR = {"SAM_added": (220, 50, 50), "V3C_correct": (40, 110, 220)}


def grid_tile_files(grid: str) -> list[Path]:
    return sorted((TILES_ROOT / grid).glob(f"{grid}_*_geo.tif"))


def crop_polygon(grid: str, geom, pad_m: float) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    minx, miny, maxx, maxy = geom.bounds
    minx -= pad_m; miny -= pad_m; maxx += pad_m; maxy += pad_m
    bbox = box(minx, miny, maxx, maxy)

    tile_files = grid_tile_files(grid)
    if not tile_files:
        return None

    covering = []
    for p in tile_files:
        with rasterio.open(p) as src:
            tb = box(*src.bounds)
            if tb.intersects(bbox):
                covering.append(p)
    if not covering:
        return None

    sources = [rasterio.open(p) for p in covering]
    try:
        mosaic, mtransform = rio_merge(sources, bounds=(minx, miny, maxx, maxy))
    finally:
        for s in sources:
            s.close()

    arr = mosaic[:3]
    arr = np.transpose(arr, (1, 2, 0))
    return arr, (minx, miny, maxx, maxy)


def render_polygon(grid: str, geom, pool: str, area_m2: float, idx: int) -> Path | None:
    cropped = crop_polygon(grid, geom, PAD_M)
    if cropped is None:
        return None
    arr, (minx, miny, maxx, maxy) = cropped
    h, w = arr.shape[:2]
    img = Image.fromarray(arr)

    scale = THUMB_LONG_EDGE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    draw = ImageDraw.Draw(img, "RGBA")
    color = POOL_COLOR[pool] + (255,)

    def proj(x, y):
        px = (x - minx) / (maxx - minx) * new_w
        py = (maxy - y) / (maxy - miny) * new_h
        return (px, py)

    geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for poly in geoms:
        coords = [proj(x, y) for x, y in poly.exterior.coords]
        draw.line(coords, fill=color, width=3)
        for interior in poly.interiors:
            ic = [proj(x, y) for x, y in interior.coords]
            draw.line(ic, fill=color + (), width=2)

    label = f"{grid} | {pool} | {area_m2:.0f} m²"
    draw.rectangle([(0, 0), (len(label) * 7 + 8, 18)], fill=(0, 0, 0, 180))
    draw.text((4, 2), label, fill=(255, 255, 255))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{idx:02d}_{grid}_{pool}_{int(area_m2)}.png"
    img.save(out_path, format="PNG", optimize=True)
    return out_path


def build_html(records: list[dict]) -> None:
    sam = [r for r in records if r["pool"] == "SAM_added"]
    v3c = [r for r in records if r["pool"] == "V3C_correct"]

    def card(r):
        return (
            f'<figure class="card">'
            f'<img src="ge600_thumbs/{r["png"]}" alt="{r["grid"]}">'
            f'<figcaption>{r["grid"]} · {r["area_m2"]:.0f} m² · MRR {r["mrr_fill"]:.3f}</figcaption>'
            f'</figure>'
        )

    sam_html = "\n".join(card(r) for r in sorted(sam, key=lambda x: -x["area_m2"]))
    v3c_html = "\n".join(card(r) for r in sorted(v3c, key=lambda x: -x["area_m2"]))

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>≥600 m² polygons — SAM_added vs V3C_correct review</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #1a1a1a; color: #eee; margin: 16px; }}
h1 {{ margin-bottom: 4px; }}
h2 {{ margin-top: 28px; padding-bottom: 4px; border-bottom: 2px solid #444; }}
h2.sam {{ border-color: #dc3232; }}
h2.v3c {{ border-color: #2870dc; }}
.summary {{ color: #aaa; font-size: 13px; margin-bottom: 8px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; }}
.card {{ margin: 0; background: #2a2a2a; border-radius: 6px; overflow: hidden; }}
.card img {{ width: 100%; display: block; }}
figcaption {{ padding: 6px 8px; font-size: 13px; color: #ccc; }}
</style></head>
<body>
<h1>JHB CBD 25 grid · ≥600 m² polygons review</h1>
<p class="summary">Audit context: SAM_added p50 MRR-fill drops below V3C_correct in this bucket
(SAM 0.800 vs V3C 0.852). Visual check whether SAM polygons here show 吞屋顶 / over-extension
and whether V3C polygons show clean rectangles or large halo. Outline = polygon boundary,
imagery = Vexcel za-gp-johannesburg-2024 (6.7 cm GSD), 8 m padding.</p>

<h2 class="sam">SAM_added — n={len(sam)}, expected 吞屋顶 tail</h2>
<div class="grid">
{sam_html}
</div>

<h2 class="v3c">V3C_correct — n={len(v3c)}, comparison</h2>
<div class="grid">
{v3c_html}
</div>
</body></html>
"""
    OUT_HTML.write_text(html)


def main() -> None:
    gdf = gpd.read_file(INPUT_GPKG)
    if str(gdf.crs) != "EPSG:32735":
        gdf = gdf.to_crs("EPSG:32735")

    # Compute mrr_fill in metric CRS for caption (true m² geometry)
    mrr_area = gdf.geometry.minimum_rotated_rectangle().area.replace(0, np.nan)
    gdf["mrr_fill"] = (gdf.geometry.area / mrr_area).clip(0, 1).fillna(0)

    # Reproject geometry to tile CRS for crop/draw
    gdf_tile = gdf.to_crs(TILE_CRS)
    gdf["geom_tile"] = gdf_tile.geometry
    gdf = gdf.sort_values(["pool", "area_m2_calc"], ascending=[True, False]).reset_index(drop=True)

    records = []
    for idx, row in gdf.iterrows():
        png = render_polygon(row["grid"], row["geom_tile"], row["pool"], row["area_m2_calc"], idx)
        if png is None:
            print(f"[skip] {row['grid']} — no covering tiles")
            continue
        records.append({
            "grid": row["grid"], "pool": row["pool"],
            "area_m2": row["area_m2_calc"], "mrr_fill": row["mrr_fill"],
            "png": png.name,
        })
        print(f"[{idx:02d}] {row['grid']} {row['pool']:<12} area={row['area_m2_calc']:7.0f} mrr={row['mrr_fill']:.3f} → {png.name}")

    build_html(records)
    print(f"\nWrote HTML → {OUT_HTML}")
    print(f"Open with:  xdg-open {OUT_HTML}")


if __name__ == "__main__":
    main()
