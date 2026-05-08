#!/usr/bin/env python3
"""Side-by-side comparison: G0816 ≥600 m² polygons under V3C overlap=0.25 vs 0.5.

Renders one row per polygon with:
  left  = current production V3C predictions (overlap=0.25)
  right = overlap=0.5 probe predictions

If chip-edge cutoffs (cat 2) close up in the right column, hypothesis confirmed
that --overlap 0.5 is a viable inference-only fix. If gaps remain, the issue
is deeper than chip stride.

Output:
  results/analysis/sam_supp_audit/g0816_overlap_compare.html
  results/analysis/sam_supp_audit/g0816_overlap_compare/*.png
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.merge import merge as rio_merge
from shapely.geometry import box

REPO = Path("/home/gaosh/projects/ZAsolar")
DATA = Path("/home/gaosh/zasolar_data")
TILES_ROOT = DATA / "tiles/johannesburg/vexcel_2024"
INPUT_GPKG = REPO / "results/analysis/sam_supp_audit/ge600_polygons.gpkg"
V3C_25 = REPO / "results/johannesburg/v3c_vexcel_2024/G0816/predictions_metric.gpkg"
V3C_50 = REPO / "results/diag/v3c_overlap50_G0816/predictions_metric.gpkg"
OUT_DIR = REPO / "results/analysis/sam_supp_audit/g0816_overlap_compare"
OUT_HTML = REPO / "results/analysis/sam_supp_audit/g0816_overlap_compare.html"

TILE_CRS = "EPSG:3857"
PAD_M = 25.0
THUMB_LONG_EDGE = 720
CHIP_STRIDE_PX_25 = 300
CHIP_STRIDE_PX_50 = 200

V3C_FILL = (0, 200, 220, 70)
V3C_EDGE = (0, 220, 240, 220)
GT_COLOR = (220, 50, 50, 255)
CHIP_GRID = (255, 255, 255, 220)
TIF_BOUNDARY = (255, 150, 0, 240)


def grid_tile_files(grid: str) -> list[Path]:
    return sorted((TILES_ROOT / grid).glob(f"{grid}_*_geo.tif"))


def covering_tifs(grid, bbox):
    bb = box(*bbox)
    out = []
    for p in grid_tile_files(grid):
        with rasterio.open(p) as src:
            if box(*src.bounds).intersects(bb):
                out.append(p)
    return out


def crop(tifs, pad_bbox):
    sources = [rasterio.open(p) for p in tifs]
    try:
        mosaic, _ = rio_merge(sources, bounds=pad_bbox)
    finally:
        for s in sources:
            s.close()
    return np.transpose(mosaic[:3], (1, 2, 0))


def _dashed(draw, p0, p1, color, dash=12, width=2):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L = (dx * dx + dy * dy) ** 0.5
    if L == 0:
        return
    n = max(1, int(L / dash))
    for i in range(0, n, 2):
        t0 = i / n
        t1 = min(1, (i + 1) / n)
        draw.line([(x0 + dx * t0, y0 + dy * t0),
                   (x0 + dx * t1, y0 + dy * t1)], fill=color, width=width)


def render_one(gt_geom_3857, v3c_geoms_3857, tifs, stride_px: int) -> Image.Image:
    minx, miny, maxx, maxy = gt_geom_3857.bounds
    minx -= PAD_M; miny -= PAD_M; maxx += PAD_M; maxy += PAD_M
    arr = crop(tifs, (minx, miny, maxx, maxy))
    h, w = arr.shape[:2]
    img = Image.fromarray(arr)
    scale = THUMB_LONG_EDGE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    draw = ImageDraw.Draw(img, "RGBA")

    def proj(x, y):
        return ((x - minx) / (maxx - minx) * new_w,
                (maxy - y) / (maxy - miny) * new_h)

    # chip stride grid (per TIF)
    for tif in tifs:
        with rasterio.open(tif) as src:
            tt = src.transform
            ox, oy = tt.c, tt.f
            px_w, px_h = tt.a, abs(tt.e)
            tw, th = src.width * px_w, src.height * px_h
        sx = stride_px * px_w
        sy = stride_px * px_h
        kx0 = int(np.floor((minx - ox) / sx))
        kx1 = int(np.ceil((maxx - ox) / sx))
        ky0 = int(np.floor((oy - maxy) / sy))
        ky1 = int(np.ceil((oy - miny) / sy))
        for k in range(kx0, kx1 + 1):
            x = ox + k * sx
            if minx <= x <= maxx:
                _dashed(draw, proj(x, miny), proj(x, maxy), CHIP_GRID, dash=14, width=2)
        for k in range(ky0, ky1 + 1):
            y = oy - k * sy
            if miny <= y <= maxy:
                _dashed(draw, proj(minx, y), proj(maxx, y), CHIP_GRID, dash=14, width=2)
        # TIF boundary
        bx0, bx1, by0, by1 = ox, ox + tw, oy - th, oy
        for (xa, ya, xb, yb) in [(bx0, by0, bx1, by0), (bx0, by1, bx1, by1),
                                  (bx0, by0, bx0, by1), (bx1, by0, bx1, by1)]:
            if xa == xb and not (minx <= xa <= maxx):
                continue
            if ya == yb and not (miny <= ya <= maxy):
                continue
            xa_ = max(minx, min(maxx, xa)); ya_ = max(miny, min(maxy, ya))
            xb_ = max(minx, min(maxx, xb)); yb_ = max(miny, min(maxy, yb))
            draw.line([proj(xa_, ya_), proj(xb_, yb_)], fill=TIF_BOUNDARY, width=2)

    # V3C polygons
    for v in v3c_geoms_3857:
        polys = [v] if v.geom_type == "Polygon" else list(v.geoms)
        for p in polys:
            draw.polygon([proj(x, y) for x, y in p.exterior.coords],
                         fill=V3C_FILL, outline=V3C_EDGE)

    # GT outline
    polys = [gt_geom_3857] if gt_geom_3857.geom_type == "Polygon" else list(gt_geom_3857.geoms)
    for p in polys:
        draw.line([proj(x, y) for x, y in p.exterior.coords], fill=GT_COLOR, width=4)

    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(INPUT_GPKG)
    if str(gdf.crs) != "EPSG:32735":
        gdf = gdf.to_crs("EPSG:32735")
    g0816 = gdf[gdf["grid"] == "G0816"].copy().reset_index(drop=True)
    if g0816.empty:
        raise SystemExit("no G0816 ≥600 m² polygons in input gpkg")

    v25 = gpd.read_file(V3C_25).to_crs(TILE_CRS)
    v50 = gpd.read_file(V3C_50).to_crs(TILE_CRS) if V3C_50.exists() else None
    if v50 is None:
        raise SystemExit(f"missing {V3C_50}; run probe script first")
    print(f"V3C overlap25 features: {len(v25)}, overlap50 features: {len(v50)}")

    g0816_tile = g0816.to_crs(TILE_CRS)
    rows = []
    for i, row in g0816_tile.iterrows():
        gt = row.geometry
        bbox = gt.bounds
        pad_bbox = (bbox[0] - PAD_M, bbox[1] - PAD_M, bbox[2] + PAD_M, bbox[3] + PAD_M)
        tifs = covering_tifs(row["grid"], pad_bbox)
        view = box(*pad_bbox)
        v25_in = v25[v25.geometry.intersects(view)].geometry
        v50_in = v50[v50.geometry.intersects(view)].geometry

        img25 = render_one(gt, v25_in, tifs, CHIP_STRIDE_PX_25)
        img50 = render_one(gt, v50_in, tifs, CHIP_STRIDE_PX_50)

        # Side by side
        gap = 8
        W = img25.width + img50.width + gap
        H = max(img25.height, img50.height) + 28
        canvas = Image.new("RGB", (W, H), (26, 26, 26))
        canvas.paste(img25, (0, 28))
        canvas.paste(img50, (img25.width + gap, 28))
        d = ImageDraw.Draw(canvas)
        area = row["area_m2_calc"]
        d.text((6, 6), f"overlap=0.25 (current)  |  G0816 SAM_added {area:.0f} m²", fill=(220, 220, 220))
        d.text((img25.width + gap + 6, 6), "overlap=0.50 (probe)", fill=(220, 220, 220))

        out_path = OUT_DIR / f"g0816_{int(area)}.png"
        canvas.save(out_path, format="PNG", optimize=True)
        rows.append({"area": area, "png": out_path.name,
                     "n25": len(v25_in), "n50": len(v50_in)})
        print(f"  area={area:.0f} m²  v3c25={len(v25_in)}  v3c50={len(v50_in)}  → {out_path.name}")

    # HTML
    cards = "\n".join(
        f'<figure class="card"><img src="g0816_overlap_compare/{r["png"]}" alt="{r["area"]}">'
        f'<figcaption>{r["area"]:.0f} m² · v3c@0.25={r["n25"]}, v3c@0.50={r["n50"]}</figcaption></figure>'
        for r in sorted(rows, key=lambda x: -x["area"]))

    OUT_HTML.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>G0816 overlap=0.25 vs 0.50</title>
<style>
body {{ font-family: sans-serif; background: #1a1a1a; color: #eee; margin: 16px; }}
.card {{ margin: 0 0 14px; background: #2a2a2a; border-radius: 6px; overflow: hidden; }}
.card img {{ width: 100%; display: block; }}
figcaption {{ padding: 6px 8px; font-size: 13px; color: #ccc; }}
.summary {{ color: #aaa; font-size: 13px; margin-bottom: 12px; }}
</style></head><body>
<h1>G0816 chip-overlap probe — overlap 0.25 vs 0.50</h1>
<p class="summary">Left: production V3C@overlap=0.25 (stride=300 px = 20.1 m). Right: probe at overlap=0.50
(stride=200 px = 13.4 m). Same model, same tiles, same canonical post-proc. White dashed = chip stride
seam (note grid density doubles on right).</p>
{cards}
</body></html>
""")
    print(f"\nWrote → {OUT_HTML}")


if __name__ == "__main__":
    main()
