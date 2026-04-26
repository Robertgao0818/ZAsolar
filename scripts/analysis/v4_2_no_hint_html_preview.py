"""Render HTML preview of V4.2 no-hint predictions with map context.

Samples N predictions per grid (stratified by confidence), crops each from
its underlying tile chunk with padding, overlays the polygon, and embeds
PNG thumbnails in a single HTML file for fast visual review.

Usage:
  python3 scripts/analysis/v4_2_no_hint_html_preview.py [--per-grid 5]
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import from_bounds
from shapely.geometry import box

PROJECT = Path(__file__).resolve().parents[2]
TILE_ROOT = Path("/home/gaosh/zasolar_data/tiles/johannesburg/aerial_2023")
NOHINT_DIR = PROJECT / "results/analysis/v4_2_conf015_sandton_no_hint"
OUT_HTML = PROJECT / "results/analysis/v4_2_conf015_sandton_no_hint_preview.html"
THUMB_CRS = "EPSG:4326"
PAD_M = 6.0  # padding in metric CRS (meters) around pred bbox

SANDTON = [
    "G1110", "G1111", "G1112", "G1113", "G1114",
    "G1144", "G1145", "G1146", "G1147", "G1148",
    "G1179", "G1180", "G1181", "G1182", "G1183",
    "G1214", "G1215", "G1216", "G1217", "G1218",
    "G1250", "G1251", "G1252", "G1253", "G1254",
]


def find_covering_chunk(grid: str, geom_4326) -> Path | None:
    """Return the tile chunk whose bounds fully contain the geom bbox."""
    minx, miny, maxx, maxy = geom_4326.bounds
    center = ((minx + maxx) / 2, (miny + maxy) / 2)
    best = None
    for f in (TILE_ROOT / grid).glob(f"{grid}_*_*_geo.tif"):
        with rasterio.open(f) as src:
            b = src.bounds
            if b.left <= center[0] <= b.right and b.bottom <= center[1] <= b.top:
                best = f
                break
    return best


def render_thumb(grid: str, pred_row, gt_gdf_4326: gpd.GeoDataFrame) -> str | None:
    """Return base64 PNG of a crop around the pred, with pred polygon outlined."""
    geom_4326 = pred_row.geometry_4326
    chunk = find_covering_chunk(grid, geom_4326)
    if chunk is None:
        return None
    pad_deg = PAD_M / 111_000.0
    minx, miny, maxx, maxy = geom_4326.bounds
    win_bounds = (minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg)
    with rasterio.open(chunk) as src:
        try:
            window = from_bounds(*win_bounds, transform=src.transform)
            arr = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
        except Exception:
            return None
        win_trans = src.window_transform(window)
    if arr.size == 0 or arr.shape[1] < 4 or arr.shape[2] < 4:
        return None

    img = np.transpose(arr, (1, 2, 0))
    pil = Image.fromarray(img, mode="RGB")
    draw = ImageDraw.Draw(pil, "RGBA")

    def to_px(x, y):
        col, row = ~win_trans * (x, y)
        return col, row

    def draw_geom(geom, color, width):
        if geom.geom_type == "Polygon":
            pts = [to_px(x, y) for x, y in geom.exterior.coords]
            draw.line(pts + [pts[0]], fill=color, width=width)
        elif geom.geom_type == "MultiPolygon":
            for g in geom.geoms:
                draw_geom(g, color, width)

    # yellow = any nearby GT (there shouldn't be any overlapping, but neighbors ok)
    win_box = box(*win_bounds)
    for _, gt_row in gt_gdf_4326.iterrows():
        if gt_row.geometry.intersects(win_box):
            draw_geom(gt_row.geometry, (255, 215, 0, 255), 2)
    # red = the no-hint prediction itself
    draw_geom(geom_4326, (255, 40, 40, 255), 3)

    # resize for web
    scale = min(256 / pil.width, 256 / pil.height, 1.0) if max(pil.width, pil.height) > 256 else 1.0
    if scale < 1.0:
        pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def sample_per_grid(gdf: gpd.GeoDataFrame, n: int) -> gpd.GeoDataFrame:
    """Stratified sample: confidence terciles, evenly."""
    if len(gdf) <= n:
        return gdf
    gdf = gdf.sort_values("confidence").reset_index(drop=True)
    thirds = np.array_split(gdf.index, 3)
    rng = np.random.default_rng(42)
    take = []
    per = max(1, n // 3)
    for idxs in thirds:
        if len(idxs):
            pick = rng.choice(idxs, size=min(per, len(idxs)), replace=False)
            take.extend(pick.tolist())
    # top up with highest-confidence if short
    leftover = [i for i in gdf.index if i not in take]
    while len(take) < n and leftover:
        take.append(leftover.pop())
    return gdf.loc[sorted(take[:n])].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-grid", type=int, default=5)
    args = parser.parse_args()

    cards_html = []
    totals = 0
    for grid in SANDTON:
        nh_gpkg = NOHINT_DIR / f"{grid}_no_hint.gpkg"
        if not nh_gpkg.exists():
            continue
        nh = gpd.read_file(nh_gpkg)
        if len(nh) == 0:
            continue
        nh["area_m2"] = nh.to_crs("EPSG:32735").area.round(2)
        sample = sample_per_grid(nh, args.per_grid)
        sample_4326 = sample.to_crs(THUMB_CRS).reset_index(drop=True)
        sample = sample.reset_index(drop=True)
        sample["geometry_4326"] = sample_4326.geometry

        gt_gpkg = PROJECT / f"data/annotations/Joburg/{grid}_V4_260421.gpkg"
        gt_4326 = gpd.read_file(gt_gpkg).to_crs(THUMB_CRS) if gt_gpkg.exists() else \
            gpd.GeoDataFrame({"geometry": []}, crs=THUMB_CRS)

        for _, row in sample.iterrows():
            b64 = render_thumb(grid, row, gt_4326)
            if b64 is None:
                continue
            totals += 1
            conf = float(row.get("confidence", 0.0))
            area = float(row.get("area_m2", 0.0))
            cards_html.append(f"""
  <div class="card">
    <img src="data:image/png;base64,{b64}"/>
    <div class="meta">
      <b>{grid}</b> · conf={conf:.3f} · {area:.1f} m²
    </div>
  </div>""")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>V4.2 no-hint preview</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#111; color:#eee; margin:16px; }}
  h1 {{ font-size:18px; margin:0 0 4px; }}
  .sub {{ color:#aaa; font-size:13px; margin-bottom:16px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:10px; }}
  .card {{ background:#1c1c1c; border-radius:6px; padding:6px; }}
  .card img {{ width:100%; display:block; border-radius:4px; image-rendering:pixelated; }}
  .meta {{ font-size:12px; padding:6px 2px 2px; color:#ddd; }}
  .legend {{ font-size:12px; color:#bbb; margin-bottom:12px; }}
  .legend span.pred {{ color:#ff4040; }}
  .legend span.gt {{ color:#ffd700; }}
</style></head><body>
<h1>V4.2 @ conf=0.15 — no-hint predictions preview ({totals} samples)</h1>
<div class="sub">{args.per_grid} stratified-by-confidence samples per grid × 25 Sandton grids (aerial_2023)</div>
<div class="legend"><span class="pred">■ red</span> = no-hint prediction (no overlap with current GT)&nbsp;·&nbsp;
                    <span class="gt">■ yellow</span> = nearby existing GT (for context)</div>
<div class="grid">{''.join(cards_html)}
</div>
</body></html>
"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[write] {OUT_HTML}  ({totals} thumbnails)")


if __name__ == "__main__":
    main()
