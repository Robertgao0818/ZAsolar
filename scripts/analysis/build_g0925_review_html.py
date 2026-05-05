"""Build a static HTML review of G0925 Vexcel mosaic with GT + V3-C + V3-C+SAM overlays."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pyproj import Transformer

REPO = Path("/home/gaosh/projects/ZAsolar")
DATA = Path("/home/gaosh/zasolar_data")

DEFAULT_TILES = DATA / "tiles/johannesburg/vexcel_2024/G0925"
DEFAULT_GT = REPO / "data/annotations_channel2_clean/G0925/G0925_clean_gt.gpkg"
DEFAULT_V3C = REPO / "results/johannesburg/v3c_vexcel_2024/G0925/predictions_metric.gpkg"
DEFAULT_V3C_SAM = REPO / "results/johannesburg/v3c_sam_maskbox_vexcel_2024/G0925/predictions_metric.gpkg"
DEFAULT_OUT = REPO / "results/analysis/g0925_review"


def build_mosaic_jpg(tiles_dir: Path, out_path: Path, max_pixels: int) -> dict:
    tile_files = sorted(tiles_dir.glob("*_geo.tif"))
    if not tile_files:
        raise FileNotFoundError(f"no tiles in {tiles_dir}")

    sources = [rasterio.open(p) for p in tile_files]
    try:
        mosaic, transform_full = rio_merge(sources, resampling=Resampling.nearest)
        crs = sources[0].crs
    finally:
        for s in sources:
            s.close()

    _, h, w = mosaic.shape
    scale = min(1.0, max_pixels / max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Mosaic is (bands, H, W); rasterio merge returns first 3 bands typically RGB
    arr = mosaic[:3]
    arr = np.transpose(arr, (1, 2, 0))
    img = Image.fromarray(arr)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=85)

    # Pixel size in CRS units after resize
    px_w = transform_full.a * (w / new_w)
    px_h = transform_full.e * (h / new_h)  # negative
    origin_x = transform_full.c
    origin_y = transform_full.f

    return {
        "image": str(out_path.name),
        "width": new_w,
        "height": new_h,
        "crs": crs.to_string(),
        "origin_x": origin_x,
        "origin_y": origin_y,
        "px_w": px_w,
        "px_h": px_h,
        "bounds_world": [
            origin_x,
            origin_y + px_h * new_h,
            origin_x + px_w * new_w,
            origin_y,
        ],
    }


def reproject_polygons(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("input geodataframe has no CRS")
    if str(gdf.crs) == target_crs:
        return gdf
    return gdf.to_crs(target_crs)


def world_to_pixel(geom, origin_x, origin_y, px_w, px_h):
    def to_pix(x, y, z=None):
        return ((x - origin_x) / px_w, (y - origin_y) / px_h)

    return transform(to_pix, geom)


def polygon_to_svg_paths(geom):
    """Return list of SVG path 'd' strings for a (Multi)Polygon in pixel space."""
    if geom is None or geom.is_empty:
        return []
    polys = []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    else:
        return []
    paths = []
    for p in polys:
        rings = [p.exterior] + list(p.interiors)
        d_parts = []
        for ring in rings:
            coords = list(ring.coords)
            if not coords:
                continue
            d_parts.append(
                "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in coords) + " Z"
            )
        if d_parts:
            paths.append(" ".join(d_parts))
    return paths


def build_layer(gpkg_path: Path, target_crs: str, mosaic_meta: dict, label_field: str | None):
    gdf = gpd.read_file(gpkg_path)
    gdf = reproject_polygons(gdf, target_crs)
    bx, by, bx2, by2 = mosaic_meta["bounds_world"]
    canvas = box(bx, by, bx2, by2)
    gdf = gdf[gdf.geometry.intersects(canvas)].copy()

    items = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        pix_geom = world_to_pixel(
            geom,
            mosaic_meta["origin_x"],
            mosaic_meta["origin_y"],
            mosaic_meta["px_w"],
            mosaic_meta["px_h"],
        )
        paths = polygon_to_svg_paths(pix_geom)
        if not paths:
            continue
        cx, cy = pix_geom.centroid.x, pix_geom.centroid.y
        label = ""
        if label_field and label_field in row.index:
            v = row[label_field]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                label = str(v) if not isinstance(v, float) else f"{v:.2f}"
        area_m2 = None
        if "area_m2" in row.index:
            area_m2 = row.get("area_m2")
        elif geom.geom_type in ("Polygon", "MultiPolygon"):
            area_m2 = geom.area
        items.append(
            {
                "id": int(idx),
                "paths": paths,
                "cx": cx,
                "cy": cy,
                "label": label,
                "area_m2": float(area_m2) if area_m2 is not None else None,
            }
        )
    return items


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>G0925 Vexcel Review</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #111; color: #eee; }
  header { padding: 8px 16px; background: #1c1c1c; position: sticky; top: 0; z-index: 10; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  .toggle { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; padding: 4px 10px; border-radius: 4px; background: #2a2a2a; }
  .toggle input { cursor: pointer; }
  .legend-swatch { display: inline-block; width: 14px; height: 14px; border: 2px solid; border-radius: 2px; }
  #viewport { position: relative; width: 100%; overflow: auto; background: #000; }
  #stage { position: relative; transform-origin: 0 0; }
  #base { display: block; }
  svg.layer { position: absolute; top: 0; left: 0; pointer-events: none; }
  .gt path { fill: rgba(255, 60, 60, 0.18); stroke: #ff3c3c; stroke-width: 2px; vector-effect: non-scaling-stroke; }
  .v3c path { fill: rgba(0, 200, 255, 0.10); stroke: #00c8ff; stroke-width: 2px; vector-effect: non-scaling-stroke; stroke-dasharray: 4 3; }
  .sam path { fill: rgba(255, 215, 0, 0.12); stroke: #ffd700; stroke-width: 2px; vector-effect: non-scaling-stroke; }
  .layer.hidden { display: none; }
  .label { font: 11px/1 monospace; fill: #fff; paint-order: stroke; stroke: #000; stroke-width: 2px; }
  .controls { display: flex; gap: 8px; align-items: center; }
  .info { font-size: 12px; color: #aaa; }
  button { background: #333; color: #eee; border: 1px solid #555; padding: 4px 10px; border-radius: 4px; cursor: pointer; }
  button:hover { background: #444; }
</style>
</head>
<body>
<header>
  <h1>G0925 — Vexcel 2024 (V3-C outlier diagnostic, bulk pred/GT=0.52)</h1>
  <label class="toggle"><input type="checkbox" id="t-gt" checked /><span class="legend-swatch" style="border-color:#ff3c3c;background:rgba(255,60,60,.18)"></span>Clean GT (__GT_N__)</label>
  <label class="toggle"><input type="checkbox" id="t-v3c" checked /><span class="legend-swatch" style="border-color:#00c8ff;background:rgba(0,200,255,.10)"></span>V3-C raw (__V3C_N__)</label>
  <label class="toggle"><input type="checkbox" id="t-sam" checked /><span class="legend-swatch" style="border-color:#ffd700;background:rgba(255,215,0,.12)"></span>V3-C+SAM (__SAM_N__)</label>
  <label class="toggle"><input type="checkbox" id="t-labels" /><span>Labels (area m²)</span></label>
  <div class="controls">
    <button id="zoom-out">−</button>
    <button id="zoom-reset">100%</button>
    <button id="zoom-in">+</button>
    <span class="info" id="zoom-readout">100%</span>
  </div>
  <span class="info">image __W__×__H__ · CRS __CRS__</span>
</header>
<div id="viewport">
  <div id="stage">
    <img id="base" src="__IMG__" width="__W__" height="__H__" />
    <svg class="layer gt" id="layer-gt" width="__W__" height="__H__" viewBox="0 0 __W__ __H__"></svg>
    <svg class="layer v3c" id="layer-v3c" width="__W__" height="__H__" viewBox="0 0 __W__ __H__"></svg>
    <svg class="layer sam" id="layer-sam" width="__W__" height="__H__" viewBox="0 0 __W__ __H__"></svg>
  </div>
</div>
<script>
const DATA = __DATA__;

function renderLayer(svgEl, items, withLabels) {
  const ns = "http://www.w3.org/2000/svg";
  while (svgEl.firstChild) svgEl.removeChild(svgEl.firstChild);
  for (const it of items) {
    const g = document.createElementNS(ns, "g");
    for (const d of it.paths) {
      const p = document.createElementNS(ns, "path");
      p.setAttribute("d", d);
      g.appendChild(p);
    }
    if (withLabels && it.area_m2 != null) {
      const t = document.createElementNS(ns, "text");
      t.setAttribute("x", it.cx);
      t.setAttribute("y", it.cy);
      t.setAttribute("class", "label");
      t.setAttribute("text-anchor", "middle");
      t.textContent = it.area_m2.toFixed(0);
      g.appendChild(t);
    }
    svgEl.appendChild(g);
  }
}

const layers = {
  gt: document.getElementById("layer-gt"),
  v3c: document.getElementById("layer-v3c"),
  sam: document.getElementById("layer-sam"),
};

function renderAll() {
  const lab = document.getElementById("t-labels").checked;
  renderLayer(layers.gt, DATA.gt, lab);
  renderLayer(layers.v3c, DATA.v3c, lab);
  renderLayer(layers.sam, DATA.sam, lab);
}

document.getElementById("t-gt").addEventListener("change", e => layers.gt.classList.toggle("hidden", !e.target.checked));
document.getElementById("t-v3c").addEventListener("change", e => layers.v3c.classList.toggle("hidden", !e.target.checked));
document.getElementById("t-sam").addEventListener("change", e => layers.sam.classList.toggle("hidden", !e.target.checked));
document.getElementById("t-labels").addEventListener("change", renderAll);

let zoom = 1.0;
const stage = document.getElementById("stage");
const readout = document.getElementById("zoom-readout");
function applyZoom() {
  stage.style.transform = `scale(${zoom})`;
  stage.style.width = (DATA.width * zoom) + "px";
  stage.style.height = (DATA.height * zoom) + "px";
  readout.textContent = Math.round(zoom * 100) + "%";
}
document.getElementById("zoom-in").onclick = () => { zoom = Math.min(8, zoom * 1.25); applyZoom(); };
document.getElementById("zoom-out").onclick = () => { zoom = Math.max(0.1, zoom / 1.25); applyZoom(); };
document.getElementById("zoom-reset").onclick = () => { zoom = 1.0; applyZoom(); };

renderAll();
applyZoom();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", type=Path, default=DEFAULT_TILES)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--v3c", type=Path, default=DEFAULT_V3C)
    ap.add_argument("--v3c-sam", type=Path, default=DEFAULT_V3C_SAM)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-pixels", type=int, default=6000,
                    help="max long-edge pixels for the rendered mosaic JPG")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mosaic_meta = build_mosaic_jpg(args.tiles_dir, args.out_dir / "mosaic.jpg", args.max_pixels)
    target_crs = mosaic_meta["crs"]

    gt_items = build_layer(args.gt, target_crs, mosaic_meta, "source")
    v3c_items = build_layer(args.v3c, target_crs, mosaic_meta, "confidence")
    sam_items = build_layer(args.v3c_sam, target_crs, mosaic_meta, "confidence")

    data = {
        "width": mosaic_meta["width"],
        "height": mosaic_meta["height"],
        "gt": gt_items,
        "v3c": v3c_items,
        "sam": sam_items,
    }

    html = (
        HTML_TEMPLATE
        .replace("__IMG__", mosaic_meta["image"])
        .replace("__W__", str(mosaic_meta["width"]))
        .replace("__H__", str(mosaic_meta["height"]))
        .replace("__CRS__", mosaic_meta["crs"])
        .replace("__GT_N__", str(len(gt_items)))
        .replace("__V3C_N__", str(len(v3c_items)))
        .replace("__SAM_N__", str(len(sam_items)))
        .replace("__DATA__", json.dumps(data))
    )

    out_html = args.out_dir / "index.html"
    out_html.write_text(html, encoding="utf-8")

    print(f"wrote {out_html}")
    print(f"  mosaic: {mosaic_meta['width']}x{mosaic_meta['height']} ({mosaic_meta['crs']})")
    print(f"  GT: {len(gt_items)}  V3-C: {len(v3c_items)}  V3-C+SAM: {len(sam_items)}")


if __name__ == "__main__":
    main()
