"""Render per-case Vexcel crops with GT vs V3-C vs failed-weight predictions.

Cases per grid:
  - large panels: clean_gt polygon area >= 300 m²
  - high-density clusters: GT centroid with >=5 neighbors within 30m

Renders 1 panel per layer (Vexcel + overlay) and assembles an HTML index.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from PIL import Image, ImageDraw

ROOT = Path("/home/gaosh/projects/ZAsolar")
TILES_ROOT = Path("/home/gaosh/zasolar_data/tiles/johannesburg/vexcel_2024")
OUT_DIR = ROOT / "results/analysis/v3c_failed_weight_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)

GRIDS = ["G0816", "G0817", "G0925"]

LAYERS = {
    "GT (clean_gt)": {
        "path_tpl": str(ROOT / "data/annotations_channel2_clean/{grid}/{grid}_clean_gt.gpkg"),
        "color": (40, 220, 80),
    },
    "V3-C raw": {
        "path_tpl": str(ROOT / "results/johannesburg/v3c_vexcel_2024/{grid}/predictions_metric.gpkg"),
        "color": (255, 220, 0),
    },
    "train20_val5_hn (FAILED)": {
        "path_tpl": str(ROOT / "results/johannesburg/train20_val5_hn_20260508_v3c/{grid}/predictions_metric.gpkg"),
        "color": (255, 80, 80),
    },
    "jhb_phaseA (FAILED)": {
        "path_tpl": str(ROOT / "results/johannesburg/jhb_phaseA_vexcel_2024/{grid}/predictions_metric.gpkg"),
        "color": (200, 100, 255),
    },
}


def find_tiles(grid: str) -> list[Path]:
    return sorted((TILES_ROOT / grid).glob(f"{grid}_*_geo.tif"))


def crop_mosaic_3857(tiles: list[Path], bbox_3857: tuple[float, float, float, float]) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Mosaic tiles into a single PIL image covering bbox (3857). Returns image and effective bbox."""
    # Find tiles intersecting bbox
    minx, miny, maxx, maxy = bbox_3857
    parts = []
    for t in tiles:
        with rasterio.open(t) as r:
            tb = r.bounds
            if tb.right < minx or tb.left > maxx or tb.top < miny or tb.bottom > maxy:
                continue
            inter = (max(tb.left, minx), max(tb.bottom, miny),
                     min(tb.right, maxx), min(tb.top, maxy))
            win = from_bounds(*inter, transform=r.transform)
            arr = r.read(window=win, boundless=True, fill_value=0,
                         out_shape=(3,
                                    max(1, round((inter[3] - inter[1]) / r.res[1])),
                                    max(1, round((inter[2] - inter[0]) / r.res[0]))))
            parts.append((inter, arr))
    if not parts:
        return Image.new("RGB", (10, 10), (40, 40, 40)), bbox_3857

    res_x = (parts[0][0][2] - parts[0][0][0]) / parts[0][1].shape[2]
    res_y = (parts[0][0][3] - parts[0][0][1]) / parts[0][1].shape[1]
    W = max(1, round((maxx - minx) / res_x))
    H = max(1, round((maxy - miny) / res_y))
    canvas = np.zeros((3, H, W), dtype=np.uint8)
    for (ix0, iy0, ix1, iy1), arr in parts:
        col = round((ix0 - minx) / res_x)
        row = round((maxy - iy1) / res_y)
        ah, aw = arr.shape[1], arr.shape[2]
        # clip
        col0, row0 = max(0, col), max(0, row)
        col1, row1 = min(W, col + aw), min(H, row + ah)
        if col1 <= col0 or row1 <= row0:
            continue
        ac0, ar0 = col0 - col, row0 - row
        ac1, ar1 = ac0 + (col1 - col0), ar0 + (row1 - row0)
        canvas[:, row0:row1, col0:col1] = arr[:3, ar0:ar1, ac0:ac1]
    img = Image.fromarray(np.transpose(canvas, (1, 2, 0)))
    return img, bbox_3857


def draw_polys(img: Image.Image, polys_3857, bbox_3857, color, width=3, alpha_fill=70):
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    minx, miny, maxx, maxy = bbox_3857
    W, H = img.size
    sx = W / (maxx - minx)
    sy = H / (maxy - miny)

    def to_px(coords):
        return [(int((x - minx) * sx), int((maxy - y) * sy)) for x, y in coords]

    for geom in polys_3857:
        if geom is None or geom.is_empty:
            continue
        gs = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for g in gs:
            if g.geom_type != "Polygon":
                continue
            ext = to_px(list(g.exterior.coords))
            if len(ext) >= 3:
                od.polygon(ext, fill=color + (alpha_fill,), outline=color + (255,), width=width)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def label_image(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    pad = 6
    tw = max(120, 7 * len(text))
    d.rectangle((0, 0, tw + pad * 2, 22), fill=(0, 0, 0))
    d.text((pad, 4), text, fill=(255, 255, 255))
    return out


def find_dense_centroids(gt: gpd.GeoDataFrame, radius_m=30.0, min_neighbors=5, top_k=3):
    """Pick centroids with most neighbors within radius (UTM 35S, meters)."""
    cents = gt.geometry.centroid
    coords = np.array([(p.x, p.y) for p in cents])
    if len(coords) == 0:
        return []
    counts = []
    for i, c in enumerate(coords):
        d = np.hypot(coords[:, 0] - c[0], coords[:, 1] - c[1])
        counts.append((i, int((d < radius_m).sum() - 1)))
    counts.sort(key=lambda t: -t[1])
    picked, taken = [], np.zeros(len(coords), dtype=bool)
    for i, n in counts:
        if n < min_neighbors:
            break
        if taken[i]:
            continue
        picked.append((i, n, coords[i]))
        d = np.hypot(coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1])
        taken[d < radius_m * 1.5] = True
        if len(picked) >= top_k:
            break
    return picked


def main():
    cases = []  # list of dicts
    for grid in GRIDS:
        gt_path = LAYERS["GT (clean_gt)"]["path_tpl"].format(grid=grid)
        gt = gpd.read_file(gt_path).to_crs(32735)
        gt_3857 = gt.to_crs(3857)

        layer_polys_3857 = {"GT (clean_gt)": list(gt_3857.geometry)}
        for name, cfg in LAYERS.items():
            if name == "GT (clean_gt)":
                continue
            p = cfg["path_tpl"].format(grid=grid)
            try:
                gp = gpd.read_file(p)
                if gp.crs is None or str(gp.crs) != "EPSG:3857":
                    gp = gp.to_crs(3857)
                layer_polys_3857[name] = list(gp.geometry)
            except Exception as e:
                print(f"[{grid}] missing {name}: {e}")
                layer_polys_3857[name] = []

        tiles = find_tiles(grid)
        # Large panels (top 3 by area)
        gt_with_idx = gt.assign(_orig_area=gt.geometry.area)
        large = gt_with_idx.sort_values("_orig_area", ascending=False).head(3)

        # Dense clusters
        dense = find_dense_centroids(gt, radius_m=30.0, min_neighbors=4, top_k=3)

        case_specs = []
        for _, row in large.iterrows():
            geom_utm = row.geometry
            case_specs.append({
                "type": "large",
                "label": f"{grid}: large panel {row['_orig_area']:.0f} m²",
                "center_utm": (geom_utm.centroid.x, geom_utm.centroid.y),
                "buf_m": max(20.0, geom_utm.bounds[2] - geom_utm.bounds[0]) * 0.6 + 10,
            })
        for i, n, c in dense:
            case_specs.append({
                "type": "dense",
                "label": f"{grid}: dense cluster ({n} neighbors / 30m)",
                "center_utm": tuple(c),
                "buf_m": 35.0,
            })

        for ci, spec in enumerate(case_specs):
            cx, cy = spec["center_utm"]
            buf = spec["buf_m"]
            bbox_utm = (cx - buf, cy - buf, cx + buf, cy + buf)
            # Reproject bbox to 3857 (corner-by-corner is acceptable for such small extents)
            corners = gpd.GeoSeries.from_xy([bbox_utm[0], bbox_utm[2]], [bbox_utm[1], bbox_utm[3]], crs=32735).to_crs(3857)
            xs = [c.x for c in corners]
            ys = [c.y for c in corners]
            bbox_3857 = (min(xs), min(ys), max(xs), max(ys))

            base_img, _ = crop_mosaic_3857(tiles, bbox_3857)
            if base_img.size[0] < 4 or base_img.size[1] < 4:
                continue

            case_id = f"{grid}_{spec['type']}_{ci}"
            panel_paths = []
            # Combined panel with all overlays
            combined = base_img.copy().convert("RGB")
            for name, cfg in LAYERS.items():
                combined = draw_polys(combined, layer_polys_3857[name], bbox_3857, cfg["color"], width=2, alpha_fill=40)
            combined = label_image(combined, "ALL OVERLAYS")
            cp = IMG_DIR / f"{case_id}_combined.jpg"
            combined.save(cp, quality=88)
            panel_paths.append(("ALL OVERLAYS", cp.name))

            for name, cfg in LAYERS.items():
                im = draw_polys(base_img.copy().convert("RGB"), layer_polys_3857[name], bbox_3857, cfg["color"], width=3, alpha_fill=80)
                im = label_image(im, name)
                pn = IMG_DIR / f"{case_id}_{name.replace(' ', '_').replace('(','').replace(')','').replace('/','-')}.jpg"
                im.save(pn, quality=88)
                panel_paths.append((name, pn.name))

            cases.append({
                "case_id": case_id,
                "grid": grid,
                "type": spec["type"],
                "label": spec["label"],
                "panels": panel_paths,
            })
            print(f"[{grid}] case {case_id}: {spec['label']} → {len(panel_paths)} panels")

    # Build HTML
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>V3-C vs failed weights — large + dense PV cases</title>",
            "<style>",
            "body{font-family:system-ui,sans-serif;background:#1b1b1b;color:#eee;margin:18px}",
            "h1{margin:0 0 6px}",
            ".sub{color:#aaa;margin-bottom:18px}",
            ".case{background:#222;border-radius:8px;padding:12px;margin:14px 0;}",
            ".case h2{margin:0 0 6px;font-size:16px}",
            ".grid{display:grid;grid-template-columns:repeat(5,1fr);gap:6px}",
            ".grid img{width:100%;border-radius:4px;cursor:zoom-in}",
            ".legend{font-size:13px;color:#aaa;margin-bottom:10px}",
            ".chip{display:inline-block;padding:2px 8px;border-radius:4px;margin-right:8px;font-weight:600}",
            "#zoom{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;align-items:center;justify-content:center;cursor:zoom-out;z-index:99}",
            "#zoom img{max-width:96vw;max-height:96vh}",
            "</style></head><body>",
            "<h1>V3-C vs 失败权重 — 大装机 + 高密度 case 比对</h1>",
            "<div class='sub'>Vexcel 2024 ortho · G0816 / G0817 / G0925 · 每个 case 顺序：ALL · GT · V3-C · train20_val5_hn (FAILED) · jhb_phaseA (FAILED)</div>",
            "<div class='legend'>",
            "<span class='chip' style='background:#28dc50;color:#000'>GT</span>",
            "<span class='chip' style='background:#ffdc00;color:#000'>V3-C raw</span>",
            "<span class='chip' style='background:#ff5050;color:#fff'>train20_val5_hn</span>",
            "<span class='chip' style='background:#c864ff;color:#fff'>jhb_phaseA</span>",
            "</div>"]
    for c in cases:
        html.append(f"<div class='case'><h2>[{c['type']}] {c['label']}</h2><div class='grid'>")
        for name, fname in c["panels"]:
            html.append(f"<div><img src='images/{fname}' alt='{name}' onclick=\"zoomTo(this.src)\"/></div>")
        html.append("</div></div>")
    html.append("<div id='zoom' onclick='this.style.display=\"none\"'><img id='zoomimg'/></div>")
    html.append("<script>function zoomTo(s){document.getElementById('zoomimg').src=s;document.getElementById('zoom').style.display='flex'}</script>")
    html.append("</body></html>")
    (OUT_DIR / "index.html").write_text("\n".join(html))
    print(f"\nWrote {OUT_DIR / 'index.html'} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
