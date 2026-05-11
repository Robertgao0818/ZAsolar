"""Side-by-side HTML compare for small-B JHB CBD grids.

Renders Vexcel base + GT + V3-C+SAM (v4_agg) + train20 per-det+SAM panels per
case. Cases per grid:
  - every GT polygon (centered chip, buf scaled to polygon size)
  - up to N train20 predictions that have no GT match at IoU>=0.3 ("unique FP")

If the rebuilt train20 pixel-or+v4_agg artifacts exist, they're added as a 5th panel.

Output: results/analysis/train20_smallgrid_compare/index.html
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
OUT_DIR = ROOT / "results/analysis/train20_smallgrid_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)

GRIDS = ["G0776", "G0853", "G0815", "G0774"]
TOP_FP_PER_GRID = 8

LAYERS = {
    "GT (clean_gt)": {
        "path_tpl": str(ROOT / "data/annotations_channel2_clean/{grid}/{grid}_clean_gt.gpkg"),
        "color": (40, 220, 80),
    },
    "V3-C+SAM (v4_agg)": {
        "path_tpl": str(ROOT / "results/johannesburg/v3c_sam_maskbox_vexcel_2024_v4_agg/{grid}/predictions_metric.gpkg"),
        "color": (255, 220, 0),
    },
    "train20 per-det+SAM": {
        "path_tpl": str(ROOT / "results/analysis/v3c_failed_weight_compare/perdet/train20_val5_hn_perdet_sam_maskbox/{grid}/predictions_metric.gpkg"),
        "color": (255, 80, 80),
    },
    "train20 pixel-or+v4_agg (REPRO)": {
        "path_tpl": str(ROOT / "results/analysis/v3c_failed_weight_compare/pixelor/train20_val5_hn_pixelor_sam_maskbox_v4agg/{grid}/predictions_metric.gpkg"),
        "color": (120, 200, 255),
        "optional": True,
    },
}


def find_tiles(grid: str) -> list[Path]:
    return sorted((TILES_ROOT / grid).glob(f"{grid}_*_geo.tif"))


def crop_mosaic_3857(tiles, bbox_3857):
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
        return Image.new("RGB", (10, 10), (40, 40, 40))
    res_x = (parts[0][0][2] - parts[0][0][0]) / parts[0][1].shape[2]
    res_y = (parts[0][0][3] - parts[0][0][1]) / parts[0][1].shape[1]
    W = max(1, round((maxx - minx) / res_x))
    H = max(1, round((maxy - miny) / res_y))
    canvas = np.zeros((3, H, W), dtype=np.uint8)
    for (ix0, iy0, ix1, iy1), arr in parts:
        col = round((ix0 - minx) / res_x)
        row = round((maxy - iy1) / res_y)
        ah, aw = arr.shape[1], arr.shape[2]
        col0, row0 = max(0, col), max(0, row)
        col1, row1 = min(W, col + aw), min(H, row + ah)
        if col1 <= col0 or row1 <= row0:
            continue
        ac0, ar0 = col0 - col, row0 - row
        ac1, ar1 = ac0 + (col1 - col0), ar0 + (row1 - row0)
        canvas[:, row0:row1, col0:col1] = arr[:3, ar0:ar1, ac0:ac1]
    return Image.fromarray(np.transpose(canvas, (1, 2, 0)))


def draw_polys(img, polys, bbox, color, width=3, alpha_fill=70):
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    minx, miny, maxx, maxy = bbox
    W, H = img.size
    sx = W / (maxx - minx)
    sy = H / (maxy - miny)

    def to_px(coords):
        return [(int((x - minx) * sx), int((maxy - y) * sy)) for x, y in coords]

    for geom in polys:
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


def label_image(img, text):
    out = img.copy()
    d = ImageDraw.Draw(out)
    pad = 6
    tw = max(120, 7 * len(text))
    d.rectangle((0, 0, tw + pad * 2, 22), fill=(0, 0, 0))
    d.text((pad, 4), text, fill=(255, 255, 255))
    return out


def utm_to_3857_bbox(bbox_utm):
    corners = gpd.GeoSeries.from_xy(
        [bbox_utm[0], bbox_utm[2]], [bbox_utm[1], bbox_utm[3]], crs=32735
    ).to_crs(3857)
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def find_unique_fps(t20_utm: gpd.GeoDataFrame, gt_utm: gpd.GeoDataFrame, iou_thresh=0.3, top_k=8):
    """Return train20 polygons with no GT match at IoU >= thresh, sorted by area desc."""
    if len(t20_utm) == 0:
        return []
    if len(gt_utm) == 0:
        return list(t20_utm.geometry)
    gt_sidx = gt_utm.sindex
    out = []
    for _, row in t20_utm.iterrows():
        pg = row.geometry
        is_tp = False
        for gi in gt_sidx.intersection(pg.bounds):
            gg = gt_utm.iloc[gi].geometry
            inter = pg.intersection(gg).area
            if inter <= 0:
                continue
            union = pg.union(gg).area
            if union > 0 and inter / union >= iou_thresh:
                is_tp = True
                break
        if not is_tp:
            out.append((pg.area, pg))
    out.sort(key=lambda t: -t[0])
    return [g for _, g in out[:top_k]]


def main():
    cases = []
    for grid in GRIDS:
        gt_path = LAYERS["GT (clean_gt)"]["path_tpl"].format(grid=grid)
        gt_utm = gpd.read_file(gt_path).to_crs(32735)

        layer_polys_3857 = {}
        layer_present = {}
        for name, cfg in LAYERS.items():
            if name == "GT (clean_gt)":
                gp = gt_utm.to_crs(3857)
                layer_polys_3857[name] = list(gp.geometry)
                layer_present[name] = True
                continue
            p = Path(cfg["path_tpl"].format(grid=grid))
            if not p.exists():
                if cfg.get("optional"):
                    layer_polys_3857[name] = []
                    layer_present[name] = False
                    print(f"[{grid}] (optional) missing {name}: {p}")
                    continue
                else:
                    print(f"[{grid}] missing required {name}: {p}")
                    layer_polys_3857[name] = []
                    layer_present[name] = False
                    continue
            try:
                gp = gpd.read_file(p).to_crs(3857)
                layer_polys_3857[name] = list(gp.geometry)
                layer_present[name] = True
            except Exception as e:
                print(f"[{grid}] failed to load {name}: {e}")
                layer_polys_3857[name] = []
                layer_present[name] = False

        # Load train20 in UTM for FP detection
        t20_p = Path(LAYERS["train20 per-det+SAM"]["path_tpl"].format(grid=grid))
        t20_utm = gpd.read_file(t20_p).to_crs(32735) if t20_p.exists() else gpd.GeoDataFrame(geometry=[], crs=32735)

        tiles = find_tiles(grid)

        case_specs = []
        # 1. Every GT polygon
        for gi, row in gt_utm.iterrows():
            gm = row.geometry
            cx, cy = gm.centroid.x, gm.centroid.y
            ext = max(gm.bounds[2] - gm.bounds[0], gm.bounds[3] - gm.bounds[1])
            buf = max(20.0, ext * 0.7 + 10)
            case_specs.append({
                "type": "GT",
                "label": f"{grid} · GT #{gi+1} ({gm.area:.0f} m²)",
                "center_utm": (cx, cy),
                "buf_m": buf,
            })
        # 2. train20 unique FPs (top by area)
        unique_fps = find_unique_fps(t20_utm, gt_utm, iou_thresh=0.3, top_k=TOP_FP_PER_GRID)
        for fi, gm in enumerate(unique_fps):
            cx, cy = gm.centroid.x, gm.centroid.y
            ext = max(gm.bounds[2] - gm.bounds[0], gm.bounds[3] - gm.bounds[1])
            buf = max(20.0, ext * 0.7 + 10)
            case_specs.append({
                "type": "train20-FP",
                "label": f"{grid} · train20 unique FP #{fi+1} ({gm.area:.0f} m²)",
                "center_utm": (cx, cy),
                "buf_m": buf,
            })

        for ci, spec in enumerate(case_specs):
            cx, cy = spec["center_utm"]
            buf = spec["buf_m"]
            bbox_utm = (cx - buf, cy - buf, cx + buf, cy + buf)
            bbox_3857 = utm_to_3857_bbox(bbox_utm)
            base_img = crop_mosaic_3857(tiles, bbox_3857)
            if base_img.size[0] < 4 or base_img.size[1] < 4:
                continue

            case_id = f"{grid}_{spec['type']}_{ci:03d}"
            panel_paths = []

            combined = base_img.copy().convert("RGB")
            for name, cfg in LAYERS.items():
                if not layer_present.get(name, False):
                    continue
                combined = draw_polys(combined, layer_polys_3857[name], bbox_3857, cfg["color"], width=2, alpha_fill=40)
            combined = label_image(combined, "ALL OVERLAYS")
            cp = IMG_DIR / f"{case_id}_combined.jpg"
            combined.save(cp, quality=88)
            panel_paths.append(("ALL OVERLAYS", cp.name, True))

            for name, cfg in LAYERS.items():
                present = layer_present.get(name, False)
                im = base_img.copy().convert("RGB")
                if present:
                    im = draw_polys(im, layer_polys_3857[name], bbox_3857, cfg["color"], width=3, alpha_fill=80)
                tag = name if present else f"{name} (n.a.)"
                im = label_image(im, tag)
                pn = IMG_DIR / f"{case_id}_{name.replace(' ','_').replace('(','').replace(')','').replace('/','-').replace('+','plus').replace('=','-eq-').replace('>','gt')}.jpg"
                im.save(pn, quality=88)
                panel_paths.append((name, pn.name, present))

            cases.append({
                "case_id": case_id,
                "grid": grid,
                "type": spec["type"],
                "label": spec["label"],
                "panels": panel_paths,
            })
            print(f"[{grid}] case {case_id}: {spec['label']} → {len(panel_paths)} panels")

    n_panels = 1 + len(LAYERS)
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>train20 vs V3-C+SAM small-B grid compare</title>",
            "<style>",
            "body{font-family:system-ui,sans-serif;background:#1b1b1b;color:#eee;margin:18px}",
            "h1{margin:0 0 6px}",
            ".sub{color:#aaa;margin-bottom:18px}",
            ".case{background:#222;border-radius:8px;padding:12px;margin:14px 0;}",
            ".case h2{margin:0 0 6px;font-size:15px}",
            f".grid{{display:grid;grid-template-columns:repeat({n_panels},1fr);gap:6px}}",
            ".grid img{width:100%;border-radius:4px;cursor:zoom-in}",
            ".legend{font-size:13px;color:#aaa;margin-bottom:10px}",
            ".chip{display:inline-block;padding:2px 8px;border-radius:4px;margin-right:8px;font-weight:600}",
            ".tag{display:inline-block;padding:1px 6px;border-radius:3px;background:#444;color:#eee;font-size:11px;margin-left:6px}",
            "#zoom{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;align-items:center;justify-content:center;cursor:zoom-out;z-index:99}",
            "#zoom img{max-width:96vw;max-height:96vh}",
            "</style></head><body>",
            "<h1>train20 vs V3-C+SAM — 小 B grid 对比</h1>",
            f"<div class='sub'>Vexcel 2024 ortho · {' / '.join(GRIDS)} · 每个 case 顺序：ALL · GT · V3-C+SAM (v4_agg) · train20 per-det+SAM · train20 pixel-or+v4_agg (如已 rebuild)</div>",
            "<div class='legend'>",
            "<span class='chip' style='background:#28dc50;color:#000'>GT</span>",
            "<span class='chip' style='background:#ffdc00;color:#000'>V3-C+SAM</span>",
            "<span class='chip' style='background:#ff5050;color:#fff'>train20 per-det+SAM</span>",
            "<span class='chip' style='background:#78c8ff;color:#000'>train20 pixel-or+v4_agg</span>",
            "</div>"]
    cur_grid = None
    for c in cases:
        if c["grid"] != cur_grid:
            cur_grid = c["grid"]
            html.append(f"<h2 style='margin-top:24px;border-bottom:1px solid #444'>{cur_grid}</h2>")
        html.append(f"<div class='case'><h2>[{c['type']}] {c['label']}</h2><div class='grid'>")
        for name, fname, present in c["panels"]:
            tag = "" if present else "<span class='tag'>missing</span>"
            html.append(f"<div><img src='images/{fname}' alt='{name}' onclick=\"zoomTo(this.src)\"/>{tag}</div>")
        html.append("</div></div>")
    html.append("<div id='zoom' onclick='this.style.display=\"none\"'><img id='zoomimg'/></div>")
    html.append("<script>function zoomTo(s){document.getElementById('zoomimg').src=s;document.getElementById('zoom').style.display='flex'}</script>")
    html.append("</body></html>")
    (OUT_DIR / "index.html").write_text("\n".join(html))
    print(f"\nWrote {OUT_DIR / 'index.html'} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
