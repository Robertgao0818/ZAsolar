#!/usr/bin/env python3
"""Render failed-weight merge-mode diagnostics.

This is a local visual audit for the 2026-05-10 merge-mode finding. It
compares the same large/dense clean-GT cases under:

- GT clean_gt
- V3-C raw (current finalized artifact)
- train20_val5_hn pixel-or (original failed artifact)
- train20_val5_hn per-detection (re-finalized from raw_detections.pkl)
- jhb_phaseA pixel-or (raw_detections.pkl not available locally)

The script assumes the per-detection train20 artifacts already exist under
``results/analysis/failed_weights_merge_mode_compare/``.
"""
from __future__ import annotations

import html
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import from_bounds
from shapely.ops import unary_union


ROOT = Path("/home/gaosh/projects/ZAsolar")
TILES_ROOT = Path("/home/gaosh/zasolar_data/tiles/johannesburg/vexcel_2024")
OUT_DIR = ROOT / "results/analysis/failed_weights_merge_mode_compare"
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

GRIDS = ["G0816", "G0817", "G0925"]
METRIC_CRS = "EPSG:32735"

LAYERS = {
    "GT clean_gt": {
        "path_tpl": str(ROOT / "data/annotations_channel2_clean/{grid}/{grid}_clean_gt.gpkg"),
        "color": (40, 220, 80),
    },
    "V3-C pixel-or": {
        "path_tpl": str(ROOT / "results/johannesburg/v3c_vexcel_2024/{grid}/predictions_metric.gpkg"),
        "color": (255, 220, 0),
    },
    "train20 pixel-or": {
        "path_tpl": str(ROOT / "results/johannesburg/train20_val5_hn_20260508_v3c/{grid}/predictions_metric.gpkg"),
        "color": (255, 80, 80),
    },
    "train20 per-det": {
        "path_tpl": str(OUT_DIR / "train20_val5_hn_20260508_v3c_per_detection/{grid}/predictions_metric.gpkg"),
        "color": (255, 155, 60),
    },
    "phaseA pixel-or": {
        "path_tpl": str(ROOT / "results/johannesburg/jhb_phaseA_vexcel_2024/{grid}/predictions_metric.gpkg"),
        "color": (200, 100, 255),
    },
}


def find_tiles(grid: str) -> list[Path]:
    return sorted((TILES_ROOT / grid).glob(f"{grid}_*_geo.tif"))


def crop_mosaic_3857(
    tiles: list[Path],
    bbox_3857: tuple[float, float, float, float],
) -> Image.Image:
    minx, miny, maxx, maxy = bbox_3857
    parts = []
    for tile in tiles:
        with rasterio.open(tile) as src:
            tb = src.bounds
            if tb.right < minx or tb.left > maxx or tb.top < miny or tb.bottom > maxy:
                continue
            inter = (
                max(tb.left, minx),
                max(tb.bottom, miny),
                min(tb.right, maxx),
                min(tb.top, maxy),
            )
            win = from_bounds(*inter, transform=src.transform)
            arr = src.read(
                window=win,
                boundless=True,
                fill_value=0,
                out_shape=(
                    3,
                    max(1, round((inter[3] - inter[1]) / src.res[1])),
                    max(1, round((inter[2] - inter[0]) / src.res[0])),
                ),
            )
            parts.append((inter, arr))
    if not parts:
        return Image.new("RGB", (10, 10), (40, 40, 40))

    res_x = (parts[0][0][2] - parts[0][0][0]) / parts[0][1].shape[2]
    res_y = (parts[0][0][3] - parts[0][0][1]) / parts[0][1].shape[1]
    width = max(1, round((maxx - minx) / res_x))
    height = max(1, round((maxy - miny) / res_y))
    canvas = np.zeros((3, height, width), dtype=np.uint8)
    for (ix0, iy0, ix1, iy1), arr in parts:
        col = round((ix0 - minx) / res_x)
        row = round((maxy - iy1) / res_y)
        ah, aw = arr.shape[1], arr.shape[2]
        col0, row0 = max(0, col), max(0, row)
        col1, row1 = min(width, col + aw), min(height, row + ah)
        if col1 <= col0 or row1 <= row0:
            continue
        ac0, ar0 = col0 - col, row0 - row
        ac1, ar1 = ac0 + (col1 - col0), ar0 + (row1 - row0)
        canvas[:, row0:row1, col0:col1] = arr[:3, ar0:ar1, ac0:ac1]
    return Image.fromarray(np.transpose(canvas, (1, 2, 0)))


def draw_polys(
    img: Image.Image,
    geoms_3857,
    bbox_3857: tuple[float, float, float, float],
    color: tuple[int, int, int],
    *,
    width: int = 3,
    alpha_fill: int = 70,
) -> Image.Image:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    minx, miny, maxx, maxy = bbox_3857
    w, h = img.size
    sx = w / (maxx - minx)
    sy = h / (maxy - miny)

    def to_px(coords):
        return [(int((x - minx) * sx), int((maxy - y) * sy)) for x, y in coords]

    for geom in geoms_3857:
        if geom is None or geom.is_empty:
            continue
        parts = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        for part in parts:
            if part.geom_type != "Polygon":
                continue
            ext = to_px(list(part.exterior.coords))
            if len(ext) >= 3:
                draw.polygon(ext, fill=color + (alpha_fill,), outline=color + (255,), width=width)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def label_image(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    pad = 6
    tw = max(120, 7 * len(text))
    draw.rectangle((0, 0, tw + pad * 2, 22), fill=(0, 0, 0))
    draw.text((pad, 4), text, fill=(255, 255, 255))
    return out


def load_layer_geoms(grid: str) -> dict[str, list]:
    out = {}
    for name, cfg in LAYERS.items():
        path = Path(cfg["path_tpl"].format(grid=grid))
        if not path.exists():
            out[name] = []
            continue
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(METRIC_CRS)
        out[name] = list(gdf.to_crs(3857).geometry)
    return out


def find_dense_centroids(gt: gpd.GeoDataFrame, radius_m: float = 30.0, top_k: int = 3):
    cents = gt.geometry.centroid
    coords = np.array([(p.x, p.y) for p in cents])
    if len(coords) == 0:
        return []
    counts = []
    for i, c in enumerate(coords):
        d = np.hypot(coords[:, 0] - c[0], coords[:, 1] - c[1])
        counts.append((i, int((d < radius_m).sum() - 1)))
    counts.sort(key=lambda t: -t[1])
    picked = []
    taken = np.zeros(len(coords), dtype=bool)
    for i, n in counts:
        if n < 4 or taken[i]:
            continue
        picked.append((i, n, coords[i]))
        d = np.hypot(coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1])
        taken[d < radius_m * 1.5] = True
        if len(picked) >= top_k:
            break
    return picked


def area_metrics(pred_path: Path, gt_path: Path) -> dict[str, float]:
    pred = gpd.read_file(pred_path)
    gt = gpd.read_file(gt_path)
    if pred.crs is None:
        pred = pred.set_crs(METRIC_CRS)
    if gt.crs is None:
        gt = gt.set_crs(METRIC_CRS)
    pred = pred.to_crs(METRIC_CRS)
    gt = gt.to_crs(METRIC_CRS)
    pred_geoms = [g for g in pred.geometry if g is not None and not g.is_empty]
    gt_geoms = [g for g in gt.geometry if g is not None and not g.is_empty]
    pred_sum_area = float(sum(g.area for g in pred_geoms))
    gt_sum_area = float(sum(g.area for g in gt_geoms))
    pred_union = unary_union(pred_geoms) if pred_geoms else None
    gt_union = unary_union(gt_geoms) if gt_geoms else None
    pred_area = float(pred_union.area) if pred_union is not None else 0.0
    gt_area = float(gt_union.area) if gt_union is not None else 0.0
    inter = 0.0
    if pred_union is not None and gt_union is not None:
        inter = float(pred_union.intersection(gt_union).area)
    p = inter / pred_area if pred_area else 0.0
    r = inter / gt_area if gt_area else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "pred_n": float(len(pred)),
        "gt_n": float(len(gt)),
        "pred_area_m2": pred_area,
        "pred_sum_area_m2": pred_sum_area,
        "gt_area_m2": gt_area,
        "gt_sum_area_m2": gt_sum_area,
        "inter_area_m2": inter,
        "area_p": p,
        "area_r": r,
        "area_f1": f1,
        "bulk_ratio": pred_area / gt_area if gt_area else 0.0,
        "overlap_multiplier": pred_sum_area / pred_area if pred_area else 0.0,
    }


def build_metrics() -> pd.DataFrame:
    rows = []
    for grid in GRIDS:
        gt_path = ROOT / f"data/annotations_channel2_clean/{grid}/{grid}_clean_gt.gpkg"
        for name, cfg in LAYERS.items():
            if name == "GT clean_gt":
                continue
            pred_path = Path(cfg["path_tpl"].format(grid=grid))
            if not pred_path.exists():
                continue
            row = {"grid": grid, "variant": name, "path": str(pred_path.relative_to(ROOT))}
            row.update(area_metrics(pred_path, gt_path))
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        total_rows = []
        for variant, grp in df.groupby("variant"):
            pred_area = float(grp["pred_area_m2"].sum())
            gt_area = float(grp["gt_area_m2"].sum())
            inter_area = float(grp["inter_area_m2"].sum())
            area_p = inter_area / pred_area if pred_area else 0.0
            area_r = inter_area / gt_area if gt_area else 0.0
            area_f1 = 2 * area_p * area_r / (area_p + area_r) if (area_p + area_r) else 0.0
            total_rows.append({
                "grid": "AGG",
                "variant": variant,
                "path": "",
                "pred_n": float(grp["pred_n"].sum()),
                "gt_n": float(grp["gt_n"].sum()),
                "pred_area_m2": pred_area,
                "pred_sum_area_m2": float(grp["pred_sum_area_m2"].sum()),
                "gt_area_m2": gt_area,
                "gt_sum_area_m2": float(grp["gt_sum_area_m2"].sum()),
                "inter_area_m2": inter_area,
                "area_p": area_p,
                "area_r": area_r,
                "area_f1": area_f1,
                "bulk_ratio": pred_area / gt_area if gt_area else 0.0,
                "overlap_multiplier": (
                    float(grp["pred_sum_area_m2"].sum()) / pred_area if pred_area else 0.0
                ),
            })
        df = pd.concat([df, pd.DataFrame(total_rows)], ignore_index=True)
    df.to_csv(OUT_DIR / "merge_mode_area_metrics.csv", index=False)
    return df


def metric_text(metrics: pd.DataFrame, grid: str, variant: str) -> str:
    row = metrics[(metrics["grid"] == grid) & (metrics["variant"] == variant)]
    if row.empty:
        return ""
    r = row.iloc[0]
    return f"F1 {r['area_f1']:.3f} / bulk {r['bulk_ratio']:.2f} / n {int(r['pred_n'])}"


def main() -> None:
    metrics = build_metrics()
    cases = []
    for grid in GRIDS:
        gt_path = ROOT / f"data/annotations_channel2_clean/{grid}/{grid}_clean_gt.gpkg"
        gt = gpd.read_file(gt_path).to_crs(32735)
        layer_geoms = load_layer_geoms(grid)
        tiles = find_tiles(grid)

        gt_with_area = gt.assign(_orig_area=gt.geometry.area)
        large = gt_with_area.sort_values("_orig_area", ascending=False).head(3)
        dense = find_dense_centroids(gt, radius_m=30.0, top_k=3)

        case_specs = []
        for _, row in large.iterrows():
            geom = row.geometry
            case_specs.append({
                "type": "large",
                "label": f"{grid}: large panel {row['_orig_area']:.0f} m2",
                "center_utm": (geom.centroid.x, geom.centroid.y),
                "buf_m": max(20.0, geom.bounds[2] - geom.bounds[0]) * 0.6 + 10,
            })
        for _, n, coord in dense:
            case_specs.append({
                "type": "dense",
                "label": f"{grid}: dense cluster ({n} neighbors / 30m)",
                "center_utm": tuple(coord),
                "buf_m": 35.0,
            })

        for ci, spec in enumerate(case_specs):
            cx, cy = spec["center_utm"]
            buf = spec["buf_m"]
            bbox_utm = (cx - buf, cy - buf, cx + buf, cy + buf)
            corners = gpd.GeoSeries.from_xy(
                [bbox_utm[0], bbox_utm[2]],
                [bbox_utm[1], bbox_utm[3]],
                crs=32735,
            ).to_crs(3857)
            xs = [p.x for p in corners]
            ys = [p.y for p in corners]
            bbox_3857 = (min(xs), min(ys), max(xs), max(ys))
            base = crop_mosaic_3857(tiles, bbox_3857)
            if base.size[0] < 4 or base.size[1] < 4:
                continue

            case_id = f"{grid}_{spec['type']}_{ci}"
            panel_paths = []
            combined = base.copy().convert("RGB")
            for name, cfg in LAYERS.items():
                combined = draw_polys(
                    combined,
                    layer_geoms.get(name, []),
                    bbox_3857,
                    cfg["color"],
                    width=2,
                    alpha_fill=35,
                )
            combined = label_image(combined, "ALL OVERLAYS")
            cp = IMG_DIR / f"{case_id}_combined.jpg"
            combined.save(cp, quality=88)
            panel_paths.append(("ALL OVERLAYS", cp.name, ""))

            for name, cfg in LAYERS.items():
                im = draw_polys(
                    base.copy().convert("RGB"),
                    layer_geoms.get(name, []),
                    bbox_3857,
                    cfg["color"],
                    width=3,
                    alpha_fill=75,
                )
                im = label_image(im, name)
                safe = name.replace(" ", "_").replace("/", "-")
                pn = IMG_DIR / f"{case_id}_{safe}.jpg"
                im.save(pn, quality=88)
                panel_paths.append((name, pn.name, metric_text(metrics, grid, name)))

            cases.append({**spec, "case_id": case_id, "panels": panel_paths})
            print(f"[{grid}] {case_id}: {spec['label']}")

    agg = metrics[metrics["grid"] == "AGG"].copy()
    metric_rows = []
    if not agg.empty:
        agg = agg.sort_values("area_f1", ascending=False)
        for _, row in agg.iterrows():
            metric_rows.append(
                f"<tr><td>{html.escape(row['variant'])}</td>"
                f"<td>{row['area_f1']:.3f}</td><td>{row['bulk_ratio']:.2f}</td>"
                f"<td>{int(row['pred_n'])}</td></tr>"
            )

    styles = """
body{font-family:system-ui,sans-serif;background:#1b1b1b;color:#eee;margin:18px}
h1{margin:0 0 6px}.sub{color:#aaa;margin-bottom:18px}
.case{background:#222;border-radius:8px;padding:12px;margin:14px 0}
.case h2{margin:0 0 6px;font-size:16px}
.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:6px}
.panel img{width:100%;border-radius:4px;cursor:zoom-in}
.cap{font-size:12px;color:#ccc;line-height:1.25;margin-top:3px}
.legend{font-size:13px;color:#aaa;margin-bottom:10px}
.chip{display:inline-block;padding:2px 8px;border-radius:4px;margin-right:8px;font-weight:600}
table{border-collapse:collapse;margin:10px 0 18px}td,th{border:1px solid #555;padding:5px 8px}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;align-items:center;justify-content:center;cursor:zoom-out;z-index:99}
#zoom img{max-width:96vw;max-height:96vh}
"""
    doc = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Failed weights merge-mode compare</title>",
        f"<style>{styles}</style></head><body>",
        "<h1>Failed weights merge-mode compare</h1>",
        "<div class='sub'>Vexcel 2024 · G0816/G0817/G0925 · train20 per-detection was re-finalized locally from raw_detections.pkl. phaseA raw_detections.pkl is not available locally, so only its pixel-or artifact is shown.</div>",
        "<div class='legend'>",
        "<span class='chip' style='background:#28dc50;color:#000'>GT</span>",
        "<span class='chip' style='background:#ffdc00;color:#000'>V3-C pixel-or</span>",
        "<span class='chip' style='background:#ff5050;color:#fff'>train20 pixel-or</span>",
        "<span class='chip' style='background:#ff9b3c;color:#000'>train20 per-det</span>",
        "<span class='chip' style='background:#c864ff;color:#fff'>phaseA pixel-or</span>",
        "</div>",
        "<h2>3-grid aggregate</h2>",
        "<table><tr><th>variant</th><th>area_F1 weighted</th><th>bulk</th><th>n_pred</th></tr>",
        *metric_rows,
        "</table>",
    ]
    for case in cases:
        doc.append(f"<div class='case'><h2>[{case['type']}] {html.escape(case['label'])}</h2><div class='grid'>")
        for name, fname, cap in case["panels"]:
            doc.append(
                f"<div class='panel'><img src='images/{html.escape(fname)}' alt='{html.escape(name)}' "
                "onclick=\"zoomTo(this.src)\"/>"
                f"<div class='cap'>{html.escape(name)}<br>{html.escape(cap)}</div></div>"
            )
        doc.append("</div></div>")
    doc.append("<div id='zoom' onclick='this.style.display=\"none\"'><img id='zoomimg'/></div>")
    doc.append("<script>function zoomTo(s){document.getElementById('zoomimg').src=s;document.getElementById('zoom').style.display='flex'}</script>")
    doc.append("</body></html>")
    (OUT_DIR / "index.html").write_text("\n".join(doc), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'index.html'} ({len(cases)} cases)")
    print(f"Wrote {OUT_DIR / 'merge_mode_area_metrics.csv'}")


if __name__ == "__main__":
    main()
