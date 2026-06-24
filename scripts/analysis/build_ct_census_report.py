#!/usr/bin/env python3
"""Build the Cape Town rooftop-solar census web report.

Pipeline:
  1. Load the merged per-grid inventory (concat of cls-filtered per-detection
     polygons, EPSG:32734).
  2. UNION-merge overlapping polygons at IoU>--iou (de-fragmentation; default
     0.1, JNB-consistent) so chunk-boundary / overlapping-chip duplicates fuse
     into one footprint. Area is preserved (double-counted overlap removed).
  3. Aggregate per CPT task-grid cell: installation count + footprint area.
  4. Render a self-contained HTML: headline stats panel + a Leaflet/folium map
     with toggleable choropleths (by count, by area) and a smooth heatmap.

The merged inventory is also written to --out-merged (the de-duplicated census
deliverable; the raw concat double-counts ~31% of area under per-detection).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import branca.colormap as cm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from posthoc_merge_and_spatial_eval import merge_overlapping  # noqa: E402

METRIC_CRS = "EPSG:32734"  # Cape Town UTM 34S — NOT JHB's 32735


def quantile_colormap(values, caption, palette):
    """StepColormap on quantile bins of the strictly-positive values."""
    pos = np.asarray([v for v in values if v > 0], dtype=float)
    if len(pos) == 0:
        pos = np.array([0.0, 1.0])
    qs = np.unique(np.percentile(pos, [0, 20, 40, 60, 80, 90, 95, 100]))
    if len(qs) < 2:
        qs = np.array([pos.min(), pos.max() + 1e-6])
    colors = palette[: len(qs) - 1] if len(palette) >= len(qs) - 1 else palette
    cmap = cm.StepColormap(colors, index=list(qs), vmin=float(qs[0]),
                           vmax=float(qs[-1]), caption=caption)
    return cmap


def style_fn(value, cmap):
    if value is None or value <= 0:
        return {"fillColor": "#d9d9d9", "color": "#bdbdbd", "weight": 0.2,
                "fillOpacity": 0.05}
    return {"fillColor": cmap(value)[:7], "color": "#666", "weight": 0.2,
            "fillOpacity": 0.78}


def make_tip():
    """A fresh tooltip per layer — folium tooltips bind to a single parent;
    reusing one object across GeoJson layers throws bindTooltip-on-undefined."""
    return folium.GeoJsonTooltip(
        fields=["gridcell_id", "n_install", "area_disp"],
        aliases=["网格", "装机数", "足迹 m²"], localize=True, sticky=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--task-grid", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--iou", type=float, default=0.1)
    ap.add_argument("--out-html", required=True, type=Path)
    ap.add_argument("--out-merged", required=True, type=Path)
    ap.add_argument("--force-merge", action="store_true",
                    help="re-run the merge even if --out-merged exists")
    args = ap.parse_args()

    print(f"[1/5] loading inventory {args.inventory}")
    inv = gpd.read_file(args.inventory)
    if inv.crs is None or inv.crs.is_geographic:
        inv = inv.to_crs(METRIC_CRS)
    n_raw = len(inv)
    area_raw = float(inv.geometry.area.sum())
    score_col = "confidence" if "confidence" in inv.columns else "score"

    if args.out_merged.exists() and not args.force_merge:
        print(f"[2/5] reusing cached merged inventory {args.out_merged}")
        merged = gpd.read_file(args.out_merged)
        if merged.crs is None or merged.crs.is_geographic:
            merged = merged.to_crs(METRIC_CRS)
    else:
        print(f"[2/5] union-merge @IoU>{args.iou}")
        merged = merge_overlapping(inv, iou_thresh=args.iou, score_col=score_col)
        merged = merged.set_geometry("geometry")
        if merged.crs is None:
            merged = merged.set_crs(METRIC_CRS)
        args.out_merged.parent.mkdir(parents=True, exist_ok=True)
        merged.to_file(str(args.out_merged), driver="GPKG")
    merged["area_m2"] = merged.geometry.area
    n_merged = len(merged)
    area_merged = float(merged.geometry.area.sum())
    print(f"      {n_raw} -> {n_merged} polygons; area {area_raw/1e6:.3f} -> "
          f"{area_merged/1e6:.3f} km2; merged={args.out_merged}")

    print("[3/5] per-cell aggregation")
    agg = (merged.groupby("gridcell_id")
           .agg(n_install=("area_m2", "size"),
                area_m2=("area_m2", "sum"))
           .reset_index())
    cells = gpd.read_file(args.task_grid)[["gridcell_id", "geometry"]]
    cells = cells.merge(agg, on="gridcell_id", how="left")
    cells["n_install"] = cells["n_install"].fillna(0).astype(int)
    cells["area_m2"] = cells["area_m2"].fillna(0.0)
    cells["area_disp"] = cells["area_m2"].round(0)
    # centroids computed in metric CRS then reprojected (correct + warning-free).
    # task grid is 4326, so project to metric first.
    cent_ll = cells.to_crs(METRIC_CRS).geometry.centroid.to_crs(4326)
    cells["clat"] = cent_ll.y.values
    cells["clon"] = cent_ll.x.values
    cells = cells.to_crs(4326)

    man = pd.read_csv(args.manifest)
    n_surveyed = len(man)
    n_pv_cells = int((cells["n_install"] > 0).sum())
    med_install = float(merged["area_m2"].median())
    sum_raw = int(man["n_raw"].sum())
    sum_cls = int(man["n_cls"].sum())
    cls_removed = sum_raw - sum_cls
    top_count = cells.nlargest(5, "n_install")[["gridcell_id", "n_install", "area_m2"]]
    top_area = cells.nlargest(5, "area_m2")[["gridcell_id", "n_install", "area_m2"]]

    print("[4/5] building map")
    ctr = [float(cells["clat"].mean()), float(cells["clon"].mean())]
    m = folium.Map(location=ctr, zoom_start=10, tiles="CartoDB positron",
                   control_scale=True)

    cmap_n = quantile_colormap(cells["n_install"], "装机数 / 网格 (installations per cell)",
                               ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c",
                                "#fc4e2a", "#e31a1c", "#b10026"])
    cmap_a = quantile_colormap(cells["area_m2"], "足迹面积 m² / 网格 (footprint m² per cell)",
                               ["#edf8fb", "#bfd3e6", "#9ebcda", "#8c96c6",
                                "#8c6bb1", "#88419d", "#6e016b"])

    gj = cells.to_json()

    fg_n = folium.FeatureGroup(name="热力图·按装机数 (choropleth)", show=True)
    folium.GeoJson(gj, style_function=lambda f: style_fn(
        f["properties"]["n_install"], cmap_n), tooltip=make_tip()).add_to(fg_n)
    fg_n.add_to(m)

    fg_a = folium.FeatureGroup(name="热力图·按足迹面积 (choropleth)", show=False)
    folium.GeoJson(gj, style_function=lambda f: style_fn(
        f["properties"]["area_m2"], cmap_a), tooltip=make_tip()).add_to(fg_a)
    fg_a.add_to(m)

    # smooth density heatmaps from cell centroids (2083 weighted points — light)
    cents = cells[cells["n_install"] > 0].copy()
    cents["lat"] = cents["clat"]
    cents["lon"] = cents["clon"]
    nmax = max(cents["n_install"].max(), 1)
    amax = max(cents["area_m2"].max(), 1)
    fg_hn = folium.FeatureGroup(name="平滑热力·按装机数 (kernel density)", show=False)
    HeatMap([[r.lat, r.lon, float(r.n_install) / nmax] for r in cents.itertuples()],
            radius=14, blur=20, min_opacity=0.25).add_to(fg_hn)
    fg_hn.add_to(m)
    fg_ha = folium.FeatureGroup(name="平滑热力·按面积 (kernel density)", show=False)
    HeatMap([[r.lat, r.lon, float(r.area_m2) / amax] for r in cents.itertuples()],
            radius=14, blur=20, min_opacity=0.25).add_to(fg_ha)
    fg_ha.add_to(m)

    cmap_n.add_to(m)
    cmap_a.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    def rows(df, unit):
        out = ""
        for r in df.itertuples():
            out += (f"<tr><td>{r.gridcell_id}</td><td style='text-align:right'>"
                    f"{r.n_install:,}</td><td style='text-align:right'>"
                    f"{r.area_m2:,.0f}{unit}</td></tr>")
        return out

    panel = f"""
    <div style="position:fixed;top:10px;left:10px;z-index:9999;max-width:430px;
      background:rgba(255,255,255,.95);border:1px solid #ccc;border-radius:8px;
      padding:14px 16px;font-family:-apple-system,Segoe UI,Roboto,sans-serif;
      font-size:13px;box-shadow:0 2px 8px rgba(0,0,0,.2);max-height:92vh;overflow:auto">
      <div style="font-size:16px;font-weight:700;margin-bottom:2px">
        开普敦屋顶光伏普查 · Cape Town Rooftop Solar Census</div>
      <div style="color:#666;margin-bottom:10px">2025 航拍底图 (aerial_2025) · 全市 2,083 网格全覆盖 · 完成 2026-06-16</div>
      <table style="width:100%;border-collapse:collapse;line-height:1.5">
        <tr><td>勘测网格</td><td style="text-align:right"><b>{n_surveyed:,} / 2,083</b> (100%)</td></tr>
        <tr><td>有光伏的网格</td><td style="text-align:right"><b>{n_pv_cells:,}</b> ({100*n_pv_cells/n_surveyed:.0f}%)</td></tr>
        <tr><td>装机数 (去重, per-det 合并 @IoU{args.iou})</td><td style="text-align:right"><b>{n_merged:,}</b></td></tr>
        <tr><td style="color:#999">合并前原始检测 (重复计数)</td><td style="text-align:right;color:#999">{n_raw:,}</td></tr>
        <tr><td>总足迹面积</td><td style="text-align:right"><b>{area_merged/1e6:.2f} km²</b></td></tr>
        <tr><td>中位单装机足迹</td><td style="text-align:right">{med_install:.1f} m²</td></tr>
        <tr><td>cls 假阳抑制</td><td style="text-align:right">−{cls_removed:,} ({100*cls_removed/sum_raw:.1f}%)</td></tr>
      </table>
      <div style="margin-top:10px;font-weight:600">装机数 Top 5 网格</div>
      <table style="width:100%;border-collapse:collapse">
        <tr style="color:#666"><td>网格</td><td style="text-align:right">装机</td><td style="text-align:right">面积</td></tr>
        {rows(top_count, " m²")}
      </table>
      <div style="margin-top:8px;font-weight:600">足迹面积 Top 5 网格</div>
      <table style="width:100%;border-collapse:collapse">
        <tr style="color:#666"><td>网格</td><td style="text-align:right">装机</td><td style="text-align:right">面积</td></tr>
        {rows(top_area, " m²")}
      </table>
      <div style="margin-top:10px;color:#888;font-size:11px;line-height:1.4">
        引擎: unified_reviewall_A 检测器 → finalize per-detection (v4_canonical)
        → DINOv2 cls adaptive_v1 假阳抑制。per-det 重叠多边形按 IoU&gt;{args.iou}
        union 合并 (去碎片, 与 JHB 一致), 去除约 {100*(area_raw-area_merged)/area_raw:.0f}% 重复计数面积。
        右上角图层可切换「按装机数 / 按面积」。
      </div>
    </div>"""
    m.get_root().html.add_child(folium.Element(panel))

    print(f"[5/5] writing {args.out_html}")
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(args.out_html))
    print("DONE")


if __name__ == "__main__":
    main()
