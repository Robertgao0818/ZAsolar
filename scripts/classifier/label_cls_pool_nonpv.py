#!/usr/bin/env python3
"""非-PV 子类打标器（cascade pool 版）

适配 `build_cls_dataset_cascade.py` 输出的 manifest.csv + manifest.gpkg，
生成自包含 HTML 打标页面，复用 `label_cls_nonpv_subtype.py` 的 8 类
schema 与交互（1-8 标注 / S 跳过 / B 回退 / 青色 bbox）。

目的：先审 V3-C ∩ V4.2 的共同 FP 核心（source_detector=both,
label=nonpv, detector=v3c → 462 个），确认是否以水暖 / 屋顶设施 /
阴影 / 路标为主，作为 backbone ablation 的 sanity check 和分类器
SoT 子类分布的依据。

用法：
    python scripts/classifier/label_cls_pool_nonpv.py \\
        --pool-dir data/cls_pv_nonpv_v3c_v42_cascade \\
        --output-dir data/cls_pv_nonpv_v3c_v42_cascade/labeler/shared_fp_v3c

默认 filter: detector=v3c AND label=nonpv AND source_detector=both
导出: nonpv_subtype_labeled.csv（与 label_cls_nonpv_subtype.py 同 schema）
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BBOX_COLOR_RGB = (0, 255, 255)  # cyan
BBOX_THICKNESS = 3
CHIP_SIZE_DEFAULT = 224
PAD_RATIO_DEFAULT = 0.5
MIN_SPAN_M_DEFAULT = 18.0  # 18 m × 0.13-0.15 m/px ≈ 130 raw px → 224 (~1.7× upsample)

LABELS = [
    ("1", "solar_thermal_water_heater", "太阳能热水器"),
    ("2", "pergola_carport_shadow", "车棚/遮阳棚"),
    ("3", "skylight_roof_window", "天窗/屋顶窗"),
    ("4", "roof_shadow_dark_fixture", "屋顶阴影/深色物"),
    ("5", "blue_tarp_or_pool", "蓝色防水布/泳池"),
    ("6", "hvac_rooftop_equipment", "HVAC/屋顶设备"),
    ("7", "actually_pv_mislabeled", "实为PV(误标)"),
    ("8", "ground_road_marking", "地面标识/路标"),
    ("9", "corrugated_metal_roof", "彩钢/瓦楞屋顶"),
    ("0", "other_unknown", "其他/不确定"),
]


def encode_chip_png(chip_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(chip_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def extract_chip_with_outline(
    src: rasterio.io.DatasetReader,
    geom_metric,
    metric_crs: str,
    chip_size: int,
    pad_ratio: float,
    min_span_m: float,
) -> np.ndarray | None:
    """Crop a square chip centered on geom_metric and outline the polygon in cyan.

    Window span = max(polygon_bbox * (1+pad_ratio), min_span_m). The min_span_m
    floor stops 1-3 m FPs from being upsampled 8x to fill 224x224.
    """
    minx, miny, maxx, maxy = geom_metric.bounds
    span = max(maxx - minx, maxy - miny) * (1.0 + pad_ratio)
    span = max(span, min_span_m)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = span / 2

    raster_crs = str(src.crs)
    box_metric = (cx - half, cy - half, cx + half, cy + half)
    if raster_crs != metric_crs:
        left, bottom, right, top = transform_bounds(
            metric_crs, raster_crs, *box_metric, densify_pts=21
        )
    else:
        left, bottom, right, top = box_metric

    rb = src.bounds
    if right < rb.left or left > rb.right or top < rb.bottom or bottom > rb.top:
        return None

    win = from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()
    if win.width <= 0 or win.height <= 0:
        return None

    arr = src.read(
        indexes=[1, 2, 3] if src.count >= 3 else None,
        window=win, boundless=True, fill_value=0,
    )
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return None

    # The actual window that rasterio gave us may differ from the requested one
    # by 1 px on either side; recover its true raster-CRS bounds for an accurate
    # affine.
    actual_left, actual_top = src.transform * (win.col_off, win.row_off)
    actual_right, actual_bottom = src.transform * (
        win.col_off + win.width, win.row_off + win.height
    )

    h_pre, w_pre = arr.shape[:2]
    arr = cv2.resize(arr, (chip_size, chip_size), interpolation=cv2.INTER_AREA)
    sx = chip_size / w_pre
    sy = chip_size / h_pre

    # Project the polygon's exterior ring vertices into raster CRS, then to
    # resized chip pixel space, and stroke the outline.
    from shapely.geometry import MultiPolygon, Polygon
    geoms = (
        list(geom_metric.geoms)
        if isinstance(geom_metric, MultiPolygon)
        else [geom_metric]
    )
    pixel_dx = src.transform.a
    pixel_dy = src.transform.e
    for poly in geoms:
        if not isinstance(poly, Polygon):
            continue
        rings = [poly.exterior, *poly.interiors]
        for ring in rings:
            if ring is None or ring.is_empty:
                continue
            xs_metric, ys_metric = zip(*ring.coords)
            if raster_crs != metric_crs:
                from rasterio.warp import transform as warp_pts
                xs_raster, ys_raster = warp_pts(
                    metric_crs, raster_crs, list(xs_metric), list(ys_metric)
                )
            else:
                xs_raster, ys_raster = list(xs_metric), list(ys_metric)
            pts = []
            for xr, yr in zip(xs_raster, ys_raster):
                px = (xr - actual_left) / pixel_dx
                py = (yr - actual_top) / pixel_dy
                pts.append([int(round(px * sx)), int(round(py * sy))])
            arr_pts = np.array(pts, dtype=np.int32)
            cv2.polylines(
                arr, [arr_pts], isClosed=True,
                color=BBOX_COLOR_RGB, thickness=BBOX_THICKNESS,
            )
    return arr


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>non-PV Subtype Labeler — cascade pool</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee;
       display: flex; flex-direction: column; height: 100vh; }
.header { padding: 8px 16px; background: #16213e; display: flex;
          align-items: center; gap: 16px; flex-shrink: 0; }
.header h1 { font-size: 16px; }
.progress { font-size: 14px; color: #aaa; }
.progress .done { color: #4ecca3; font-weight: bold; }
.bucket-tag { padding: 3px 10px; border-radius: 10px; font-size: 12px;
              background: #0f3460; color: #4ecca3; font-weight: bold; }
.main { flex: 1; display: flex; overflow: hidden; }
.viewer { flex: 1; display: flex; align-items: center; justify-content: center;
          padding: 8px; position: relative; }
.viewer img { max-width: 100%; max-height: 100%; object-fit: contain;
              border: 2px solid #333; border-radius: 4px;
              image-rendering: pixelated; }
.sidebar { width: 280px; background: #16213e; padding: 12px; overflow-y: auto;
           display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }
.info { font-size: 13px; line-height: 1.6; padding: 8px; background: #0f3460;
        border-radius: 6px; }
.info .val { color: #4ecca3; }
.label-btn { display: flex; align-items: center; gap: 8px; padding: 8px 10px;
             border: 1px solid #333; border-radius: 6px; cursor: pointer;
             font-size: 13px; transition: all 0.15s; background: transparent; color: #eee;
             width: 100%; text-align: left; }
.label-btn:hover { background: #0f3460; border-color: #4ecca3; }
.label-btn.active { background: #4ecca3; color: #1a1a2e; font-weight: bold;
                    border-color: #4ecca3; }
.label-btn .key { display: inline-block; width: 22px; height: 22px;
                  line-height: 22px; text-align: center; background: #333;
                  border-radius: 4px; font-weight: bold; font-size: 12px; flex-shrink: 0; }
.label-btn.active .key { background: #1a1a2e; color: #4ecca3; }
.nav { display: flex; gap: 8px; margin-top: auto; padding-top: 8px; }
.nav button { flex: 1; padding: 8px; border: 1px solid #333; border-radius: 6px;
              background: #0f3460; color: #eee; cursor: pointer; font-size: 13px; }
.nav button:hover { background: #4ecca3; color: #1a1a2e; }
.export-btn { padding: 10px; border: none; border-radius: 6px; background: #e94560;
              color: #fff; cursor: pointer; font-size: 14px; font-weight: bold;
              margin-top: 4px; }
.export-btn:hover { background: #c73e54; }
.hint { font-size: 11px; color: #666; text-align: center; margin-top: 4px; }
.badge { position: absolute; top: 16px; left: 16px; padding: 4px 12px;
         border-radius: 12px; font-size: 13px; font-weight: bold; }
.badge.labeled { background: #4ecca3; color: #1a1a2e; }
.badge.unlabeled { background: #e94560; color: #fff; }
.per-grid { font-size: 11px; color: #aaa; margin-top: 4px;
            padding: 4px; background: #0f3460; border-radius: 4px;
            max-height: 160px; overflow-y: auto; }
</style>
</head>
<body>
<div class="header">
  <h1>non-PV Subtype Audit — cascade pool</h1>
  <div class="progress">
    <span class="done" id="labeledCount">0</span> / <span id="totalCount">0</span> 已标注
    &nbsp;|&nbsp; 当前 <span id="currentIdx">1</span>
  </div>
  <div class="bucket-tag" id="bucketTag"></div>
</div>
<div class="main">
  <div class="viewer">
    <img id="chipImg" src="" />
    <div class="badge" id="badge"></div>
  </div>
  <div class="sidebar">
    <div class="info" id="chipInfo"></div>
    <div id="labelButtons"></div>
    <div class="per-grid" id="perGrid"></div>
    <div class="nav">
      <button onclick="prev()">&larr; B 回退</button>
      <button onclick="skip()">S 跳过 &rarr;</button>
    </div>
    <button class="export-btn" onclick="exportCSV()">导出 CSV</button>
    <div class="hint">1-9, 0 标注 · S 跳过 · B 回退 · 青色轮廓 = 检测多边形</div>
  </div>
</div>
<script>
const LABELS = %%LABELS_JSON%%;
const CHIPS = %%CHIPS_JSON%%;
let idx = 0;
for (let i = 0; i < CHIPS.length; i++) {
  if (!CHIPS[i].human_label) { idx = i; break; }
}

function render() {
  const c = CHIPS[idx];
  document.getElementById("chipImg").src = "data:image/png;base64," + c.img;
  document.getElementById("currentIdx").textContent = idx + 1;
  document.getElementById("totalCount").textContent = CHIPS.length;
  document.getElementById("labeledCount").textContent =
    CHIPS.filter(x => x.human_label).length;
  document.getElementById("bucketTag").textContent =
    c.detector + " · " + c.source_detector;
  document.getElementById("chipInfo").innerHTML =
    `<b>chip_id:</b> <span class="val">${c.chip_id}</span><br>` +
    `<b>Detector:</b> <span class="val">${c.detector}</span><br>` +
    `<b>Source:</b> <span class="val">${c.source_detector}</span><br>` +
    `<b>Grid:</b> <span class="val">${c.grid_id}</span><br>` +
    `<b>Pred idx:</b> <span class="val">${c.pred_idx}</span><br>` +
    `<b>Area:</b> <span class="val">${c.area_m2.toFixed(1)} m²</span><br>` +
    `<b>IoU vs GT:</b> <span class="val">${c.iou_to_gt.toFixed(3)}</span>`;

  const badge = document.getElementById("badge");
  if (c.human_label) {
    const lbl = LABELS.find(l => l[1] === c.human_label);
    badge.textContent = lbl ? lbl[2] : c.human_label;
    badge.className = "badge labeled";
  } else {
    badge.textContent = "未标注";
    badge.className = "badge unlabeled";
  }

  const container = document.getElementById("labelButtons");
  container.innerHTML = "";
  for (const [key, en, zh] of LABELS) {
    const btn = document.createElement("button");
    btn.className = "label-btn" + (c.human_label === en ? " active" : "");
    btn.innerHTML = `<span class="key">${key}</span> ${zh}`;
    btn.onclick = () => applyLabel(en);
    container.appendChild(btn);
  }

  const byGrid = {};
  for (const x of CHIPS) {
    if (!byGrid[x.grid_id]) byGrid[x.grid_id] = {done: 0, total: 0};
    byGrid[x.grid_id].total += 1;
    if (x.human_label) byGrid[x.grid_id].done += 1;
  }
  let html = "<b>Per-grid 进度:</b><br>";
  for (const [g, s] of Object.entries(byGrid)) {
    html += `${g}: ${s.done}/${s.total}<br>`;
  }
  document.getElementById("perGrid").innerHTML = html;
}

function applyLabel(label) {
  CHIPS[idx].human_label = label;
  if (idx < CHIPS.length - 1) idx++;
  render();
}
function skip() { if (idx < CHIPS.length - 1) { idx++; render(); } }
function prev() { if (idx > 0) { idx--; render(); } }

document.addEventListener("keydown", e => {
  const k = e.key;
  if (k >= "1" && k <= "9") { applyLabel(LABELS[parseInt(k) - 1][1]); }
  else if (k === "0") { applyLabel(LABELS[9][1]); }
  else if (k.toLowerCase() === "s") { skip(); }
  else if (k.toLowerCase() === "b") { prev(); }
});

function exportCSV() {
  let csv = "chip_id,detector,source_detector,grid_id,pred_idx,iou_to_gt,area_m2,human_label\n";
  for (const c of CHIPS) {
    csv += [c.chip_id, c.detector, c.source_detector, c.grid_id, c.pred_idx,
            c.iou_to_gt, c.area_m2, c.human_label || ""].join(",") + "\n";
  }
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "nonpv_subtype_labeled.csv";
  a.click();
}

render();
</script>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pool-dir", type=Path, required=True,
                   help="cascade pool 输出目录（含 manifest.csv + manifest.gpkg）")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="HTML 输出目录（默认 <pool>/labeler/<filter_id>）")
    p.add_argument(
        "--detectors", nargs="+", default=["v3c"],
        help="保留的 detector，默认仅 v3c（V3-C 是主路线，V4.2-side 后续靠 source_detector=both 传播）",
    )
    p.add_argument(
        "--source-detector", nargs="+", default=["both"],
        help="保留的 source_detector，默认 both（共同 FP 核心）",
    )
    p.add_argument(
        "--max-rows", type=int, default=None, help="限制最大样本数，方便分批"
    )
    p.add_argument("--metric-crs", default="EPSG:32735")
    p.add_argument(
        "--imagery-root", type=Path,
        default=Path.home() / "zasolar_data/tiles/johannesburg/geid_2024_02",
    )
    p.add_argument("--chip-size", type=int, default=CHIP_SIZE_DEFAULT)
    p.add_argument("--pad-ratio", type=float, default=PAD_RATIO_DEFAULT)
    p.add_argument(
        "--min-span-m", type=float, default=MIN_SPAN_M_DEFAULT,
        help="floor on chip window span in metric units; protects small "
             "polygons from being upsampled to fill the chip",
    )
    args = p.parse_args()

    manifest_csv = args.pool_dir / "manifest.csv"
    manifest_gpkg = args.pool_dir / "manifest.gpkg"
    if not manifest_csv.exists() or not manifest_gpkg.exists():
        print(f"ERROR: {manifest_csv} 或 {manifest_gpkg} 不存在")
        return 1

    print(f"[1/4] 读取 manifest.gpkg ...")
    gdf = gpd.read_file(manifest_gpkg)
    if str(gdf.crs) != args.metric_crs:
        gdf = gdf.to_crs(args.metric_crs)

    n0 = len(gdf)
    gdf = gdf[gdf["label"] == "nonpv"]
    gdf = gdf[gdf["detector"].isin(args.detectors)]
    gdf = gdf[gdf["source_detector"].isin(args.source_detector)]
    print(f"  filter: label=nonpv ∧ detector∈{args.detectors} ∧ source_detector∈{args.source_detector}")
    print(f"  {n0} → {len(gdf)} rows")

    if args.max_rows is not None:
        gdf = gdf.sort_values(["grid_id", "pred_idx"]).head(args.max_rows)
        print(f"  truncated to first {len(gdf)} rows")

    if len(gdf) == 0:
        print("  no rows after filter; nothing to label")
        return 0

    filter_id = "_".join(args.detectors) + "__" + "_".join(args.source_detector)
    out_dir = args.output_dir or (args.pool_dir / "labeler" / filter_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/4] 提取 chip + 画 bbox + 内嵌 base64 ...")
    chips = []
    skipped = 0
    src_cache: dict[str, rasterio.io.DatasetReader] = {}

    try:
        for _, r in gdf.iterrows():
            grid = r["grid_id"]
            mosaic = args.imagery_root / f"{grid}_mosaic.tif"
            if not mosaic.exists():
                skipped += 1
                continue
            key = str(mosaic)
            if key not in src_cache:
                src_cache[key] = rasterio.open(mosaic)
            chip = extract_chip_with_outline(
                src_cache[key], r.geometry, args.metric_crs,
                args.chip_size, args.pad_ratio, args.min_span_m,
            )
            if chip is None:
                skipped += 1
                continue
            b64 = encode_chip_png(chip)
            if not b64:
                skipped += 1
                continue
            chips.append({
                "chip_id": str(r["chip_id"]),
                "detector": str(r["detector"]),
                "source_detector": str(r["source_detector"]),
                "grid_id": str(grid),
                "pred_idx": int(r["pred_idx"]),
                "iou_to_gt": float(r["iou_to_gt"]),
                "area_m2": float(r["area_m2"]),
                "human_label": "",
                "img": b64,
            })
    finally:
        for h in src_cache.values():
            h.close()

    print(f"  内嵌 {len(chips)} 张 chip, skipped {skipped}")

    print(f"\n[3/4] 写 template.csv ...")
    template = pd.DataFrame([{k: v for k, v in c.items() if k != "img"} for c in chips])
    template.to_csv(out_dir / "template.csv", index=False)

    print(f"\n[4/4] 渲染 HTML ...")
    html = HTML_TEMPLATE.replace(
        "%%LABELS_JSON%%", json.dumps(LABELS, ensure_ascii=False)
    ).replace(
        "%%CHIPS_JSON%%", json.dumps(chips, ensure_ascii=False)
    )
    out_path = out_dir / "nonpv_subtype_labeler.html"
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n✓ 生成 {out_path} ({size_mb:.1f} MB)")
    wsl_path = str(out_path.resolve())
    if wsl_path.startswith("/home/"):
        win_path = "\\\\wsl$\\Ubuntu" + wsl_path.replace("/", "\\")
        print(f"  Windows 路径: {win_path}")
    print(f"\n  快捷键: 1-9, 0 标注, S 跳过, B 回退")
    print(f"  标完点「导出 CSV」下载 nonpv_subtype_labeled.csv → 放回 {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
