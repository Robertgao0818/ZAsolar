#!/usr/bin/env python3
"""非-PV 子类打标器（classifier 训练池偏差诊断）

读取 `cls_nonpv_subtype_audit.py` 产出的 `annotated_nonpv.csv`，
对 `subtype == "unknown"` 的样本做**按 bucket 分层 + 置信度降序**
采样，重新从 tile 提取 chip，生成自包含 HTML 打标页面。

目的：
  验证 CT batch003 / CT batch004 / JHB Sandton 三个 bucket 的 non-PV
  子类分布是否一致。若 JHB 明显更多天窗 / HVAC / 遮阳棚等非热水器子
  类，则 "77% 热水器" 的结论不可外推，需要对 non-PV 做 subtype-
  stratified reweighting。

用法：
    python scripts/analysis/label_cls_nonpv_subtype.py \\
        --run-id 2026-04-23 --per-bucket 80

生成：
    results/analysis/classifier_nonpv_audit/<run_id>/labeler/
      nonpv_subtype_labeler.html
      template.csv

在 Windows 浏览器中打开 HTML：
    快捷键 1-8 标注, S 跳过, B 回退
    标完点「导出 CSV」下载 labeled.csv
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
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.classifier.build_cls_dataset import (  # noqa: E402
    CHIP_SIZE,
    PROJECT_ROOT,
    _find_tile,
)

BBOX_COLOR_RGB = (0, 255, 255)  # cyan — best contrast on aerial roofs
BBOX_THICKNESS = 3

AUDIT_ROOT = PROJECT_ROOT / "results" / "analysis" / "classifier_nonpv_audit"

LABELS = [
    ("1", "solar_thermal_water_heater", "太阳能热水器"),
    ("2", "pergola_carport_shadow", "车棚/遮阳棚"),
    ("3", "skylight_roof_window", "天窗/屋顶窗"),
    ("4", "roof_shadow_dark_fixture", "屋顶阴影/深色物"),
    ("5", "blue_tarp_roof_cover", "蓝色防水布"),
    ("6", "hvac_rooftop_equipment", "HVAC/屋顶设备"),
    ("7", "actually_pv_mislabeled", "实为PV(误标)"),
    ("8", "other_unknown", "其他/不确定"),
]


def stratified_sample(
    df: pd.DataFrame, per_bucket: int, subtype_filter: str = "unknown",
) -> pd.DataFrame:
    """Per-bucket top-confidence slice from rows matching `subtype_filter`.

    High-confidence unknowns first because those are the riskiest FPs —
    the detector is most sure about them, so they dominate training signal
    for the classifier as a downstream filter.
    """
    pool = df[df["subtype"] == subtype_filter]
    parts = []
    for bucket, part in pool.groupby("source_bucket"):
        sorted_part = part.sort_values("confidence", ascending=False).reset_index(drop=True)
        n = min(per_bucket, len(sorted_part))
        parts.append(sorted_part.head(n))
    return pd.concat(parts, ignore_index=True) if parts else pool.iloc[:0]


def encode_chip_png(chip) -> str:
    bgr = cv2.cvtColor(chip, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def load_geometries_for_sample(sample: pd.DataFrame) -> dict:
    """Pre-load geometry for every (bucket, grid, pred_id) in the sample.

    Reads each grid's `<results_path>/<grid>/review/<grid>_reviewed.gpkg`
    once and indexes by pred_id (row index in the reviewed gpkg, matching
    how `load_reviewed_predictions` generated pred_ids in annotated_nonpv.csv).
    Returns {(bucket, grid_id, pred_id) -> shapely polygon in EPSG:4326}.
    """
    geom_by_key: dict = {}
    gpkg_cache: dict = {}
    missing = 0

    for _, r in sample.iterrows():
        key = (r["source_bucket"], r["grid_id"], int(r["pred_id"]))
        gpkg_key = (r["source_bucket"], r["grid_id"])
        if gpkg_key not in gpkg_cache:
            results_path = PROJECT_ROOT / r["results_path"]
            gpkg_path = results_path / r["grid_id"] / "review" / f"{r['grid_id']}_reviewed.gpkg"
            if not gpkg_path.exists():
                gpkg_cache[gpkg_key] = None
            else:
                try:
                    gdf = gpd.read_file(gpkg_path)
                    if gdf.crs and gdf.crs.to_epsg() != 4326:
                        gdf = gdf.to_crs(epsg=4326)
                    gpkg_cache[gpkg_key] = gdf
                except Exception as e:  # noqa: BLE001
                    print(f"  WARN: failed to read {gpkg_path}: {e}")
                    gpkg_cache[gpkg_key] = None
        gdf = gpkg_cache[gpkg_key]
        if gdf is None or int(r["pred_id"]) >= len(gdf):
            missing += 1
            continue
        geom_by_key[key] = gdf.iloc[int(r["pred_id"])].geometry

    if missing:
        print(f"  WARN: {missing} sample rows could not resolve geometry (gpkg or pred_id out of range)")
    return geom_by_key


def extract_chip_with_bbox(
    geom_4326,
    grid_id: str,
    region: str,
    tiles_root,
    tile_cache: dict,
    chip_size: int = CHIP_SIZE,
):
    """Extract chip centered on geometry centroid, draw polygon bbox.

    Bbox is computed from the polygon's bounds in EPSG:4326, transformed to
    tile pixel coordinates via the tile's affine, then offset into chip
    coordinates using the same (x0, y0) anchor that `extract_chip` uses.
    Color: cyan (high contrast vs red/brown rooftops).
    """
    lon = geom_4326.centroid.x
    lat = geom_4326.centroid.y
    tile_path = _find_tile(lon, lat, grid_id, region, tiles_root)
    if tile_path is None:
        return None

    key = str(tile_path)
    if key not in tile_cache:
        tile_cache[key] = rasterio.open(tile_path)
    src = tile_cache[key]

    py, px = src.index(lon, lat)
    x0 = max(0, int(px - chip_size // 2))
    y0 = max(0, int(py - chip_size // 2))
    x0 = min(x0, max(0, src.width - chip_size))
    y0 = min(y0, max(0, src.height - chip_size))
    w = min(chip_size, src.width - x0)
    h = min(chip_size, src.height - y0)
    if w < chip_size * 0.5 or h < chip_size * 0.5:
        return None

    window = Window(x0, y0, w, h)
    data = src.read(window=window)

    if w < chip_size or h < chip_size:
        padded = np.zeros((data.shape[0], chip_size, chip_size), dtype=data.dtype)
        padded[:, :h, :w] = data
        data = padded

    if np.all(data >= 245):
        return None

    img = data[:3].transpose(1, 2, 0).copy()  # HWC RGB uint8

    minx, miny, maxx, maxy = geom_4326.bounds
    ul_row, ul_col = src.index(minx, maxy)  # upper-left corner pixel
    lr_row, lr_col = src.index(maxx, miny)  # lower-right corner pixel
    x1 = int(ul_col - x0)
    y1 = int(ul_row - y0)
    x2 = int(lr_col - x0)
    y2 = int(lr_row - y0)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1 = max(0, min(chip_size - 1, x1))
    y1 = max(0, min(chip_size - 1, y1))
    x2 = max(0, min(chip_size - 1, x2))
    y2 = max(0, min(chip_size - 1, y2))

    cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR_RGB, BBOX_THICKNESS)
    return img


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>non-PV Subtype Labeler</title>
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
.per-bucket { font-size: 11px; color: #aaa; margin-top: 4px;
              padding: 4px; background: #0f3460; border-radius: 4px; }
</style>
</head>
<body>
<div class="header">
  <h1>non-PV Subtype Audit</h1>
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
    <div class="per-bucket" id="perBucket"></div>
    <div class="nav">
      <button onclick="prev()">&larr; B 回退</button>
      <button onclick="skip()">S 跳过 &rarr;</button>
    </div>
    <button class="export-btn" onclick="exportCSV()">导出 CSV</button>
    <div class="hint">1-8 标注 · S 跳过 · B 回退 · 青色方框 = 检测范围</div>
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
  document.getElementById("bucketTag").textContent = c.source_bucket;
  document.getElementById("chipInfo").innerHTML =
    `<b>Bucket:</b> <span class="val">${c.source_bucket}</span><br>` +
    `<b>Region:</b> <span class="val">${c.region}</span><br>` +
    `<b>Grid:</b> <span class="val">${c.grid_id}</span><br>` +
    `<b>Pred:</b> <span class="val">${c.pred_id}</span><br>` +
    `<b>Conf:</b> <span class="val">${c.confidence.toFixed(3)}</span><br>` +
    `<b>Area:</b> <span class="val">${c.area_m2.toFixed(1)} m²</span>`;

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

  // per-bucket progress
  const byBucket = {};
  for (const x of CHIPS) {
    if (!byBucket[x.source_bucket]) byBucket[x.source_bucket] = {done: 0, total: 0};
    byBucket[x.source_bucket].total += 1;
    if (x.human_label) byBucket[x.source_bucket].done += 1;
  }
  let bucketHtml = "<b>Per-bucket 进度:</b><br>";
  for (const [b, s] of Object.entries(byBucket)) {
    bucketHtml += `${b}: ${s.done}/${s.total}<br>`;
  }
  document.getElementById("perBucket").innerHTML = bucketHtml;
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
  if (k >= "1" && k <= "8") { applyLabel(LABELS[parseInt(k) - 1][1]); }
  else if (k.toLowerCase() === "s") { skip(); }
  else if (k.toLowerCase() === "b") { prev(); }
});

function exportCSV() {
  let csv = "source_bucket,region,grid_id,pred_id,confidence,area_m2,centroid_lon,centroid_lat,human_label\n";
  for (const c of CHIPS) {
    csv += [c.source_bucket, c.region, c.grid_id, c.pred_id,
            c.confidence, c.area_m2, c.centroid_lon, c.centroid_lat,
            c.human_label || ""].join(",") + "\n";
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
    p.add_argument("--run-id", required=True, help="cls_nonpv_subtype_audit run_id")
    p.add_argument("--per-bucket", type=int, default=80,
                   help="per-bucket 采样数量（按置信度降序）")
    p.add_argument("--tiles-root", type=Path, default=None,
                   help="tile 根目录 override（默认 per-grid 解析）")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="HTML 输出目录（默认 <audit_root>/<run_id>/labeler）")
    args = p.parse_args()

    run_dir = AUDIT_ROOT / args.run_id
    annotated_csv = run_dir / "annotated_nonpv.csv"
    if not annotated_csv.exists():
        print(f"ERROR: {annotated_csv} 不存在。先跑 cls_nonpv_subtype_audit.py --run-id {args.run_id}")
        return 1

    out_dir = args.output_dir or (run_dir / "labeler")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 读取 {annotated_csv.name}...")
    df = pd.read_csv(annotated_csv)
    print(f"  {len(df)} non-PV 条目, buckets={sorted(df['source_bucket'].unique())}")

    print(f"\n[2/3] Per-bucket 置信度降序采样 (per_bucket={args.per_bucket})...")
    sample = stratified_sample(df, args.per_bucket, subtype_filter="unknown")
    print(f"  采样总数: {len(sample)}")
    for b, part in sample.groupby("source_bucket"):
        print(f"    {b}: {len(part)}")

    sample.to_csv(out_dir / "template.csv", index=False)

    print(f"\n[3/3] 加载每条 prediction 的 geometry（reviewed.gpkg）...")
    geom_by_key = load_geometries_for_sample(sample)
    print(f"  resolved {len(geom_by_key)} / {len(sample)} geometries")

    print(f"\n[4/4] 提取 chip + 画 bbox + 内嵌 base64 → HTML...")
    chips = []
    tile_cache: dict = {}
    skipped = 0
    try:
        for _, r in sample.iterrows():
            key = (r["source_bucket"], r["grid_id"], int(r["pred_id"]))
            geom = geom_by_key.get(key)
            if geom is None:
                skipped += 1
                continue
            chip = extract_chip_with_bbox(
                geom, r["grid_id"], r["region"],
                args.tiles_root, tile_cache,
            )
            if chip is None:
                skipped += 1
                continue
            b64 = encode_chip_png(chip)
            if not b64:
                skipped += 1
                continue
            chips.append({
                "source_bucket": r["source_bucket"],
                "region": r["region"],
                "grid_id": r["grid_id"],
                "pred_id": int(r["pred_id"]),
                "confidence": float(r["confidence"]),
                "area_m2": float(r["area_m2"]),
                "centroid_lon": float(r["centroid_lon"]),
                "centroid_lat": float(r["centroid_lat"]),
                "human_label": "",
                "img": b64,
            })
    finally:
        for h in tile_cache.values():
            h.close()

    print(f"  内嵌 {len(chips)} 张 chip, skipped {skipped}")

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
    print(f"\n  快捷键: 1-8 标注, S 跳过, B 回退")
    print(f"  标完点「导出 CSV」下载 nonpv_subtype_labeled.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
