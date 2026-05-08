"""Render per-polygon chips for large-installation review (≥200 m²) and emit a
self-contained HTML labeler.

For each polygon in ``data/coco_train20_val5_qa/large_polygons.csv`` we:
  1. Find the source tiles for the polygon's (region, grid, imagery_layer).
  2. Mosaic the tiles in-memory restricted to the polygon's padded bbox.
  3. Save raw PNG + overlay PNG (cyan polygon outline, magenta bbox).
  4. Embed all metadata into ``index.html`` as a JS array; the page renders
     image + sidebar + keyboard label buttons + localStorage progress + CSV
     export.

Output:  data/coco_train20_val5_qa/large_polygon_review/
  ├─ index.html
  ├─ chips/<polygon_id>_raw.png
  ├─ chips/<polygon_id>_overlay.png
  └─ polygons.json   (also embedded in HTML for offline use)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from PIL import Image, ImageDraw
from rasterio.merge import merge as rio_merge

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.grid_utils import resolve_tiles_dir  # noqa: E402

QA_DIR = PROJECT_ROOT / "data/coco_train20_val5_qa"
SPEC_PATH = PROJECT_ROOT / "configs/datasets/train20_val5.yaml"
OUT_DIR = QA_DIR / "large_polygon_review"
CHIPS_DIR = OUT_DIR / "chips"

CRS_BY_REGION = {"johannesburg": "EPSG:32735", "cape_town": "EPSG:32734"}


def parse_grid_entry(entry):
    if isinstance(entry, dict):
        return entry["grid_id"], entry.get("file")
    return entry, None


def load_grid_to_layer():
    """grid_id → (region_key, imagery_layer, annotation_path) from the spec."""
    spec = yaml.safe_load(SPEC_PATH.read_text())
    out = {}
    for split, regions in spec["splits"].items():
        for region_key, cfg in regions.items():
            ann_root = PROJECT_ROOT / cfg["annotation_root"]
            for entry in cfg.get("grids", []):
                grid_id, fname = parse_grid_entry(entry)
                if fname:
                    path = ann_root / fname
                else:
                    path = ann_root / cfg["annotation_pattern"].format(grid_id=grid_id)
                out[grid_id] = {
                    "region": region_key,
                    "imagery_layer": cfg["imagery_layer"],
                    "annotation_path": path,
                    "split": split,
                }
    return out


def list_tiles(grid_id, region, imagery_layer):
    tiles_dir = resolve_tiles_dir(grid_id, region=region, imagery_layer=imagery_layer)
    if tiles_dir.is_file():
        return [tiles_dir]
    tiles = sorted(tiles_dir.glob(f"{grid_id}_*_*_geo.tif"))
    if not tiles:
        tiles = sorted(p for p in tiles_dir.glob(f"{grid_id}_*.tif")
                       if "mosaic" not in p.stem)
    return tiles


def render_chip(poly_geom_4326, region, tiles, padding_factor=1.0,
                min_pad_m=8.0, target_size=512):
    """Render a chip centered on the polygon, with optional polygon overlay.

    Returns (raw_pil, overlay_pil, geo_meta).
    """
    metric_crs = CRS_BY_REGION[region]
    poly_metric = (gpd.GeoSeries([poly_geom_4326], crs="EPSG:4326")
                   .to_crs(metric_crs).iloc[0])
    minx, miny, maxx, maxy = poly_metric.bounds
    w_m = max(maxx - minx, 1)
    h_m = max(maxy - miny, 1)
    pad_m = max(min_pad_m, padding_factor * max(w_m, h_m))
    ext_minx, ext_miny = minx - pad_m, miny - pad_m
    ext_maxx, ext_maxy = maxx + pad_m, maxy + pad_m

    # Reproject extent to tile native CRS once
    with rasterio.open(tiles[0]) as src0:
        tile_crs = src0.crs
    ext_in_tile_crs = (gpd.GeoSeries(
        [gpd.GeoSeries([
            __import__("shapely").geometry.box(ext_minx, ext_miny, ext_maxx, ext_maxy)
        ], crs=metric_crs).iloc[0]],
        crs=metric_crs,
    ).to_crs(tile_crs).iloc[0])
    tx_minx, ty_miny, tx_maxx, ty_maxy = ext_in_tile_crs.bounds

    # Pick intersecting tiles
    relevant = []
    for t in tiles:
        with rasterio.open(t) as src:
            b = src.bounds
            if not (tx_maxx < b.left or tx_minx > b.right or
                    ty_maxy < b.bottom or ty_miny > b.top):
                relevant.append(t)
    if not relevant:
        return None, None, {"error": "no_intersecting_tiles"}

    srcs = [rasterio.open(t) for t in relevant]
    try:
        arr, out_transform = rio_merge(srcs, bounds=(tx_minx, ty_miny, tx_maxx, ty_maxy))
    finally:
        for s in srcs:
            s.close()
    # arr shape: (bands, h, w)
    if arr.shape[0] >= 3:
        rgb = arr[:3].transpose(1, 2, 0)
    else:
        rgb = np.repeat(arr[:1].transpose(1, 2, 0), 3, axis=2)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    raw_pil = Image.fromarray(rgb)
    if max(raw_pil.size) > target_size:
        scale = target_size / max(raw_pil.size)
        new_w = int(raw_pil.width * scale)
        new_h = int(raw_pil.height * scale)
        raw_pil_disp = raw_pil.resize((new_w, new_h), Image.LANCZOS)
    else:
        scale = 1.0
        raw_pil_disp = raw_pil.copy()

    # Overlay polygon (and bbox) in display-pixel space
    overlay = raw_pil_disp.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay, "RGBA")

    inv = ~out_transform
    poly_tile = (gpd.GeoSeries([poly_geom_4326], crs="EPSG:4326")
                 .to_crs(tile_crs).iloc[0])

    def to_disp_px(x, y):
        col, row = inv * (x, y)
        return col * scale, row * scale

    def draw_ring(coords, color, width):
        pts = [to_disp_px(x, y) for x, y in coords]
        draw.line(pts + [pts[0]], fill=color, width=width)

    # bbox (magenta)
    bx_coords = [(tx_minx, ty_miny), (tx_maxx, ty_miny),
                 (tx_maxx, ty_maxy), (tx_minx, ty_maxy)]
    # Note: use polygon's actual bbox in tile CRS (not the padded extent)
    pminx, pminy, pmaxx, pmaxy = poly_tile.bounds
    bx_coords = [(pminx, pminy), (pmaxx, pminy),
                 (pmaxx, pmaxy), (pminx, pmaxy)]
    draw_ring(bx_coords, (255, 0, 255, 200), 1)

    if poly_tile.geom_type == "Polygon":
        draw_ring(list(poly_tile.exterior.coords), (0, 255, 255, 230), 2)
        for ring in poly_tile.interiors:
            draw_ring(list(ring.coords), (0, 255, 255, 180), 2)
    elif poly_tile.geom_type == "MultiPolygon":
        for sub in poly_tile.geoms:
            draw_ring(list(sub.exterior.coords), (0, 255, 255, 230), 2)

    return raw_pil_disp, overlay, {
        "tile_count": len(relevant),
        "extent_m": [w_m, h_m],
        "scale": scale,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-area", type=float, default=200.0,
                    help="Only render polygons with area_m2 >= this (default 200)")
    ap.add_argument("--padding-factor", type=float, default=1.0,
                    help="Padding around polygon bbox as fraction of bbox side")
    ap.add_argument("--target-size", type=int, default=512,
                    help="Long edge of saved chip in pixels")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit polygon count for quick testing")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHIPS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(QA_DIR / "large_polygons.csv")
    df = df[df.area_m2 >= args.min_area].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"Rendering {len(df)} polygons (≥{args.min_area} m²) → {CHIPS_DIR}")

    grid_meta = load_grid_to_layer()

    # Cache geometries by grid (re-load only if grid changes)
    cached_grid = None
    cached_gdf_4326 = None

    items = []
    for i, row in df.iterrows():
        gid = row.grid_id
        meta = grid_meta.get(gid)
        if not meta:
            print(f"[SKIP] {gid}: not in spec")
            continue
        if cached_grid != gid:
            ann_path = meta["annotation_path"]
            g = gpd.read_file(ann_path)
            if g.crs is None:
                g = g.set_crs("EPSG:4326")
            cached_gdf_4326 = g.to_crs("EPSG:4326")
            cached_grid = gid

        # match polygon by area (≤1 cm² tolerance)
        target_area = row.area_m2
        metric_crs = CRS_BY_REGION[meta["region"]]
        cached_metric = cached_gdf_4326.to_crs(metric_crs)
        areas = cached_metric.geometry.area
        # find polygon with area ≈ target
        diff = (areas - target_area).abs()
        # rank candidates and pick the one whose 4326 centroid matches src_field too
        cand_idx = diff.idxmin()
        if diff.loc[cand_idx] > 0.5:
            print(f"[WARN] {row.polygon_id} ({gid} area={target_area}): closest match "
                  f"diff={diff.loc[cand_idx]:.2f} m²")
        poly = cached_gdf_4326.geometry.iloc[cand_idx]

        tiles = list_tiles(gid, meta["region"], meta["imagery_layer"])
        if not tiles:
            print(f"[SKIP] {gid}: no tiles")
            continue

        try:
            raw, overlay, geo_meta = render_chip(
                poly, meta["region"], tiles,
                padding_factor=args.padding_factor,
                target_size=args.target_size,
            )
        except Exception as e:
            print(f"[ERR ] polygon {row.polygon_id} ({gid}): {e}")
            continue
        if raw is None:
            continue

        pid = int(row.polygon_id)
        raw.save(CHIPS_DIR / f"{pid:04d}_raw.png")
        overlay.save(CHIPS_DIR / f"{pid:04d}_overlay.png")

        items.append({
            "polygon_id": pid,
            "split": row.split,
            "region": row.region,
            "grid_id": gid,
            "imagery_layer": meta["imagery_layer"],
            "source": row.src_field if pd.notna(row.src_field) else None,
            "area_m2": float(row.area_m2),
            "flag": row.flag if pd.notna(row.flag) else "",
            "raw": f"chips/{pid:04d}_raw.png",
            "overlay": f"chips/{pid:04d}_overlay.png",
            "tile_count": geo_meta["tile_count"],
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(df)} rendered")

    items.sort(key=lambda x: -x["area_m2"])
    (OUT_DIR / "polygons.json").write_text(
        json.dumps(items, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n[SAVE] {OUT_DIR/'polygons.json'} ({len(items)} polygons)")

    # Build the HTML
    html = build_html(items)
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"[SAVE] {OUT_DIR/'index.html'}")
    print(f"\nOpen file://{OUT_DIR/'index.html'} to start review")


def build_html(items: list[dict]) -> str:
    LABELS = [
        ("ok", "1", "正确合并 (single installation, boundary 合理)", "#4ecca3"),
        ("roof_swallow", "2", "屋顶吞噬 (含非 PV 区域)", "#e94560"),
        ("multi_install_merged", "3", "多装机合并 (应拆为多个)", "#f39c12"),
        ("fragmented_subarray", "4", "子阵列分割不全 (邻近还有同装机片段)", "#e67e22"),
        ("non_pv", "5", "非 PV (热水器/天窗等)", "#e94560"),
        ("uncertain", "9", "不确定", "#888"),
    ]
    items_json = json.dumps(items, ensure_ascii=False)
    labels_json = json.dumps(LABELS, ensure_ascii=False)
    return r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Large Polygon Review — train20_val5</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee;
       display: flex; flex-direction: column; height: 100vh; }
.header { padding: 8px 16px; background: #16213e; display: flex;
          align-items: center; gap: 16px; flex-shrink: 0; flex-wrap: wrap; }
.header h1 { font-size: 16px; }
.progress { font-size: 14px; color: #aaa; }
.progress .done { color: #4ecca3; font-weight: bold; }
.filter-tag { padding: 3px 10px; border-radius: 10px; font-size: 12px;
              background: #0f3460; cursor: pointer; user-select: none; }
.filter-tag.active { background: #4ecca3; color: #1a1a2e; font-weight: bold; }
.main { flex: 1; display: flex; overflow: hidden; }
.viewer { flex: 1; display: flex; align-items: center; justify-content: center;
          padding: 8px; position: relative; flex-direction: column; gap: 8px; }
.viewer img { max-width: 100%; max-height: calc(100vh - 120px); object-fit: contain;
              border: 2px solid #333; border-radius: 4px;
              image-rendering: pixelated; }
.toggle-bar { display: flex; gap: 8px; align-items: center; }
.toggle-bar button { padding: 6px 12px; border: 1px solid #333;
                     background: #16213e; color: #eee; border-radius: 4px;
                     cursor: pointer; font-size: 12px; }
.toggle-bar button.active { background: #4ecca3; color: #1a1a2e; border-color: #4ecca3; }
.sidebar { width: 320px; background: #16213e; padding: 12px; overflow-y: auto;
           display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }
.info { font-size: 13px; line-height: 1.6; padding: 8px; background: #0f3460;
        border-radius: 6px; }
.info .val { color: #4ecca3; }
.info .key { color: #aaa; display: inline-block; width: 88px; }
.label-btn { display: flex; align-items: flex-start; gap: 8px; padding: 8px 10px;
             border: 1px solid #333; border-radius: 6px; cursor: pointer;
             font-size: 12px; transition: all 0.1s; background: transparent; color: #eee;
             width: 100%; text-align: left; line-height: 1.4; }
.label-btn:hover { background: #0f3460; }
.label-btn.active { font-weight: bold; }
.label-btn .key { display: inline-block; min-width: 22px; height: 22px;
                  line-height: 22px; text-align: center; background: #333;
                  border-radius: 4px; font-weight: bold; font-size: 12px; flex-shrink: 0; }
.label-btn.active .key { background: #1a1a2e; }
.nav { display: flex; gap: 8px; margin-top: auto; padding-top: 8px; }
.nav button { flex: 1; padding: 8px; border: 1px solid #333; border-radius: 6px;
              background: #0f3460; color: #eee; cursor: pointer; font-size: 13px; }
.nav button:hover { background: #4ecca3; color: #1a1a2e; }
.export-btn { padding: 10px; border: none; border-radius: 6px; background: #e94560;
              color: #fff; cursor: pointer; font-size: 14px; font-weight: bold;
              margin-top: 4px; }
.hint { font-size: 11px; color: #666; text-align: center; margin-top: 4px; line-height: 1.6; }
.badge { position: absolute; top: 16px; left: 16px; padding: 4px 12px;
         border-radius: 12px; font-size: 13px; font-weight: bold; }
.badge.labeled { background: #4ecca3; color: #1a1a2e; }
.badge.unlabeled { background: #555; color: #fff; }
.badge.flagged { background: #f39c12; color: #1a1a2e; }
.summary { font-size: 11px; padding: 6px; background: #0f3460; border-radius: 4px;
           max-height: 200px; overflow-y: auto; }
.summary table { width: 100%; border-collapse: collapse; font-size: 11px; }
.summary td { padding: 2px 4px; }
.jump { width: 100%; padding: 6px; background: #0f3460; color: #eee;
        border: 1px solid #333; border-radius: 4px; }
</style>
</head>
<body>
<div class="header">
  <h1>Large Polygon Review</h1>
  <div class="progress">
    <span class="done" id="labeledCount">0</span> / <span id="totalCount">0</span> 已标注 ·
    当前 <span id="currentIdx">1</span> / <span id="filterCount">0</span>
  </div>
  <span class="filter-tag" id="filterAll" onclick="setFilter('all')">全部</span>
  <span class="filter-tag" id="filterUnlabeled" onclick="setFilter('unlabeled')">未标注</span>
  <span class="filter-tag" id="filterFlagged" onclick="setFilter('flagged')">flag 标记</span>
  <span class="filter-tag" id="filterGE500" onclick="setFilter('ge500')">≥500 m²</span>
  <span class="filter-tag" id="filterJHB" onclick="setFilter('johannesburg')">JHB</span>
  <span class="filter-tag" id="filterCT" onclick="setFilter('cape_town')">CT</span>
</div>
<div class="main">
  <div class="viewer">
    <img id="chipImg" src="" />
    <div class="badge unlabeled" id="badge">未标注</div>
    <div class="toggle-bar">
      <button id="btnOverlay" class="active" onclick="setMode('overlay')">overlay [T]</button>
      <button id="btnRaw" onclick="setMode('raw')">raw</button>
    </div>
  </div>
  <div class="sidebar">
    <div class="info" id="chipInfo"></div>
    <div id="labelButtons"></div>
    <div class="nav">
      <button onclick="prev()">&larr; B 上一个</button>
      <button onclick="skip()">S 下一个 &rarr;</button>
    </div>
    <input type="number" class="jump" id="jumpInput" placeholder="跳到 polygon_id (Enter)"
           onkeydown="if(event.key==='Enter') jumpToId(this.value)" />
    <button class="export-btn" onclick="exportCSV()">导出 CSV</button>
    <div class="summary" id="summary"></div>
    <div class="hint">
      1-5, 9 标注 · S 下一个 · B 上一个 · T 切 raw/overlay<br>
      标注存于 localStorage，导出 CSV 即可
    </div>
  </div>
</div>
<script>
const POLYGONS = __ITEMS__;
const LABELS = __LABELS__;
const STORAGE_KEY = 'large_polygon_review_v1';
let labels = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
let mode = 'overlay';
let filter = 'all';
let cur = 0;
let visiblePolys = POLYGONS.slice();

function applyFilter() {
  if (filter === 'all') visiblePolys = POLYGONS.slice();
  else if (filter === 'unlabeled') visiblePolys = POLYGONS.filter(p => !labels[p.polygon_id]);
  else if (filter === 'flagged') visiblePolys = POLYGONS.filter(p => p.flag && p.flag.length > 0);
  else if (filter === 'ge500') visiblePolys = POLYGONS.filter(p => p.area_m2 >= 500);
  else if (filter === 'johannesburg') visiblePolys = POLYGONS.filter(p => p.region === 'johannesburg');
  else if (filter === 'cape_town') visiblePolys = POLYGONS.filter(p => p.region === 'cape_town');
  for (const f of ['all', 'unlabeled', 'flagged', 'ge500', 'johannesburg', 'cape_town']) {
    const el = document.getElementById('filter' + f.charAt(0).toUpperCase() + f.slice(1).replace('johannesburg','JHB').replace('cape_town','CT'));
    if (el) el.classList.toggle('active', filter === f);
  }
  if (cur >= visiblePolys.length) cur = 0;
  render();
}

function setFilter(f) { filter = f; applyFilter(); }
function setMode(m) {
  mode = m;
  document.getElementById('btnOverlay').classList.toggle('active', m === 'overlay');
  document.getElementById('btnRaw').classList.toggle('active', m === 'raw');
  render();
}

function render() {
  if (!visiblePolys.length) {
    document.getElementById('chipImg').src = '';
    document.getElementById('chipInfo').innerHTML = '<em>(此筛选下没有 polygon)</em>';
    return;
  }
  const p = visiblePolys[cur];
  document.getElementById('chipImg').src = mode === 'overlay' ? p.overlay : p.raw;
  document.getElementById('chipInfo').innerHTML = `
    <div><span class="key">polygon</span> <span class="val">#${p.polygon_id}</span></div>
    <div><span class="key">split</span> ${p.split}</div>
    <div><span class="key">region</span> ${p.region}</div>
    <div><span class="key">grid</span> ${p.grid_id}</div>
    <div><span class="key">layer</span> ${p.imagery_layer}</div>
    <div><span class="key">source</span> ${p.source ?? '(none)'}</div>
    <div><span class="key">area</span> <span class="val">${p.area_m2.toFixed(1)} m²</span></div>
    <div><span class="key">flag</span> ${p.flag || '—'}</div>
    <div><span class="key">tiles</span> ${p.tile_count}</div>
  `;
  const cur_label = labels[p.polygon_id];
  const lb = document.getElementById('labelButtons');
  lb.innerHTML = '';
  for (const [code, key, desc, color] of LABELS) {
    const btn = document.createElement('button');
    btn.className = 'label-btn' + (cur_label === code ? ' active' : '');
    btn.style.borderColor = cur_label === code ? color : '#333';
    btn.style.background = cur_label === code ? color : 'transparent';
    btn.style.color = cur_label === code ? '#1a1a2e' : '#eee';
    btn.innerHTML = `<span class="key">${key}</span><span>${desc}</span>`;
    btn.onclick = () => labelCurrent(code);
    lb.appendChild(btn);
  }
  const badge = document.getElementById('badge');
  if (cur_label) {
    badge.className = 'badge labeled';
    const desc = LABELS.find(l => l[0] === cur_label);
    badge.textContent = desc ? desc[2] : cur_label;
  } else {
    badge.className = 'badge unlabeled';
    badge.textContent = '未标注';
  }
  document.getElementById('currentIdx').textContent = cur + 1;
  document.getElementById('filterCount').textContent = visiblePolys.length;
  document.getElementById('totalCount').textContent = POLYGONS.length;
  document.getElementById('labeledCount').textContent = Object.keys(labels).length;
  renderSummary();
}

function renderSummary() {
  const counts = {};
  for (const code of LABELS.map(l => l[0])) counts[code] = 0;
  for (const v of Object.values(labels)) counts[v] = (counts[v] || 0) + 1;
  let html = '<table>';
  for (const [code, key, desc] of LABELS) {
    html += `<tr><td>${desc.split(' (')[0]}</td><td style="text-align:right;color:#4ecca3">${counts[code]}</td></tr>`;
  }
  html += '</table>';
  document.getElementById('summary').innerHTML = html;
}

function labelCurrent(code) {
  if (!visiblePolys.length) return;
  const pid = visiblePolys[cur].polygon_id;
  if (labels[pid] === code) {
    delete labels[pid];
  } else {
    labels[pid] = code;
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
  // auto-advance on label, but stay (and refresh) if already at last
  if (labels[pid] && cur < visiblePolys.length - 1) {
    next();
  } else {
    render();
  }
}

function next() { if (cur < visiblePolys.length - 1) { cur++; render(); } }
function prev() { if (cur > 0) { cur--; render(); } }
function skip() { next(); }

function jumpToId(id) {
  id = parseInt(id);
  if (isNaN(id)) return;
  const idx = visiblePolys.findIndex(p => p.polygon_id === id);
  if (idx >= 0) { cur = idx; render(); }
  else alert('polygon_id ' + id + ' not in current filter');
}

function exportCSV() {
  const rows = [['polygon_id','grid_id','region','split','area_m2','source','flag','label','labeled_at']];
  const ts = new Date().toISOString();
  for (const p of POLYGONS) {
    rows.push([p.polygon_id, p.grid_id, p.region, p.split, p.area_m2.toFixed(1),
               p.source ?? '', p.flag ?? '', labels[p.polygon_id] ?? '', labels[p.polygon_id] ? ts : '']);
  }
  const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g,'""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'large_polygon_review_labels.csv';
  a.click();
}

document.addEventListener('keydown', e => {
  if (document.activeElement && document.activeElement.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  if (k === 's' || k === 'arrowright') { e.preventDefault(); skip(); }
  else if (k === 'b' || k === 'arrowleft') { e.preventDefault(); prev(); }
  else if (k === 't') { e.preventDefault(); setMode(mode === 'overlay' ? 'raw' : 'overlay'); }
  else {
    const m = LABELS.find(l => l[1] === k);
    if (m) { e.preventDefault(); labelCurrent(m[0]); }
  }
});

applyFilter();
</script>
</body>
</html>
""".replace("__ITEMS__", items_json).replace("__LABELS__", labels_json)


if __name__ == "__main__":
    main()
