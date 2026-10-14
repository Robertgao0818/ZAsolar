"""Expand 12 fragmented "cut-in-half" polygons via SAM2 mask+box prompting.

Each cut polygon is a true sub-array of a larger installation; SAM is asked
to fill the rest by passing the original as a low-res mask prompt and an
expanded bbox. Multiple expansion factors are tried and the candidate is
chosen by area-growth + non-roof-swallow heuristics, but ALL candidates are
saved so the human reviewer can pick.

Outputs to ``data/coco_train20_val5_qa/expand_cut/``:
  - candidates.gpkg     # all expanded polygon candidates per seed
  - manifest.json       # seed metadata + per-candidate area/score
  - chips/<pid>_seed.png      # original polygon outline (cyan)
  - chips/<pid>_<strategy>.png # candidate outline (cyan = new, magenta = seed)
  - polygons.json       # metadata for HTML reviewer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from PIL import Image, ImageDraw
from rasterio.features import rasterize, shapes as rio_shapes
from rasterio.merge import merge as rio_merge
from rasterio.windows import Window
from shapely.geometry import box as shapely_box, shape

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.grid_utils import resolve_tiles_dir  # noqa: E402

QA_DIR = PROJECT_ROOT / "data/coco_train20_val5_qa"
SPEC_PATH = PROJECT_ROOT / "configs/datasets/train20_val5.yaml"
OUT_DIR = QA_DIR / "expand_cut"
CHIPS_DIR = OUT_DIR / "chips"

SAM2_CHECKPOINT = Path("/home/gaosh/zasolar_data/models/sam2/checkpoints/sam2.1_hiera_large.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l"

CRS_BY_REGION = {"johannesburg": "EPSG:32735", "cape_town": "EPSG:32734"}

# Cut polygon list — derived from labels.csv where label == fragmented_subarray
# AND visual review confirmed cut-from-larger
CUT_POLYGON_IDS = [25, 27, 39, 47, 71, 76, 86, 120, 187, 194, 221, 238]

# Expansion strategies: (name, bbox_scale, use_mask_prompt, n_points)
STRATEGIES = [
    ("box15_mask",  1.5, True,  0),
    ("box20_mask",  2.0, True,  0),
    ("box30_mask",  3.0, True,  0),
    ("box20_mask_pts", 2.0, True, 5),
]


def parse_grid_entry(entry):
    if isinstance(entry, dict):
        return entry["grid_id"], entry.get("file")
    return entry, None


def load_grid_meta():
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


def list_tiles(grid_id, region, layer):
    d = resolve_tiles_dir(grid_id, region=region, imagery_layer=layer)
    if d.is_file():
        return [d]
    tiles = sorted(d.glob(f"{grid_id}_*_*_geo.tif"))
    if not tiles:
        tiles = sorted(p for p in d.glob(f"{grid_id}_*.tif") if "mosaic" not in p.stem)
    return tiles


def load_sam():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM2 → {device}...")
    model = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=device)
    pred = SAM2ImagePredictor(model)
    return pred, device


def crop_chip_for_polygon(poly_4326, region, tiles, scale=3.0, max_size_px=1500):
    """Crop a chip around the polygon at `scale × bbox` in metric units.

    Returns: (chip_rgb [H,W,3] uint8, transform, metric_crs, tile_crs).
    The transform is in the tile's native CRS (pixel → coord).
    """
    metric_crs = CRS_BY_REGION[region]
    poly_metric = (gpd.GeoSeries([poly_4326], crs="EPSG:4326")
                   .to_crs(metric_crs).iloc[0])
    minx, miny, maxx, maxy = poly_metric.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = max(maxx - minx, maxy - miny) * scale / 2
    half = max(half, 12.0)  # ≥12 m radius minimum
    ext = (cx - half, cy - half, cx + half, cy + half)

    with rasterio.open(tiles[0]) as src0:
        tile_crs = src0.crs
    ext_tile = (gpd.GeoSeries([shapely_box(*ext)], crs=metric_crs)
                .to_crs(tile_crs).iloc[0]).bounds

    relevant = []
    for t in tiles:
        with rasterio.open(t) as src:
            b = src.bounds
            if not (ext_tile[2] < b.left or ext_tile[0] > b.right or
                    ext_tile[3] < b.bottom or ext_tile[1] > b.top):
                relevant.append(t)
    if not relevant:
        return None, None, metric_crs, tile_crs

    srcs = [rasterio.open(t) for t in relevant]
    try:
        arr, transform = rio_merge(srcs, bounds=ext_tile)
    finally:
        for s in srcs:
            s.close()
    if arr.shape[0] >= 3:
        rgb = arr[:3].transpose(1, 2, 0)
    else:
        rgb = np.repeat(arr[:1].transpose(1, 2, 0), 3, axis=2)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    if max(rgb.shape[:2]) > max_size_px:
        s = max_size_px / max(rgb.shape[:2])
        new_w = int(rgb.shape[1] * s)
        new_h = int(rgb.shape[0] * s)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # rebuild transform
        from rasterio.transform import Affine
        a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        transform = Affine(a / s, b, c, d, e / s, f)

    return rgb, transform, metric_crs, tile_crs


def poly_to_pixel_mask(poly_tile_crs, transform, h, w):
    return rasterize([(poly_tile_crs, 1)], out_shape=(h, w), transform=transform,
                     fill=0, all_touched=False, dtype=np.uint8)


def expand_with_sam(predictor, chip_rgb, seed_mask, scale, use_mask, n_points, device):
    h, w = chip_rgb.shape[:2]
    ys, xs = np.where(seed_mask > 0)
    if len(xs) == 0:
        return None, 0.0

    cx_seed = float(xs.mean())
    cy_seed = float(ys.mean())
    sw = xs.max() - xs.min() + 1
    sh = ys.max() - ys.min() + 1
    bw = sw * scale
    bh = sh * scale
    bx0 = max(0, int(cx_seed - bw / 2))
    by0 = max(0, int(cy_seed - bh / 2))
    bx1 = min(w, int(cx_seed + bw / 2))
    by1 = min(h, int(cy_seed + bh / 2))
    box = np.array([[bx0, by0, bx1, by1]], dtype=np.float32)

    point_coords = None
    point_labels = None
    if n_points > 0:
        idx = np.linspace(0, len(xs) - 1, n_points).astype(int)
        point_coords = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
        point_labels = np.ones(n_points, dtype=np.int32)

    mask_input = None
    if use_mask:
        # SAM2 expects mask_input at 256x256 in logit space (positive ~ +10, neg ~ -10)
        m256 = cv2.resize(seed_mask.astype(np.uint8) * 255, (256, 256),
                          interpolation=cv2.INTER_NEAREST)
        m256 = (m256 > 127).astype(np.float32)
        m_logits = m256 * 20.0 - 10.0
        # SAM2 ImagePredictor expects shape (1, 256, 256)
        mask_input = m_logits[None, :, :]

    predictor.set_image(chip_rgb)
    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=True,
        )

    # Pick mask: prefer one that contains the seed centroid AND has area > seed
    seed_area = float(seed_mask.sum())
    best_mask = None
    best_score = -1.0
    for i, (m, sc) in enumerate(zip(masks, scores)):
        m_bool = m > 0.5
        # Must overlap seed substantially
        overlap_seed = (m_bool & (seed_mask > 0)).sum() / max(seed_area, 1)
        if overlap_seed < 0.5:
            continue
        new_area = float(m_bool.sum())
        # Prefer larger but not roof-swallow (cap at chip area * 0.6)
        if new_area > h * w * 0.6:
            continue
        # composite score: SAM confidence + log(growth)
        growth = max(new_area / max(seed_area, 1), 1.0)
        composite = float(sc) + 0.4 * np.log(growth)
        if composite > best_score:
            best_score = composite
            best_mask = m_bool
    if best_mask is None:
        return None, 0.0
    return best_mask, float(best_score)


def mask_to_polygon(mask, transform):
    if mask.sum() == 0:
        return None
    polys = []
    for geom, val in rio_shapes(mask.astype(np.uint8), transform=transform):
        if val == 1:
            polys.append(shape(geom))
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    from shapely.ops import unary_union
    return unary_union(polys)


def render_overlay(chip_rgb, seed_mask, new_mask, transform):
    img = Image.fromarray(chip_rgb).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    if seed_mask is not None and seed_mask.sum() > 0:
        contours, _ = cv2.findContours(seed_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            pts = [(int(p[0][0]), int(p[0][1])) for p in c]
            if len(pts) >= 3:
                draw.line(pts + [pts[0]], fill=(255, 0, 255, 255), width=2)
    if new_mask is not None and new_mask.sum() > 0:
        contours, _ = cv2.findContours(new_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            pts = [(int(p[0][0]), int(p[0][1])) for p in c]
            if len(pts) >= 3:
                draw.line(pts + [pts[0]], fill=(0, 255, 255, 255), width=3)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polygon-ids", nargs="+", type=int, default=CUT_POLYGON_IDS)
    ap.add_argument("--chip-scale", type=float, default=3.5,
                    help="Chip extent vs seed bbox (default 3.5x)")
    ap.add_argument("--target-size", type=int, default=1024,
                    help="SAM2 image size (max chip dim, default 1024)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHIPS_DIR.mkdir(parents=True, exist_ok=True)

    df_large = pd.read_csv(QA_DIR / "large_polygons.csv")
    df = df_large[df_large.polygon_id.isin(args.polygon_ids)].reset_index(drop=True)
    print(f"Processing {len(df)} cut polygons")

    grid_meta = load_grid_meta()
    predictor, device = load_sam()

    out_records = []
    candidate_features = []

    cached_grid = None
    cached_gdf_4326 = None

    for _, row in df.iterrows():
        pid = int(row.polygon_id)
        gid = row.grid_id
        meta = grid_meta[gid]
        print(f"\n--- polygon {pid} ({gid}, {row.area_m2:.0f} m²) ---")

        if cached_grid != gid:
            g = gpd.read_file(meta["annotation_path"])
            if g.crs is None:
                g = g.set_crs("EPSG:4326")
            cached_gdf_4326 = g.to_crs("EPSG:4326")
            cached_grid = gid

        # Match polygon by area
        cached_metric = cached_gdf_4326.to_crs(CRS_BY_REGION[meta["region"]])
        diff = (cached_metric.geometry.area - row.area_m2).abs()
        cand_idx = diff.idxmin()
        seed_4326 = cached_gdf_4326.geometry.iloc[cand_idx]

        tiles = list_tiles(gid, meta["region"], meta["imagery_layer"])
        chip_rgb, transform, metric_crs, tile_crs = crop_chip_for_polygon(
            seed_4326, meta["region"], tiles,
            scale=args.chip_scale, max_size_px=args.target_size,
        )
        if chip_rgb is None:
            print(f"  SKIP: no chip")
            continue

        seed_tile = (gpd.GeoSeries([seed_4326], crs="EPSG:4326")
                     .to_crs(tile_crs).iloc[0])
        h, w = chip_rgb.shape[:2]
        seed_mask = poly_to_pixel_mask(seed_tile, transform, h, w)
        seed_area_m2 = float(row.area_m2)
        print(f"  chip: {w}x{h}, seed_mask px: {int(seed_mask.sum())}")

        # Save seed-only image
        seed_img = render_overlay(chip_rgb, seed_mask, None, transform)
        seed_img.save(CHIPS_DIR / f"{pid:04d}_seed.png")

        candidates_this = []
        for name, scale, use_mask, n_points in STRATEGIES:
            new_mask, sc = expand_with_sam(predictor, chip_rgb, seed_mask,
                                           scale=scale, use_mask=use_mask,
                                           n_points=n_points, device=device)
            if new_mask is None:
                print(f"  {name}: no valid mask")
                continue
            new_poly_tile = mask_to_polygon(new_mask, transform)
            if new_poly_tile is None or new_poly_tile.is_empty:
                continue
            new_poly_metric = (gpd.GeoSeries([new_poly_tile], crs=tile_crs)
                               .to_crs(metric_crs).iloc[0])
            new_area_m2 = float(new_poly_metric.area)
            growth = new_area_m2 / seed_area_m2
            print(f"  {name}: area {new_area_m2:.0f} m² (growth {growth:.2f}x, score {sc:.3f})")

            ovl = render_overlay(chip_rgb, seed_mask, new_mask, transform)
            ovl_path = CHIPS_DIR / f"{pid:04d}_{name}.png"
            ovl.save(ovl_path)

            new_poly_4326 = (gpd.GeoSeries([new_poly_metric], crs=metric_crs)
                             .to_crs("EPSG:4326").iloc[0])
            candidates_this.append({
                "polygon_id": pid,
                "strategy": name,
                "score": float(sc),
                "seed_area_m2": seed_area_m2,
                "new_area_m2": new_area_m2,
                "growth_factor": float(growth),
                "image": f"chips/{pid:04d}_{name}.png",
            })
            candidate_features.append({
                "polygon_id": pid,
                "strategy": name,
                "seed_area_m2": seed_area_m2,
                "new_area_m2": new_area_m2,
                "growth_factor": float(growth),
                "score": float(sc),
                "geometry": new_poly_4326,
            })

        out_records.append({
            "polygon_id": pid,
            "grid_id": gid,
            "region": meta["region"],
            "split": meta["split"],
            "imagery_layer": meta["imagery_layer"],
            "seed_area_m2": seed_area_m2,
            "source": row.src_field if pd.notna(row.src_field) else None,
            "seed_image": f"chips/{pid:04d}_seed.png",
            "candidates": candidates_this,
        })

    # Save GPKG of candidates
    if candidate_features:
        cdf = gpd.GeoDataFrame(candidate_features, crs="EPSG:4326")
        cdf.to_file(OUT_DIR / "candidates.gpkg", driver="GPKG")

    (OUT_DIR / "manifest.json").write_text(json.dumps(out_records, indent=2) + "\n")
    print(f"\n[SAVE] {OUT_DIR/'manifest.json'} ({len(out_records)} seeds, "
          f"{len(candidate_features)} candidates)")

    # Build review HTML
    html = build_review_html(out_records)
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"[SAVE] {OUT_DIR/'index.html'}")


def build_review_html(records: list[dict]) -> str:
    items_json = json.dumps(records, ensure_ascii=False)
    return r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8">
<title>Cut-Polygon Expansion Review</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: system-ui, sans-serif; background:#1a1a2e; color:#eee;
       display:flex; flex-direction:column; height:100vh; }
.header { padding:8px 16px; background:#16213e; display:flex; align-items:center;
          gap:16px; flex-shrink:0; flex-wrap:wrap; }
.main { flex:1; display:flex; overflow:hidden; }
.candidates { flex:1; display:grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
              gap:8px; padding:12px; overflow-y:auto; }
.cand { border:2px solid #333; border-radius:6px; background:#0f1830;
        padding:6px; cursor:pointer; }
.cand img { width:100%; image-rendering:pixelated; border-radius:4px; }
.cand .meta { font-size:12px; color:#aaa; padding:6px 4px; line-height:1.5; }
.cand.selected { border-color:#4ecca3; box-shadow:0 0 0 2px #4ecca366; }
.cand.seed { border-color:#888; }
.cand .label { display:inline-block; background:#16213e; color:#4ecca3;
               padding:2px 8px; border-radius:8px; font-size:11px; }
.sidebar { width:300px; background:#16213e; padding:12px; overflow-y:auto;
           display:flex; flex-direction:column; gap:8px; flex-shrink:0; }
.info { font-size:13px; line-height:1.6; padding:8px; background:#0f3460; border-radius:6px; }
.info .key { color:#aaa; display:inline-block; width:88px; }
.info .val { color:#4ecca3; }
.btn { padding:10px; border:none; border-radius:6px; background:#0f3460; color:#eee;
       cursor:pointer; font-size:13px; }
.btn.accept { background:#4ecca3; color:#1a1a2e; font-weight:bold; }
.btn.reject { background:#e94560; color:#fff; font-weight:bold; }
.btn.export { background:#f39c12; color:#1a1a2e; font-weight:bold; margin-top:8px; }
.nav { display:flex; gap:8px; }
.nav .btn { flex:1; }
.hint { font-size:11px; color:#666; line-height:1.5; }
</style></head><body>
<div class="header">
  <h1 style="font-size:16px">Cut-Polygon Expansion Review</h1>
  <span id="prog" style="color:#aaa">0 / 0</span>
  <span id="counts" style="color:#aaa"></span>
</div>
<div class="main">
  <div class="candidates" id="candList"></div>
  <div class="sidebar">
    <div class="info" id="info"></div>
    <div class="nav">
      <button class="btn" onclick="prev()">← B 上一</button>
      <button class="btn" onclick="next()">S 下一 →</button>
    </div>
    <button class="btn reject" onclick="rejectAll()">R 全部拒绝（保留原 polygon）</button>
    <button class="btn export" onclick="exportDecisions()">导出决策 CSV</button>
    <div class="hint">
      点击候选选中 / 取消，每个 polygon 最多选 1 个候选。<br>
      1-4 = 选第 N 个候选 · 0 = seed-only · R = 全部拒绝<br>
      S/B 切换上下一个 polygon<br>
      青色 = 候选扩展 · 紫色 = 原 polygon
    </div>
  </div>
</div>
<script>
const RECORDS = __ITEMS__;
const STORAGE_KEY = 'expand_cut_review_v1';
let decisions = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
let cur = 0;

function render() {
  const r = RECORDS[cur];
  if (!r) { document.getElementById('candList').innerHTML = '<em>no records</em>'; return; }
  const dec = decisions[r.polygon_id] || null;

  const cl = document.getElementById('candList');
  cl.innerHTML = '';
  // seed first (clickable to mean "reject all expansions, keep seed")
  const sd = document.createElement('div');
  sd.className = 'cand seed' + (dec === 'seed' || dec === null ? ' selected' : '');
  sd.innerHTML = `<img src="${r.seed_image}" /><div class="meta">
    <span class="label">seed</span> ${r.seed_area_m2.toFixed(0)} m² · 原 polygon
  </div>`;
  sd.onclick = () => decide('seed');
  cl.appendChild(sd);

  for (let i = 0; i < r.candidates.length; i++) {
    const c = r.candidates[i];
    const card = document.createElement('div');
    card.className = 'cand' + (dec === c.strategy ? ' selected' : '');
    card.innerHTML = `<img src="${c.image}" /><div class="meta">
      <span class="label">${c.strategy}</span>
      ${c.new_area_m2.toFixed(0)} m² · growth ${c.growth_factor.toFixed(2)}x
      · score ${c.score.toFixed(2)}
    </div>`;
    card.onclick = () => decide(c.strategy);
    cl.appendChild(card);
  }
  document.getElementById('info').innerHTML = `
    <div><span class="key">polygon</span> <span class="val">#${r.polygon_id}</span></div>
    <div><span class="key">grid</span> ${r.grid_id}</div>
    <div><span class="key">split</span> ${r.split}</div>
    <div><span class="key">region</span> ${r.region}</div>
    <div><span class="key">layer</span> ${r.imagery_layer}</div>
    <div><span class="key">seed area</span> ${r.seed_area_m2.toFixed(0)} m²</div>
    <div><span class="key">source</span> ${r.source ?? '(none)'}</div>
    <div><span class="key">decision</span> <span class="val">${dec || '(未决)'}</span></div>
  `;
  document.getElementById('prog').textContent = `${cur+1} / ${RECORDS.length}`;
  const dn = Object.keys(decisions).length;
  document.getElementById('counts').textContent = `已决 ${dn} / ${RECORDS.length}`;
}
function decide(strategy) {
  const r = RECORDS[cur];
  if (decisions[r.polygon_id] === strategy) {
    delete decisions[r.polygon_id];
  } else {
    decisions[r.polygon_id] = strategy;
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(decisions));
  render();
}
function rejectAll() { decide('seed'); }
function next() { if (cur < RECORDS.length - 1) cur++; render(); }
function prev() { if (cur > 0) cur--; render(); }
function exportDecisions() {
  const rows = [['polygon_id','grid_id','region','split','seed_area_m2','decision','chosen_area_m2','growth_factor']];
  for (const r of RECORDS) {
    const d = decisions[r.polygon_id] || 'seed';
    let area = r.seed_area_m2, growth = 1.0;
    if (d !== 'seed') {
      const c = r.candidates.find(x => x.strategy === d);
      if (c) { area = c.new_area_m2; growth = c.growth_factor; }
    }
    rows.push([r.polygon_id, r.grid_id, r.region, r.split, r.seed_area_m2.toFixed(1),
               d, area.toFixed(1), growth.toFixed(2)]);
  }
  const csv = rows.map(r=>r.map(v=>`"${String(v).replace(/"/g,'""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'expand_cut_decisions.csv';
  a.click();
}
document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (k === 's' || k === 'arrowright') { e.preventDefault(); next(); }
  else if (k === 'b' || k === 'arrowleft') { e.preventDefault(); prev(); }
  else if (k === 'r') { e.preventDefault(); rejectAll(); }
  else if (k === '0') { e.preventDefault(); decide('seed'); }
  else if (['1','2','3','4'].includes(k)) {
    const i = parseInt(k) - 1;
    const r = RECORDS[cur];
    if (r && r.candidates[i]) { e.preventDefault(); decide(r.candidates[i].strategy); }
  }
});
render();
</script></body></html>
""".replace("__ITEMS__", items_json)


if __name__ == "__main__":
    main()
