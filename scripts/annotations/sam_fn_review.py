#!/usr/bin/env python3
"""
SAM FN Review Server — interactive review of SAM 2.1 segmentation for FN markers.

Loads SAM 2.1 Large, pre-runs initial segmentation on all remaining FN markers,
then serves an HTML interface where the user can:
  - View tile image with original Edit polygon (if any) and SAM result
  - Left-click to add positive prompt points
  - Right-click to add negative prompt points
  - Re-segment with updated points
  - Accept / Skip / Delete each marker

Usage:
    python scripts/annotations/sam_fn_review.py [--port 8770]
"""

import argparse
import io
import json
import sys
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse, parse_qs

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image, ImageDraw
from shapely.geometry import shape, mapping, Point
from shapely.ops import transform
from shapely.validation import make_valid
import pyproj

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TILES_ROOT = Path("/mnt/d/ZAsolar/tiles")
SAM_CHECKPOINT = Path("/mnt/c/Users/gaosh/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/GeoOSAM/sam2/checkpoints/sam2.1_hiera_large.pt")
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_l"
METRIC_CRS = "EPSG:32734"
CROP_HALF = 300  # pixels around marker for preview

proj_to_metric = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32734", always_xy=True)
proj_to_4326 = pyproj.Transformer.from_crs("EPSG:32734", "EPSG:4326", always_xy=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_fn_markers() -> list[dict]:
    """Load all remaining FN markers with metadata."""
    markers = []
    idx = 0
    for base in [Path("results"), Path("/mnt/d/ZAsolar/results")]:
        for gpkg_path in sorted(base.glob("*/review/*_fn_markers.gpkg")):
            grid = gpkg_path.parent.parent.name
            gdf = gpd.read_file(gpkg_path)
            if len(gdf) == 0:
                continue

            # Load Edit polygons for this grid
            dec_csv = gpkg_path.parent / "detection_review_decisions.csv"
            pred_gpkg = gpkg_path.parent.parent / "predictions_metric.gpkg"
            edit_polys = []
            if dec_csv.exists() and pred_gpkg.exists():
                decisions = pd.read_csv(dec_csv)
                edit_ids = set(decisions[decisions["status"] == "edit"]["pred_id"])
                if edit_ids:
                    preds = gpd.read_file(pred_gpkg)
                    edits = preds[preds.index.isin(edit_ids)]
                    if edits.crs and edits.crs.to_epsg() != 4326:
                        edits = edits.to_crs(epsg=4326)
                    edit_polys = list(edits.geometry)

            for _, row in gdf.iterrows():
                geom = row.geometry
                tile_key = row.get("tile_key", "")
                parts = tile_key.split("_")
                if len(parts) < 3:
                    continue
                col, r = int(parts[-2]), int(parts[-1])
                tile_path = TILES_ROOT / grid / f"{grid}_{col}_{r}_geo.tif"

                # Find which Edit polygon contains this marker (if any)
                edit_poly_wkt = None
                if geom and edit_polys:
                    geom_4326 = geom if geom.geom_type == "Point" else geom.centroid
                    for ep in edit_polys:
                        if ep.contains(geom_4326) or ep.distance(geom_4326) < 0.0001:
                            edit_poly_wkt = ep.wkt
                            break

                # Get pixel coords from GPKG geometry + tile transform
                px, py = 0, 0
                if tile_path.exists() and geom:
                    with rasterio.open(tile_path) as src:
                        if geom.geom_type == "Point":
                            col_px, row_px = ~src.transform * (geom.x, geom.y)
                        else:
                            c = geom.centroid
                            col_px, row_px = ~src.transform * (c.x, c.y)
                        px, py = int(col_px), int(row_px)

                markers.append({
                    "id": idx,
                    "grid_id": grid,
                    "tile_key": tile_key,
                    "tile_path": str(tile_path),
                    "px": px,
                    "py": py,
                    "edit_poly_wkt": edit_poly_wkt,
                    "status": "pending",  # pending / accepted / skipped / deleted
                    "positive_points": [[px, py]],
                    "negative_points": [],
                    "sam_wkt": None,
                    "sam_area_m2": None,
                })
                idx += 1

    return markers


# ---------------------------------------------------------------------------
# SAM model
# ---------------------------------------------------------------------------
class SAMSegmenter:
    def __init__(self):
        self.model = None
        self.lock = Lock()

    def load(self):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("Loading SAM 2.1 Large...")
        sam2 = build_sam2(SAM_CONFIG, str(SAM_CHECKPOINT), device="cuda")
        self.predictor = SAM2ImagePredictor(sam2)
        print("SAM loaded.")

    def segment(self, image_array: np.ndarray,
                positive_points: list, negative_points: list) -> tuple:
        """Run SAM with given points. Returns (mask, score, polygon_wkt, area_m2)."""
        with self.lock:
            self.predictor.set_image(image_array)

            points = positive_points + negative_points
            labels = [1] * len(positive_points) + [0] * len(negative_points)

            masks, scores, _ = self.predictor.predict(
                point_coords=np.array(points, dtype=np.float32),
                point_labels=np.array(labels, dtype=np.int32),
                multimask_output=True,
            )

            # Pick best mask (highest score)
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = float(scores[best_idx])

            return mask, score


sam_model = SAMSegmenter()


def mask_to_polygon_wkt(mask: np.ndarray, tile_path: str) -> tuple:
    """Convert binary mask to WKT polygon in EPSG:4326, return (wkt, area_m2)."""
    from rasterio.features import shapes as rio_shapes
    with rasterio.open(tile_path) as src:
        tf = src.transform

    mask_uint8 = mask.astype(np.uint8)
    polys = []
    for geom, val in rio_shapes(mask_uint8, mask=mask_uint8 > 0, transform=tf):
        if val > 0:
            poly = shape(geom)
            if poly.is_valid and poly.area > 0:
                polys.append(poly)

    if not polys:
        return None, 0

    biggest = max(polys, key=lambda p: p.area)
    # Area in metric CRS
    metric_poly = transform(proj_to_metric.transform, biggest)
    area_m2 = metric_poly.area

    return biggest.wkt, area_m2


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
class ReviewHandler(BaseHTTPRequestHandler):
    markers = []
    current_idx = 0
    output_path = None

    def log_message(self, fmt, *args):
        pass  # suppress logs

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _png(self, img: Image.Image):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        body = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._serve_html()
        elif path == "/api/status":
            self._api_status()
        elif path.startswith("/api/marker/") and path.endswith("/image"):
            mid = int(path.split("/")[3])
            self._api_image(mid)
        elif path.startswith("/api/marker/") and path.endswith("/data"):
            mid = int(path.split("/")[3])
            self._api_marker_data(mid)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if path.startswith("/api/marker/") and path.endswith("/segment"):
            mid = int(path.split("/")[3])
            self._api_segment(mid, body)
        elif path.startswith("/api/marker/") and path.endswith("/decide"):
            mid = int(path.split("/")[3])
            self._api_decide(mid, body)
        elif path == "/api/save":
            self._api_save()
        else:
            self.send_error(404)

    def _api_status(self):
        counts = {"total": len(self.markers)}
        for s in ["pending", "accepted", "skipped", "deleted"]:
            counts[s] = sum(1 for m in self.markers if m["status"] == s)
        self._json(counts)

    def _api_marker_data(self, mid):
        if mid < 0 or mid >= len(self.markers):
            self.send_error(404)
            return
        m = self.markers[mid]
        self._json({
            "id": m["id"],
            "grid_id": m["grid_id"],
            "tile_key": m["tile_key"],
            "px": m["px"], "py": m["py"],
            "status": m["status"],
            "has_edit_poly": m["edit_poly_wkt"] is not None,
            "positive_points": m["positive_points"],
            "negative_points": m["negative_points"],
            "sam_area_m2": m["sam_area_m2"],
        })

    def _api_image(self, mid):
        if mid < 0 or mid >= len(self.markers):
            self.send_error(404)
            return
        m = self.markers[mid]
        tile_path = m["tile_path"]
        if not Path(tile_path).exists():
            self.send_error(404)
            return

        img = Image.open(tile_path).convert("RGB")
        px, py = m["px"], m["py"]

        # Crop
        x1 = max(0, px - CROP_HALF)
        y1 = max(0, py - CROP_HALF)
        x2 = min(img.width, px + CROP_HALF)
        y2 = min(img.height, py + CROP_HALF)
        crop = img.crop((x1, y1, x2, y2))
        draw = ImageDraw.Draw(crop)

        with rasterio.open(tile_path) as src:
            tile_tf = src.transform

        # Draw Edit polygon (yellow dashed)
        if m["edit_poly_wkt"]:
            from shapely import wkt
            edit_geom = wkt.loads(m["edit_poly_wkt"])
            if edit_geom.geom_type == "Polygon":
                coords = list(edit_geom.exterior.coords)
                pxc = [(~tile_tf * (lon, lat)) for lon, lat in coords]
                pxc = [(c - x1, r - y1) for c, r in pxc]
                draw.polygon(pxc, outline="yellow", width=2)

        # Draw SAM polygon (cyan)
        if m["sam_wkt"]:
            from shapely import wkt
            sam_geom = wkt.loads(m["sam_wkt"])
            if sam_geom.geom_type == "Polygon":
                coords = list(sam_geom.exterior.coords)
                pxc = [(~tile_tf * (lon, lat)) for lon, lat in coords]
                pxc = [(c - x1, r - y1) for c, r in pxc]
                draw.polygon(pxc, outline="cyan", width=2)

        # Draw positive points (green)
        for pt in m["positive_points"]:
            cx, cy = pt[0] - x1, pt[1] - y1
            r = 8
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill="lime", outline="white", width=1)

        # Draw negative points (red)
        for pt in m["negative_points"]:
            cx, cy = pt[0] - x1, pt[1] - y1
            r = 8
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill="red", outline="white", width=1)

        self._png(crop)

    def _api_segment(self, mid, body):
        if mid < 0 or mid >= len(self.markers):
            self.send_error(404)
            return
        m = self.markers[mid]

        # Update points from request
        if "positive_points" in body:
            m["positive_points"] = body["positive_points"]
        if "negative_points" in body:
            m["negative_points"] = body["negative_points"]

        tile_path = m["tile_path"]
        if not Path(tile_path).exists():
            self._json({"error": "tile not found"}, 404)
            return

        with rasterio.open(tile_path) as src:
            img_array = np.moveaxis(src.read()[:3], 0, -1)

        pos = m["positive_points"]
        neg = m["negative_points"]
        if not pos:
            self._json({"error": "no positive points"}, 400)
            return

        mask, score = sam_model.segment(img_array, pos, neg)
        wkt_str, area_m2 = mask_to_polygon_wkt(mask, tile_path)

        m["sam_wkt"] = wkt_str
        m["sam_area_m2"] = area_m2

        self._json({
            "score": score,
            "area_m2": round(area_m2, 1),
            "has_polygon": wkt_str is not None,
        })

    def _api_decide(self, mid, body):
        if mid < 0 or mid >= len(self.markers):
            self.send_error(404)
            return
        decision = body.get("decision", "skip")
        self.markers[mid]["status"] = decision
        self._json({"ok": True, "status": decision})

    def _api_save(self):
        """Save accepted SAM polygons back to reviewed GPKGs."""
        accepted = [m for m in self.markers if m["status"] == "accepted" and m["sam_wkt"]]
        if not accepted:
            self._json({"saved": 0})
            return

        # Group by grid
        by_grid = {}
        for m in accepted:
            by_grid.setdefault(m["grid_id"], []).append(m)

        total_saved = 0
        for grid_id, grid_markers in by_grid.items():
            # Find reviewed gpkg
            for base in [Path("results"), Path("/mnt/d/ZAsolar/results")]:
                reviewed_path = base / grid_id / "review" / f"{grid_id}_reviewed.gpkg"
                if reviewed_path.exists():
                    break
            else:
                continue

            gdf = gpd.read_file(str(reviewed_path))
            crs = gdf.crs

            # Remove old SAM FN polygons
            if "source" in gdf.columns:
                gdf = gdf[gdf["source"] != "sam_fn_review"].copy()

            new_polys = []
            for m in grid_markers:
                from shapely import wkt
                poly = wkt.loads(m["sam_wkt"])
                new_polys.append({
                    "geometry": poly,
                    "source": "sam_fn_review",
                    "tile_key": m["tile_key"],
                    "confidence": 1.0,
                    "review_status": "correct",
                })

            if new_polys:
                fn_gdf = gpd.GeoDataFrame(new_polys, crs=crs)
                for col in gdf.columns:
                    if col not in fn_gdf.columns and col != "geometry":
                        fn_gdf[col] = None
                for col in fn_gdf.columns:
                    if col not in gdf.columns and col != "geometry":
                        gdf[col] = None
                merged = gpd.GeoDataFrame(
                    pd.concat([gdf, fn_gdf], ignore_index=True), crs=crs
                )
                merged.to_file(str(reviewed_path), driver="GPKG")
                total_saved += len(new_polys)

        # Also save decisions log
        log = []
        for m in self.markers:
            log.append({
                "id": m["id"], "grid_id": m["grid_id"],
                "tile_key": m["tile_key"], "status": m["status"],
                "sam_area_m2": m["sam_area_m2"],
                "n_pos_points": len(m["positive_points"]),
                "n_neg_points": len(m["negative_points"]),
            })
        log_path = self.output_path / "sam_fn_review_decisions.csv"
        pd.DataFrame(log).to_csv(log_path, index=False)

        self._json({"saved": total_saved, "log": str(log_path)})

    def _serve_html(self):
        html = HTML_TEMPLATE
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>SAM FN Review</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, sans-serif; background: #1a1a1a; color: #eee; display: flex; height: 100vh; }
  #sidebar { width: 320px; background: #252525; padding: 16px; overflow-y: auto; border-right: 1px solid #444; }
  #main { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; }
  canvas { border: 1px solid #555; cursor: crosshair; max-width: 95vw; max-height: 85vh; }
  h2 { font-size: 16px; margin-bottom: 12px; color: #8cf; }
  .info { font-size: 13px; color: #aaa; margin-bottom: 8px; line-height: 1.6; }
  .info b { color: #eee; }
  .btn-row { display: flex; gap: 8px; margin: 12px 0; flex-wrap: wrap; }
  button { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: 600; }
  .btn-accept { background: #2a7; color: #fff; }
  .btn-skip { background: #666; color: #fff; }
  .btn-delete { background: #a33; color: #fff; }
  .btn-reseg { background: #47a; color: #fff; }
  .btn-reset { background: #555; color: #fff; }
  .btn-save { background: #a72; color: #fff; width: 100%; margin-top: 16px; padding: 12px; font-size: 15px; }
  .btn-nav { background: #444; color: #fff; }
  .status-bar { display: flex; gap: 12px; font-size: 12px; margin: 12px 0; }
  .status-bar span { padding: 2px 8px; border-radius: 3px; }
  .s-pending { background: #555; }
  .s-accepted { background: #2a7; }
  .s-skipped { background: #666; }
  .s-deleted { background: #a33; }
  #marker-status { font-size: 14px; padding: 4px 12px; border-radius: 4px; margin: 8px 0; display: inline-block; }
  .legend { font-size: 11px; color: #888; margin-top: 12px; line-height: 1.8; }
  .legend span { display: inline-block; width: 12px; height: 12px; border-radius: 2px; vertical-align: middle; margin-right: 4px; }
  .point-mode { font-size: 13px; margin: 8px 0; padding: 8px; background: #333; border-radius: 4px; }
  .point-mode label { cursor: pointer; margin-right: 12px; }
  #toolbar { padding: 8px 16px; background: #252525; border-bottom: 1px solid #444; width: 100%; display: flex; align-items: center; gap: 16px; }
</style>
</head><body>

<div id="sidebar">
  <h2>SAM FN Review</h2>
  <div class="status-bar" id="status-bar"></div>
  <div class="info" id="marker-info"></div>
  <div id="marker-status"></div>

  <div class="point-mode">
    <b>Click mode:</b><br>
    <label><input type="radio" name="mode" value="positive" checked> Left: + point</label>
    <label><input type="radio" name="mode" value="negative"> Left: − point</label>
    <p style="font-size:11px;color:#888;margin-top:4px">Right-click always adds − point</p>
  </div>

  <div class="btn-row">
    <button class="btn-reseg" onclick="resegment()">Re-segment (S)</button>
    <button class="btn-reset" onclick="resetPoints()">Reset points (R)</button>
  </div>

  <div class="btn-row">
    <button class="btn-accept" onclick="decide('accepted')">Accept (A)</button>
    <button class="btn-skip" onclick="decide('skipped')">Skip (K)</button>
    <button class="btn-delete" onclick="decide('deleted')">Delete (D)</button>
  </div>

  <div class="btn-row">
    <button class="btn-nav" onclick="nav(-1)">← Prev (←)</button>
    <button class="btn-nav" onclick="nav(1)">Next → (→)</button>
    <button class="btn-nav" onclick="navNextPending()">Next pending (N)</button>
  </div>

  <div class="legend">
    <span style="background:lime"></span> Positive point<br>
    <span style="background:red"></span> Negative point<br>
    <span style="background:cyan"></span> SAM polygon<br>
    <span style="background:yellow"></span> Edit polygon (original)<br>
  </div>

  <button class="btn-save" onclick="saveAll()">Save all accepted → GPKG</button>
  <div id="save-msg" style="margin-top:8px;font-size:12px;color:#8cf;"></div>
</div>

<div id="main">
  <div id="toolbar">
    <span id="nav-label" style="font-size:14px;"></span>
  </div>
  <canvas id="canvas"></canvas>
</div>

<script>
let markers = [];
let currentIdx = 0;
let totalMarkers = 0;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let tileImg = null;

async function init() {
  const res = await fetch('/api/status');
  const status = await res.json();
  totalMarkers = status.total;
  updateStatusBar(status);
  loadMarker(0);
}

function updateStatusBar(s) {
  document.getElementById('status-bar').innerHTML =
    `<span class="s-pending">Pending: ${s.pending}</span>` +
    `<span class="s-accepted">Accepted: ${s.accepted}</span>` +
    `<span class="s-skipped">Skipped: ${s.skipped}</span>` +
    `<span class="s-deleted">Deleted: ${s.deleted}</span>`;
}

async function loadMarker(idx) {
  if (idx < 0 || idx >= totalMarkers) return;
  currentIdx = idx;

  // Load data
  const res = await fetch(`/api/marker/${idx}/data`);
  const data = await res.json();

  // Update info
  document.getElementById('marker-info').innerHTML =
    `<b>${data.grid_id} / ${data.tile_key}</b><br>` +
    `Marker: (${data.px}, ${data.py})<br>` +
    `Edit polygon: ${data.has_edit_poly ? 'Yes' : 'No'}<br>` +
    `SAM area: ${data.sam_area_m2 ? data.sam_area_m2.toFixed(1) + ' m²' : '—'}<br>` +
    `Points: +${data.positive_points.length} / −${data.negative_points.length}`;

  const statusEl = document.getElementById('marker-status');
  statusEl.textContent = data.status.toUpperCase();
  statusEl.className = 's-' + data.status;
  statusEl.style.padding = '4px 12px';
  statusEl.style.borderRadius = '4px';

  document.getElementById('nav-label').textContent = `${idx + 1} / ${totalMarkers}`;

  // Load image
  tileImg = new window.Image();
  tileImg.onload = () => {
    canvas.width = tileImg.width;
    canvas.height = tileImg.height;
    ctx.drawImage(tileImg, 0, 0);
  };
  tileImg.src = `/api/marker/${idx}/image?t=${Date.now()}`;
}

function getClickMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = Math.round((e.clientX - rect.left) * scaleX);
  const y = Math.round((e.clientY - rect.top) * scaleY);

  // Convert canvas coords back to tile coords
  // Canvas shows a crop starting at (px - CROP_HALF, py - CROP_HALF)
  addPoint(x, y, getClickMode());
});

canvas.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = Math.round((e.clientX - rect.left) * scaleX);
  const y = Math.round((e.clientY - rect.top) * scaleY);
  addPoint(x, y, 'negative');
});

async function addPoint(canvasX, canvasY, type) {
  // Get marker data to compute tile coords
  const res = await fetch(`/api/marker/${currentIdx}/data`);
  const data = await res.json();
  const cropX1 = Math.max(0, data.px - """ + str(CROP_HALF) + """);
  const cropY1 = Math.max(0, data.py - """ + str(CROP_HALF) + """);
  const tileX = canvasX + cropX1;
  const tileY = canvasY + cropY1;

  if (type === 'positive') {
    data.positive_points.push([tileX, tileY]);
  } else {
    data.negative_points.push([tileX, tileY]);
  }

  // Re-segment with updated points
  const segRes = await fetch(`/api/marker/${currentIdx}/segment`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      positive_points: data.positive_points,
      negative_points: data.negative_points,
    }),
  });
  const segData = await segRes.json();

  // Refresh image and info
  loadMarker(currentIdx);
}

async function resegment() {
  const res = await fetch(`/api/marker/${currentIdx}/data`);
  const data = await res.json();
  await fetch(`/api/marker/${currentIdx}/segment`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      positive_points: data.positive_points,
      negative_points: data.negative_points,
    }),
  });
  loadMarker(currentIdx);
}

async function resetPoints() {
  const res = await fetch(`/api/marker/${currentIdx}/data`);
  const data = await res.json();
  // Reset to only the original marker point
  await fetch(`/api/marker/${currentIdx}/segment`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      positive_points: [[data.px, data.py]],
      negative_points: [],
    }),
  });
  loadMarker(currentIdx);
}

async function decide(decision) {
  await fetch(`/api/marker/${currentIdx}/decide`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({decision}),
  });
  // Update status bar
  const statusRes = await fetch('/api/status');
  updateStatusBar(await statusRes.json());
  // Auto-advance to next pending
  navNextPending();
}

function nav(delta) { loadMarker(currentIdx + delta); }

async function navNextPending() {
  for (let i = currentIdx + 1; i < totalMarkers; i++) {
    const res = await fetch(`/api/marker/${i}/data`);
    const data = await res.json();
    if (data.status === 'pending') { loadMarker(i); return; }
  }
  // Wrap around
  for (let i = 0; i <= currentIdx; i++) {
    const res = await fetch(`/api/marker/${i}/data`);
    const data = await res.json();
    if (data.status === 'pending') { loadMarker(i); return; }
  }
  document.getElementById('save-msg').textContent = 'All markers reviewed!';
}

async function saveAll() {
  const res = await fetch('/api/save', {method: 'POST'});
  const data = await res.json();
  document.getElementById('save-msg').textContent = `Saved ${data.saved} polygons.`;
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  switch(e.key) {
    case 'a': case 'A': decide('accepted'); break;
    case 'k': case 'K': decide('skipped'); break;
    case 'd': case 'D': decide('deleted'); break;
    case 's': case 'S': resegment(); break;
    case 'r': case 'R': resetPoints(); break;
    case 'n': case 'N': navNextPending(); break;
    case 'ArrowLeft': nav(-1); break;
    case 'ArrowRight': nav(1); break;
  }
});

init();
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--output-dir", default="results/analysis/sam_fn_review")
    parser.add_argument("--no-presegment", action="store_true",
                        help="Skip initial segmentation, start with empty results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FN markers...")
    markers = load_fn_markers()
    print(f"Loaded {len(markers)} markers")

    sam_model.load()

    # Pre-segment all markers with initial single/multi-point
    if not args.no_presegment:
        print(f"\nPre-segmenting {len(markers)} markers...")
        for i, m in enumerate(markers):
            tile_path = m["tile_path"]
            if not Path(tile_path).exists():
                continue
            try:
                with rasterio.open(tile_path) as src:
                    img_array = np.moveaxis(src.read()[:3], 0, -1)

                pos = m["positive_points"]
                neg = m["negative_points"]
                mask, score = sam_model.segment(img_array, pos, neg)
                wkt_str, area_m2 = mask_to_polygon_wkt(mask, tile_path)
                m["sam_wkt"] = wkt_str
                m["sam_area_m2"] = area_m2

                status = f"[{i+1}/{len(markers)}] {m['grid_id']}/{m['tile_key']}: {area_m2:.0f}m² score={score:.3f}"
                print(f"  {status}")
            except Exception as e:
                print(f"  [{i+1}/{len(markers)}] {m['grid_id']}/{m['tile_key']}: ERROR {e}")

    ReviewHandler.markers = markers
    ReviewHandler.output_path = output_dir

    server = HTTPServer(("0.0.0.0", args.port), ReviewHandler)
    print(f"\n{'='*60}")
    print(f"SAM FN Review server: http://localhost:{args.port}")
    print(f"  {len(markers)} markers loaded")
    print(f"  Shortcuts: A=accept K=skip D=delete S=resegment R=reset N=next")
    print(f"{'='*60}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
