"""
Detection review GUI: review model predictions on tiles, classify as correct/delete/edit.

Workflow:
  1. Run detect_and_evaluate.py on a grid (detection-only, no GT needed)
  2. Run this GUI to review predictions per tile
  3. Export reviewed predictions as GPKG + QGIS QML style for SAM2 correction

Usage:
  python scripts/annotations/review_detections.py --grid-id G1686
  python scripts/annotations/review_detections.py --grid-id G1686 --predictions results/G1686/predictions_metric.gpkg

Keyboard shortcuts:
  A = mark ALL predictions on current tile as correct
  D = mark ALL predictions on current tile as delete
  1-9 = toggle individual prediction status (correct → delete → edit → correct)
  Arrow keys = prev/next tile
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.grid_utils import (
    TILES_ROOT,
    get_grid_spec,
    get_tile_bounds,
    normalize_grid_id,
)

STATUS_VALUES = {"correct", "delete", "edit", ""}
STATUS_COLORS = {
    "correct": (34, 197, 94, 160),    # green
    "delete": (239, 68, 68, 160),     # red
    "edit": (234, 179, 8, 160),       # yellow
    "": (156, 163, 175, 120),         # gray (unreviewed)
}
STATUS_OUTLINE = {
    "correct": (22, 163, 74),
    "delete": (220, 38, 38),
    "edit": (202, 138, 4),
    "": (107, 114, 128),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class DetectionReviewStore:
    """Manages prediction review state."""

    def __init__(self, grid_id: str, predictions_path: Path, tiles_dir: Path):
        self.grid_id = grid_id
        self.tiles_dir = tiles_dir
        self.review_dir = predictions_path.parent / "review"
        self.review_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_path = self.review_dir / "detection_review_decisions.csv"

        # Load predictions
        self.pred_gdf = gpd.read_file(str(predictions_path))
        if self.pred_gdf.crs and self.pred_gdf.crs.to_epsg() != 4326:
            self.pred_gdf = self.pred_gdf.to_crs(epsg=4326)
        print(f"  Loaded {len(self.pred_gdf)} predictions from {predictions_path.name}")

        # Build tile → predictions index
        self.spec = get_grid_spec(grid_id)
        self.tile_preds = self._index_by_tile()
        self.decisions = self._load_decisions()
        # If GPKG has review_status column, use it to seed decisions
        if "review_status" in self.pred_gdf.columns:
            for idx, row in self.pred_gdf.iterrows():
                st = str(row["review_status"]).strip()
                if st and st != "unreviewed" and str(idx) not in self.decisions:
                    self.decisions[str(idx)] = st
            print(f"  Loaded {len(self.decisions)} decisions from review_status column")
        self.fn_markers = self._load_fn_markers()

        # Pre-render tile images with predictions
        self.tile_images: dict[str, bytes] = {}

    def _index_by_tile(self) -> dict[str, list[int]]:
        """Map each tile to prediction indices."""
        tile_preds: dict[str, list[int]] = {}
        for idx, row in self.pred_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            cx, cy = geom.centroid.x, geom.centroid.y
            # Find which tile this prediction falls in
            for col in range(self.spec.n_cols):
                for r in range(self.spec.n_rows):
                    txmin, tymin, txmax, tymax = get_tile_bounds(self.spec, col, r)
                    if txmin <= cx <= txmax and tymin <= cy <= tymax:
                        tile_key = f"{self.grid_id}_{col}_{r}"
                        tile_preds.setdefault(tile_key, []).append(idx)
                        break
                else:
                    continue
                break
        return tile_preds

    def _load_decisions(self) -> dict[str, str]:
        """Load per-prediction decisions: pred_index -> status."""
        if not self.decisions_path.exists():
            return {}
        decisions = {}
        with self.decisions_path.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                decisions[row["pred_id"]] = row.get("status", "")
        return decisions

    def _write_decisions(self) -> None:
        with self.decisions_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["pred_id", "status", "updated_at"])
            writer.writeheader()
            for pred_id in sorted(self.decisions, key=lambda x: int(x)):
                writer.writerow({
                    "pred_id": pred_id,
                    "status": self.decisions[pred_id],
                    "updated_at": utc_now_iso(),
                })

    def get_tile_list(self, include_empty: bool = False) -> list[dict]:
        """Return tiles sorted by tile key. If include_empty, also list tiles with no predictions."""
        tiles = []
        seen = set()
        for tile_key in sorted(self.tile_preds.keys()):
            pred_indices = self.tile_preds[tile_key]
            n_total = len(pred_indices)
            n_reviewed = sum(1 for i in pred_indices if str(i) in self.decisions and self.decisions[str(i)])
            tiles.append({
                "tile_key": tile_key,
                "n_predictions": n_total,
                "n_reviewed": n_reviewed,
            })
            seen.add(tile_key)

        if include_empty:
            # Add all downloaded tiles that have no predictions
            for f in sorted(self.tiles_dir.glob(f"{self.grid_id}_*_geo.tif")):
                tile_key = f.stem.replace("_geo", "")
                if tile_key not in seen:
                    tiles.append({
                        "tile_key": tile_key,
                        "n_predictions": 0,
                        "n_reviewed": 0,
                    })
            tiles.sort(key=lambda t: t["tile_key"])
        return tiles

    def get_tile_predictions(self, tile_key: str) -> list[dict]:
        """Return predictions for a tile with their review status."""
        indices = self.tile_preds.get(tile_key, [])
        preds = []
        for idx in indices:
            row = self.pred_gdf.iloc[idx]
            status = self.decisions.get(str(idx), "")
            pred = {
                "pred_id": int(idx),
                "status": status,
                "confidence": float(row.get("confidence", row.get("score", 0))),
                "area_m2": float(row.get("area_m2", 0)),
            }
            preds.append(pred)
        return preds

    def save_decision(self, pred_id: str, status: str) -> None:
        if status not in STATUS_VALUES:
            raise ValueError(f"invalid status: {status}")
        if status:
            self.decisions[pred_id] = status
        else:
            self.decisions.pop(pred_id, None)
        self._write_decisions()

    def save_tile_decisions(self, tile_key: str, status: str) -> None:
        """Mark all predictions on a tile with the same status."""
        for idx in self.tile_preds.get(tile_key, []):
            if status:
                self.decisions[str(idx)] = status
            else:
                self.decisions.pop(str(idx), None)
        self._write_decisions()

    # ── FN markers ──
    def _fn_markers_path(self) -> Path:
        return self.review_dir / "fn_markers.csv"

    def _load_fn_markers(self) -> list[dict]:
        p = self._fn_markers_path()
        if not p.exists():
            return []
        markers = []
        with p.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                markers.append({
                    "tile_key": row["tile_key"],
                    "px": float(row["px"]),
                    "py": float(row["py"]),
                })
        return markers

    def _save_fn_markers(self) -> None:
        p = self._fn_markers_path()
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["tile_key", "px", "py"])
            writer.writeheader()
            for m in self.fn_markers:
                writer.writerow(m)

    def add_fn_marker(self, tile_key: str, px: float, py: float) -> None:
        self.fn_markers.append({"tile_key": tile_key, "px": px, "py": py})
        self._save_fn_markers()

    def remove_fn_marker(self, tile_key: str, px: float, py: float, radius: float = 30) -> bool:
        """Remove the nearest marker within radius (in pixel coords)."""
        best_i, best_d = -1, radius
        for i, m in enumerate(self.fn_markers):
            if m["tile_key"] != tile_key:
                continue
            d = ((m["px"] - px)**2 + (m["py"] - py)**2) ** 0.5
            if d < best_d:
                best_i, best_d = i, d
        if best_i >= 0:
            self.fn_markers.pop(best_i)
            self._save_fn_markers()
            return True
        return False

    def get_fn_markers(self, tile_key: str) -> list[dict]:
        return [m for m in self.fn_markers if m["tile_key"] == tile_key]

    def _tile_pixel_to_geo(self, tile_key: str, px: float, py: float) -> tuple[float, float] | None:
        """Convert pixel coords to geographic coords for a tile."""
        geo_path = self.tiles_dir / f"{tile_key}_geo.tif"
        if not geo_path.exists():
            alt_path = self.tiles_dir / f"{tile_key}.tif"
            if alt_path.exists():
                geo_path = alt_path
            else:
                return None
        with rasterio.open(geo_path) as ds:
            bounds = ds.bounds
            w, h = ds.width, ds.height
        lon = bounds.left + (px / w) * (bounds.right - bounds.left)
        lat = bounds.top - (py / h) * (bounds.top - bounds.bottom)
        return (lon, lat)

    def export_fn_markers_gpkg(self) -> Path | None:
        """Export FN markers as point GPKG for QGIS."""
        if not self.fn_markers:
            return None
        from shapely.geometry import Point
        points, tile_keys = [], []
        for m in self.fn_markers:
            geo = self._tile_pixel_to_geo(m["tile_key"], m["px"], m["py"])
            if geo:
                points.append(Point(geo[0], geo[1]))
                tile_keys.append(m["tile_key"])
        if not points:
            return None
        gdf = gpd.GeoDataFrame({"tile_key": tile_keys, "type": "fn_marker"},
                               geometry=points, crs="EPSG:4326")
        out = self.review_dir / f"{self.grid_id}_fn_markers.gpkg"
        gdf.to_file(str(out), driver="GPKG")
        print(f"  Exported {len(gdf)} FN markers to {out}")
        return out

    def render_base_tile(self, tile_key: str) -> bytes:
        """Return base tile image as JPEG (no overlays — overlays drawn client-side)."""
        if tile_key in self.tile_images:
            return self.tile_images[tile_key]

        geo_path = self.tiles_dir / f"{tile_key}_geo.tif"
        if not geo_path.exists():
            alt_path = self.tiles_dir / f"{tile_key}.tif"
            if alt_path.exists():
                geo_path = alt_path
            else:
                img = Image.new("RGB", (512, 512), (40, 40, 40))
                draw = ImageDraw.Draw(img)
                draw.text((200, 250), "No tile", fill=(200, 200, 200))
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=85)
                return buf.getvalue()

        with rasterio.open(geo_path) as ds:
            data = ds.read()
            bounds = ds.bounds

        if data.shape[0] >= 3:
            rgb = np.transpose(data[:3], (1, 2, 0))
        else:
            rgb = np.transpose(np.stack([data[0]] * 3), (1, 2, 0))
        img = Image.fromarray(rgb.astype(np.uint8))

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        result = buf.getvalue()
        self.tile_images[tile_key] = result
        return result

    def get_tile_polygons(self, tile_key: str) -> list[dict]:
        """Return prediction polygons as pixel-coordinate arrays for client-side rendering."""
        geo_path = self.tiles_dir / f"{tile_key}_geo.tif"
        if not geo_path.exists():
            alt_path = self.tiles_dir / f"{tile_key}.tif"
            if alt_path.exists():
                geo_path = alt_path
            else:
                return []

        with rasterio.open(geo_path) as ds:
            bounds = ds.bounds
            w, h = ds.width, ds.height

        indices = self.tile_preds.get(tile_key, [])
        polys = []
        for num, idx in enumerate(indices, start=1):
            pred_row = self.pred_gdf.iloc[idx]
            geom = pred_row.geometry
            if geom is None:
                continue

            if geom.geom_type == "Polygon":
                ring = list(geom.exterior.coords)
            elif geom.geom_type == "MultiPolygon":
                ring = list(geom.geoms[0].exterior.coords)
            else:
                continue

            coords = []
            for x, y in ring:
                px = (x - bounds.left) / (bounds.right - bounds.left) * w
                py = (bounds.top - y) / (bounds.top - bounds.bottom) * h
                coords.append([round(px, 1), round(py, 1)])

            status = self.decisions.get(str(idx), "")
            confidence = float(pred_row.get("confidence", pred_row.get("score", 0)))
            area = float(pred_row.get("area_m2", 0))
            polys.append({
                "pred_id": int(idx),
                "num": num,
                "coords": coords,
                "status": status,
                "confidence": confidence,
                "area_m2": area,
            })
        return polys

    def export_gpkg(self, output_path: Path | None = None) -> Path:
        """Export predictions with review_status to GPKG."""
        if output_path is None:
            output_path = self.review_dir / f"{self.grid_id}_reviewed.gpkg"

        gdf = self.pred_gdf.copy()
        gdf["review_status"] = gdf.index.map(
            lambda idx: self.decisions.get(str(idx), "unreviewed")
        )
        gdf.to_file(str(output_path), driver="GPKG")
        print(f"  Exported {len(gdf)} predictions to {output_path}")

        # Also write QML style
        qml_path = output_path.with_suffix(".qml")
        write_qml_style(qml_path)
        print(f"  QML style: {qml_path}")

        # Export FN markers
        fn_path = self.export_fn_markers_gpkg()
        return output_path


def write_qml_style(qml_path: Path) -> None:
    """Write a QGIS QML style file for color-coded review_status."""
    qml = """<?xml version="1.0" encoding="UTF-8"?>
<qgis version="3.34" styleCategories="Symbology">
  <renderer-v2 type="categorizedSymbol" attr="review_status" enableorderby="0">
    <categories>
      <category symbol="0" value="correct" label="Correct" render="true"/>
      <category symbol="1" value="delete" label="Delete (FP)" render="true"/>
      <category symbol="2" value="edit" label="Needs Edit" render="true"/>
      <category symbol="3" value="unreviewed" label="Unreviewed" render="true"/>
    </categories>
    <symbols>
      <symbol type="fill" name="0" alpha="0.6">
        <layer class="SimpleFill">
          <Option type="Map">
            <Option type="QString" value="22,163,74,153" name="color"/>
            <Option type="QString" value="solid" name="style"/>
            <Option type="QString" value="15,118,50,255" name="outline_color"/>
            <Option type="QString" value="0.4" name="outline_width"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="1" alpha="0.6">
        <layer class="SimpleFill">
          <Option type="Map">
            <Option type="QString" value="220,38,38,153" name="color"/>
            <Option type="QString" value="solid" name="style"/>
            <Option type="QString" value="180,20,20,255" name="outline_color"/>
            <Option type="QString" value="0.4" name="outline_width"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="2" alpha="0.6">
        <layer class="SimpleFill">
          <Option type="Map">
            <Option type="QString" value="234,179,8,153" name="color"/>
            <Option type="QString" value="solid" name="style"/>
            <Option type="QString" value="180,130,0,255" name="outline_color"/>
            <Option type="QString" value="0.4" name="outline_width"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="3" alpha="0.4">
        <layer class="SimpleFill">
          <Option type="Map">
            <Option type="QString" value="156,163,175,100" name="color"/>
            <Option type="QString" value="solid" name="style"/>
            <Option type="QString" value="107,114,128,255" name="outline_color"/>
            <Option type="QString" value="0.3" name="outline_width"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>"""
    qml_path.write_text(qml, encoding="utf-8")


def build_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Detection Review</title>
  <style>
    :root {
      --bg: #f4f0e8; --panel: #fffaf2; --ink: #1e2328; --muted: #6a6f75;
      --line: #d8cdbd; --correct: #16a34a; --delete: #dc2626; --edit: #ca8a04;
    }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Georgia, serif; color: var(--ink);
      background: radial-gradient(circle at top left, #fff8ea 0, #f4f0e8 42%, #e8e3d7 100%);
    }
    .wrap { max-width:1600px; margin:0 auto; padding:8px; }
    .bar, .panel { background:rgba(255,250,242,0.92); border:1px solid var(--line);
      border-radius:14px; box-shadow:0 10px 30px rgba(60,52,41,0.08); }
    .bar { display:flex; gap:8px; align-items:center; padding:8px 12px; margin-bottom:8px; flex-wrap:wrap; }
    .title { font-size:22px; font-weight:700; }
    .meta { color:var(--muted); font-size:13px; }
    .controls { display:flex; gap:6px; flex-wrap:wrap; }
    button { border:1px solid var(--line); border-radius:8px; background:#fff;
      color:var(--ink); padding:8px 12px; cursor:pointer; font:inherit; font-size:14px; }
    button:hover { border-color:#bfa78c; }
    .btn-correct { border-color: var(--correct); color: var(--correct); }
    .btn-delete { border-color: var(--delete); color: var(--delete); }
    .btn-edit { border-color: var(--edit); color: var(--edit); }
    .grid { display:grid; grid-template-columns:1fr 280px; gap:8px; height:calc(100vh - 60px); }
    .panel { padding:10px; overflow-y:auto; }
    .canvas-wrap { position:relative; height:100%; display:flex; align-items:center;
      justify-content:center; background:#1a1a1a; border-radius:10px; overflow:hidden; cursor:crosshair; }
    canvas { width:100%; height:100%; display:block; }
    .pred-item { display:flex; align-items:center; gap:8px; padding:7px 10px;
      border-bottom:1px solid #efe5d6; cursor:pointer; border-radius:6px; font-size:14px; }
    .pred-item:hover { background:#f5f0e6; }
    .pred-item.selected { background:#e8f0fe; border:1px solid #93b4e8; }
    .pred-num { font-weight:700; min-width:24px; text-align:center;
      border-radius:50%; width:26px; height:26px; line-height:26px; font-size:12px; color:#fff; }
    .pred-conf { color:var(--muted); font-size:13px; }
    .pred-status { font-weight:600; font-size:12px; padding:2px 8px; border-radius:4px; }
    .pred-status.correct { background:#dcfce7; color:var(--correct); }
    .pred-status.delete { background:#fee2e2; color:var(--delete); }
    .pred-status.edit { background:#fef9c3; color:var(--edit); }
    .pred-status.unreviewed { background:#f3f4f6; color:var(--muted); }
    .progress-bar { height:4px; background:#e5e7eb; border-radius:2px; margin:8px 0; }
    .progress-fill { height:100%; background:var(--correct); border-radius:2px; transition:width 0.3s; }
    .stats { display:grid; grid-template-columns:1fr 1fr; gap:4px; font-size:13px; margin:8px 0; }
    .stats .label { color:var(--muted); }
    .selected-info { background:#e8f0fe; border:1px solid #93b4e8; border-radius:8px;
      padding:10px; margin:10px 0; display:none; }
    .selected-info.visible { display:block; }
    .selected-actions { display:flex; gap:6px; margin-top:8px; }
    .kbd { display:inline-block; min-width:1.5em; padding:1px 5px; border:1px solid var(--line);
      border-radius:4px; background:#fff; text-align:center; font-size:12px; }
    .pred-list-wrap { max-height:50vh; overflow-y:auto; }
    @media (max-width:980px) { .grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<div class="wrap">
  <div class="bar">
    <div class="title" id="gridTitle">Detection Review</div>
    <div class="meta" id="summary">Loading...</div>
    <div style="flex:1"></div>
    <div class="controls">
      <button id="prevBtn">&#9664; Prev</button>
      <button id="nextBtn">Next &#9654;</button>
    </div>
    <div class="controls">
      <button class="btn-correct" id="allCorrectBtn">All Correct <span class="kbd">Q</span></button>
      <button id="markerBtn" style="border-color:#f59e0b;color:#b45309;">Mark FN <span class="kbd">M</span></button>
      <button id="exportBtn">Export GPKG</button>
    </div>
    <div class="controls">
      <label class="meta" for="filterSelect">Filter</label>
      <select id="filterSelect">
        <option value="all">All tiles</option>
        <option value="all_incl_empty">All + empty tiles</option>
        <option value="unreviewed">Unreviewed</option>
        <option value="reviewed">Reviewed</option>
        <option value="empty">Empty only</option>
      </select>
    </div>
  </div>

  <div class="grid">
    <div class="panel">
      <div class="canvas-wrap" id="canvasWrap">
        <canvas id="tileCanvas"></canvas>
      </div>
    </div>
    <div class="panel">
      <div class="meta">Tile</div>
      <div class="title" id="tileKey" style="font-size:18px;">-</div>
      <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
      <div class="stats">
        <div class="label">Predictions</div><div id="nPreds">-</div>
        <div class="label">Reviewed</div><div id="nReviewed">-</div>
      </div>

      <div class="selected-info" id="selectedInfo">
        <div><strong>Selected: #<span id="selNum">-</span></strong>
          <span class="pred-conf" id="selDetails"></span></div>
        <div class="selected-actions">
          <button class="btn-correct" id="selCorrectBtn">Correct <span class="kbd">A</span></button>
          <button class="btn-delete" id="selDeleteBtn">Delete <span class="kbd">D</span></button>
          <button class="btn-edit" id="selEditBtn">Edit <span class="kbd">E</span></button>
          <button id="selClearBtn">Clear <span class="kbd">C</span></button>
        </div>
      </div>

      <hr style="border:0;border-top:1px solid #efe5d6;margin:10px 0;">
      <div class="meta" style="margin-bottom:6px;">Predictions (click on image or list):</div>
      <div class="pred-list-wrap" id="predList"></div>
      <hr style="border:0;border-top:1px solid #efe5d6;margin:12px 0;">
      <div class="meta">
        Click polygon to select, then:<br>
        <span class="kbd">A</span> correct,
        <span class="kbd">D</span> delete,
        <span class="kbd">E</span> edit,
        <span class="kbd">C</span> clear<br>
        <span class="kbd">Q</span> all correct (quick pass)<br>
        <span class="kbd">M</span> toggle FN marker mode (click to place, right-click to remove)<br>
        <span class="kbd">&larr;</span>/<span class="kbd">&rarr;</span> prev/next tile
      </div>
    </div>
  </div>
</div>

<script>
const COLORS = {
  correct:    {fill:"rgba(34,197,94,0.35)", stroke:"#16a34a", num:"#15803d"},
  delete:     {fill:"rgba(239,68,68,0.35)", stroke:"#dc2626", num:"#b91c1c"},
  edit:       {fill:"rgba(234,179,8,0.35)", stroke:"#ca8a04", num:"#a16207"},
  unreviewed: {fill:"rgba(156,163,175,0.25)", stroke:"#6b7280", num:"#6b7280"},
  selected:   {fill:"rgba(59,130,246,0.4)", stroke:"#2563eb", num:"#1d4ed8"},
};

let tiles=[], visible=[], currentIdx=0;
let polys=[], selectedIdx=-1;
let baseImage=null, markers=[];
let markerMode=false;
const canvas = document.getElementById("tileCanvas");
const ctx = canvas.getContext("2d");

// ── Zoom & Pan state ──
let scale=1, panX=0, panY=0;
let isDragging=false, dragStartX=0, dragStartY=0, panStartX=0, panStartY=0;
const MIN_SCALE=0.5, MAX_SCALE=12;

function updateMarkerBtn() {
  const btn=document.getElementById("markerBtn");
  if(markerMode) {
    btn.style.background="#fef3c7"; btn.style.fontWeight="700";
    canvas.style.cursor="cell";
  } else {
    btn.style.background="#fff"; btn.style.fontWeight="normal";
    canvas.style.cursor="crosshair";
  }
}

function resetView() { scale=1; panX=0; panY=0; resizeCanvas(); }

function resizeCanvas() {
  if(!baseImage) return;
  const wrap = document.getElementById("canvasWrap");
  const w = wrap.clientWidth, h = wrap.clientHeight;
  canvas.width = w; canvas.height = h;
  drawCanvas();
}

// Convert screen coords to image coords
function screenToImage(sx, sy) {
  const rect = canvas.getBoundingClientRect();
  const cx = (sx - rect.left);
  const cy = (sy - rect.top);
  return [(cx - panX) / scale, (cy - panY) / scale];
}

// Zoom to fit a bounding box (with padding)
function zoomToBBox(xmin, ymin, xmax, ymax, padding=80) {
  if(!baseImage) return;
  const bw = xmax - xmin, bh = ymax - ymin;
  if(bw < 1 || bh < 1) return;
  const cw = canvas.width, ch = canvas.height;
  scale = Math.min((cw - padding*2) / bw, (ch - padding*2) / bh, MAX_SCALE);
  const bcx = (xmin + xmax) / 2, bcy = (ymin + ymax) / 2;
  panX = cw/2 - bcx * scale;
  panY = ch/2 - bcy * scale;
  drawCanvas();
}

// Zoom to selected polygon
function zoomToSelected() {
  if(selectedIdx < 0 || selectedIdx >= polys.length) return;
  const coords = polys[selectedIdx].coords;
  let xmin=Infinity, ymin=Infinity, xmax=-Infinity, ymax=-Infinity;
  coords.forEach(([x,y]) => { xmin=Math.min(xmin,x); ymin=Math.min(ymin,y); xmax=Math.max(xmax,x); ymax=Math.max(ymax,y); });
  // Add margin around polygon
  const margin = Math.max(xmax-xmin, ymax-ymin) * 1.5 + 60;
  const cx=(xmin+xmax)/2, cy=(ymin+ymax)/2;
  zoomToBBox(cx-margin, cy-margin, cx+margin, cy+margin, 20);
}

async function loadTiles(includeEmpty) {
  const url = includeEmpty ? "/api/tiles?include_empty=1" : "/api/tiles";
  tiles = await (await fetch(url)).json();
  applyFilter();
}

function applyFilter() {
  const mode = document.getElementById("filterSelect").value;
  // Reload with empty tiles if needed
  const needEmpty = (mode === "all_incl_empty" || mode === "empty");
  const hasEmpty = tiles.some(t => t.n_predictions === 0);
  if (needEmpty && !hasEmpty) { loadTiles(true); return; }
  if (!needEmpty && hasEmpty) { loadTiles(false); return; }

  if (mode==="all") visible=tiles.filter(t=>t.n_predictions>0);
  else if (mode==="all_incl_empty") visible=[...tiles];
  else if (mode==="unreviewed") visible=tiles.filter(t=>t.n_reviewed<t.n_predictions && t.n_predictions>0);
  else if (mode==="reviewed") visible=tiles.filter(t=>t.n_reviewed===t.n_predictions && t.n_predictions>0);
  else if (mode==="empty") visible=tiles.filter(t=>t.n_predictions===0);
  else visible=[...tiles];
  currentIdx = Math.min(currentIdx, Math.max(visible.length-1,0));
  loadCurrentTile();
}

function updateSummary() {
  const total=tiles.length, done=tiles.filter(t=>t.n_reviewed===t.n_predictions).length;
  const tp=tiles.reduce((s,t)=>s+t.n_predictions,0);
  const tr=tiles.reduce((s,t)=>s+t.n_reviewed,0);
  const pos=visible.length?currentIdx+1:0;
  document.getElementById("summary").textContent=
    `Tile ${pos}/${visible.length} | ${done}/${total} tiles done | ${tr}/${tp} preds reviewed`;
}

async function loadCurrentTile() {
  selectedIdx=-1;
  updateSummary();
  const tile=visible[currentIdx];
  if(!tile){document.getElementById("tileKey").textContent="No tiles";return;}
  document.getElementById("tileKey").textContent=tile.tile_key;
  updateTileStats(tile);

  const [imgBlob, polyData, markerData] = await Promise.all([
    fetch(`/api/tile_image/${tile.tile_key}`).then(r=>r.blob()),
    fetch(`/api/tile_polys/${tile.tile_key}`).then(r=>r.json()),
    fetch(`/api/tile_markers/${tile.tile_key}`).then(r=>r.json()),
  ]);
  polys = polyData;
  markers = markerData;

  if(baseImage && baseImage.src) URL.revokeObjectURL(baseImage.src);
  baseImage = new Image();
  baseImage.onload = () => {
    resetView();
    // Auto-select first unreviewed prediction (or first prediction if all reviewed)
    if(polys.length > 0) {
      const firstUnrev = polys.findIndex(p => !p.status);
      selectedIdx = firstUnrev >= 0 ? firstUnrev : 0;
      zoomToSelected();
    }
    // Pre-fetch next tile image
    const nextTile = visible[currentIdx+1];
    if(nextTile) { const p = new Image(); p.src = `/api/tile_image/${nextTile.tile_key}`; }
  };
  baseImage.onerror = () => { console.error("Failed to load tile image"); };
  baseImage.src = URL.createObjectURL(imgBlob);

  renderPredList();
}

function updateTileStats(tile) {
  document.getElementById("nPreds").textContent=tile.n_predictions;
  document.getElementById("nReviewed").textContent=tile.n_reviewed;
  document.getElementById("progressFill").style.width=
    tile.n_predictions?`${(tile.n_reviewed/tile.n_predictions*100).toFixed(0)}%`:"0%";
}

function drawCanvas() {
  if(!baseImage) return;
  const cw=canvas.width, ch=canvas.height;
  ctx.clearRect(0,0,cw,ch);
  ctx.save();
  ctx.translate(panX, panY);
  ctx.scale(scale, scale);

  ctx.drawImage(baseImage, 0, 0);

  // Adjust line widths for zoom level
  const lw = Math.max(1, 2/scale);
  const selLw = Math.max(2, 3/scale);
  const fontSize = Math.max(8, Math.min(14, 14/scale));

  polys.forEach((p,i) => {
    const isSelected = (i===selectedIdx);
    const st = p.status||"unreviewed";
    const c = COLORS[st]||COLORS.unreviewed;

    ctx.beginPath();
    p.coords.forEach(([x,y],j) => { j===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
    ctx.closePath();

    if(isSelected) {
      // Selected: no fill, thick bright outline only
      ctx.strokeStyle="#2563eb";
      ctx.lineWidth=Math.max(3, 5/scale);
      ctx.stroke();
      // Outer glow
      ctx.strokeStyle="rgba(37,99,235,0.3)";
      ctx.lineWidth=Math.max(6, 10/scale);
      ctx.stroke();
    } else {
      // Normal: light fill + outline
      ctx.fillStyle=c.fill;
      ctx.fill();
      ctx.strokeStyle=c.stroke;
      ctx.lineWidth=lw;
      ctx.stroke();
    }

    // Number label: top-left corner of bbox, not center
    let xmin=Infinity, ymin=Infinity;
    p.coords.forEach(([x,y]) => { xmin=Math.min(xmin,x); ymin=Math.min(ymin,y); });
    const label=String(p.num);
    ctx.font=`bold ${fontSize}px Georgia`;
    const m=ctx.measureText(label);
    const tw=m.width, th=fontSize;
    const pad=3/scale;
    const lx=xmin-pad, ly=ymin-th-pad*3;
    const bgColor = isSelected ? "rgba(37,99,235,0.85)" : "rgba(0,0,0,0.7)";
    ctx.fillStyle=bgColor;
    ctx.fillRect(lx-pad, ly-pad, tw+pad*3, th+pad*2);
    ctx.fillStyle="#fff";
    ctx.textAlign="left"; ctx.textBaseline="top";
    ctx.fillText(label, lx, ly);
  });

  // Draw FN markers
  markers.forEach(m => {
    const mx=m.px, my=m.py, r=Math.max(6, 10/scale);
    // Orange diamond marker
    ctx.beginPath();
    ctx.moveTo(mx, my-r); ctx.lineTo(mx+r, my); ctx.lineTo(mx, my+r); ctx.lineTo(mx-r, my);
    ctx.closePath();
    ctx.fillStyle="rgba(249,115,22,0.8)";
    ctx.fill();
    ctx.strokeStyle="#c2410c";
    ctx.lineWidth=Math.max(1, 2/scale);
    ctx.stroke();
    // "FN" label
    const fs=Math.max(7, 10/scale);
    ctx.font=`bold ${fs}px Georgia`;
    ctx.fillStyle="#fff";
    ctx.textAlign="center"; ctx.textBaseline="middle";
    ctx.fillText("FN",mx,my);
  });

  ctx.restore();

  // Marker mode indicator
  if(markerMode) {
    ctx.fillStyle="rgba(249,115,22,0.85)";
    ctx.fillRect(cw-130,8,122,26);
    ctx.fillStyle="#fff";
    ctx.font="bold 13px Georgia";
    ctx.textAlign="center"; ctx.textBaseline="middle";
    ctx.fillText("FN MARKER MODE",cw-69,21);
  }

  // Zoom indicator
  if(scale !== 1) {
    ctx.fillStyle="rgba(0,0,0,0.6)";
    ctx.fillRect(8,ch-28,70,22);
    ctx.fillStyle="#fff";
    ctx.font="13px Georgia";
    ctx.textAlign="left"; ctx.textBaseline="middle";
    ctx.fillText(`${Math.round(scale*100)}%`, 14, ch-17);
  }

  updateSelectedInfo();
}

function updateSelectedInfo() {
  const el=document.getElementById("selectedInfo");
  if(selectedIdx<0||selectedIdx>=polys.length) { el.classList.remove("visible"); return; }
  const p=polys[selectedIdx];
  el.classList.add("visible");
  document.getElementById("selNum").textContent=p.num;
  document.getElementById("selDetails").textContent=
    `${(p.confidence*100).toFixed(0)}% conf | ${p.area_m2.toFixed(1)}m² | ${p.status||"unreviewed"}`;
  document.querySelectorAll(".pred-item").forEach((el,i)=>{
    el.classList.toggle("selected",i===selectedIdx);
  });
  // Scroll selected into view
  const selEl = document.querySelector(".pred-item.selected");
  if(selEl) selEl.scrollIntoView({block:"nearest"});
}

function renderPredList() {
  const el=document.getElementById("predList");
  el.innerHTML="";
  polys.forEach((p,i) => {
    const sc=p.status||"unreviewed";
    const numColor = COLORS[sc]?COLORS[sc].stroke:"#6b7280";
    const div=document.createElement("div");
    div.className="pred-item"+(i===selectedIdx?" selected":"");
    div.innerHTML=`
      <div class="pred-num" style="background:${numColor}">${p.num}</div>
      <div class="pred-conf">${(p.confidence*100).toFixed(0)}% | ${p.area_m2.toFixed(1)}m²</div>
      <div style="flex:1"></div>
      <div class="pred-status ${sc}">${sc}</div>`;
    div.addEventListener("click",()=>{
      selectedIdx=i;
      zoomToSelected();
      renderPredList();
    });
    el.appendChild(div);
  });
}

// Hit test: point in polygon (ray casting)
function pointInPoly(px,py,coords) {
  let inside=false;
  for(let i=0,j=coords.length-1;i<coords.length;j=i++) {
    const [xi,yi]=coords[i],[xj,yj]=coords[j];
    if(((yi>py)!==(yj>py))&&(px<(xj-xi)*(py-yi)/(yj-yi)+xi)) inside=!inside;
  }
  return inside;
}

// ── Mouse interactions ──

// Scroll wheel zoom
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
  const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale * factor));
  // Zoom centered on cursor
  panX = mx - (mx - panX) * (newScale / scale);
  panY = my - (my - panY) * (newScale / scale);
  scale = newScale;
  drawCanvas();
}, {passive: false});

// Click to select, drag to pan
canvas.addEventListener("mousedown", e => {
  isDragging = true;
  dragStartX = e.clientX; dragStartY = e.clientY;
  panStartX = panX; panStartY = panY;
  canvas.style.cursor = "grabbing";
});

canvas.addEventListener("mousemove", e => {
  if(!isDragging) return;
  panX = panStartX + (e.clientX - dragStartX);
  panY = panStartY + (e.clientY - dragStartY);
  drawCanvas();
});

canvas.addEventListener("mouseup", e => {
  const dx = Math.abs(e.clientX - dragStartX), dy = Math.abs(e.clientY - dragStartY);
  isDragging = false;
  canvas.style.cursor = "crosshair";

  // If barely moved, treat as click
  if(dx < 4 && dy < 4) {
    const [ix, iy] = screenToImage(e.clientX, e.clientY);

    if(markerMode) {
      // Place FN marker
      const tile=visible[currentIdx];
      if(tile) {
        fetch("/api/add_marker", {
          method:"POST", headers:{"Content-Type":"application/json"},
          body:JSON.stringify({tile_key:tile.tile_key, px:Math.round(ix), py:Math.round(iy)})
        }).then(()=>fetch(`/api/tile_markers/${tile.tile_key}`))
          .then(r=>r.json()).then(d=>{markers=d; drawCanvas();});
      }
    } else {
      // Select polygon
      let found = -1;
      for(let i=polys.length-1; i>=0; i--) {
        if(pointInPoly(ix, iy, polys[i].coords)) { found=i; break; }
      }
      if(found >= 0) {
        selectedIdx = found;
        zoomToSelected();
      } else {
        selectedIdx = -1;
        drawCanvas();
      }
      renderPredList();
    }
  }
});

canvas.addEventListener("mouseleave", () => { isDragging=false; canvas.style.cursor=markerMode?"cell":"crosshair"; });

// Right-click to remove nearest FN marker
canvas.addEventListener("contextmenu", e => {
  e.preventDefault();
  if(!markerMode) return;
  const [ix, iy] = screenToImage(e.clientX, e.clientY);
  const tile=visible[currentIdx];
  if(!tile) return;
  fetch("/api/remove_marker", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({tile_key:tile.tile_key, px:ix, py:iy})
  }).then(()=>fetch(`/api/tile_markers/${tile.tile_key}`))
    .then(r=>r.json()).then(d=>{markers=d; drawCanvas();});
});

// Double-click to reset view
canvas.addEventListener("dblclick", e => { e.preventDefault(); resetView(); });

async function setSelectedStatus(status) {
  if(selectedIdx<0||selectedIdx>=polys.length) return;
  const p=polys[selectedIdx];
  await fetch("/api/decision", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({pred_id:String(p.pred_id), status})
  });
  p.status=status||"";
  const tile=visible[currentIdx];
  if(tile) { tile.n_reviewed=polys.filter(x=>x.status).length; updateTileStats(tile); }
  updateSummary();
  // Auto-advance to next unreviewed prediction
  if(status) {
    const nextUnrev = polys.findIndex((pp,ii) => ii > selectedIdx && !pp.status);
    if(nextUnrev >= 0) {
      selectedIdx = nextUnrev;
      zoomToSelected();
    } else {
      drawCanvas();
    }
  } else {
    drawCanvas();
  }
  renderPredList();
}

async function markAllTile(status) {
  const tile=visible[currentIdx];
  if(!tile) return;
  await fetch("/api/tile_decision", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({tile_key:tile.tile_key, status})
  });
  polys.forEach(p=>{p.status=status;});
  tile.n_reviewed=polys.length;
  updateTileStats(tile);
  updateSummary();
  drawCanvas();
  renderPredList();
}

// Button handlers
document.getElementById("prevBtn").addEventListener("click",()=>{if(currentIdx>0){currentIdx--;loadCurrentTile();}});
document.getElementById("nextBtn").addEventListener("click",()=>{if(currentIdx<visible.length-1){currentIdx++;loadCurrentTile();}});
document.getElementById("allCorrectBtn").addEventListener("click",()=>markAllTile("correct"));
document.getElementById("filterSelect").addEventListener("change",applyFilter);
document.getElementById("exportBtn").addEventListener("click",async()=>{
  const r=await fetch("/api/export",{method:"POST"});
  const d=await r.json(); alert("Exported:\\n"+d.path);
});
document.getElementById("selCorrectBtn").addEventListener("click",()=>setSelectedStatus("correct"));
document.getElementById("selDeleteBtn").addEventListener("click",()=>setSelectedStatus("delete"));
document.getElementById("selEditBtn").addEventListener("click",()=>setSelectedStatus("edit"));
document.getElementById("selClearBtn").addEventListener("click",()=>setSelectedStatus(""));
document.getElementById("markerBtn").addEventListener("click",()=>{markerMode=!markerMode; updateMarkerBtn(); drawCanvas();});

// Keyboard
document.addEventListener("keydown", e => {
  if(e.target.tagName==="TEXTAREA"||e.target.tagName==="INPUT"||e.target.tagName==="SELECT") return;
  switch(e.key) {
    case "ArrowLeft":  if(currentIdx>0){currentIdx--;loadCurrentTile();} e.preventDefault(); break;
    case "ArrowRight": if(currentIdx<visible.length-1){currentIdx++;loadCurrentTile();} e.preventDefault(); break;
    case "a": case "A": setSelectedStatus("correct"); e.preventDefault(); break;
    case "d":           setSelectedStatus("delete"); e.preventDefault(); break;
    case "e": case "E": setSelectedStatus("edit"); e.preventDefault(); break;
    case "c":           setSelectedStatus(""); e.preventDefault(); break;
    case "q": case "Q": markAllTile("correct"); e.preventDefault(); break;
    case "m": case "M": markerMode=!markerMode; updateMarkerBtn(); drawCanvas(); e.preventDefault(); break;
    case "Escape":      selectedIdx=-1; markerMode=false; updateMarkerBtn(); resetView(); renderPredList(); e.preventDefault(); break;
    case "f": case "F": if(selectedIdx>=0) zoomToSelected(); else resetView(); e.preventDefault(); break;
  }
});

// Handle window resize
window.addEventListener("resize", () => { resizeCanvas(); });

loadTiles(false);
</script>
</body>
</html>"""


class ReviewHandler(BaseHTTPRequestHandler):
    store: DetectionReviewStore

    def log_message(self, format, *args):
        pass  # suppress request logs

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "":
            html = build_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        elif path == "/api/tiles":
            qs = parse_qs(parsed.query)
            include_empty = qs.get("include_empty", ["0"])[0] == "1"
            self._json_response(self.store.get_tile_list(include_empty=include_empty))

        elif path.startswith("/api/tile_preds/"):
            tile_key = path.split("/")[-1]
            self._json_response(self.store.get_tile_predictions(tile_key))

        elif path.startswith("/api/tile_image/"):
            tile_key = path.split("/")[-1].split("?")[0]
            img_bytes = self.store.render_base_tile(tile_key)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(img_bytes)))
            self.end_headers()
            self.wfile.write(img_bytes)

        elif path.startswith("/api/tile_polys/"):
            tile_key = path.split("/")[-1].split("?")[0]
            self._json_response(self.store.get_tile_polygons(tile_key))

        elif path.startswith("/api/tile_markers/"):
            tile_key = path.split("/")[-1].split("?")[0]
            self._json_response(self.store.get_fn_markers(tile_key))

        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/decision":
            data = json.loads(self._read_body())
            self.store.save_decision(data["pred_id"], data["status"])
            self._json_response({"ok": True})

        elif path == "/api/tile_decision":
            data = json.loads(self._read_body())
            self.store.save_tile_decisions(data["tile_key"], data["status"])
            self._json_response({"ok": True})

        elif path == "/api/add_marker":
            data = json.loads(self._read_body())
            self.store.add_fn_marker(data["tile_key"], data["px"], data["py"])
            self._json_response({"ok": True})

        elif path == "/api/remove_marker":
            data = json.loads(self._read_body())
            removed = self.store.remove_fn_marker(data["tile_key"], data["px"], data["py"])
            self._json_response({"ok": True, "removed": removed})

        elif path == "/api/export":
            path = self.store.export_gpkg()
            self._json_response({"ok": True, "path": str(path)})

        else:
            self.send_error(HTTPStatus.NOT_FOUND)


def main():
    parser = argparse.ArgumentParser(description="Review model detections per tile")
    parser.add_argument("--grid-id", required=True, help="Grid ID, e.g. G1686")
    parser.add_argument("--predictions", type=Path, default=None,
                       help="Path to predictions GPKG (default: results/<grid>/predictions_metric.gpkg)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    grid_id = normalize_grid_id(args.grid_id)
    base_dir = Path(__file__).resolve().parent.parent.parent

    if args.predictions:
        pred_path = args.predictions
    else:
        pred_path = base_dir / "results" / grid_id / "predictions_metric.gpkg"

    if not pred_path.exists():
        print(f"[ERROR] Predictions not found: {pred_path}")
        print(f"  Run: python detect_and_evaluate.py --grid-id {grid_id} --model-path checkpoints_cleaned/best_model.pth")
        sys.exit(1)

    tiles_dir = TILES_ROOT / grid_id
    if not tiles_dir.exists():
        print(f"[ERROR] Tiles not found: {tiles_dir}")
        sys.exit(1)

    print(f"[INIT] Grid: {grid_id}")
    print(f"[INIT] Predictions: {pred_path}")
    print(f"[INIT] Tiles: {tiles_dir}")

    store = DetectionReviewStore(grid_id, pred_path, tiles_dir)
    tile_list = store.get_tile_list()
    print(f"[INIT] {len(tile_list)} tiles with predictions")

    ReviewHandler.store = store

    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Open in browser: {url}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("\n[DONE] Server stopped")


if __name__ == "__main__":
    main()
