"""
Build a thumbnail HTML gallery of TP-lost predictions per classifier backbone.

A TP-lost prediction is one where:
  - the raw detector found it AND it overlaps a GT installation (max IoU >= 0.3)
  - the classifier's cls_score < 0.5 (so it gets removed by the cascade filter)
  - cls_applied=True (i.e. small enough to be classified, not a large bypass)

This is the population worth eyeballing to understand WHY recall drops on
filtered: are the lost TPs visually entangled with thermal heaters / dark
roofs (entanglement loss) or are they unambiguous PVs the classifier got wrong?

Usage:
    python scripts/analysis/build_tp_lost_gallery.py \
        --backbone convnext \
        --backbone dinov2 \
        --out-root results/analysis/cls_cascade_holdout/tp_lost_gallery
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.grid_utils import resolve_tiles_dir, _resolve_gt_gpkg  # noqa: E402
from scripts.classifier.classify_predictions import (  # noqa: E402
    extract_detection_chips,
    classify_chips,
    load_classifier,
    CHIP_SIZE,
    IMG_SIZE,
)


HOLDOUT_GRIDS: dict[str, list[tuple[str, str]]] = {
    # region -> list of (grid_id, model_run, imagery_layer)
    "ct": [
        ("G1971", "v3c_targeted_hn_aerial_2025", "aerial_2025"),
        ("G1973", "v3c_targeted_hn_aerial_2025", "aerial_2025"),
        ("G1981", "v3c_targeted_hn_aerial_2025", "aerial_2025"),
        ("G2027", "v3c_targeted_hn_aerial_2025", "aerial_2025"),
        ("G2029", "v3c_targeted_hn_aerial_2025", "aerial_2025"),
        ("G2032", "v3c_targeted_hn_aerial_2025", "aerial_2025"),
    ],
    "jhb": [
        ("G0856", "v4_aerial_2023", "aerial_2023"),
        ("G0890", "v4_aerial_2023", "aerial_2023"),
        ("G0892", "v4_aerial_2023", "aerial_2023"),
        ("G1110", "v4_aerial_2023", "aerial_2023"),
        ("G1111", "v4_aerial_2023", "aerial_2023"),
        ("G1144", "v4_aerial_2023", "aerial_2023"),
        ("G1146", "v4_aerial_2023", "aerial_2023"),
        ("G1183", "v4_aerial_2023", "aerial_2023"),
        ("G1250", "v4_aerial_2023", "aerial_2023"),
        ("G1253", "v4_aerial_2023", "aerial_2023"),
    ],
}

REGION_DIR = {"ct": "cape_town", "jhb": "johannesburg"}

BACKBONE_CONFIGS_V1: dict[str, dict] = {
    "effb0": {
        "model_path": "checkpoints/cls_pv_thermal_v1_effb0/best_cls.pth",
        "arch": "efficientnet_b0",
        "label": "EfficientNet-B0 (v1)",
    },
    "convnext": {
        "model_path": "checkpoints/cls_pv_thermal_v1_convnext_tiny/best_cls.pth",
        "arch": "convnext_tiny",
        "label": "ConvNeXt-Tiny (v1)",
    },
    "dinov2": {
        "model_path": "checkpoints/cls_pv_thermal_v1_dinov2_vits14/best_cls.pth",
        "arch": "dinov2_vits14",
        "label": "DINOv2-ViT-S/14 (v1)",
    },
}

BACKBONE_CONFIGS_V2: dict[str, dict] = {
    "effb0": {
        "model_path": "checkpoints/cls_pv_thermal_v2_efficientnet_b0/best_cls.pth",
        "arch": "efficientnet_b0",
        "label": "EfficientNet-B0 (v2 per-imagery)",
    },
    "convnext": {
        "model_path": "checkpoints/cls_pv_thermal_v2_convnext_tiny/best_cls.pth",
        "arch": "convnext_tiny",
        "label": "ConvNeXt-Tiny (v2 per-imagery)",
    },
    "dinov2": {
        "model_path": "checkpoints/cls_pv_thermal_v2_dinov2_vits14/best_cls.pth",
        "arch": "dinov2_vits14",
        "label": "DINOv2-ViT-S/14 (v2 per-imagery)",
    },
}

# Default v1 for backward compat; switched to v2 via --version v2.
BACKBONE_CONFIGS: dict[str, dict] = BACKBONE_CONFIGS_V1

PV_THRESHOLD = 0.5  # v1 single-threshold default; v2 overrides per imagery layer.
AREA_CUTOFF = 30.0
MATCH_IOU = 0.3  # any pred with IoU >= this against any GT counts as TP-ish
THUMB_SIZE = 256  # px per chip in the gallery


def match_preds_to_gt(
    pred_gdf: gpd.GeoDataFrame, gt_gdf: gpd.GeoDataFrame
) -> dict[int, dict]:
    """Return {pred_idx: {max_iou, gt_idx_of_max}}.

    Uses metric CRS already on pred_gdf. Reprojects GT to match.
    Polygon-level IoU per pred against every GT, keep max.
    """
    if gt_gdf.crs != pred_gdf.crs:
        gt_gdf = gt_gdf.to_crs(pred_gdf.crs)

    gt_geoms = list(gt_gdf.geometry)
    gt_areas = [g.area for g in gt_geoms]

    sindex = gt_gdf.sindex

    matches = {}
    for idx in pred_gdf.index:
        pred_geom = pred_gdf.loc[idx].geometry
        if pred_geom is None or pred_geom.is_empty:
            matches[idx] = {"max_iou": 0.0, "gt_idx": -1}
            continue

        candidates = list(sindex.intersection(pred_geom.bounds))
        if not candidates:
            matches[idx] = {"max_iou": 0.0, "gt_idx": -1}
            continue

        pred_area = pred_geom.area
        best_iou = 0.0
        best_gt = -1
        for ci in candidates:
            gt_geom = gt_geoms[ci]
            inter = pred_geom.intersection(gt_geom).area
            if inter <= 0:
                continue
            union = pred_area + gt_areas[ci] - inter
            if union <= 0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_gt = ci
        matches[idx] = {"max_iou": float(best_iou), "gt_idx": int(best_gt)}

    return matches


def render_chip_with_gt(
    chip_rgb: np.ndarray,
    pred_geom_metric,
    gt_geom_metric,
    chip_center_lonlat: tuple[float, float],
    src_transform,
    src_crs_to_metric_transformer,
    chip_x0: int,
    chip_y0: int,
    chip_w: int,
    chip_h: int,
    inv_transform,
    cls_score: float,
    pred_area_m2: float,
    max_iou: float,
) -> np.ndarray:
    """Draw GT polygon (green) + pred polygon (yellow) on a chip.

    chip_rgb is HWC RGB uint8 (CHIP_SIZE x CHIP_SIZE).

    For polygon overlay we reproject metric -> tile pixel using inv_transform.
    """
    img = chip_rgb.copy()

    def metric_to_chip_px(geom):
        # geom is in metric CRS; transform to tile CRS (EPSG:4326), then to tile pixel
        from shapely.ops import transform as shp_transform
        geom_4326 = shp_transform(src_crs_to_metric_transformer, geom)
        # geom_4326 is in tile CRS; project to pixel coords via tile transform
        coords = []
        if geom_4326.geom_type == "Polygon":
            polys = [geom_4326]
        elif geom_4326.geom_type == "MultiPolygon":
            polys = list(geom_4326.geoms)
        else:
            return []

        all_rings = []
        for poly in polys:
            ring = []
            for x, y in poly.exterior.coords:
                col, row = inv_transform * (x, y)
                # subtract chip origin to get chip-local pixel coords
                cx = col - chip_x0
                cy = row - chip_y0
                ring.append((cx, cy))
            all_rings.append(ring)
        return all_rings

    # GT polygon — semi-transparent green fill + thin outline (drawn first, BELOW pred)
    if gt_geom_metric is not None:
        try:
            gt_rings = metric_to_chip_px(gt_geom_metric)
            if gt_rings:
                overlay = img.copy()
                gt_pts_list = [np.array(r, dtype=np.int32).reshape(-1, 1, 2) for r in gt_rings]
                cv2.fillPoly(overlay, gt_pts_list, (0, 220, 0))
                img = cv2.addWeighted(overlay, 0.32, img, 0.68, 0)
                for pts in gt_pts_list:
                    cv2.polylines(img, [pts], isClosed=True,
                                  color=(0, 200, 0), thickness=1, lineType=cv2.LINE_AA)
        except Exception:
            pass

    # Pred polygon — yellow outline on top
    if pred_geom_metric is not None:
        try:
            for ring in metric_to_chip_px(pred_geom_metric):
                pts = np.array(ring, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], isClosed=True,
                              color=(255, 220, 0), thickness=2, lineType=cv2.LINE_AA)
        except Exception:
            pass

    # Caption strip
    text1 = f"cls={cls_score:.2f}  area={pred_area_m2:.0f}m^2  iou={max_iou:.2f}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text1, (4, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Resize for thumbnail
    img = cv2.resize(img, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)
    return img


def process_grid(
    grid_id: str,
    region: str,
    model_run: str,
    imagery_layer: str,
    backbones: dict[str, dict],
    out_dirs: dict[str, Path],
    threshold_lookup=None,
) -> dict[str, list[dict]]:
    """Run all backbones on one grid; return tp_lost records per backbone."""
    import rasterio
    from rasterio.windows import Window
    from pyproj import Transformer

    region_dir = REGION_DIR[region]
    grid_results = PROJECT_ROOT / "results" / region_dir / model_run / grid_id
    raw_path = grid_results / "predictions_metric.gpkg"
    if not raw_path.exists():
        print(f"  [{grid_id}] missing {raw_path}, skip")
        return {bk: [] for bk in backbones}

    pred_gdf = gpd.read_file(raw_path)
    if len(pred_gdf) == 0:
        return {bk: [] for bk in backbones}

    # Resolve GT
    try:
        gt_path = _resolve_gt_gpkg(grid_id, region=region)
    except Exception as e:
        print(f"  [{grid_id}] GT resolve failed: {e}, skip")
        return {bk: [] for bk in backbones}
    if not gt_path.exists():
        print(f"  [{grid_id}] GT missing, skip")
        return {bk: [] for bk in backbones}

    gt_gdf = gpd.read_file(gt_path)
    if pred_gdf.crs is None:
        pred_gdf.set_crs(epsg=32734 if region == "ct" else 32735, inplace=True)
    if gt_gdf.crs is None:
        gt_gdf.set_crs(epsg=4326, inplace=True)

    # Match raw preds to GT in metric CRS
    matches = match_preds_to_gt(pred_gdf, gt_gdf)
    n_tp_ish = sum(1 for m in matches.values() if m["max_iou"] >= MATCH_IOU)

    # Extract chips once (shared across backbones)
    tiles_root = PROJECT_ROOT / "tiles"  # symlink/legacy fallback
    # but classify_predictions._find_tile uses tiles_root/grid_id, so resolve via region-aware
    grid_tiles_dir = resolve_tiles_dir(grid_id, region=region, imagery_layer=imagery_layer)
    if not grid_tiles_dir.exists():
        print(f"  [{grid_id}] tiles dir missing: {grid_tiles_dir}, skip")
        return {bk: [] for bk in backbones}

    # We need tiles_root such that tiles_root/<grid_id> = grid_tiles_dir
    tiles_root = grid_tiles_dir.parent

    print(f"  [{grid_id}] {len(pred_gdf)} preds, {n_tp_ish} match GT (IoU>={MATCH_IOU}), "
          f"tiles={tiles_root.name}/{grid_id}")

    chips, classified_idx, _skipped = extract_detection_chips(
        pred_gdf, grid_id, tiles_root, AREA_CUTOFF,
    )
    classified_set = set(classified_idx)

    # For overlay rendering we need tile transform per chip
    # Re-walk preds and remember (tile_path, chip_x0, chip_y0, chip_w, chip_h) per classified idx
    chip_meta: dict[int, dict] = {}
    pred_4326 = pred_gdf.to_crs(epsg=4326) if pred_gdf.crs.to_epsg() != 4326 else pred_gdf
    tile_handles: dict[str, rasterio.DatasetReader] = {}
    try:
        for idx in classified_idx:
            row = pred_gdf.loc[idx]
            row_4326 = pred_4326.loc[idx]
            lon, lat = row_4326.geometry.centroid.x, row_4326.geometry.centroid.y
            # Find tile (mirror _find_tile logic)
            from scripts.classifier.classify_predictions import _find_tile
            tile_path = _find_tile(lon, lat, grid_id, tiles_root)
            if tile_path is None:
                continue
            key = str(tile_path)
            if key not in tile_handles:
                tile_handles[key] = rasterio.open(tile_path)
            src = tile_handles[key]
            py, px = src.index(lon, lat)
            x0 = max(0, int(px - CHIP_SIZE // 2))
            y0 = max(0, int(py - CHIP_SIZE // 2))
            x0 = min(x0, max(0, src.width - CHIP_SIZE))
            y0 = min(y0, max(0, src.height - CHIP_SIZE))
            w = min(CHIP_SIZE, src.width - x0)
            h = min(CHIP_SIZE, src.height - y0)
            chip_meta[idx] = {
                "tile_path": str(tile_path),
                "x0": x0, "y0": y0, "w": w, "h": h,
                "src_transform": src.transform,
                "src_crs": src.crs,
            }
    finally:
        for h in tile_handles.values():
            h.close()

    # Pre-build metric->tile_crs transformer (per region)
    # We'll lazily build per-pred since CRS may vary, but typically ct=32734, jhb=32735, tile=4326
    metric_to_tile_transformers: dict[tuple[int, int], object] = {}

    def get_transformer(metric_epsg: int, tile_epsg: int):
        key = (metric_epsg, tile_epsg)
        if key not in metric_to_tile_transformers:
            t = Transformer.from_crs(
                f"EPSG:{metric_epsg}", f"EPSG:{tile_epsg}", always_xy=True
            ).transform
            metric_to_tile_transformers[key] = t
        return metric_to_tile_transformers[key]

    # Run each backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_records: dict[str, list[dict]] = {bk: [] for bk in backbones}

    for backbone_key, cfg in backbones.items():
        model_path = PROJECT_ROOT / cfg["model_path"]
        if not model_path.exists():
            print(f"    [{backbone_key}] model missing: {model_path}, skip")
            continue
        model, mcfg = load_classifier(model_path, device)
        img_size = mcfg.get("img_size", IMG_SIZE)
        preprocessing = mcfg.get("preprocessing")

        scores = classify_chips(
            model, chips, classified_idx, device, img_size, 64,
            preprocessing=preprocessing,
        )
        # Free model GPU mem before next backbone
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Identify tp_lost: matched_TP AND classified AND cls_score < threshold.
        # Threshold is per-imagery in v2; falls back to PV_THRESHOLD for v1.
        if threshold_lookup is not None:
            cur_thr = threshold_lookup(cfg.get("arch"), imagery_layer)
        else:
            cur_thr = PV_THRESHOLD
        n_lost = 0
        for idx, score in scores.items():
            if score >= cur_thr:
                continue
            m = matches.get(idx)
            if not m or m["max_iou"] < MATCH_IOU:
                continue
            n_lost += 1

            # Render thumbnail
            cm = chip_meta.get(idx)
            if cm is None:
                continue

            # We have the chip array already; find which one corresponds to idx
            try:
                chip_arr_pos = classified_idx.index(idx)
            except ValueError:
                continue
            chip_rgb = chips[chip_arr_pos]

            # Build inverse tile transform for metric->pixel overlay
            tile_transform = cm["src_transform"]
            tile_crs = cm["src_crs"]
            inv_transform = ~tile_transform

            metric_epsg = pred_gdf.crs.to_epsg()
            tile_epsg = tile_crs.to_epsg() if tile_crs else 4326
            transformer = get_transformer(metric_epsg, tile_epsg)

            pred_geom_metric = pred_gdf.loc[idx].geometry
            gt_idx = m["gt_idx"]
            gt_geom_metric = None
            if gt_idx >= 0:
                gt_metric = gt_gdf.to_crs(pred_gdf.crs).iloc[gt_idx].geometry
                gt_geom_metric = gt_metric

            try:
                rendered = render_chip_with_gt(
                    chip_rgb=chip_rgb,
                    pred_geom_metric=pred_geom_metric,
                    gt_geom_metric=gt_geom_metric,
                    chip_center_lonlat=(0, 0),
                    src_transform=tile_transform,
                    src_crs_to_metric_transformer=transformer,
                    chip_x0=cm["x0"],
                    chip_y0=cm["y0"],
                    chip_w=cm["w"],
                    chip_h=cm["h"],
                    inv_transform=inv_transform,
                    cls_score=score,
                    pred_area_m2=float(pred_gdf.loc[idx].get("area_m2", 0.0)),
                    max_iou=m["max_iou"],
                )
            except Exception as e:
                print(f"    [{backbone_key}] render failed for {grid_id} pred {idx}: {e}")
                continue

            chip_dir = out_dirs[backbone_key] / "chips" / region / grid_id
            chip_dir.mkdir(parents=True, exist_ok=True)
            chip_filename = f"{grid_id}_p{int(idx):04d}.jpg"
            chip_path = chip_dir / chip_filename
            cv2.imwrite(str(chip_path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])

            out_records[backbone_key].append({
                "grid_id": grid_id,
                "region": region,
                "pred_idx": int(idx),
                "cls_score": float(score),
                "max_iou": float(m["max_iou"]),
                "area_m2": float(pred_gdf.loc[idx].get("area_m2", 0.0)),
                "chip_rel": f"chips/{region}/{grid_id}/{chip_filename}",
            })

        print(f"    [{backbone_key}] tp_lost = {n_lost}")

    return out_records


def write_html(
    backbone_key: str,
    backbone_label: str,
    records: list[dict],
    out_dir: Path,
):
    """Write a single-page gallery with region tabs."""
    by_region = {"ct": [], "jhb": []}
    for r in records:
        by_region[r["region"]].append(r)
    for k in by_region:
        by_region[k].sort(key=lambda r: r["cls_score"])

    n_ct = len(by_region["ct"])
    n_jhb = len(by_region["jhb"])
    n_total = n_ct + n_jhb

    def render_chip_card(r: dict) -> str:
        cap = (
            f"{r['grid_id']}  p{r['pred_idx']}<br>"
            f"cls={r['cls_score']:.2f}&nbsp;&nbsp;iou={r['max_iou']:.2f}<br>"
            f"area={r['area_m2']:.0f} m²"
        )
        return (
            f'<div class="card">'
            f'<img src="{escape(r["chip_rel"])}" loading="lazy"/>'
            f'<div class="cap">{cap}</div>'
            f'</div>'
        )

    ct_cards = "\n".join(render_chip_card(r) for r in by_region["ct"])
    jhb_cards = "\n".join(render_chip_card(r) for r in by_region["jhb"])

    def render_chip_card_with_buttons(r: dict) -> str:
        chip_id = f"{r['region']}_{r['grid_id']}_p{r['pred_idx']:04d}"
        cap = (
            f"{r['grid_id']}  p{r['pred_idx']}<br>"
            f"cls={r['cls_score']:.2f}&nbsp;&nbsp;iou={r['max_iou']:.2f}&nbsp;&nbsp;"
            f"area={r['area_m2']:.0f} m²"
        )
        return (
            f'<div class="card" data-id="{chip_id}">'
            f'<img src="{escape(r["chip_rel"])}" loading="lazy"/>'
            f'<div class="cap">{cap}</div>'
            f'<div class="btns">'
            f'<button class="btn btn-A" data-label="A" title="Ambiguous / heater-like (1)">A 灰区</button>'
            f'<button class="btn btn-B" data-label="B" title="Clearly PV, classifier wrong (2)">B 明显PV</button>'
            f'<button class="btn btn-C" data-label="C" title="GT polygon itself is wrong (3)">C GT错</button>'
            f'<button class="btn btn-D" data-label="D" title="Detector mis-located on non-PV object e.g. car (4)">D 检测错</button>'
            f'<button class="btn btn-X" data-label="" title="Clear (0)">×</button>'
            f'</div>'
            f'</div>'
        )

    ct_cards = "\n".join(render_chip_card_with_buttons(r) for r in by_region["ct"])
    jhb_cards = "\n".join(render_chip_card_with_buttons(r) for r in by_region["jhb"])

    storage_key = f"tp_lost_labels_{backbone_key}"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>TP-lost gallery — {escape(backbone_label)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f7f7f7; color: #222; }}
  h1 {{ margin: 0 0 6px 0; font-size: 22px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 18px; }}
  .legend {{ font-size: 12px; color: #555; background: #fff; padding: 8px 12px; border-radius: 6px; margin-bottom: 12px; display: inline-block; }}
  .legend span.gt {{ color: #00b300; font-weight: 600; }}
  .legend span.pr {{ color: #e0a000; font-weight: 600; }}
  .toolbar {{ background: #fff; padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
  .toolbar .stats {{ font-size: 13px; color: #444; }}
  .toolbar .stats b {{ color: #2266cc; }}
  .toolbar button {{ padding: 6px 14px; border: 1px solid #bbb; background: #fafafa; border-radius: 4px; cursor: pointer; font-size: 13px; }}
  .toolbar button:hover {{ background: #eee; }}
  .toolbar button.danger {{ border-color: #c44; color: #c44; }}
  .tabs {{ display: flex; gap: 8px; margin-bottom: 16px; }}
  .tab {{ padding: 8px 18px; background: #ddd; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }}
  .tab.active {{ background: #2266cc; color: #fff; }}
  .pane {{ display: none; }}
  .pane.active {{ display: block; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }}
  .card {{ background: #fff; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); position: relative; }}
  .card img {{ width: 100%; display: block; }}
  .card .cap {{ padding: 6px 10px; font-size: 12px; line-height: 1.4; color: #333; border-top: 1px solid #eee; }}
  .card .btns {{ display: flex; gap: 4px; padding: 6px 8px; background: #fafafa; border-top: 1px solid #eee; }}
  .card .btn {{ flex: 1; padding: 5px 4px; border: 1px solid #ccc; background: #fff; border-radius: 3px; cursor: pointer; font-size: 11px; font-weight: 600; }}
  .card .btn:hover {{ background: #f0f0f0; }}
  .card .btn-X {{ flex: 0 0 28px; color: #888; }}
  .card[data-label="A"] {{ outline: 3px solid #f5a623; }}
  .card[data-label="A"] .btn-A {{ background: #f5a623; color: #fff; border-color: #f5a623; }}
  .card[data-label="B"] {{ outline: 3px solid #d0021b; }}
  .card[data-label="B"] .btn-B {{ background: #d0021b; color: #fff; border-color: #d0021b; }}
  .card[data-label="C"] {{ outline: 3px solid #888; }}
  .card[data-label="C"] .btn-C {{ background: #888; color: #fff; border-color: #888; }}
  .card[data-label="D"] {{ outline: 3px solid #4a90e2; }}
  .card[data-label="D"] .btn-D {{ background: #4a90e2; color: #fff; border-color: #4a90e2; }}
  .empty {{ color: #888; padding: 40px; text-align: center; font-style: italic; }}
  .kbd {{ font-family: monospace; background: #eee; padding: 1px 6px; border-radius: 3px; font-size: 11px; }}
</style>
</head>
<body>
<h1>TP-lost gallery — {escape(backbone_label)}</h1>
<div class="meta">
  Total: <b>{n_total}</b> &middot; CT: <b>{n_ct}</b> &middot; JHB: <b>{n_jhb}</b>
  &middot; threshold = {PV_THRESHOLD} &middot; match IoU ≥ {MATCH_IOU}
</div>
<div class="legend">
  <span class="gt">▢ green</span> = GT polygon &nbsp;|&nbsp;
  <span class="pr">▢ yellow</span> = predicted polygon &nbsp;|&nbsp;
  sorted by cls_score ascending
  &nbsp;|&nbsp; keys: <span class="kbd">1</span>=A灰区 <span class="kbd">2</span>=B明显PV <span class="kbd">3</span>=C GT错 <span class="kbd">4</span>=D 检测错 <span class="kbd">0</span>=clear
</div>
<div class="toolbar">
  <div class="stats" id="stats">Loading…</div>
  <button onclick="exportCSV()">Export CSV</button>
  <button onclick="copyCSV()">Copy CSV to clipboard</button>
  <button class="danger" onclick="clearAll()">Clear all labels</button>
</div>
<div class="tabs">
  <button class="tab active" data-pane="ct" onclick="showPane('ct', this)">CT ({n_ct})</button>
  <button class="tab" data-pane="jhb" onclick="showPane('jhb', this)">JHB ({n_jhb})</button>
</div>
<div id="pane-ct" class="pane active">
  {'<div class="grid">' + ct_cards + '</div>' if ct_cards else '<div class="empty">No CT tp_lost.</div>'}
</div>
<div id="pane-jhb" class="pane">
  {'<div class="grid">' + jhb_cards + '</div>' if jhb_cards else '<div class="empty">No JHB tp_lost.</div>'}
</div>
<script>
const STORAGE_KEY = {json.dumps(storage_key)};
const BACKBONE = {json.dumps(backbone_key)};

function loadLabels() {{
  try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }}
  catch (e) {{ return {{}}; }}
}}
function saveLabels(labels) {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
}}
function applyLabels() {{
  const labels = loadLabels();
  document.querySelectorAll('.card').forEach(c => {{
    const id = c.dataset.id;
    if (labels[id]) c.dataset.label = labels[id];
    else delete c.dataset.label;
  }});
  updateStats();
}}
function setLabel(id, label) {{
  const labels = loadLabels();
  if (label) labels[id] = label;
  else delete labels[id];
  saveLabels(labels);
  applyLabels();
}}
function updateStats() {{
  const labels = loadLabels();
  const total = document.querySelectorAll('.card').length;
  const counts = {{A: 0, B: 0, C: 0, D: 0}};
  for (const k in labels) if (counts[labels[k]] !== undefined) counts[labels[k]]++;
  const labeled = counts.A + counts.B + counts.C + counts.D;
  document.getElementById('stats').innerHTML =
    `Labeled: <b>${{labeled}}/${{total}}</b> &middot; ` +
    `A 灰区: <b>${{counts.A}}</b> &middot; ` +
    `B 明显PV: <b>${{counts.B}}</b> &middot; ` +
    `C GT错: <b>${{counts.C}}</b> &middot; ` +
    `D 检测错: <b>${{counts.D}}</b>`;
}}
document.addEventListener('click', (e) => {{
  const btn = e.target.closest('.btn');
  if (!btn) return;
  const card = btn.closest('.card');
  if (!card) return;
  setLabel(card.dataset.id, btn.dataset.label);
}});

let focusedCard = null;
document.addEventListener('mouseover', (e) => {{
  const card = e.target.closest('.card');
  if (card) focusedCard = card;
}});
document.addEventListener('keydown', (e) => {{
  if (!focusedCard) return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const map = {{'1': 'A', '2': 'B', '3': 'C', '4': 'D', '0': ''}};
  if (map[e.key] !== undefined) {{
    setLabel(focusedCard.dataset.id, map[e.key]);
    e.preventDefault();
  }}
}});

function showPane(name, btn) {{
  document.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('pane-' + name).classList.add('active');
  if (btn) btn.classList.add('active');
}}

function buildCSV() {{
  const labels = loadLabels();
  const rows = [["backbone","region","grid_id","pred_idx","cls_score","max_iou","area_m2","label"]];
  document.querySelectorAll('.card').forEach(c => {{
    const id = c.dataset.id;
    const [region, grid, predTok] = id.split('_');
    const cap = c.querySelector('.cap').innerText;
    const m = cap.match(/cls=([\\d.]+)\\s+iou=([\\d.]+)\\s+area=(\\d+)/);
    const cls = m ? m[1] : '';
    const iou = m ? m[2] : '';
    const area = m ? m[3] : '';
    rows.push([BACKBONE, region, grid, predTok.replace('p',''), cls, iou, area, labels[id] || '']);
  }});
  return rows.map(r => r.join(',')).join('\\n');
}}
function exportCSV() {{
  const csv = buildCSV();
  const blob = new Blob([csv], {{type: 'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `tp_lost_labels_${{BACKBONE}}.csv`;
  a.click();
}}
async function copyCSV() {{
  await navigator.clipboard.writeText(buildCSV());
  alert('CSV copied to clipboard');
}}
function clearAll() {{
  if (!confirm('Clear ALL labels for this backbone?')) return;
  localStorage.removeItem(STORAGE_KEY);
  applyLabels();
}}

applyLabels();
</script>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    # Also write manifest.json
    (out_dir / "manifest.json").write_text(
        json.dumps({"backbone": backbone_key, "records": records}, indent=2),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", action="append", required=True,
                    choices=list(BACKBONE_CONFIGS_V1.keys()),
                    help="Backbone(s) to process; pass --backbone repeatedly")
    ap.add_argument("--version", choices=["v1", "v2"], default="v1",
                    help="Classifier dataset version (v1=single thr 0.5, "
                         "v2=per-imagery thr from thresholds_v2.json)")
    ap.add_argument("--thresholds-json", type=Path,
                    default=PROJECT_ROOT / "configs" / "classifier"
                    / "thresholds_v2.json",
                    help="Per-imagery threshold JSON (used when --version v2)")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--grids", nargs="*", default=None,
                    help="Optional subset of grid IDs to process")
    args = ap.parse_args()

    if args.version == "v2":
        configs = BACKBONE_CONFIGS_V2
        thr_data = json.loads(args.thresholds_json.read_text())

        def threshold_lookup(arch: str, imagery_layer: str) -> float:
            return float(thr_data["by_backbone"][arch]["thresholds"]
                         [imagery_layer]["threshold"])
        default_out = (PROJECT_ROOT / "results" / "analysis"
                       / "cls_cascade_holdout_v2" / "tp_lost_gallery")
    else:
        configs = BACKBONE_CONFIGS_V1
        threshold_lookup = None
        default_out = (PROJECT_ROOT / "results" / "analysis"
                       / "cls_cascade_holdout" / "tp_lost_gallery")

    out_root = args.out_root or default_out
    backbones = {k: configs[k] for k in args.backbone}
    out_root.mkdir(parents=True, exist_ok=True)
    out_dirs = {bk: out_root / bk for bk in backbones}
    for d in out_dirs.values():
        (d / "chips").mkdir(parents=True, exist_ok=True)

    all_records: dict[str, list[dict]] = {bk: [] for bk in backbones}

    for region, grids in HOLDOUT_GRIDS.items():
        for grid_id, model_run, layer in grids:
            if args.grids and grid_id not in args.grids:
                continue
            print(f"[{region}] {grid_id} ({model_run})")
            recs = process_grid(grid_id, region, model_run, layer,
                                backbones, out_dirs, threshold_lookup)
            for bk, rs in recs.items():
                all_records[bk].extend(rs)

    for bk, recs in all_records.items():
        out_dir = out_dirs[bk]
        write_html(bk, configs[bk]["label"], recs, out_dir)
        print(f"[{bk}] gallery -> {out_dir / 'index.html'} ({len(recs)} chips)")


if __name__ == "__main__":
    main()
