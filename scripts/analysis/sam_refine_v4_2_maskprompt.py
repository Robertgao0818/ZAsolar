#!/usr/bin/env python3
"""SAM2 refinement using V4.2 polygon as low-res MASK prompt (option 4).

Difference from sam_refine_v4_2.py:
  - Rasterize V4.2 mask polygon to binary in window pixel coords
  - Resize to 256x256 (SAM2 mask input size for image_size=1024)
  - Convert to logits and pass as `input_masks` to model.forward
  - Also keep input_boxes as anchor (helps SAM2 stability)

Output gpkg layout matches sam_refine_v4_2.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.features import rasterize, shapes as rio_shapes
from rasterio.windows import Window
from shapely.geometry import shape

CBD_GRIDS = [
    "G0772","G0773","G0774","G0775","G0776","G0814","G0815","G0816","G0817","G0818",
    "G0853","G0854","G0855","G0856","G0857","G0888","G0889","G0890","G0891","G0892",
    "G0922","G0923","G0924","G0925","G0926",
]
METRIC_CRS = "EPSG:32735"
CROP_MARGIN_PX = 64
MIN_MASK_AREA_PX = 4
MASK_LOGIT_POS = 10.0
MASK_LOGIT_NEG = -10.0
SAM_MASK_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-results-root", type=Path, required=True)
    p.add_argument("--tiles-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--grids", nargs="*", default=CBD_GRIDS)
    p.add_argument("--sam-model-id", default="facebook/sam2.1-hiera-large")
    p.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16","float16","float32"])
    p.add_argument("--prompt-mode", default="mask_box", choices=["mask_only","mask_box","box_only"])
    return p.parse_args()


def find_chunk_for_geom(grid_id, tiles_root, cent_metric):
    """Locate the chunk containing `cent_metric` (Point in METRIC_CRS).

    Reprojects the centroid to each chunk's CRS so this works regardless of
    whether the chunks are EPSG:4326 (aerial_2023) or EPSG:3857 (vexcel_2024).
    """
    grid_dir = tiles_root / grid_id
    if not grid_dir.is_dir():
        return None
    cent_series = gpd.GeoSeries([cent_metric], crs=METRIC_CRS)
    for tif in grid_dir.glob(f"{grid_id}_*_*_geo.tif"):
        with rasterio.open(tif) as src:
            b = src.bounds
            chunk_crs = str(src.crs)
        c = cent_series.to_crs(chunk_crs).iloc[0]
        if b.left <= c.x <= b.right and b.bottom <= c.y <= b.top:
            return tif
    return None


def geom_to_pixel_bbox(geom_metric, src):
    series = gpd.GeoSeries([geom_metric], crs=METRIC_CRS).to_crs(str(src.crs))
    g = series.iloc[0]
    minx, miny, maxx, maxy = g.bounds
    row_min, col_min = src.index(minx, maxy)
    row_max, col_max = src.index(maxx, miny)
    return (
        float(min(col_min, col_max)),
        float(min(row_min, row_max)),
        float(max(col_min, col_max)),
        float(max(row_min, row_max)),
    ), g


def build_window(bbox_px, src_w, src_h):
    x0, y0, x1, y1 = bbox_px
    margin = max(CROP_MARGIN_PX, int(max(x1 - x0, y1 - y0) * 0.2))
    a = max(0, int(x0 - margin))
    b = max(0, int(y0 - margin))
    c = min(src_w, int(x1 + margin))
    d = min(src_h, int(y1 + margin))
    return Window(a, b, max(1, c - a), max(1, d - b)), (a, b)


def rasterize_polygon_window(geom_chunk_crs, window_transform, window_h, window_w):
    if geom_chunk_crs is None or geom_chunk_crs.is_empty:
        return None
    out = rasterize(
        [(geom_chunk_crs, 1)],
        out_shape=(window_h, window_w),
        transform=window_transform,
        fill=0,
        dtype=np.uint8,
    )
    return out


def polygon_from_mask(mask, transform):
    best = None
    best_area = 0.0
    for geom, val in rio_shapes(mask.astype(np.uint8), transform=transform):
        if val != 1:
            continue
        poly = shape(geom)
        if poly.is_valid and not poly.is_empty and poly.area > best_area:
            best = poly
            best_area = poly.area
    return best, best_area


def run_one_grid(grid_id, src_root, tiles_root, out_root, processor, model, device, dtype, prompt_mode):
    pred_path = src_root / grid_id / "predictions_metric.gpkg"
    if not pred_path.exists():
        return None
    preds = gpd.read_file(pred_path)
    if len(preds) == 0:
        return {"grid": grid_id, "n_in": 0, "n_out": 0}

    out_records = []
    n_in = len(preds)
    n_no_chunk = 0
    n_empty_mask = 0

    for i, row in preds.iterrows():
        geom_metric = row.geometry
        if geom_metric is None or geom_metric.is_empty:
            continue
        chunk = find_chunk_for_geom(grid_id, tiles_root, geom_metric.centroid)
        if chunk is None:
            n_no_chunk += 1
            continue
        with rasterio.open(chunk) as src:
            bbox_px, geom_chunk_crs = geom_to_pixel_bbox(geom_metric, src)
            window, (ox, oy) = build_window(bbox_px, src.width, src.height)
            data = src.read(window=window)
            window_transform = src.window_transform(window)
            chunk_crs = str(src.crs)
        rgb = np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)
        win_h, win_w = rgb.shape[:2]

        # bbox in window pixel coords
        bx0 = max(0.0, min(win_w - 1, bbox_px[0] - ox))
        by0 = max(0.0, min(win_h - 1, bbox_px[1] - oy))
        bx1 = max(0.0, min(win_w - 1, bbox_px[2] - ox))
        by1 = max(0.0, min(win_h - 1, bbox_px[3] - oy))

        # Rasterize V4.2 polygon in window pixel coords
        v42_mask_window = rasterize_polygon_window(geom_chunk_crs, window_transform, win_h, win_w)
        if v42_mask_window is None or v42_mask_window.sum() < MIN_MASK_AREA_PX:
            n_empty_mask += 1
            continue

        # Process image (this scales image to 1024x1024 and box accordingly)
        inputs = processor(
            images=Image.fromarray(rgb),
            input_boxes=[[[bx0, by0, bx1, by1]]],
            return_tensors="pt",
        )

        # Build input_masks: rescale V4.2 mask the same way processor rescaled the image
        # Processor non-uniformly resizes (win_h, win_w) → (1024, 1024). We'll resize mask to (256,256)
        mask_256 = cv2.resize(v42_mask_window, (SAM_MASK_SIZE, SAM_MASK_SIZE), interpolation=cv2.INTER_NEAREST)
        mask_logits = np.where(mask_256 > 0, MASK_LOGIT_POS, MASK_LOGIT_NEG).astype(np.float32)
        # Shape per Sam2 expectation: [batch=1, num_objects=1, 1, 256, 256] OR [1,1,256,256]
        # Test indicates conv1 expects [B,1,256,256]. The processor packages boxes as [B, num_obj, 4]; mask should match num_obj dim.
        mask_tensor = torch.from_numpy(mask_logits)[None, None, :, :]  # [1,1,256,256]

        if device == "cuda":
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            mask_tensor = mask_tensor.to(device)

        kwargs = dict(inputs)
        if prompt_mode == "mask_only":
            kwargs.pop("input_boxes", None)
            kwargs["input_masks"] = mask_tensor
        elif prompt_mode == "mask_box":
            kwargs["input_masks"] = mask_tensor
        elif prompt_mode == "box_only":
            pass  # no mask

        with torch.no_grad(), torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                              dtype=dtype, enabled=device == "cuda"):
            outputs = model(**kwargs, multimask_output=True)
        masks_list = processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().to(torch.float32).cpu(),
            inputs["original_sizes"].detach().cpu(),
        )
        scores = outputs.iou_scores.detach().to(torch.float32).cpu().numpy()[0][0]
        masks_t = masks_list[0][0]
        masks = [masks_t[k].to(torch.uint8).numpy() for k in range(masks_t.shape[0])]
        # Pick best by score
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        if mask.sum() < MIN_MASK_AREA_PX:
            order = np.argsort(-scores)
            chosen = None
            for idx in order[1:]:
                if masks[int(idx)].sum() >= MIN_MASK_AREA_PX:
                    chosen = masks[int(idx)]
                    best_idx = int(idx)
                    break
            if chosen is None:
                n_empty_mask += 1
                continue
            mask = chosen

        poly_chunk_crs, _ = polygon_from_mask(mask, window_transform)
        if poly_chunk_crs is None or poly_chunk_crs.is_empty:
            n_empty_mask += 1
            continue
        gs = gpd.GeoSeries([poly_chunk_crs], crs=chunk_crs).to_crs(METRIC_CRS)
        poly_metric = gs.iloc[0]
        if not poly_metric.is_valid or poly_metric.is_empty:
            n_empty_mask += 1
            continue
        rec = {k: row[k] for k in preds.columns if k != "geometry"}
        rec["geometry"] = poly_metric
        rec["sam_score"] = float(scores[best_idx])
        rec["sam_mask_idx"] = best_idx
        rec["orig_area_m2"] = float(geom_metric.area)
        rec["sam_area_m2"] = float(poly_metric.area)
        out_records.append(rec)

    out_dir = out_root / grid_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions_metric.gpkg"
    if out_records:
        gdf = gpd.GeoDataFrame(out_records, crs=METRIC_CRS)
        gdf.to_file(out_path, driver="GPKG")
    else:
        gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=METRIC_CRS)
        gdf.to_file(out_path, driver="GPKG")

    cfg_out = {
        "grid_id": grid_id, "source": str(pred_path), "tiles_root": str(tiles_root),
        "model": "v4_2_sam_maskprompt_PoC", "prompt_mode": prompt_mode,
        "n_in": n_in, "n_out": len(out_records),
        "n_no_chunk": n_no_chunk, "n_empty_mask": n_empty_mask,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_out, indent=2))
    print(f"[{grid_id}] in={n_in}  refined={len(out_records)}  no_chunk={n_no_chunk}  empty_mask={n_empty_mask}")
    return cfg_out


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Loading SAM2 ({args.sam_model_id})...")
    from transformers import Sam2Model, Sam2Processor
    processor = Sam2Processor.from_pretrained(args.sam_model_id)
    model = Sam2Model.from_pretrained(args.sam_model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.to(device)
    model.eval()
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.torch_dtype]
    print(f"  ready on {device} (dtype={args.torch_dtype}, prompt_mode={args.prompt_mode})")

    summaries = []
    for g in args.grids:
        try:
            s = run_one_grid(g, args.src_results_root, args.tiles_root, args.output_root,
                             processor, model, device, dtype, args.prompt_mode)
            if s:
                summaries.append(s)
        except Exception as e:
            print(f"[{g}] ERROR: {e!r}")
            import traceback; traceback.print_exc()

    summary_path = args.output_root / "_run_summary.json"
    summary_path.write_text(json.dumps({"grids": summaries, "prompt_mode": args.prompt_mode,
                                         "timestamp_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"\nDONE — summary: {summary_path}")


if __name__ == "__main__":
    main()
