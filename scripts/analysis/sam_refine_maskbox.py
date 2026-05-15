#!/usr/bin/env python3
"""SAM2 mask+box refinement of detector polygons (region-agnostic).

Reads `predictions_metric.gpkg` produced by `finalize.py` (or legacy
`detect_and_evaluate.py`) for each requested grid, re-segments every polygon
with SAM2 using the polygon's rasterized mask as `input_masks` (low-res mask
prompt) plus its bbox as `input_boxes`, and writes a new
`predictions_metric.gpkg` per grid.

Multi-region: region + grid-list are required; metric CRS is resolved per grid
via `core.grid_utils.get_metric_crs`; tiles_root may be passed explicitly
(`--tiles-root`) or auto-resolved from `regions.yaml` when `--imagery-layer`
is given. Both chunked and mosaic file_layouts are supported.
"""
from __future__ import annotations

import argparse
import json
import os
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import region_registry
from core.grid_utils import get_metric_crs, normalize_grid_id

CROP_MARGIN_PX = 64
MIN_MASK_AREA_PX = 4
MASK_LOGIT_POS = 10.0
MASK_LOGIT_NEG = -10.0
SAM_MASK_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--region", required=True,
                   help="region key or alias (ct / cape_town / jhb / johannesburg)")
    p.add_argument("--grids", nargs="+", required=True,
                   help="grid IDs to refine (e.g. G0772 G0773 ...)")
    p.add_argument("--src-results-root", type=Path, required=True,
                   help="detector results root containing <grid>/predictions_metric.gpkg")
    p.add_argument("--output-root", type=Path, required=True,
                   help="output root; SAM-refined gpkgs land at <output-root>/<grid>/predictions_metric.gpkg")
    p.add_argument("--tiles-root", type=Path, default=None,
                   help="override tiles root (chunked: <root>/<grid>/*.tif; "
                        "mosaic: <root>/<grid>_mosaic.tif). If omitted, resolves "
                        "from regions.yaml via --imagery-layer.")
    p.add_argument("--imagery-layer", default=None,
                   help="imagery_layer id from regions.yaml; used to resolve "
                        "tiles_root and file_layout when --tiles-root is omitted.")
    p.add_argument("--sam-model-id", default="facebook/sam2.1-hiera-large")
    p.add_argument("--torch-dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--prompt-mode", default="mask_box",
                   choices=["mask_only", "mask_box", "box_only"])
    p.add_argument("--sam-batch-size", type=int, default=8,
                   help="polygons per SAM forward pass; reduce if OOM")
    p.add_argument("--label", default="sam_maskbox",
                   help="tag written into per-grid config.json `model` field")
    return p.parse_args()


def resolve_tiles_root(args, region_key):
    if args.tiles_root is not None:
        return args.tiles_root, None
    if args.imagery_layer is None:
        raise SystemExit("[sam_refine] must pass either --tiles-root or --imagery-layer")
    layer = region_registry.get_imagery_layer(region_key, args.imagery_layer)
    tiles_root = region_registry.get_imagery_layer_path(region_key, args.imagery_layer)
    return tiles_root, layer.file_layout


def resolve_grid_tif(grid_id, tiles_root, file_layout):
    """Return list of source TIFs for this grid (chunked → many, mosaic → one).

    Honors ``SOLAR_TILES_ROOT`` env override (RunPod /dev/shm fast-path),
    matching ``detect_direct.py``: env path holds chunked dirs or mosaic tifs
    directly under it, no region/layer subdivision.
    """
    env_root = os.environ.get("SOLAR_TILES_ROOT")
    if env_root:
        env_path = Path(env_root)
        if file_layout == "mosaic":
            env_mosaic = env_path / f"{grid_id}_mosaic.tif"
            if env_mosaic.exists():
                return [env_mosaic]
        else:
            env_chunk = env_path / grid_id
            if env_chunk.is_dir():
                chunks = sorted(env_chunk.glob(f"{grid_id}_*_*_geo.tif"))
                if chunks:
                    return chunks
    if file_layout == "mosaic":
        mosaic = tiles_root / f"{grid_id}_mosaic.tif"
        return [mosaic] if mosaic.exists() else []
    grid_dir = tiles_root / grid_id
    if not grid_dir.is_dir():
        return []
    return sorted(grid_dir.glob(f"{grid_id}_*_*_geo.tif"))


def find_tif_for_geom(tifs, metric_crs, cent_metric):
    """Locate the TIF containing `cent_metric` (Point in `metric_crs`).

    Reprojects the centroid to each TIF's native CRS so chunked imagery in
    EPSG:4326 or EPSG:3857 both work.
    """
    cent_series = gpd.GeoSeries([cent_metric], crs=metric_crs)
    for tif in tifs:
        with rasterio.open(tif) as src:
            b = src.bounds
            tif_crs = str(src.crs)
        c = cent_series.to_crs(tif_crs).iloc[0]
        if b.left <= c.x <= b.right and b.bottom <= c.y <= b.top:
            return tif
    return None


def geom_to_pixel_bbox(geom_metric, metric_crs, src):
    series = gpd.GeoSeries([geom_metric], crs=metric_crs).to_crs(str(src.crs))
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
    return rasterize(
        [(geom_chunk_crs, 1)],
        out_shape=(window_h, window_w),
        transform=window_transform,
        fill=0,
        dtype=np.uint8,
    )


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


def _pick_best_mask(masks, scores):
    """Return (mask, best_idx) using IoU score, falling back to next-best if
    the top mask is below MIN_MASK_AREA_PX. None if all are empty."""
    best_idx = int(np.argmax(scores))
    if masks[best_idx].sum() >= MIN_MASK_AREA_PX:
        return masks[best_idx], best_idx
    for idx in np.argsort(-scores)[1:]:
        if masks[int(idx)].sum() >= MIN_MASK_AREA_PX:
            return masks[int(idx)], int(idx)
    return None, None


def _run_sam_batch(processor, model, device, dtype, batch, prompt_mode):
    """Run SAM2 on a batch of polygon records. Returns list of (mask, best_idx, score)
    or None entries (when no mask passes the MIN_MASK_AREA_PX threshold)."""
    images = [Image.fromarray(r["rgb"]) for r in batch]
    boxes = [[[r["bx0"], r["by0"], r["bx1"], r["by1"]]] for r in batch]
    inputs = processor(images=images, input_boxes=boxes, return_tensors="pt")

    mask_logits_stack = np.stack([r["mask_logits"] for r in batch], axis=0)  # [B,256,256]
    mask_tensor = torch.from_numpy(mask_logits_stack)[:, None, :, :]         # [B,1,256,256]

    if device == "cuda":
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        mask_tensor = mask_tensor.to(device)

    kwargs = dict(inputs)
    if prompt_mode == "mask_only":
        kwargs.pop("input_boxes", None)
        kwargs["input_masks"] = mask_tensor
    elif prompt_mode == "mask_box":
        kwargs["input_masks"] = mask_tensor

    with torch.no_grad(), torch.autocast(
        device_type="cuda" if device == "cuda" else "cpu",
        dtype=dtype,
        enabled=device == "cuda",
    ):
        outputs = model(**kwargs, multimask_output=True)

    masks_list = processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().to(torch.float32).cpu(),
        inputs["original_sizes"].detach().cpu(),
    )
    # masks_list: list of len B, each [num_obj=1, num_masks, H, W]
    scores_all = outputs.iou_scores.detach().to(torch.float32).cpu().numpy()  # [B, 1, num_masks]

    results = []
    for b_idx in range(len(batch)):
        scores = scores_all[b_idx][0]
        masks_t = masks_list[b_idx][0]
        masks = [masks_t[k].to(torch.uint8).numpy() for k in range(masks_t.shape[0])]
        mask, best_idx = _pick_best_mask(masks, scores)
        if mask is None:
            results.append(None)
        else:
            results.append((mask, best_idx, float(scores[best_idx])))
    return results


def run_one_grid(
    grid_id,
    *,
    region_key,
    metric_crs,
    src_root,
    tiles_root,
    file_layout,
    out_root,
    processor,
    model,
    device,
    dtype,
    prompt_mode,
    label,
    sam_batch_size,
):
    pred_path = src_root / grid_id / "predictions_metric.gpkg"
    if not pred_path.exists():
        return None
    preds = gpd.read_file(pred_path)
    n_in = len(preds)
    if n_in == 0:
        out_dir = out_root / grid_id
        out_dir.mkdir(parents=True, exist_ok=True)
        gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=metric_crs)
        gdf.to_file(out_dir / "predictions_metric.gpkg", driver="GPKG")
        return {"grid": grid_id, "n_in": 0, "n_out": 0}

    grid_tifs = resolve_grid_tif(grid_id, tiles_root, file_layout or "chunked")
    if not grid_tifs:
        print(f"[{grid_id}] WARN: no source TIFs under {tiles_root} (file_layout={file_layout})")
        return {"grid": grid_id, "n_in": n_in, "n_out": 0, "n_no_chunk": n_in}

    n_no_chunk = 0
    n_empty_mask = 0

    # ── Pass 1: prepare per-polygon crops, masks, prompts (no SAM yet) ──
    poly_records = []
    for i, row in preds.iterrows():
        geom_metric = row.geometry
        if geom_metric is None or geom_metric.is_empty:
            continue
        tif = find_tif_for_geom(grid_tifs, metric_crs, geom_metric.centroid)
        if tif is None:
            n_no_chunk += 1
            continue
        with rasterio.open(tif) as src:
            bbox_px, geom_chunk_crs = geom_to_pixel_bbox(geom_metric, metric_crs, src)
            window, (ox, oy) = build_window(bbox_px, src.width, src.height)
            data = src.read(window=window)
            window_transform = src.window_transform(window)
            chunk_crs = str(src.crs)
        rgb = np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)
        win_h, win_w = rgb.shape[:2]

        bx0 = max(0.0, min(win_w - 1, bbox_px[0] - ox))
        by0 = max(0.0, min(win_h - 1, bbox_px[1] - oy))
        bx1 = max(0.0, min(win_w - 1, bbox_px[2] - ox))
        by1 = max(0.0, min(win_h - 1, bbox_px[3] - oy))

        pred_mask_window = rasterize_polygon_window(geom_chunk_crs, window_transform, win_h, win_w)
        if pred_mask_window is None or pred_mask_window.sum() < MIN_MASK_AREA_PX:
            n_empty_mask += 1
            continue

        mask_256 = cv2.resize(pred_mask_window, (SAM_MASK_SIZE, SAM_MASK_SIZE),
                              interpolation=cv2.INTER_NEAREST)
        mask_logits = np.where(mask_256 > 0, MASK_LOGIT_POS, MASK_LOGIT_NEG).astype(np.float32)

        poly_records.append({
            "row": row,
            "geom_metric": geom_metric,
            "chunk_crs": chunk_crs,
            "window_transform": window_transform,
            "rgb": rgb,
            "bx0": bx0, "by0": by0, "bx1": bx1, "by1": by1,
            "mask_logits": mask_logits,
        })

    # ── Pass 2: batched SAM forward, post-process to metric-CRS polygons ──
    out_records = []
    for start in range(0, len(poly_records), sam_batch_size):
        batch = poly_records[start:start + sam_batch_size]
        results = _run_sam_batch(processor, model, device, dtype, batch, prompt_mode)
        for rec, res in zip(batch, results):
            if res is None:
                n_empty_mask += 1
                continue
            mask, best_idx, score = res
            poly_chunk_crs, _ = polygon_from_mask(mask, rec["window_transform"])
            if poly_chunk_crs is None or poly_chunk_crs.is_empty:
                n_empty_mask += 1
                continue
            gs = gpd.GeoSeries([poly_chunk_crs], crs=rec["chunk_crs"]).to_crs(metric_crs)
            poly_metric = gs.iloc[0]
            if not poly_metric.is_valid or poly_metric.is_empty:
                n_empty_mask += 1
                continue
            row = rec["row"]
            out_row = {k: row[k] for k in preds.columns if k != "geometry"}
            out_row["geometry"] = poly_metric
            out_row["sam_score"] = score
            out_row["sam_mask_idx"] = best_idx
            out_row["orig_area_m2"] = float(rec["geom_metric"].area)
            out_row["sam_area_m2"] = float(poly_metric.area)
            out_records.append(out_row)

    out_dir = out_root / grid_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions_metric.gpkg"
    if out_records:
        gdf = gpd.GeoDataFrame(out_records, crs=metric_crs)
        gdf.to_file(out_path, driver="GPKG")
    else:
        gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=metric_crs)
        gdf.to_file(out_path, driver="GPKG")

    cfg_out = {
        "grid_id": grid_id,
        "region": region_key,
        "metric_crs": metric_crs,
        "source": str(pred_path),
        "tiles_root": str(tiles_root),
        "file_layout": file_layout,
        "model": label,
        "prompt_mode": prompt_mode,
        "n_in": n_in,
        "n_out": len(out_records),
        "n_no_chunk": n_no_chunk,
        "n_empty_mask": n_empty_mask,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_out, indent=2))
    print(f"[{grid_id}] in={n_in}  refined={len(out_records)}  "
          f"no_chunk={n_no_chunk}  empty_mask={n_empty_mask}")
    return cfg_out


def main():
    args = parse_args()

    region_key = region_registry.normalize_region_key(args.region)
    if region_key is None:
        raise SystemExit(f"[sam_refine] unknown region alias: {args.region!r}")

    tiles_root, file_layout = resolve_tiles_root(args, region_key)
    print(f"[sam_refine] region={region_key} tiles_root={tiles_root} file_layout={file_layout}")

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
    for raw_g in args.grids:
        g = normalize_grid_id(raw_g)
        try:
            metric_crs = get_metric_crs(g, region=region_key)
        except Exception as e:
            print(f"[{g}] ERROR resolving metric CRS: {e!r}")
            continue
        try:
            s = run_one_grid(
                g,
                region_key=region_key,
                metric_crs=metric_crs,
                src_root=args.src_results_root,
                tiles_root=tiles_root,
                file_layout=file_layout,
                out_root=args.output_root,
                processor=processor,
                model=model,
                device=device,
                dtype=dtype,
                prompt_mode=args.prompt_mode,
                label=args.label,
                sam_batch_size=max(1, int(args.sam_batch_size)),
            )
            if s:
                summaries.append(s)
        except Exception as e:
            print(f"[{g}] ERROR: {e!r}")
            import traceback
            traceback.print_exc()

    summary_path = args.output_root / "_run_summary.json"
    summary_path.write_text(json.dumps({
        "region": region_key,
        "grids": summaries,
        "prompt_mode": args.prompt_mode,
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"\nDONE — summary: {summary_path}")


if __name__ == "__main__":
    main()
