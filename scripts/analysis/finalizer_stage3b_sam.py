#!/usr/bin/env python3
"""Stage 3B: cluster envelope -> SAM2 mask refinement (cross-grid sweep).

Builds mutual-IoU clusters from a per-detection (pre-NMS) finalize gpkg, then
replaces each cluster with a single SAM2-tightened polygon. The cluster's
union of member masks supplies an optional mask prompt; the cluster bbox
supplies the box prompt. Designed as a follow-up to the Stage 3 envelope-group
fill ablation (G0817 win that did not transport to G0816/G0925).

Variants:

- ``stage3b_all_maskbox``    every cluster -> SAM (mask + box prompt)
- ``stage3b_all_boxonly``    every cluster -> SAM (box only)
- ``stage3b_multi_maskbox``  n>=2 -> SAM, n=1 keeps per-detection geom
- ``stage3b_multi_boxonly``  same gating, box-only prompt

Baselines mirrored from finalizer_envelope_group_sweep.py: ``old_pixel_or``
(input pixel-or gpkg), ``per_detection`` (input per-detection gpkg),
``mutual_iou_<t>`` (cluster pixel-or unions), and optional
``ref_sam_maskbox`` (existing per-prediction SAM refinement of pixel-or).

Outputs (under ``--output-dir``):

- ``overall_metrics.csv``                 one row per variant
- ``source_metrics.csv``                  one row per (variant, source)
- ``cluster_assignments.csv``             cluster -> envelope/SAM features
- ``stage3b_<variant>/predictions_metric.gpkg``  refined polygons
- ``summary.md``                          baselines + Stage 3B leader
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image
from rasterio.features import rasterize, shapes as rio_shapes
from rasterio.windows import Window
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.grid_utils import get_metric_crs, normalize_grid_id  # noqa: E402

from scripts.analysis.finalizer_envelope_group_sweep import (  # noqa: E402
    EnvelopeRow,
    _safe_iou,
    _unary_safe,
    assign_clusters_to_envelopes,
    compute_overall,
    compute_source_metrics,
    mutual_iou_clusters,
    _md_table,
)


CROP_MARGIN_PX = 64
MIN_MASK_AREA_PX = 4
MASK_LOGIT_POS = 10.0
MASK_LOGIT_NEG = -10.0
SAM_MASK_SIZE = 256


# ───────────────────────── chunk + SAM helpers ─────────────────────────

def find_chunk_for_geom(grid_id: str, tiles_root: Path, cent_metric, metric_crs: str):
    grid_dir = tiles_root / grid_id
    if not grid_dir.is_dir():
        return None
    cent_series = gpd.GeoSeries([cent_metric], crs=metric_crs)
    for tif in grid_dir.glob(f"{grid_id}_*_*_geo.tif"):
        with rasterio.open(tif) as src:
            b = src.bounds
            chunk_crs = str(src.crs)
        c = cent_series.to_crs(chunk_crs).iloc[0]
        if b.left <= c.x <= b.right and b.bottom <= c.y <= b.top:
            return tif
    return None


def geom_to_pixel_bbox(geom_metric, src, metric_crs: str):
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


# ───────────────────────── cluster build ─────────────────────────

@dataclass
class ClusterRow:
    cluster_id: int
    members: list[int]
    geom: BaseGeometry         # union of member geoms (per-detection pixel-or)
    area: float
    score: float


def build_clusters_from_per_detection(
    per_det: gpd.GeoDataFrame,
    iou_threshold: float,
) -> list[ClusterRow]:
    polys = list(per_det.geometry.values)
    if "confidence" in per_det.columns:
        scores = per_det["confidence"].astype(float).values
    elif "score" in per_det.columns:
        scores = per_det["score"].astype(float).values
    else:
        scores = np.ones(len(per_det))
    cluster_id = mutual_iou_clusters(polys, iou_threshold)
    members_by_cid: dict[int, list[int]] = {}
    for i, cid in enumerate(cluster_id):
        members_by_cid.setdefault(cid, []).append(i)
    out: list[ClusterRow] = []
    for new_id, (_, members) in enumerate(sorted(members_by_cid.items())):
        geoms = [polys[i] for i in members]
        merged = _unary_safe(geoms)
        if merged is None or merged.is_empty:
            continue
        out.append(ClusterRow(
            cluster_id=new_id,
            members=members,
            geom=merged,
            area=float(merged.area),
            score=float(max(scores[i] for i in members)),
        ))
    return out


# ───────────────────────── SAM call ─────────────────────────

@dataclass
class SamResult:
    cluster_id: int
    geom_metric: BaseGeometry | None
    sam_score: float
    sam_mask_idx: int
    n_members: int
    src_area_m2: float
    refined_area_m2: float
    chunk_path: str | None
    failure: str | None


def refine_cluster_with_sam(
    cluster: ClusterRow,
    grid_id: str,
    tiles_root: Path,
    metric_crs: str,
    processor,
    model,
    device: str,
    autocast_dtype,
    use_mask_prompt: bool,
    box_geom_metric: BaseGeometry | None = None,
) -> SamResult:
    """Run SAM2 on `cluster`. By default cluster.geom is used both for the
    bbox prompt and (when ``use_mask_prompt``) the mask prompt. Pass
    ``box_geom_metric`` to override the bbox source (e.g. enclosing pixel-or
    envelope for the hybrid Stage 3B variant); the cluster geom is still
    used as the mask prompt in that case.
    """
    cluster_geom = cluster.geom
    box_geom = box_geom_metric if box_geom_metric is not None else cluster_geom
    cent = cluster_geom.centroid
    chunk = find_chunk_for_geom(grid_id, tiles_root, cent, metric_crs)
    if chunk is None:
        return SamResult(
            cluster_id=cluster.cluster_id, geom_metric=None, sam_score=0.0,
            sam_mask_idx=-1, n_members=len(cluster.members),
            src_area_m2=cluster.area, refined_area_m2=0.0,
            chunk_path=None, failure="no_chunk",
        )

    with rasterio.open(chunk) as src:
        bbox_px, _ = geom_to_pixel_bbox(box_geom, src, metric_crs)
        _, cluster_geom_chunk = geom_to_pixel_bbox(cluster_geom, src, metric_crs)
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

    cluster_mask_window = rasterize_polygon_window(
        cluster_geom_chunk, window_transform, win_h, win_w,
    )
    if cluster_mask_window is None or cluster_mask_window.sum() < MIN_MASK_AREA_PX:
        return SamResult(
            cluster_id=cluster.cluster_id, geom_metric=None, sam_score=0.0,
            sam_mask_idx=-1, n_members=len(cluster.members),
            src_area_m2=cluster.area, refined_area_m2=0.0,
            chunk_path=str(chunk), failure="empty_input_mask",
        )

    inputs = processor(
        images=Image.fromarray(rgb),
        input_boxes=[[[bx0, by0, bx1, by1]]],
        return_tensors="pt",
    )

    kwargs = dict(inputs)
    if use_mask_prompt:
        mask_256 = cv2.resize(
            cluster_mask_window, (SAM_MASK_SIZE, SAM_MASK_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_logits = np.where(mask_256 > 0, MASK_LOGIT_POS, MASK_LOGIT_NEG).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_logits)[None, None, :, :]  # [1,1,256,256]
        if device == "cuda":
            mask_tensor = mask_tensor.to(device)
        kwargs["input_masks"] = mask_tensor

    if device == "cuda":
        kwargs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in kwargs.items()}

    with torch.no_grad(), torch.autocast(
        device_type="cuda" if device == "cuda" else "cpu",
        dtype=autocast_dtype, enabled=device == "cuda",
    ):
        outputs = model(**kwargs, multimask_output=True)
    masks_list = processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().to(torch.float32).cpu(),
        inputs["original_sizes"].detach().cpu(),
    )
    scores = outputs.iou_scores.detach().to(torch.float32).cpu().numpy()[0][0]
    masks_t = masks_list[0][0]
    masks = [masks_t[k].to(torch.uint8).numpy() for k in range(masks_t.shape[0])]

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
            return SamResult(
                cluster_id=cluster.cluster_id, geom_metric=None, sam_score=float(scores[best_idx]),
                sam_mask_idx=best_idx, n_members=len(cluster.members),
                src_area_m2=cluster.area, refined_area_m2=0.0,
                chunk_path=str(chunk), failure="empty_sam_mask",
            )
        mask = chosen

    poly_chunk_crs, _ = polygon_from_mask(mask, window_transform)
    if poly_chunk_crs is None or poly_chunk_crs.is_empty:
        return SamResult(
            cluster_id=cluster.cluster_id, geom_metric=None, sam_score=float(scores[best_idx]),
            sam_mask_idx=best_idx, n_members=len(cluster.members),
            src_area_m2=cluster.area, refined_area_m2=0.0,
            chunk_path=str(chunk), failure="vectorize_empty",
        )
    gs = gpd.GeoSeries([poly_chunk_crs], crs=chunk_crs).to_crs(metric_crs)
    poly_metric = gs.iloc[0]
    if not poly_metric.is_valid or poly_metric.is_empty:
        return SamResult(
            cluster_id=cluster.cluster_id, geom_metric=None, sam_score=float(scores[best_idx]),
            sam_mask_idx=best_idx, n_members=len(cluster.members),
            src_area_m2=cluster.area, refined_area_m2=0.0,
            chunk_path=str(chunk), failure="reproject_empty",
        )
    return SamResult(
        cluster_id=cluster.cluster_id, geom_metric=poly_metric,
        sam_score=float(scores[best_idx]), sam_mask_idx=best_idx,
        n_members=len(cluster.members), src_area_m2=cluster.area,
        refined_area_m2=float(poly_metric.area), chunk_path=str(chunk),
        failure=None,
    )


# ───────────────────────── orchestration ─────────────────────────

def _load_gpkg(path: Path, metric_crs: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(path)
    if g.crs is None:
        raise ValueError(f"{path} has no CRS")
    if str(g.crs) != metric_crs:
        g = g.to_crs(metric_crs)
    return g.reset_index(drop=True)


def _write_variant_gpkg(
    out_dir: Path, name: str, geoms: list[BaseGeometry], metric_crs: str,
) -> None:
    out_sub = out_dir / name
    out_sub.mkdir(parents=True, exist_ok=True)
    if geoms:
        gdf = gpd.GeoDataFrame({"geometry": geoms}, geometry="geometry", crs=metric_crs)
    else:
        gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=metric_crs)
    gdf.to_file(out_sub / "predictions_metric.gpkg", driver="GPKG")


def main() -> int:
    ap = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-id", required=True)
    ap.add_argument("--region", default="jhb")
    ap.add_argument("--per-detection-gpkg", required=True, type=Path,
                    help="pre-NMS per-detection finalize output (nms_iou>=1.0)")
    ap.add_argument("--pixel-or-gpkg", type=Path, default=None,
                    help="pixel-or finalize output (baseline)")
    ap.add_argument("--clean-gt", required=True, type=Path)
    ap.add_argument("--ref-sam-maskbox-gpkg", type=Path, default=None,
                    help="existing per-prediction SAM refinement of pixel-or")
    ap.add_argument("--tiles-root", required=True, type=Path,
                    help="chunked tile root; expects <root>/<grid>/<grid>_*_*_geo.tif")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--mutual-iou", type=float, default=0.3)
    ap.add_argument("--cluster-in-envelope", type=float, default=0.8,
                    help="Cluster assigned to enclosing pixel-or envelope when "
                         "intersection / cluster_area >= this threshold (hybrid)")
    ap.add_argument("--sam-model-id", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--torch-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--enable-hybrid", action="store_true",
                    help="Add hybrid Stage 3B variants: enclosing pixel-or "
                         "envelope bbox as box prompt + cluster pixel-or as "
                         "mask prompt. Requires --pixel-or-gpkg.")
    ap.add_argument("--write-variant-gpkg", action="store_true",
                    help="Also write predictions_metric.gpkg per variant")
    args = ap.parse_args()

    grid_id = normalize_grid_id(args.grid_id)
    metric_crs = str(get_metric_crs(grid_id, region=args.region))
    print(f"[stage3b] grid={grid_id} region={args.region} crs={metric_crs}")

    per_det = _load_gpkg(args.per_detection_gpkg, metric_crs)
    pixel_or = _load_gpkg(args.pixel_or_gpkg, metric_crs) if args.pixel_or_gpkg else None
    clean_gt = _load_gpkg(args.clean_gt, metric_crs)
    ref_sam = (
        _load_gpkg(args.ref_sam_maskbox_gpkg, metric_crs)
        if args.ref_sam_maskbox_gpkg and args.ref_sam_maskbox_gpkg.exists()
        else None
    )
    print(f"[stage3b] per_det={len(per_det)}  "
          f"pixel_or={'-' if pixel_or is None else len(pixel_or)}  "
          f"clean_gt={len(clean_gt)}  "
          f"ref_sam={'-' if ref_sam is None else len(ref_sam)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) clusters from per_detection
    clusters = build_clusters_from_per_detection(per_det, args.mutual_iou)
    print(f"[stage3b] mutual_iou_{args.mutual_iou} clusters: {len(clusters)} "
          f"(multi: {sum(1 for c in clusters if len(c.members) >= 2)})")

    # 1b) optional: build envelopes + cluster->envelope assignment for hybrid
    envelopes: list[EnvelopeRow] = []
    cluster_to_env: dict[int, int | None] = {}
    enable_hybrid = bool(args.enable_hybrid and pixel_or is not None)
    if enable_hybrid:
        envelopes = [
            EnvelopeRow(
                env_id=i, geom=row.geometry,
                area=float(row.geometry.area),
                score=float(row.get("confidence", row.get("score", 0.0))),
            )
            for i, row in pixel_or.iterrows()
            if row.geometry is not None and not row.geometry.is_empty
        ]
        cluster_to_env = assign_clusters_to_envelopes(
            clusters, envelopes, args.cluster_in_envelope,
        )
        n_assigned = sum(1 for v in cluster_to_env.values() if v is not None)
        print(f"[stage3b]  hybrid: clusters assigned to envelope: "
              f"{n_assigned}/{len(clusters)}  (envelopes: {len(envelopes)})")
    else:
        print("[stage3b]  hybrid disabled (no --enable-hybrid or no pixel-or)")

    env_by_id = {e.env_id: e for e in envelopes}

    # 2) load SAM
    print(f"[stage3b] loading SAM2 ({args.sam_model_id})...")
    from transformers import Sam2Model, Sam2Processor
    processor = Sam2Processor.from_pretrained(args.sam_model_id)
    model = Sam2Model.from_pretrained(args.sam_model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.to(device)
    model.eval()
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    autocast_dtype = dtype_map[args.torch_dtype]
    print(f"[stage3b]  ready on {device} (dtype={args.torch_dtype})")

    # 3) SAM-refine every cluster, twice (mask+box and box-only). cluster_id is
    #    stable across both passes.
    cluster_lookup = {c.cluster_id: c for c in clusters}
    n_clusters = len(clusters)

    sam_maskbox: dict[int, SamResult] = {}
    sam_boxonly: dict[int, SamResult] = {}
    sam_hybrid: dict[int, SamResult] = {}
    cluster_env_box_used: dict[int, int | None] = {}  # cluster_id -> env_id used
    failures = {"maskbox": [], "boxonly": [], "hybrid": []}
    for i, c in enumerate(clusters):
        if i % 25 == 0 or i == n_clusters - 1:
            print(f"[stage3b]  SAM cluster {i+1}/{n_clusters}  "
                  f"n_members={len(c.members)}  area_m2={c.area:.1f}")
        r_mb = refine_cluster_with_sam(
            c, grid_id, args.tiles_root, metric_crs, processor, model,
            device, autocast_dtype, use_mask_prompt=True,
        )
        sam_maskbox[c.cluster_id] = r_mb
        if r_mb.failure:
            failures["maskbox"].append((c.cluster_id, r_mb.failure))
        r_bo = refine_cluster_with_sam(
            c, grid_id, args.tiles_root, metric_crs, processor, model,
            device, autocast_dtype, use_mask_prompt=False,
        )
        sam_boxonly[c.cluster_id] = r_bo
        if r_bo.failure:
            failures["boxonly"].append((c.cluster_id, r_bo.failure))

        if enable_hybrid:
            env_id = cluster_to_env.get(c.cluster_id)
            cluster_env_box_used[c.cluster_id] = env_id
            box_geom = env_by_id[env_id].geom if env_id is not None else None
            r_hy = refine_cluster_with_sam(
                c, grid_id, args.tiles_root, metric_crs, processor, model,
                device, autocast_dtype, use_mask_prompt=True,
                box_geom_metric=box_geom,
            )
            sam_hybrid[c.cluster_id] = r_hy
            if r_hy.failure:
                failures["hybrid"].append((c.cluster_id, r_hy.failure))

    print(f"[stage3b] SAM mask+box failures: {len(failures['maskbox'])}, "
          f"box-only failures: {len(failures['boxonly'])}, "
          f"hybrid failures: {len(failures['hybrid'])}")

    # 4) write cluster_assignments.csv
    rows = []
    for c in clusters:
        mb = sam_maskbox[c.cluster_id]
        bo = sam_boxonly[c.cluster_id]
        row = {
            "cluster_id": c.cluster_id,
            "n_members": len(c.members),
            "src_area_m2": round(c.area, 1),
            "score": round(c.score, 3),
            "sam_mb_area_m2": round(mb.refined_area_m2, 1),
            "sam_mb_iou_score": round(mb.sam_score, 3),
            "sam_mb_failure": mb.failure or "",
            "sam_bo_area_m2": round(bo.refined_area_m2, 1),
            "sam_bo_iou_score": round(bo.sam_score, 3),
            "sam_bo_failure": bo.failure or "",
        }
        if enable_hybrid:
            hy = sam_hybrid[c.cluster_id]
            env_id = cluster_env_box_used.get(c.cluster_id)
            row.update({
                "hybrid_env_id": "" if env_id is None else int(env_id),
                "hybrid_env_area_m2": (round(env_by_id[env_id].area, 1)
                                        if env_id is not None else 0.0),
                "sam_hy_area_m2": round(hy.refined_area_m2, 1),
                "sam_hy_iou_score": round(hy.sam_score, 3),
                "sam_hy_failure": hy.failure or "",
            })
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.output_dir / "cluster_assignments.csv", index=False)

    # 5) build variant geom lists
    def variant_all(sam_results: dict[int, SamResult]) -> list[BaseGeometry]:
        out = []
        for c in clusters:
            r = sam_results[c.cluster_id]
            if r.geom_metric is not None:
                out.append(r.geom_metric)
            else:
                out.append(c.geom)
        return out

    def variant_multi(sam_results: dict[int, SamResult]) -> list[BaseGeometry]:
        out = []
        for c in clusters:
            if len(c.members) >= 2:
                r = sam_results[c.cluster_id]
                if r.geom_metric is not None:
                    out.append(r.geom_metric)
                else:
                    out.append(c.geom)
            else:
                # solo cluster: keep per-detection geom (cluster.geom == single per-det)
                out.append(c.geom)
        return out

    variants: dict[str, list[BaseGeometry]] = {}
    variants["per_detection"] = list(per_det.geometry.values)
    variants[f"mutual_iou_{args.mutual_iou}"] = [c.geom for c in clusters]
    if pixel_or is not None:
        variants["old_pixel_or"] = list(pixel_or.geometry.values)
    if ref_sam is not None:
        variants["ref_sam_maskbox"] = list(ref_sam.geometry.values)
    variants["stage3b_all_maskbox"] = variant_all(sam_maskbox)
    variants["stage3b_all_boxonly"] = variant_all(sam_boxonly)
    variants["stage3b_multi_maskbox"] = variant_multi(sam_maskbox)
    variants["stage3b_multi_boxonly"] = variant_multi(sam_boxonly)
    if enable_hybrid:
        variants["stage3b_hybrid_all"] = variant_all(sam_hybrid)
        variants["stage3b_hybrid_multi"] = variant_multi(sam_hybrid)

    # 6) metrics
    sources = list(clean_gt["source"].unique()) if "source" in clean_gt.columns else []
    overall_rows = []
    source_rows = []
    for vname, geoms in variants.items():
        ovr = compute_overall(geoms, clean_gt)
        ovr["variant"] = vname
        overall_rows.append(ovr)
        for src in sources:
            sub = clean_gt[clean_gt["source"] == src].reset_index(drop=True)
            srm = compute_source_metrics(geoms, sub)
            srm["variant"] = vname
            srm["source"] = src
            source_rows.append(srm)

    overall_df = pd.DataFrame(overall_rows)
    overall_cols = [
        "variant", "n_pred", "pred_area_m2", "n_gt", "gt_area_m2", "pred_gt",
        "max_m2", "p95_m2", "p99_m2", "n_ge_200",
        "area_p", "area_r", "area_f1",
        "tp_pred_iou03", "fp_iou03", "fn_gt_iou03",
        "poly_p_iou03", "poly_r_iou03", "poly_f1_iou03",
    ]
    overall_df = overall_df[overall_cols]
    overall_df.to_csv(args.output_dir / "overall_metrics.csv", index=False)

    source_df = pd.DataFrame(source_rows)
    source_cols = [
        "variant", "source", "n_gt", "gt_area_sum",
        "any_overlap_rate", "recall_iou03", "recall_iou05",
        "median_iou", "mean_iou", "share_iou_lt_05",
        "area_recall", "median_coverage",
        "mean_n_pred_hits", "median_n_pred_hits",
    ]
    source_df = source_df[source_cols]
    source_df.to_csv(args.output_dir / "source_metrics.csv", index=False)

    # 7) optional gpkg per variant (Stage 3B variants only)
    if args.write_variant_gpkg:
        gpkg_variants = [
            "stage3b_all_maskbox", "stage3b_all_boxonly",
            "stage3b_multi_maskbox", "stage3b_multi_boxonly",
        ]
        if enable_hybrid:
            gpkg_variants.extend(["stage3b_hybrid_all", "stage3b_hybrid_multi"])
        for vname in gpkg_variants:
            _write_variant_gpkg(args.output_dir, vname, variants[vname], metric_crs)

    # 8) summary
    write_summary(args.output_dir, grid_id, overall_df, source_df, sources)

    # 9) run config
    (args.output_dir / "run_config.json").write_text(json.dumps({
        "grid_id": grid_id, "region": args.region, "metric_crs": metric_crs,
        "per_detection_gpkg": str(args.per_detection_gpkg),
        "pixel_or_gpkg": None if pixel_or is None else str(args.pixel_or_gpkg),
        "ref_sam_maskbox_gpkg": (None if ref_sam is None
                                  else str(args.ref_sam_maskbox_gpkg)),
        "tiles_root": str(args.tiles_root),
        "mutual_iou": args.mutual_iou,
        "sam_model_id": args.sam_model_id,
        "torch_dtype": args.torch_dtype,
        "n_clusters": n_clusters,
        "n_clusters_multi": sum(1 for c in clusters if len(c.members) >= 2),
        "sam_maskbox_failures": len(failures["maskbox"]),
        "sam_boxonly_failures": len(failures["boxonly"]),
        "hybrid_enabled": enable_hybrid,
        "hybrid_clusters_assigned": (
            sum(1 for v in cluster_to_env.values() if v is not None)
            if enable_hybrid else 0
        ),
        "sam_hybrid_failures": len(failures["hybrid"]),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    print(f"[stage3b] wrote overall_metrics.csv ({len(overall_df)}), "
          f"source_metrics.csv ({len(source_df)}), "
          f"cluster_assignments.csv ({n_clusters})")
    return 0


def write_summary(
    out_dir: Path, grid_id: str,
    overall_df: pd.DataFrame, source_df: pd.DataFrame,
    sources: list[str],
) -> None:
    keep_cols = ["variant", "n_pred", "pred_gt",
                 "area_p", "area_r", "area_f1",
                 "poly_r_iou03", "poly_f1_iou03"]
    lines = [f"# {grid_id} Stage 3B (cluster -> SAM) summary\n\n"]
    lines.append("Stage 3B: replace each mutual-IoU cluster with one "
                 "SAM2-tightened polygon. Box prompt = cluster bbox; "
                 "mask prompt (when used) = cluster pixel-or.\n\n")
    lines.append("## Overall (baselines + Stage 3B variants)\n\n")
    lines.append(_md_table(overall_df[keep_cols].reset_index(drop=True)))

    leader = overall_df.sort_values("area_f1", ascending=False).iloc[0]
    lines.append(f"\n**area_F1 leader**: `{leader['variant']}` "
                 f"(area_F1={leader['area_f1']:.3f}, "
                 f"poly_F1@0.3={leader['poly_f1_iou03']:.3f}, "
                 f"n_pred={int(leader['n_pred'])})\n")

    if sources:
        lines.append("\n## Source diagnostics\n")
        pivot_variants = [
            v for v in (
                "old_pixel_or", "per_detection",
                "ref_sam_maskbox",
                "stage3b_all_maskbox", "stage3b_all_boxonly",
                "stage3b_multi_maskbox", "stage3b_multi_boxonly",
                "stage3b_hybrid_all", "stage3b_hybrid_multi",
            )
            if v in set(source_df["variant"])
        ]
        # also include the mutual_iou_<t> baseline
        mu = source_df[source_df["variant"].str.startswith("mutual_iou_")]["variant"]
        if len(mu):
            pivot_variants.insert(2, mu.iloc[0])
        for src in sources:
            sub = source_df[(source_df["source"] == src)
                            & (source_df["variant"].isin(pivot_variants))]
            if len(sub) == 0:
                continue
            lines.append(f"\n### source = `{src}`\n\n")
            cols = ["variant", "n_gt", "recall_iou05", "median_iou",
                    "share_iou_lt_05", "area_recall"]
            lines.append(_md_table(sub[cols].reset_index(drop=True)))

    (out_dir / "summary.md").write_text("".join(lines))


if __name__ == "__main__":
    sys.exit(main())
