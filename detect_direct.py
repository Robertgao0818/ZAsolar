#!/usr/bin/env python3
"""Direct Mask R-CNN inference (no geoai wrapper).

Produces a raw detection artifact (`raw_detections.pkl`) consumed by
``finalize.py``. See `/home/gaosh/.claude/plans/opengeoai-detector-sam-...md`
(plan v1.4) for design.

Stage 1 of the production direct pipeline:

    detect_direct.py  →  raw_detections.pkl
                            │
                       finalize.py  →  predictions_metric.gpkg
                            │
              (optional)    ▼  scripts/analysis/sam_refine_maskbox.py
                            │
                       predictions_metric.gpkg  (SAM-refined)

The full pipeline is orchestrated by ``scripts/runpod_detect_direct_template.sh``
(Phase A: detect+finalize parallel; Phase B: SAM refine serial; Phase C: eval).

Knob-rich; saturation-friendly. Custom DataLoader + collate so torchvision's
``list[Tensor]`` input works; per-worker rasterio handles for GDAL fork-safety.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.inference.raw_artifact import (
    Chip,
    Detection,
    PIPELINE_VERSION,
    RawArtifact,
    SCHEMA_VERSION,
    SourceTile,
    utc_now_iso,
    write_artifact,
)
from core.inference.tile_dataset import (
    SlidingWindowDataset,
    list_collate,
    worker_init_fn,
)
from core.models.maskrcnn import build_solar_maskrcnn
from core import region_registry
from core.grid_utils import normalize_grid_id, normalize_region

DEFAULT_OUTPUT_BASE = Path(__file__).resolve().parent / "results" / "analysis" / "direct_maskrcnn_v1"
SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="detect_direct.py",
        description="Direct Mask R-CNN inference → raw_detections.pkl",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Identity
    p.add_argument("--grid-id", required=True, help="e.g. G0816")
    p.add_argument("--region", required=True, help="ct | jhb | cape_town | johannesburg")
    p.add_argument("--imagery-layer", required=True,
                   help="layer id from regions.yaml, e.g. aerial_2025, geid_2024_02, vexcel_2024")
    p.add_argument("--model-run", required=True,
                   help="model_run id from regions.yaml (used in output path; provenance only)")
    p.add_argument("--model-path", required=True, type=Path,
                   help=".pth checkpoint, e.g. checkpoints/exp003_C_targeted_hn/best_model.pth")
    # Knobs (Phase 1 conservative defaults)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--chip-size", type=int, default=400)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--parity-mode", choices=["direct", "geoai"], default="direct",
                   help="direct = current raw crop artifact; geoai = store full-chip "
                        "masks and use torchvision's default detections_per_img for "
                        "closer geoai.SolarPanelDetector parity")
    p.add_argument("--detector-score-threshold", type=float, default=0.05,
                   help="model.roi_heads.score_thresh — controls what enters the artifact")
    p.add_argument("--nms-iou-threshold", type=float, default=0.5,
                   help="model.roi_heads.nms_thresh (box NMS IoU). 0.99 = effectively "
                        "disabled; useful for probing whether sibling sub-arrays are "
                        "being suppressed by NMS rather than missing at proposal stage.")
    p.add_argument("--detections-per-img", type=int, default=None,
                   help="model.roi_heads.detections_per_img; default is 300 in "
                        "direct mode and torchvision/geoai default 100 in geoai mode")
    p.add_argument("--mask-threshold", type=float, default=0.3,
                   help="recorded only; soft mask binarization happens in finalize.py")
    p.add_argument("--raw-mask-storage", choices=["crop", "full_chip"], default="crop")
    p.add_argument("--device", default="auto", help="cuda | cpu | auto")
    # Output
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override output dir. Default: results/analysis/direct_maskrcnn_v1/<region>/<model_run>/<grid>/")
    p.add_argument("--output-raw", type=Path, default=None,
                   help="Override raw artifact path. Default: <output-dir>/raw_detections.pkl")
    # Smoke / debug
    p.add_argument("--profile", action="store_true",
                   help="Print per-stage timing summary at the end")
    p.add_argument("--dry-run-windows", action="store_true",
                   help="Iterate dataset only; no GPU forward, no artifact written")
    p.add_argument("--max-chips", type=int, default=None,
                   help="Cap chips for smoke testing")
    p.add_argument("--hash-tiles-content", action="store_true",
                   help="Include sha256 of source tiles in artifact (slow on big imagery)")
    return p


# ─────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────
def resolve_tile_paths(args: argparse.Namespace, region_key: str) -> list[Path]:
    """Return source TIF paths based on imagery layer's file_layout.

    Honors `SOLAR_TILES_ROOT` env var as a fast-path override (RunPod /dev/shm),
    matching `core.grid_utils.resolve_tiles_dir` semantics: if the env-rooted
    candidate path exists, it wins over the registry path.
    """
    layer = region_registry.get_imagery_layer(region_key, args.imagery_layer)
    grid_id = args.grid_id

    # On-disk tiles stay keyed under the SOURCE grid ID after the CPT regrid
    # (ADR-0002 §5): CPT1240's tiles live in the G1240 dir, with CPT1240_*
    # filenames. download_tiles.py:197 and core.grid_utils.resolve_tiles_dir:230
    # both honor this; this resolver must too, or every regridded (CPT) grid
    # FileNotFounds. Directory/mosaic paths use source_id; the chip glob keeps the
    # logical grid_id (filenames are CPT-keyed). For non-regridded regions
    # resolve_source_grid_id is a no-op (returns grid_id), so JHB etc. are unchanged.
    source_id = region_registry.resolve_source_grid_id(grid_id, region_key)

    # 1) SOLAR_TILES_ROOT fast path (RunPod /dev/shm)
    env_root = os.environ.get("SOLAR_TILES_ROOT")
    chunk_dir = None
    mosaic_path = None
    if env_root:
        env_path = Path(env_root)
        env_chunk = env_path / source_id
        env_mosaic = env_path / f"{source_id}_mosaic.tif"
        if layer.file_layout == "chunked" and env_chunk.exists():
            chunk_dir = env_chunk
        elif layer.file_layout == "mosaic" and env_mosaic.exists():
            mosaic_path = env_mosaic

    # 2) Registry path (canonical, post-restructure)
    if chunk_dir is None and mosaic_path is None:
        layer_root = region_registry.get_imagery_layer_path(region_key, args.imagery_layer)
        if layer.file_layout == "mosaic":
            mosaic_path = layer_root / f"{source_id}_mosaic.tif"
        else:
            chunk_dir = layer_root / source_id

    if layer.file_layout == "mosaic":
        if mosaic_path is None or not mosaic_path.exists():
            raise FileNotFoundError(f"mosaic TIF not found: {mosaic_path}")
        return [mosaic_path]

    if layer.file_layout == "chunked":
        if chunk_dir is None or not chunk_dir.exists():
            raise FileNotFoundError(f"chunked tile dir not found: {chunk_dir}")
        chunks = sorted(chunk_dir.glob(f"{grid_id}_*_*_geo.tif"))
        if not chunks:
            chunks = sorted(
                p for p in chunk_dir.glob(f"{grid_id}_*_*.tif")
                if "_geo" not in p.stem and "mosaic" not in p.stem and "mask" not in p.stem
            )
        if not chunks:
            raise FileNotFoundError(f"no chunks under {chunk_dir}")
        return chunks

    raise ValueError(f"unsupported file_layout: {layer.file_layout!r}")


def resolve_output_dir(args: argparse.Namespace, region_arg: str) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return DEFAULT_OUTPUT_BASE / region_arg / args.model_run / args.grid_id


# ─────────────────────────────────────────────────────────────────────────
# Mask cropping (V1.4 decision #18)
# ─────────────────────────────────────────────────────────────────────────
def _clip_box_to_int(
    box_xyxy: np.ndarray, chip_size: int
) -> tuple[int, int, int, int]:
    """Round + clip a float box to integer pixel bounds within [0, chip_size]."""
    x1 = int(max(0, np.floor(box_xyxy[0])))
    y1 = int(max(0, np.floor(box_xyxy[1])))
    x2 = int(min(chip_size, np.ceil(box_xyxy[2])))
    y2 = int(min(chip_size, np.ceil(box_xyxy[3])))
    if x2 <= x1:
        x2 = min(chip_size, x1 + 1)
    if y2 <= y1:
        y2 = min(chip_size, y1 + 1)
    return x1, y1, x2, y2


def _box_chip_to_source(
    box_xyxy: np.ndarray, window: tuple[int, int, int, int]
) -> tuple[float, float, float, float]:
    """Project chip-pixel float box → source-TIF-pixel float box."""
    col_off, row_off, _w, _h = window
    return (
        float(box_xyxy[0] + col_off),
        float(box_xyxy[1] + row_off),
        float(box_xyxy[2] + col_off),
        float(box_xyxy[3] + row_off),
    )


# ─────────────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> int:
    region_arg = args.region
    region_key = region_registry.normalize_region_key(region_arg)
    if region_key is None:
        sys.exit(f"unknown region alias: {region_arg!r}")
    args.grid_id = normalize_grid_id(args.grid_id)

    print(f"[detect_direct] grid={args.grid_id} region={region_arg}({region_key}) "
          f"layer={args.imagery_layer} run={args.model_run}")
    print(f"[detect_direct] model={args.model_path}")

    # ── Resolve tiles ────────────────────────────────────────────────
    tif_paths = resolve_tile_paths(args, region_key)
    print(f"[detect_direct] {len(tif_paths)} source TIF(s)")

    if args.detections_per_img is None:
        args.detections_per_img = 100 if args.parity_mode == "geoai" else 300
    if args.parity_mode == "geoai" and args.raw_mask_storage != "full_chip":
        print("[detect_direct] geoai parity mode: forcing --raw-mask-storage full_chip")
        args.raw_mask_storage = "full_chip"

    # ── Output dir ───────────────────────────────────────────────────
    out_dir = resolve_output_dir(args, region_arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_raw if args.output_raw is not None else out_dir / "raw_detections.pkl"
    print(f"[detect_direct] output_dir={out_dir}")
    print(f"[detect_direct] raw_path={raw_path}")
    print(f"[detect_direct] parity_mode={args.parity_mode} raw_mask_storage={args.raw_mask_storage} "
          f"detections_per_img={args.detections_per_img}")

    # ── Dataset ──────────────────────────────────────────────────────
    dataset = SlidingWindowDataset(
        tif_paths,
        chip_size=args.chip_size,
        overlap=args.overlap,
        edge_pad=True,
        max_chips=args.max_chips,
        window_origin_mode="geoai" if args.parity_mode == "geoai" else "anchored",
    )
    print(f"[detect_direct] dataset: {len(dataset)} chips, stride={dataset.stride}")

    if args.dry_run_windows:
        # Iterate the dataset only (covers tile open + window math).
        t0 = time.time()
        for i in range(len(dataset)):
            chip, meta = dataset[i]
            assert chip.shape == (3, args.chip_size, args.chip_size)
            assert meta.chip_index == i
        dataset.close()
        print(f"[detect_direct] --dry-run-windows OK in {time.time() - t0:.2f}s")
        return 0

    # ── Device ───────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[detect_direct] device={device}")

    # ── Model ────────────────────────────────────────────────────────
    model, info = build_solar_maskrcnn(
        pretrained_path=str(args.model_path),
        return_load_info=True,
    )
    if info.missing or info.unexpected:
        print(f"[WARN] checkpoint load reported missing={len(info.missing)} "
              f"unexpected={len(info.unexpected)}")
    model.roi_heads.score_thresh = float(args.detector_score_threshold)
    model.roi_heads.nms_thresh = float(args.nms_iou_threshold)
    model.roi_heads.detections_per_img = int(args.detections_per_img)
    model.eval()
    model.to(device)

    # ── DataLoader ───────────────────────────────────────────────────
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
        collate_fn=list_collate,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # ── Inference loop ───────────────────────────────────────────────
    all_chips: list[Chip] = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    n_chips_done = 0
    n_detections = 0

    with torch.inference_mode():
        for batch_tensors, batch_metas in loader:
            tensors_on_device = [t.to(device, non_blocking=True) for t in batch_tensors]
            outputs = model(tensors_on_device)
            for chip_meta, out in zip(batch_metas, outputs):
                chip = _process_one_chip(
                    chip_meta=chip_meta,
                    out=out,
                    chip_size=args.chip_size,
                    raw_mask_storage=args.raw_mask_storage,
                    geoai_binary_mask_threshold=(
                        float(args.mask_threshold) if args.parity_mode == "geoai" else None
                    ),
                )
                all_chips.append(chip)
                n_detections += len(chip.detections)
            n_chips_done += len(batch_metas)

    elapsed = time.time() - t_start
    chips_per_sec = n_chips_done / elapsed if elapsed > 0 else 0.0
    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
    )
    print(f"[detect_direct] {n_chips_done} chips → {n_detections} detections "
          f"in {elapsed:.1f}s ({chips_per_sec:.2f} chips/s, peak VRAM {peak_vram_gb:.2f} GB)")

    # ── Build artifact ───────────────────────────────────────────────
    source_tiles = []
    for tm in dataset.tif_meta:
        sha = None
        if args.hash_tiles_content:
            sha = _file_sha256(tm["path"])
        source_tiles.append(SourceTile(
            path=tm["path"],
            size_bytes=tm["size_bytes"],
            mtime=tm["mtime"],
            crs=tm["crs"],
            transform=tm["transform"],
            bounds=tm["bounds"],
            shape=tm["shape"],
            sha256=sha,
        ))

    artifact = RawArtifact(
        schema_version=SCHEMA_VERSION,
        pipeline_version=PIPELINE_VERSION,
        created_at_utc=utc_now_iso(),
        git_commit=_git_commit(),
        script_sha256=SCRIPT_SHA256,
        torch_version=torch.__version__,
        torchvision_version=_torchvision_version(),
        rasterio_version=_rasterio_version(),
        grid_id=args.grid_id,
        region_arg=region_arg,
        region_key=region_key,
        imagery_layer_id=args.imagery_layer,
        model_run_id=args.model_run,
        model_path=str(args.model_path),
        model_sha256=_file_sha256(args.model_path),
        model_builder="core.models.build_solar_maskrcnn",
        detector_score_threshold=float(args.detector_score_threshold),
        detections_per_img=int(args.detections_per_img),
        nms_thresh=float(args.nms_iou_threshold),
        mask_threshold_used=float(args.mask_threshold),
        raw_mask_storage=args.raw_mask_storage,
        chip_size=(args.chip_size, args.chip_size),
        overlap=args.overlap,
        edge_pad=True,
        source_tiles=source_tiles,
        chips=all_chips,
    )
    write_artifact(artifact, raw_path)
    print(f"[detect_direct] wrote {raw_path}")

    if args.profile:
        print(f"[profile] chips/sec       : {chips_per_sec:.3f}")
        print(f"[profile] peak VRAM (GB)  : {peak_vram_gb:.3f}")
        print(f"[profile] total detections: {n_detections}")
        print(f"[profile] total wall time : {elapsed:.2f}s")

    dataset.close()
    return 0


def _process_one_chip(
    *,
    chip_meta,
    out: dict,
    chip_size: int,
    raw_mask_storage: str,
    geoai_binary_mask_threshold: float | None = None,
) -> Chip:
    """Convert one torchvision detection output dict into a `Chip` artifact."""
    boxes = out["boxes"].detach().cpu().numpy()       # [N, 4]
    scores = out["scores"].detach().cpu().numpy()     # [N]
    labels = out["labels"].detach().cpu().numpy()     # [N]
    masks_full = out["masks"].detach().cpu().numpy()  # [N, 1, H, W]

    detections: list[Detection] = []
    for i in range(boxes.shape[0]):
        box_chip = boxes[i]
        score = float(scores[i])
        label = int(labels[i])
        x1, y1, x2, y2 = _clip_box_to_int(box_chip, chip_size)

        # Soft mask is float in [0,1] at chip resolution.
        soft = masks_full[i, 0]  # [H, W]
        soft_uint8 = (np.clip(soft, 0.0, 1.0) * 255.0).astype(np.uint8)

        if raw_mask_storage == "full_chip":
            if geoai_binary_mask_threshold is None:
                mask_chip_uint8 = soft_uint8
            else:
                # Geoai generate_masks thresholds the float mask directly
                # before painting full-raster band 1. Storing the binary
                # full-chip mask here avoids uint8 soft-mask quantization drift
                # in --parity-mode geoai while leaving crop storage untouched.
                mask_chip_uint8 = (
                    (soft > float(geoai_binary_mask_threshold)).astype(np.uint8) * 255
                )
            # Crop to box for default storage too (always populated):
            mask_crop = soft_uint8[y1:y2, x1:x2].copy()
        else:
            mask_chip_uint8 = None
            mask_crop = soft_uint8[y1:y2, x1:x2].copy()

        det = Detection(
            box_chip_xyxy=(float(box_chip[0]), float(box_chip[1]),
                           float(box_chip[2]), float(box_chip[3])),
            box_source_xyxy=_box_chip_to_source(box_chip, chip_meta.window),
            score=score,
            label=label,
            mask_crop_uint8=mask_crop,
            mask_crop_offset=(x1, y1),
            mask_crop_shape=(int(y2 - y1), int(x2 - x1)),
            source_detection_index=i,
            mask_chip_uint8=mask_chip_uint8,
        )
        detections.append(det)

    return Chip(
        chip_index=chip_meta.chip_index,
        source_tif=chip_meta.source_tif,
        source_tile_id=chip_meta.source_tile_id,
        source_crs=chip_meta.source_crs,
        source_transform=chip_meta.source_transform,
        window=chip_meta.window,
        window_transform=chip_meta.window_transform,
        valid_window=chip_meta.valid_window,
        valid_shape=chip_meta.valid_shape,
        chip_shape=chip_meta.chip_shape,
        detections=detections,
    )


# ─────────────────────────────────────────────────────────────────────────
# Provenance helpers
# ─────────────────────────────────────────────────────────────────────────
def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _torchvision_version() -> str:
    try:
        import torchvision
        return torchvision.__version__
    except Exception:
        return ""


def _rasterio_version() -> str:
    try:
        import rasterio
        return rasterio.__version__
    except Exception:
        return ""


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
