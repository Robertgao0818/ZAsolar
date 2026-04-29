"""Raw detection artifact schema (V1 / direct_maskrcnn_v1) + read/write API.

This is the on-disk contract between `detect_direct.py` and downstream
consumers (`finalize.py`, the SAM-refine adapter in Phase 1.5, any
re-postprocessing scripts in Phase 2).

V1 storage = pickle (protocol=5). The reader/writer are thin so a future
migration to parquet + .npz mask files won't break callers.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
PIPELINE_VERSION = "direct_maskrcnn_v1"


@dataclass
class Detection:
    """One Mask R-CNN detection inside one chip."""
    box_chip_xyxy: tuple[float, float, float, float]
    box_source_xyxy: tuple[float, float, float, float]
    score: float
    label: int
    # Default storage = cropped uint8 mask (V1.4 decision #13).
    # The crop is at integer pixel offset within the chip (decision #18).
    mask_crop_uint8: np.ndarray   # shape (h, w), values 0-255 soft mask
    mask_crop_offset: tuple[int, int]   # (x, y) in chip pixel space, integer
    mask_crop_shape: tuple[int, int]    # (h, w) integer
    source_detection_index: int          # original index in raw model output
    # Optional debug field; None when --raw-mask-storage crop:
    mask_chip_uint8: np.ndarray | None = None


@dataclass
class Chip:
    """One sliding-window chip."""
    chip_index: int
    source_tif: str                # absolute path
    source_tile_id: str            # filename stem
    source_crs: str
    source_transform: tuple        # affine (a, b, c, d, e, f)
    window: tuple[int, int, int, int]    # (col_off, row_off, w, h) in source TIF
    window_transform: tuple        # affine for the window
    valid_window: tuple[int, int, int, int]   # sub-rect with real raster data
    valid_shape: tuple[int, int]   # (h, w)
    chip_shape: tuple[int, int]    # always (chip_size, chip_size)
    detections: list[Detection] = field(default_factory=list)


@dataclass
class SourceTile:
    """Provenance metadata for one source TIF."""
    path: str
    size_bytes: int
    mtime: float
    crs: str
    transform: tuple
    bounds: tuple[float, float, float, float]
    shape: tuple[int, int]
    sha256: str | None = None      # only when --hash-tiles-content


@dataclass
class RawArtifact:
    """Top-level artifact written by detect_direct.py."""
    schema_version: int
    pipeline_version: str
    created_at_utc: str
    git_commit: str
    script_sha256: str
    torch_version: str
    torchvision_version: str
    rasterio_version: str
    grid_id: str
    region_arg: str            # CLI value, e.g. "jhb"
    region_key: str            # canonical, e.g. "johannesburg"
    imagery_layer_id: str
    model_run_id: str
    model_path: str
    model_sha256: str
    model_builder: str
    detector_score_threshold: float
    detections_per_img: int
    nms_thresh: float
    mask_threshold_used: float
    raw_mask_storage: str       # "crop" or "full_chip"
    chip_size: tuple[int, int]
    overlap: float
    edge_pad: bool
    source_tiles: list[SourceTile] = field(default_factory=list)
    chips: list[Chip] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────
# Read / write
# ─────────────────────────────────────────────────────────────────────────
class SchemaVersionError(ValueError):
    """Raised when reading an artifact with an incompatible schema version."""


def write_artifact(artifact: RawArtifact, path: str | Path) -> None:
    """Pickle the artifact at `path` (protocol=5)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_payload(artifact)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=5)


def read_artifact(path: str | Path) -> RawArtifact:
    """Read an artifact written by `write_artifact`. Raises
    `SchemaVersionError` if the schema version doesn't match this code."""
    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "schema_version" not in payload:
        raise SchemaVersionError(f"not a raw-detections artifact: {path}")
    sv = payload["schema_version"]
    if sv != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"artifact schema_version={sv}, code expects {SCHEMA_VERSION} "
            f"(file: {path})"
        )
    return _from_payload(payload)


def _to_payload(a: RawArtifact) -> dict[str, Any]:
    """Convert RawArtifact → plain dict-of-dicts for pickling."""
    return {
        "schema_version": a.schema_version,
        "pipeline_version": a.pipeline_version,
        "created_at_utc": a.created_at_utc,
        "git_commit": a.git_commit,
        "script_sha256": a.script_sha256,
        "torch_version": a.torch_version,
        "torchvision_version": a.torchvision_version,
        "rasterio_version": a.rasterio_version,
        "grid_id": a.grid_id,
        "region_arg": a.region_arg,
        "region_key": a.region_key,
        "imagery_layer_id": a.imagery_layer_id,
        "model_run_id": a.model_run_id,
        "model_path": a.model_path,
        "model_sha256": a.model_sha256,
        "model_builder": a.model_builder,
        "detector_score_threshold": a.detector_score_threshold,
        "detections_per_img": a.detections_per_img,
        "nms_thresh": a.nms_thresh,
        "mask_threshold_used": a.mask_threshold_used,
        "raw_mask_storage": a.raw_mask_storage,
        "chip_size": tuple(a.chip_size),
        "overlap": a.overlap,
        "edge_pad": a.edge_pad,
        "source_tiles": [asdict(s) for s in a.source_tiles],
        "chips": [
            {
                "chip_index": c.chip_index,
                "source_tif": c.source_tif,
                "source_tile_id": c.source_tile_id,
                "source_crs": c.source_crs,
                "source_transform": tuple(c.source_transform),
                "window": tuple(c.window),
                "window_transform": tuple(c.window_transform),
                "valid_window": tuple(c.valid_window),
                "valid_shape": tuple(c.valid_shape),
                "chip_shape": tuple(c.chip_shape),
                "detections": [
                    {
                        "box_chip_xyxy": tuple(d.box_chip_xyxy),
                        "box_source_xyxy": tuple(d.box_source_xyxy),
                        "score": float(d.score),
                        "label": int(d.label),
                        "mask_crop_uint8": d.mask_crop_uint8,
                        "mask_crop_offset": tuple(d.mask_crop_offset),
                        "mask_crop_shape": tuple(d.mask_crop_shape),
                        "source_detection_index": int(d.source_detection_index),
                        "mask_chip_uint8": d.mask_chip_uint8,
                    }
                    for d in c.detections
                ],
            }
            for c in a.chips
        ],
    }


def _from_payload(p: dict[str, Any]) -> RawArtifact:
    return RawArtifact(
        schema_version=p["schema_version"],
        pipeline_version=p["pipeline_version"],
        created_at_utc=p["created_at_utc"],
        git_commit=p["git_commit"],
        script_sha256=p["script_sha256"],
        torch_version=p["torch_version"],
        torchvision_version=p["torchvision_version"],
        rasterio_version=p["rasterio_version"],
        grid_id=p["grid_id"],
        region_arg=p["region_arg"],
        region_key=p["region_key"],
        imagery_layer_id=p["imagery_layer_id"],
        model_run_id=p["model_run_id"],
        model_path=p["model_path"],
        model_sha256=p["model_sha256"],
        model_builder=p["model_builder"],
        detector_score_threshold=p["detector_score_threshold"],
        detections_per_img=p["detections_per_img"],
        nms_thresh=p["nms_thresh"],
        mask_threshold_used=p["mask_threshold_used"],
        raw_mask_storage=p["raw_mask_storage"],
        chip_size=tuple(p["chip_size"]),
        overlap=p["overlap"],
        edge_pad=p["edge_pad"],
        source_tiles=[SourceTile(**s) for s in p["source_tiles"]],
        chips=[
            Chip(
                chip_index=c["chip_index"],
                source_tif=c["source_tif"],
                source_tile_id=c["source_tile_id"],
                source_crs=c["source_crs"],
                source_transform=tuple(c["source_transform"]),
                window=tuple(c["window"]),
                window_transform=tuple(c["window_transform"]),
                valid_window=tuple(c["valid_window"]),
                valid_shape=tuple(c["valid_shape"]),
                chip_shape=tuple(c["chip_shape"]),
                detections=[Detection(**d) for d in c["detections"]],
            )
            for c in p["chips"]
        ],
    )


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 string (for `created_at_utc`)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
