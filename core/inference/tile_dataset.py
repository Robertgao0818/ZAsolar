"""Sliding-window dataset for direct Mask R-CNN inference.

Handles both file layouts in the project:

  - **chunked**: each TIF is one chunk like ``G1234_0_0_geo.tif``; we slide
    windows within each chunk independently. Many chunks per grid.
  - **mosaic**: one giant TIF like ``G0816_mosaic.tif``; we slide windows
    across it with stride. One TIF per grid.

Yields ``(chip_tensor[3, chip_size, chip_size], ChipMeta)``. Chip pixels are
``float32`` in ``[0, 1]``. **No ImageNet normalization**: torchvision's
``GeneralizedRCNNTransform`` (configured via ``image_mean`` / ``image_std`` in
``build_solar_maskrcnn``) does that. Edge windows are zero-padded so the
output shape is stable; ``ChipMeta.valid_window`` records the sub-rect with
real raster data.

GDAL is not fork-safe — use ``worker_init_fn`` to open per-worker handles
when wrapping in ``DataLoader``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset


@dataclass
class ChipMeta:
    """Metadata for one sliding-window chip."""
    chip_index: int
    source_tif: str
    source_tile_id: str
    source_crs: str
    source_transform: tuple
    window: tuple[int, int, int, int]   # (col_off, row_off, w, h) in source TIF
    window_transform: tuple              # affine of the chip (after padding)
    valid_window: tuple[int, int, int, int]   # sub-rect with real raster data
    valid_shape: tuple[int, int]         # (h, w) of valid_window
    chip_shape: tuple[int, int]          # always (chip_size, chip_size)


class SlidingWindowDataset(Dataset):
    """Iterate sliding windows across one or more source TIFs.

    Args:
        tif_paths: source TIFs to iterate (chunked: one per chunk; mosaic: one).
        chip_size: width / height of each chip in pixels (square).
        overlap: fractional overlap, e.g. 0.25 → stride = chip_size * 0.75.
        edge_pad: when True (default), edge windows are zero-padded so every
            yielded chip is exactly chip_size × chip_size.
        bands: source bands to read (default first 3 = RGB).
        max_chips: optional cap (smoke-test).
    """

    def __init__(
        self,
        tif_paths: Sequence[str | Path],
        *,
        chip_size: int = 400,
        overlap: float = 0.25,
        edge_pad: bool = True,
        bands: Sequence[int] = (1, 2, 3),
        max_chips: int | None = None,
    ) -> None:
        if not tif_paths:
            raise ValueError("tile_paths is empty")
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap must be in [0, 1): got {overlap}")
        if chip_size <= 0:
            raise ValueError(f"chip_size must be > 0: got {chip_size}")

        self.tif_paths = [Path(p) for p in tif_paths]
        self.chip_size = int(chip_size)
        self.overlap = float(overlap)
        self.edge_pad = bool(edge_pad)
        self.bands = tuple(int(b) for b in bands)
        if len(self.bands) != 3:
            raise ValueError(f"bands must specify exactly 3 RGB bands; got {self.bands}")

        # Pre-compute all (tif_index, window) pairs.
        self._items: list[tuple[int, Window]] = []
        # Per-TIF metadata cached at construction (cheap; one open).
        self._tif_meta: list[dict] = []
        stride = max(1, int(round(self.chip_size * (1.0 - self.overlap))))
        self._stride = stride

        for ti, p in enumerate(self.tif_paths):
            with rasterio.open(p) as src:
                self._tif_meta.append({
                    "path": str(p),
                    "tile_id": p.stem,
                    "crs": str(src.crs) if src.crs else "",
                    "transform": tuple(src.transform)[:6],
                    "width": int(src.width),
                    "height": int(src.height),
                    "size_bytes": p.stat().st_size,
                    "mtime": p.stat().st_mtime,
                    "bounds": tuple(src.bounds),
                    "shape": (int(src.height), int(src.width)),
                })

            for col_off, row_off in self._iter_window_origins(
                self._tif_meta[ti]["width"],
                self._tif_meta[ti]["height"],
            ):
                w = min(self.chip_size, self._tif_meta[ti]["width"] - col_off)
                h = min(self.chip_size, self._tif_meta[ti]["height"] - row_off)
                if w <= 0 or h <= 0:
                    continue
                self._items.append((ti, Window(col_off, row_off, w, h)))

                if max_chips is not None and len(self._items) >= max_chips:
                    break
            if max_chips is not None and len(self._items) >= max_chips:
                break

        # Per-worker rasterio handles, opened lazily in __getitem__.
        self._dataset_handles: dict[int, rasterio.io.DatasetReader] = {}

    # ── public ────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ChipMeta]:
        ti, win = self._items[index]
        meta = self._tif_meta[ti]
        src = self._open(ti)
        # Read first 3 bands inside the (possibly partial) window.
        # When edge_pad: pad to chip_size.
        chip_w = int(win.width)
        chip_h = int(win.height)

        arr = src.read(self.bands, window=win)  # shape (3, chip_h, chip_w)
        if arr.dtype != np.float32:
            # Normalize to [0, 1]. uint8 → /255; uint16 → /65535; float passthrough.
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            elif arr.dtype == np.uint16:
                arr = arr.astype(np.float32) / 65535.0
            else:
                arr = arr.astype(np.float32)

        if self.edge_pad and (chip_h < self.chip_size or chip_w < self.chip_size):
            padded = np.zeros((3, self.chip_size, self.chip_size), dtype=np.float32)
            padded[:, :chip_h, :chip_w] = arr
            tensor = torch.from_numpy(padded)
            chip_shape = (self.chip_size, self.chip_size)
        else:
            tensor = torch.from_numpy(arr)
            chip_shape = (chip_h, chip_w)

        # Build ChipMeta. window_transform reflects the source-TIF affine for
        # the chip's top-left; rasterio computes this for us.
        win_tr = src.window_transform(win)
        chip_meta = ChipMeta(
            chip_index=index,
            source_tif=meta["path"],
            source_tile_id=meta["tile_id"],
            source_crs=meta["crs"],
            source_transform=meta["transform"],
            window=(int(win.col_off), int(win.row_off), self.chip_size if self.edge_pad else chip_w, self.chip_size if self.edge_pad else chip_h),
            window_transform=tuple(win_tr)[:6],
            valid_window=(int(win.col_off), int(win.row_off), chip_w, chip_h),
            valid_shape=(chip_h, chip_w),
            chip_shape=chip_shape,
        )
        return tensor, chip_meta

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def tif_meta(self) -> list[dict]:
        """Per-TIF metadata captured at construction (immutable copy)."""
        return [dict(m) for m in self._tif_meta]

    # ── internals ─────────────────────────────────────────────────────
    def _iter_window_origins(self, width: int, height: int):
        """Yield (col_off, row_off) covering the raster.

        Stride = ``chip_size * (1 - overlap)``. The final row/column are
        anchored at ``width - chip_size`` / ``height - chip_size`` so the
        rightmost / bottommost pixels are always covered (when the raster
        is at least ``chip_size`` wide / tall). When the raster is smaller
        than ``chip_size``, a single window starting at (0, 0) is yielded.
        """
        stride = self._stride
        cs = self.chip_size

        if width <= cs:
            col_offsets = [0]
        else:
            col_offsets = list(range(0, width - cs + 1, stride))
            if col_offsets[-1] + cs < width:
                col_offsets.append(width - cs)

        if height <= cs:
            row_offsets = [0]
        else:
            row_offsets = list(range(0, height - cs + 1, stride))
            if row_offsets[-1] + cs < height:
                row_offsets.append(height - cs)

        for r in row_offsets:
            for c in col_offsets:
                yield c, r

    def _open(self, ti: int) -> rasterio.io.DatasetReader:
        """Per-worker lazy open. Handles are not shared across processes
        because rasterio/GDAL is not fork-safe."""
        h = self._dataset_handles.get(ti)
        if h is None:
            h = rasterio.open(self.tif_paths[ti])
            self._dataset_handles[ti] = h
        return h

    def close(self) -> None:
        for h in self._dataset_handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._dataset_handles.clear()


# ─────────────────────────────────────────────────────────────────────────
# DataLoader helpers
# ─────────────────────────────────────────────────────────────────────────
def list_collate(batch):
    """Collate that yields ``(list[Tensor], list[ChipMeta])``.

    torchvision detection models take ``list[Tensor]``, not stacked
    ``Tensor[B, C, H, W]``, so we don't ``torch.stack``. Stable shapes from
    ``edge_pad=True`` are preserved by passing tensors through as-is.
    """
    tensors = [b[0] for b in batch]
    metas = [b[1] for b in batch]
    return tensors, metas


def worker_init_fn(_worker_id: int) -> None:
    """Drop any inherited rasterio handles in the worker."""
    # Each worker calls __getitem__ which opens its own per-TIF handle.
    # No shared state to set up here, but the hook is exposed so callers
    # can wire in additional GDAL config (e.g. CPL_VSIL_CURL_ALLOWED_EXTENSIONS).
    return None
