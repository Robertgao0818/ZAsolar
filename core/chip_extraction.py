"""Tile-crop primitives for HN / COCO chip export.

This module collapses the ``crop window → clamp/pad → skip-blank → LZW-write``
logic and the ``resolve tile that contains a point`` logic that was duplicated
verbatim across the hard-negative / COCO export CLIs:

  - ``scripts/training/export_targeted_hn.py``  (reviewed-FP HN)
  - ``scripts/training/export_v4_hn.py``        (curated small-FP HN)
  - ``scripts/training/build_v4_3_hn.py``       (V4.3 multi-source HN)
  - ``pipeline/hn_ops.py``                       (negative-pool HN)
  - ``export_coco_dataset.py``                   (chip-grid COCO export — write only)

Scope
-----
This is the **detector-side tile-crop** helper used by HN and COCO dataset
export: it crops a fixed-size square chip from an aerial GeoTIFF (chunked or
mosaic layout) and writes it back as an LZW GeoTIFF. It is intentionally
distinct from the sibling subrepo ``solar_cls/...chip_extraction.py``, which
crops *classifier* chips with adaptive bounding boxes + margins around detection
polygons — a different domain. Do not merge the two; they share only a name.

The tile-resolution path here honours the imagery layer's ``file_layout``
(``chunked`` vs ``mosaic``) via ``core.region_registry`` /
``core.grid_utils.resolve_tiles_dir``. Three of the legacy ``find_tile``
copies globbed ``{grid}_{col}_{row}_geo.tif`` unconditionally, so for
``mosaic`` layers (vexcel_2024 / aerial_legacy, a single
``{grid}_mosaic.tif``) they matched nothing and **silently returned an empty
result — dropping entire HN batches** (rule 06-multi-city). ``region`` is a
required, explicit argument to ``resolve_tile_for_point``; the region is never
inferred from the grid ID.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

__all__ = [
    "point_xy_in_crs",
    "crop_chip",
    "write_chip_geotiff",
    "resolve_tile_for_point",
]


def point_xy_in_crs(lon: float, lat: float, dst_crs) -> tuple[float, float]:
    """Reproject an EPSG:4326 ``(lon, lat)`` point into ``dst_crs``.

    HN source geometries are stored in EPSG:4326, but tiles may be in any
    native CRS (vexcel_2024 / aerial_legacy are EPSG:3857). The point must be
    reprojected before it can be compared against tile bounds or fed to
    ``src.index`` — comparing lon/lat against metre-scale bounds silently
    resolves nothing (rule 06-multi-city: never assume EPSG).

    When ``dst_crs`` is None or already geographic EPSG:4326 (CT/JHB aerial
    layers) the transform is a no-op and the original lon/lat is returned
    unchanged, so callers operating on 4326 tiles get byte-identical pixel
    indices to the pre-refactor direct ``src.index(lon, lat)`` call.
    """
    if dst_crs is None:
        return lon, lat
    try:
        from rasterio.crs import CRS as _CRS
        if _CRS.from_epsg(4326) == dst_crs:
            return lon, lat
    except Exception:  # noqa: BLE001
        pass
    from rasterio.warp import transform as _warp_transform
    xs, ys = _warp_transform("EPSG:4326", dst_crs, [lon], [lat])
    return xs[0], ys[0]


def crop_chip(
    src: rasterio.DatasetReader,
    lon: float,
    lat: float,
    chip_size: int,
    *,
    skip_blank: bool = True,
):
    """Crop a ``chip_size`` square centred on an EPSG:4326 ``(lon, lat)`` point.

    Reproduces the shared HN crop logic exactly:
      1. reproject the point into the tile CRS (no-op for 4326 tiles),
      2. index pixel coords, centre a ``chip_size`` window, clamp to bounds,
      3. skip tiny edge chips (< 50% of ``chip_size`` in either dimension),
      4. read the window and zero-pad up to ``chip_size`` if clamped,
      5. when ``skip_blank``, drop chips that are entirely >= 245 (blank
         tile margin).

    Returns ``(data, window, x0, y0, w, h)`` on success, or ``None`` when the
    chip is a tiny edge chip or (with ``skip_blank``) blank. ``data`` is the
    (possibly padded) ``(bands, chip_size, chip_size)`` array; ``window`` is the
    unpadded read window (use ``src.window_transform(window)`` for the geo
    transform).
    """
    x_native, y_native = point_xy_in_crs(lon, lat, src.crs)
    py, px = src.index(x_native, y_native)

    x0 = max(0, int(px - chip_size // 2))
    y0 = max(0, int(py - chip_size // 2))
    x0 = min(x0, max(0, src.width - chip_size))
    y0 = min(y0, max(0, src.height - chip_size))

    w = min(chip_size, src.width - x0)
    h = min(chip_size, src.height - y0)

    if w < chip_size * 0.5 or h < chip_size * 0.5:
        return None  # tiny edge chip

    window = Window(x0, y0, w, h)
    data = src.read(window=window)

    if w < chip_size or h < chip_size:
        padded = np.zeros((data.shape[0], chip_size, chip_size), dtype=data.dtype)
        padded[:, :h, :w] = data
        data = padded

    if skip_blank and np.all(data >= 245):
        return None  # blank tile margin

    return data, window, x0, y0, w, h


def write_chip_geotiff(
    src: rasterio.DatasetReader,
    data: np.ndarray,
    window: Window,
    chip_path: Path | str,
    chip_size: int,
) -> None:
    """Write ``data`` as an LZW-compressed GeoTIFF chip.

    Copies the source profile, strips any source-side photometric / JPEG
    settings (so JPEG-source tiles don't poison the output), fixes width /
    height to ``chip_size``, takes the geo transform from ``window`` on ``src``,
    and writes with ``compress="lzw"``. This is the byte-for-byte write path
    shared by every HN/COCO chip exporter.
    """
    profile = src.profile.copy()
    for key in ("photometric", "compress", "jpeg_quality", "jpegtablesmode"):
        profile.pop(key, None)
    profile.update(
        driver="GTiff",
        width=chip_size,
        height=chip_size,
        transform=src.window_transform(window),
        compress="lzw",
    )
    with rasterio.open(str(chip_path), "w", **profile) as dst:
        dst.write(data)


def resolve_tile_for_point(
    lon: float,
    lat: float,
    grid_id: str,
    *,
    region: str | None = None,
    imagery_layer: str | None = None,
    tiles_root: Path | None = None,
) -> Path | None:
    """Return the GeoTIFF that contains an EPSG:4326 ``(lon, lat)`` point.

    Branches on the imagery layer's ``file_layout`` (rule 06-multi-city):

    - ``mosaic`` layers (vexcel_2024 / aerial_legacy: a single
      ``{grid}_mosaic.tif``) resolve directly to that one file. The legacy
      ``find_tile`` copies that globbed ``{grid}_{col}_{row}_geo.tif`` matched
      nothing here and silently dropped the whole batch — this branch fixes
      that defect.
    - ``chunked`` layers resolve to a directory of geo chunks; the chunk whose
      native bounds contain the point is returned.

    ``region`` must be passed explicitly (or left None to let
    ``resolve_tiles_dir`` look it up); region is never inferred from the grid
    ID naming pattern. ``tiles_root`` overrides the registry (e.g. RunPod
    ``/dev/shm``) and may point either at the grid subdir or one level above it.

    The point is in EPSG:4326; each candidate chunk's bounds are in that
    chunk's native CRS, so the point is reprojected per-chunk via ``src.crs``
    before the contains-check (do not assume lon/lat == tile units).
    """
    from core.grid_utils import resolve_tiles_dir

    if tiles_root is not None:
        base: Path | None = Path(tiles_root)
    else:
        try:
            base = resolve_tiles_dir(grid_id, region=region,
                                     imagery_layer=imagery_layer)
        except Exception:  # noqa: BLE001
            return None

    if base is None:
        return None
    base = Path(base)

    # Mosaic layout: resolve_tiles_dir returns the single mosaic file directly.
    if base.is_file():
        return base

    # tiles_root override points at (or one level above) the grid subdir. The
    # legacy chunked find_tile copies treated `tiles_root/<grid_id>/` as the
    # chip dir, so descend into it when present; also tolerate a mosaic file or
    # the override already being the grid dir.
    if tiles_root is not None and base.is_dir():
        cand = base / grid_id
        if cand.is_dir():
            base = cand
        else:
            mosaic = base / f"{grid_id}_mosaic.tif"
            if mosaic.is_file():
                return mosaic

    if not base.is_dir():
        return None

    # Chunked layout: pick the chunk whose native bounds contain the point.
    for tif in sorted(base.glob(f"{grid_id}_*_*_geo.tif")):
        try:
            with rasterio.open(tif) as src:
                left, bottom, right, top = src.bounds
                x_native, y_native = point_xy_in_crs(lon, lat, src.crs)
                if left <= x_native <= right and bottom <= y_native <= top:
                    return tif
        except Exception:  # noqa: BLE001
            continue
    return None
