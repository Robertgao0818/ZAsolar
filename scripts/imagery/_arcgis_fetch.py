"""ArcGIS ImageServer chunk fetcher with sub-request stitching.

Each grid (1km × 1km in EPSG:3857 at admin_grid centroid) is split into
2×2 chunks of 700m × 700m. Each 7000-px chunk is too big for a single
exportImage call (server cap maxImage=4100), so it's built from 2×2
sub-requests of 3500px (each covers 350m × 350m), then stitched in
memory and written as a single JPEG-compressed tiled GeoTIFF.

Supports `verify_ssl=False` for eThekwini's self-signed cert.
"""

from __future__ import annotations

import io
import math
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.transform import from_bounds


# ──────────────────────────────────────────────────────────────────────
# CRS helpers (EPSG:4326 ⇄ EPSG:3857 web mercator)
# ──────────────────────────────────────────────────────────────────────
_R = 6378137.0


def lonlat_to_3857(lon: float, lat: float) -> tuple[float, float]:
    x = lon * math.pi / 180.0 * _R
    y = math.log(math.tan(math.pi / 4 + lat * math.pi / 360.0)) * _R
    return x, y


def grid_chunks_3857(centroid_lon: float, centroid_lat: float,
                     grid_size_m: float, chunks_per_side: int,
                     ) -> list[dict]:
    """Generate chunk bboxes in EPSG:3857 covering a `grid_size_m` square
    centered on (centroid_lon, centroid_lat). Returns a list of dicts:
        {col, row, xmin, ymin, xmax, ymax, width_m, height_m}
    """
    cx, cy = lonlat_to_3857(centroid_lon, centroid_lat)
    half = grid_size_m / 2.0
    xmin_grid, ymin_grid = cx - half, cy - half
    chunk_m = grid_size_m / chunks_per_side
    chunks: list[dict] = []
    for r in range(chunks_per_side):
        for c in range(chunks_per_side):
            xmin = xmin_grid + c * chunk_m
            ymax = ymin_grid + (chunks_per_side - r) * chunk_m  # rows from top
            ymin = ymax - chunk_m
            xmax = xmin + chunk_m
            chunks.append({"col": c, "row": r,
                           "xmin": xmin, "ymin": ymin,
                           "xmax": xmax, "ymax": ymax,
                           "width_m": chunk_m, "height_m": chunk_m})
    return chunks


# ──────────────────────────────────────────────────────────────────────
# Sub-request fetch + chunk stitch
# ──────────────────────────────────────────────────────────────────────
@dataclass
class FetchResult:
    ok: bool
    out_path: Path | None
    bytes_in: int
    bytes_out: int
    elapsed_s: float
    sub_requests: int
    reason: str = ""


def _exportImage_url(base_url: str, bbox: tuple[float, float, float, float],
                     size: tuple[int, int], output_sr: int) -> str:
    p = {
        "bbox": ",".join(f"{v:.3f}" for v in bbox),
        "bboxSR": str(output_sr),
        "imageSR": str(output_sr),
        "size": f"{size[0]},{size[1]}",
        "format": "tiff",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    return f"{base_url.rstrip('/')}/exportImage?{urllib.parse.urlencode(p)}"


def fetch_arcgis_chunk(
    *,
    base_url: str,
    chunk: dict,
    chunk_px: int,
    sub_max_px: int,
    output_sr: int,
    out_path: Path,
    verify_ssl: bool = True,
    timeout: int = 180,
    nodata_size_threshold: int = 50_000,
) -> FetchResult:
    """Fetch one chunk by splitting into NxN sub-requests as needed,
    stitching results, and writing as JPEG95 tiled GeoTIFF.

    Returns FetchResult with reason="empty_response" if all sub-tiles are
    empty (likely caller should attempt fallback source).
    """
    n_sub = max(1, math.ceil(chunk_px / sub_max_px))
    sub_px = chunk_px // n_sub
    sub_w_m = chunk["width_m"] / n_sub
    sub_h_m = chunk["height_m"] / n_sub

    canvas: np.ndarray | None = None
    bands_seen: int | None = None
    bytes_in = 0
    sub_count = 0
    t0 = time.perf_counter()
    empties = 0

    for sr in range(n_sub):
        for sc in range(n_sub):
            xmin = chunk["xmin"] + sc * sub_w_m
            ymin = chunk["ymin"] + (n_sub - sr - 1) * sub_h_m
            xmax = xmin + sub_w_m
            ymax = ymin + sub_h_m
            url = _exportImage_url(base_url, (xmin, ymin, xmax, ymax),
                                   (sub_px, sub_px), output_sr)
            try:
                r = requests.get(url, verify=verify_ssl, timeout=timeout)
            except Exception as e:
                return FetchResult(False, None, bytes_in, 0,
                                   time.perf_counter() - t0, sub_count,
                                   reason=f"http_exc:{e!s}[:120]")
            sub_count += 1
            bytes_in += len(r.content)

            if r.status_code != 200 or "image/tiff" not in r.headers.get("Content-Type", ""):
                return FetchResult(False, None, bytes_in, 0,
                                   time.perf_counter() - t0, sub_count,
                                   reason=f"bad_response:status={r.status_code}")

            if len(r.content) < 5000:
                empties += 1
                # Treat as empty/nodata sub-tile; fill canvas with zeros for now
                arr = np.zeros((1, sub_px, sub_px), dtype=np.uint8)
                bands = 1
            else:
                with rasterio.MemoryFile(r.content) as mem:
                    with mem.open() as src:
                        arr = src.read()  # (bands, H, W)
                        bands = src.count

            if canvas is None:
                bands_seen = bands
                # Use bands of first non-empty tile if possible
                canvas = np.zeros((bands, chunk_px, chunk_px), dtype=arr.dtype)
            elif bands != bands_seen and bands > bands_seen:
                # promote canvas to higher band count
                pad = bands - bands_seen
                canvas = np.concatenate(
                    [canvas, np.zeros((pad, chunk_px, chunk_px), dtype=canvas.dtype)],
                    axis=0)
                bands_seen = bands

            if arr.shape[0] < bands_seen:
                pad = bands_seen - arr.shape[0]
                arr = np.concatenate(
                    [arr, np.zeros((pad, sub_px, sub_px), dtype=arr.dtype)],
                    axis=0)

            row_off = sr * sub_px
            col_off = sc * sub_px
            canvas[:, row_off:row_off + sub_px, col_off:col_off + sub_px] = arr[:bands_seen]

    if empties == n_sub * n_sub:
        return FetchResult(False, None, bytes_in, 0,
                           time.perf_counter() - t0, sub_count,
                           reason="empty_response")

    # Write JPEG95 tiled GeoTIFF
    transform = from_bounds(chunk["xmin"], chunk["ymin"],
                            chunk["xmax"], chunk["ymax"],
                            chunk_px, chunk_px)
    profile = {
        "driver": "GTiff",
        "height": chunk_px, "width": chunk_px,
        "count": bands_seen, "dtype": "uint8",
        "crs": f"EPSG:{output_sr}",
        "transform": transform,
        "tiled": True, "blockxsize": 512, "blockysize": 512,
        "compress": "JPEG", "jpeg_quality": 95,
    }
    if bands_seen >= 3:
        profile["photometric"] = "YCBCR"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(canvas[:bands_seen])

    return FetchResult(True, out_path, bytes_in, out_path.stat().st_size,
                       time.perf_counter() - t0, sub_count)
