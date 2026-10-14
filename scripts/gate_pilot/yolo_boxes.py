"""Pixel boxes for the single-class YOLO gate.

A box is the axis-aligned bounds of the polygon after it has been clipped to
the chip's valid footprint, converted into the chip image. Image y grows
downward. Width or height below 2 pixels is expanded to 2 and kept. The
normalized coordinates are relative to the full chip image, including any
black padding.
"""

from __future__ import annotations


def _expand_and_clamp(start: float, end: float, minimum: float, limit: float) -> tuple[float, float]:
    if end < start:
        start, end = end, start
    if end - start < minimum:
        center = (start + end) / 2.0
        start = center - minimum / 2.0
        end = center + minimum / 2.0
    if start < 0:
        end -= start
        start = 0.0
    if end > limit:
        start -= end - limit
        end = limit
    start = min(max(start, 0.0), limit)
    end = min(max(end, 0.0), limit)
    return start, end


def yolo_box_from_geo_bounds(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    *,
    chip_minx: float,
    chip_miny: float,
    chip_maxx: float,
    chip_maxy: float,
    valid_w: int,
    valid_h: int,
    chip_size: int = 400,
    min_pixels: float = 2.0,
) -> tuple[float, float, float, float] | None:
    """Return YOLO ``cx, cy, w, h`` normalized to ``chip_size``, or None if empty."""
    if valid_w <= 0 or valid_h <= 0 or chip_size <= 0:
        raise ValueError("chip dimensions must be positive")
    geo_w = chip_maxx - chip_minx
    geo_h = chip_maxy - chip_miny
    if geo_w <= 0 or geo_h <= 0:
        raise ValueError("chip geographic extent is empty")
    if maxx <= minx or maxy <= miny:
        return None
    x1 = (minx - chip_minx) / geo_w * valid_w
    x2 = (maxx - chip_minx) / geo_w * valid_w
    y1 = (chip_maxy - maxy) / geo_h * valid_h
    y2 = (chip_maxy - miny) / geo_h * valid_h
    x1, x2 = _expand_and_clamp(x1, x2, min_pixels, float(chip_size))
    y1, y2 = _expand_and_clamp(y1, y2, min_pixels, float(chip_size))
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        ((x1 + x2) / 2.0) / chip_size,
        ((y1 + y2) / 2.0) / chip_size,
        (x2 - x1) / chip_size,
        (y2 - y1) / chip_size,
    )
