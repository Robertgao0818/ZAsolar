"""One-pass pregate rescue. No model, no raster reads.

A chip the presence pregate rejected is a rescue candidate when a detection
from a chip that the gate kept, on the same source tif, has score at least
``min_score`` and its box overlaps the rejected chip's window. The box is
shifted by that kept chip's own window offset. Windows are whatever the
chip metadata recorded. Nothing here assumes a regular stride.

Only the original kept detections are triggers. A rescued chip cannot
nominate another chip.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def rescue_candidates(
    chip_windows: Sequence[Mapping],
    kept_dets: Sequence[Mapping],
    rejected: Iterable[int],
    min_score: float,
) -> set[int]:
    """Return chip indexes that should be sent back to the detector.

    ``chip_windows`` items have ``chip_index``, ``tif_index``, and
    ``window=(col_off, row_off, width, height)`` in that tif's pixels.
    ``kept_dets`` items have ``chip_index``, ``score``, and
    ``box_chip_xyxy=(x1, y1, x2, y2)`` in the kept chip's pixels.
    ``rejected`` is the chip indexes the pregate refused.
    Overlap must have positive area. Touching an edge is not enough.
    """
    threshold = float(min_score)
    by_chip: dict[int, Mapping] = {}
    for item in chip_windows:
        by_chip[int(item["chip_index"])] = item

    # Tile-pixel boxes, grouped by source tif, from kept chips only.
    boxes_by_tif: dict[object, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for det in kept_dets:
        score = float(det["score"])
        if score < threshold:
            continue
        chip_index = int(det["chip_index"])
        chip = by_chip.get(chip_index)
        if chip is None:
            continue
        boxes_by_tif.setdefault(chip["tif_index"], []).append(
            (chip_index, _box_to_tile(det["box_chip_xyxy"], chip["window"]))
        )

    chosen: set[int] = set()
    for raw_index in rejected:
        chip_index = int(raw_index)
        chip = by_chip.get(chip_index)
        if chip is None:
            continue
        rect = _window_rect(chip["window"])
        for source_index, tile_box in boxes_by_tif.get(chip["tif_index"], []):
            if source_index == chip_index:
                continue
            if _positive_area_overlap(tile_box, rect):
                chosen.add(chip_index)
                break
    return chosen


def _window_rect(window: Sequence[float]) -> tuple[float, float, float, float]:
    col, row, width, height = (float(value) for value in window)
    return (col, row, col + width, row + height)


def _box_to_tile(
    box_chip_xyxy: Sequence[float], window: Sequence[float],
) -> tuple[float, float, float, float]:
    col_off, row_off = float(window[0]), float(window[1])
    x1, y1, x2, y2 = (float(value) for value in box_chip_xyxy)
    return (x1 + col_off, y1 + row_off, x2 + col_off, y2 + row_off)


def _positive_area_overlap(
    left: Sequence[float], right: Sequence[float],
) -> bool:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    return (x2 - x1) > 0.0 and (y2 - y1) > 0.0
