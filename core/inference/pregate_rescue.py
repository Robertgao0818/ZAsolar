"""One-pass pregate rescue. No model, no raster reads.

A chip the presence pregate rejected is a rescue candidate when a detection
from a chip that the gate kept, on the same source tif, has score at least
``min_score`` and its box overlaps the rejected chip's window. The box is
shifted by that kept chip's own window offset. Windows are whatever the
chip metadata recorded. Nothing here assumes a regular stride.

``require_edge`` adds a second condition and is off by default. The kept
box must also meet that chip's valid-window edge, within ``edge_px``, on a
side the rejected window extends past. A box that only sits in the stride
overlap, and was fully inside the kept chip, does not qualify.

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
    *,
    require_edge: bool = False,
    edge_px: int = 2,
) -> set[int]:
    """Return chip indexes that should be sent back to the detector.

    ``chip_windows`` items have ``chip_index``, ``tif_index``, and
    ``window=(col_off, row_off, width, height)`` in that tif's pixels.
    Optional ``valid_window`` is the same shape. The edge test uses it when
    it is present, and ``window`` otherwise.
    ``kept_dets`` items have ``chip_index``, ``score``, and
    ``box_chip_xyxy=(x1, y1, x2, y2)`` in the kept chip's pixels.
    ``rejected`` is the chip indexes the pregate refused.
    Overlap must have positive area. Touching an edge is not enough.

    With the defaults, ``require_edge=False``, the edge arguments are
    ignored and a positive-area overlap is the whole trigger.
    """
    threshold = float(min_score)
    tolerance = int(edge_px)
    by_chip: dict[int, Mapping] = {}
    for item in chip_windows:
        by_chip[int(item["chip_index"])] = item

    # Tile-pixel boxes, grouped by source tif, from kept chips only.
    # The chip-pixel box is kept alongside for the optional edge test.
    boxes_by_tif: dict[object, list[tuple]] = {}
    for det in kept_dets:
        score = float(det["score"])
        if score < threshold:
            continue
        chip_index = int(det["chip_index"])
        chip = by_chip.get(chip_index)
        if chip is None:
            continue
        chip_box = tuple(float(value) for value in det["box_chip_xyxy"])
        boxes_by_tif.setdefault(chip["tif_index"], []).append(
            (chip_index, _box_to_tile(chip_box, chip["window"]), chip_box)
        )

    chosen: set[int] = set()
    for raw_index in rejected:
        chip_index = int(raw_index)
        chip = by_chip.get(chip_index)
        if chip is None:
            continue
        rect = _window_rect(chip["window"])
        for source_index, tile_box, chip_box in boxes_by_tif.get(chip["tif_index"], []):
            if source_index == chip_index:
                continue
            if not _positive_area_overlap(tile_box, rect):
                continue
            if require_edge and not _cut_toward_rejected(
                chip_box, by_chip[source_index], chip, tolerance,
            ):
                continue
            chosen.add(chip_index)
            break
    return chosen


def _cut_toward_rejected(
    box_chip: Sequence[float],
    kept: Mapping,
    rejected: Mapping,
    edge_px: int,
) -> bool:
    """True when the kept box meets the valid edge that faces ``rejected``.

    Facing means the rejected window extends strictly past the kept edge
    rect on that side. The edge rect is ``valid_window`` when the kept chip
    has one, else ``window``. The box is in kept-chip pixels, whose origin
    is ``window``, so an inset valid rect is shifted by the window origin.
    A side matches only when the box interval meets that edge within
    ``edge_px`` pixels. A box that merely reaches past the edge from far
    away does not.
    """
    window = kept["window"]
    valid = kept.get("valid_window") or window
    origin_x = float(window[0])
    origin_y = float(window[1])
    left = float(valid[0]) - origin_x
    top = float(valid[1]) - origin_y
    right = left + float(valid[2])
    bottom = top + float(valid[3])
    x1, y1, x2, y2 = (float(value) for value in box_chip)
    tol = float(edge_px)

    rejected_window = rejected["window"]
    rej_col = float(rejected_window[0])
    rej_row = float(rejected_window[1])
    rej_right = rej_col + float(rejected_window[2])
    rej_bottom = rej_row + float(rejected_window[3])
    kept_col = float(valid[0])
    kept_row = float(valid[1])
    kept_right = kept_col + float(valid[2])
    kept_bottom = kept_row + float(valid[3])

    if rej_right > kept_right and x2 >= right - tol and x1 <= right + tol:
        return True
    if rej_col < kept_col and x1 <= left + tol and x2 >= left - tol:
        return True
    if rej_bottom > kept_bottom and y2 >= bottom - tol and y1 <= bottom + tol:
        return True
    if rej_row < kept_row and y1 <= top + tol and y2 >= top - tol:
        return True
    return False


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
