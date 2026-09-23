"""The YOLO box is the clipped polygon, in chip-image pixels."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gate_pilot.yolo_boxes import yolo_box_from_geo_bounds


def test_left_half_of_a_chip_is_a_full_height_box():
    # Geographic square, valid window fills the 400 px image.
    box = yolo_box_from_geo_bounds(
        0.0, 0.0, 0.5, 1.0,
        chip_minx=0.0, chip_miny=0.0, chip_maxx=1.0, chip_maxy=1.0,
        valid_w=400, valid_h=400,
    )
    assert box is not None
    cx, cy, w, h = box
    assert cx == 0.25
    assert cy == 0.5
    assert w == 0.5
    assert h == 1.0


def test_sub_two_pixel_box_is_expanded_and_kept():
    # One pixel wide at the left edge of a 400 px chip.
    box = yolo_box_from_geo_bounds(
        0.0, 0.25, 1.0 / 400.0, 0.75,
        chip_minx=0.0, chip_miny=0.0, chip_maxx=1.0, chip_maxy=1.0,
        valid_w=400, valid_h=400,
    )
    assert box is not None
    _cx, _cy, w, h = box
    assert w == 2.0 / 400.0
    assert h == 0.5
    assert _cx == 1.0 / 400.0
