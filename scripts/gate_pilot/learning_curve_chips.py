#!/usr/bin/env python3
"""Cut chips and YOLO labels for the frozen learning-curve grids.

Images are written for all 39 primary grids. YOLO boxes are written only
for the 30 training grids. A holdout grid that already has a labels
directory stops the run. The four pilot training grids reuse their
existing PNGs by hard link when those files are already complete.

Does not train and does not score the holdout.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from scripts.gate_pilot.chip_windows import chip_windows_from_tifs
from scripts.gate_pilot.common import OUT_ROOT, setup_logging
from scripts.gate_pilot.learning_curve_select import SELECTION_DIR, SELECTION_PATH
from scripts.gate_pilot.step2_chips_and_labels import (
    SMALL_AREA_M2,
    _manifest,
    _pair_table,
    _read_layer,
    _tile_paths,
    _write_images,
    _write_yolo,
)

CHIP_ROOT = SELECTION_DIR / "chips"
PILOT_CHIP_ROOT = OUT_ROOT / "chips"
VECTOR_ROOT = SELECTION_DIR / "vectors"
SUMMARY_PATH = SELECTION_DIR / "chip_summary.csv"
SENTINEL = SELECTION_DIR / "chips.done"
EXPECTED_CHIPS = 2058
REUSE_GRIDS = ("CPT2932", "CPT2373", "CPT2182", "CPT2330")


def _link_pilot_images(grid_id: str, log) -> bool:
    if grid_id not in REUSE_GRIDS:
        return False
    source = PILOT_CHIP_ROOT / grid_id / "images"
    pngs = sorted(path for path in source.glob("*.png") if path.stat().st_size > 0)
    if len(pngs) != EXPECTED_CHIPS:
        return False
    destination = CHIP_ROOT / grid_id / "images"
    destination.mkdir(parents=True, exist_ok=True)
    for path in pngs:
        target = destination / path.name
        if target.exists() and target.stat().st_size > 0:
            continue
        os.link(path, target)
    linked = len(list(destination.glob("*.png")))
    if linked != EXPECTED_CHIPS:
        raise RuntimeError(f"{grid_id} linked {linked} images, expected {EXPECTED_CHIPS}")
    log.info("%s reused %d pilot images", grid_id, linked)
    return True


def _one_grid(grid_id: str, train_ids: set[str], log) -> dict:
    label_dir = CHIP_ROOT / grid_id / "labels"
    if grid_id not in train_ids and label_dir.exists():
        raise RuntimeError(f"holdout {grid_id} already has labels; refusing to keep them")
    paths = _tile_paths(grid_id)
    chips = chip_windows_from_tifs(paths, grid_id)
    if len(chips) != EXPECTED_CHIPS:
        raise RuntimeError(f"{grid_id} has {len(chips)} chips, expected {EXPECTED_CHIPS}")
    vector_dir = VECTOR_ROOT / grid_id
    pre = _read_layer(vector_dir / "predictions_metric.gpkg")
    post_path = vector_dir / "predictions_metric_cls_filtered.gpkg"
    post = _read_layer(post_path)
    pre_hits = _pair_table(chips, pre, "pre")
    post_hits = _pair_table(chips, post, "post")
    small_ids = set(post.loc[post["area_m2"] < SMALL_AREA_M2, "poly_id"].astype(int))
    manifest = _manifest(grid_id, chips, pre_hits, post_hits, small_ids)
    out_dir = CHIP_ROOT / grid_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    pd.concat([pre_hits, post_hits], ignore_index=True).to_csv(
        out_dir / "intersections.csv", index=False
    )
    if grid_id in train_ids:
        _write_yolo(grid_id, chips, pre_hits, chip_root=CHIP_ROOT)
        n_labels = len(list(label_dir.glob("*.txt")))
        if n_labels != EXPECTED_CHIPS:
            raise RuntimeError(f"{grid_id} has {n_labels} labels, expected {EXPECTED_CHIPS}")
    if not _link_pilot_images(grid_id, log):
        _write_images(grid_id, paths, EXPECTED_CHIPS, log, chip_root=CHIP_ROOT)
    n_png = len(list((out_dir / "images").glob("*.png")))
    if n_png != EXPECTED_CHIPS:
        raise RuntimeError(f"{grid_id} png count {n_png}, expected {EXPECTED_CHIPS}")
    row = {
        "grid_id": grid_id,
        "role": "train" if grid_id in train_ids else "holdout",
        "n_chips": EXPECTED_CHIPS,
        "n_pre_chips": int(manifest["y_pre"].sum()),
        "n_post_chips": int(manifest["y_post"].sum()),
        "n_post_small_chips": int(manifest["y_post_small"].sum()),
        "n_pre_boxes": int(len(pre_hits)),
        "post_not_in_pre": int((manifest["y_post"] & ~manifest["y_pre"]).sum()),
    }
    log.info(
        "%s pre_chips=%d post_chips=%d boxes=%d",
        grid_id, row["n_pre_chips"], row["n_post_chips"], row["n_pre_boxes"],
    )
    return row


def main() -> int:
    log = setup_logging("learning_curve_chips")
    if not (SELECTION_DIR / "fetch.done").exists():
        log.error("fetch.done is missing; refusing to cut chips")
        return 1
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("schema") != "learning_curve_selection_v1":
        log.error("unexpected selection schema")
        return 1
    train_ids = set(selection["train_30"])
    holdout_ids = set(selection["holdout"])
    grids = list(dict.fromkeys([*selection["train_30"], *selection["holdout"]]))
    if len(grids) != 39 or train_ids & holdout_ids:
        log.error("selection is not 30 disjoint train grids plus 9 holdout grids")
        return 1
    if SENTINEL.exists():
        SENTINEL.unlink()
    rows = []
    consecutive = 0
    for index, grid_id in enumerate(grids, start=1):
        try:
            rows.append(_one_grid(grid_id, train_ids, log))
            consecutive = 0
            log.info("complete %d/%d %s", index, len(grids), grid_id)
        except Exception:
            consecutive += 1
            log.exception("failed %s", grid_id)
            if consecutive >= 2:
                log.error("two grids in a row failed; stopping")
                break
        pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)
    done = {row["grid_id"] for row in rows}
    if done != set(grids):
        log.error("chips incomplete: %d/%d", len(done), len(grids))
        return 1
    summary = pd.DataFrame(rows)
    if int((summary["post_not_in_pre"]).sum()) > 0:
        log.warning("some filtered chips are not inside the pre-filter chips; see chip_summary.csv")
    SENTINEL.write_text(f"ok {len(grids)}\n", encoding="utf-8")
    log.info("chips.done %d grids", len(grids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
