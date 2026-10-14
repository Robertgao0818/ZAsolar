#!/usr/bin/env python3
"""Step 2: write chip PNGs and labels from the real GeoTIFFs.

Runs only after step 1 has matched the chip grid and passed imagery parity.
Holdout chips are written so step 5 can score them later. YOLO boxes are
written only for the four training grids. This script does not train and
does not run a gate model on the holdout.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from PIL import Image

from core import region_registry
from core.inference.tile_dataset import SlidingWindowDataset
from scripts.gate_pilot.chip_windows import CHIP_SIZE, OVERLAP, chip_windows_from_tifs
from scripts.gate_pilot.common import (
    HOLDOUT_PV_GRID,
    OUT_ROOT,
    STEP0_DIR,
    TRAIN_PV_GRIDS,
    VECTOR_DIR,
    setup_logging,
)
from scripts.gate_pilot.yolo_boxes import yolo_box_from_geo_bounds

TILES_ROOT = Path.home() / "zasolar_data" / "tiles" / "cape_town" / "aerial_2025"
CHIP_ROOT = OUT_ROOT / "chips"
PARITY_DIR = OUT_ROOT / "parity"
STEP2_DIR = OUT_ROOT / "step2"
SMALL_AREA_M2 = 30.0


def _require_step1() -> dict:
    feasibility = json.loads((STEP0_DIR / "feasibility.json").read_text(encoding="utf-8"))
    compare = json.loads((OUT_ROOT / "step1" / "chip_compare.json").read_text(encoding="utf-8"))
    parity = json.loads((OUT_ROOT / "step1" / "parity.json").read_text(encoding="utf-8"))
    if feasibility["verdict"] != "pass":
        raise RuntimeError("step 0 did not pass")
    if not compare or not all(row["match"] for row in compare):
        raise RuntimeError("real-tile chip grid does not match step 0; refusing to label")
    if not parity or not all(row["pass"] for row in parity):
        raise RuntimeError("imagery parity did not pass; refusing to label")
    return feasibility


def _tile_paths(grid_id: str) -> list[Path]:
    source = region_registry.resolve_source_grid_id(grid_id, "cape_town")
    paths = sorted((TILES_ROOT / source).glob(f"{grid_id}_*_*_geo.tif"))
    if len(paths) != 42:
        raise RuntimeError(f"{grid_id} has {len(paths)} tiles, expected 42")
    return paths


def _read_layer(path: Path) -> gpd.GeoDataFrame:
    frame = pyogrio.read_dataframe(path, fid_as_index=True)
    frame = frame.reset_index(names="poly_id")
    if frame.crs is None or frame.crs.to_epsg() != 32734:
        raise RuntimeError(f"{path} CRS is {frame.crs}")
    if "area_m2" not in frame.columns:
        frame["area_m2"] = frame.geometry.area
    return frame


def _pair_table(chips: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, layer: str) -> pd.DataFrame:
    columns = [
        "chip_index", "poly_id", "layer", "area_m2", "area_deg2",
        "poly_area_m2", "minx", "miny", "maxx", "maxy",
    ]
    if polygons.empty:
        return pd.DataFrame(columns=columns)
    projected = polygons.to_crs(chips.crs)
    poly_geoms = shapely.make_valid(np.asarray(projected.geometry.values, dtype=object))
    chip_geoms = np.asarray(chips.geometry.values, dtype=object)
    tree = shapely.STRtree(poly_geoms)
    pairs = tree.query(chip_geoms, predicate="intersects")
    if pairs.size == 0:
        return pd.DataFrame(columns=columns)
    chip_index = pairs[0]
    poly_index = pairs[1]
    inter = shapely.intersection(chip_geoms[chip_index], poly_geoms[poly_index])
    area_deg2 = shapely.area(inter)
    keep = area_deg2 > 0
    if not keep.any():
        return pd.DataFrame(columns=columns)
    inter = inter[keep]
    chip_index = chip_index[keep]
    poly_index = poly_index[keep]
    area_deg2 = area_deg2[keep]
    metric = gpd.GeoSeries(list(inter), crs=chips.crs).to_crs(32734)
    bounds = shapely.bounds(inter)
    poly_ids = polygons["poly_id"].to_numpy()
    poly_area = polygons["area_m2"].to_numpy()
    return pd.DataFrame(
        {
            "chip_index": chips["chip_index"].to_numpy()[chip_index],
            "poly_id": poly_ids[poly_index],
            "layer": layer,
            "area_m2": metric.area.to_numpy(),
            "area_deg2": area_deg2,
            "poly_area_m2": poly_area[poly_index],
            "minx": bounds[:, 0],
            "miny": bounds[:, 1],
            "maxx": bounds[:, 2],
            "maxy": bounds[:, 3],
        }
    )


def _check_chip_index(grid_id: str, chips: gpd.GeoDataFrame, log) -> pd.DataFrame:
    rerun = gpd.read_file(PARITY_DIR / grid_id / "predictions_metric.gpkg")
    rerun = rerun[rerun["chip_index"] >= 0].drop_duplicates("chip_index").reset_index(drop=True)
    if rerun.empty:
        raise RuntimeError(f"{grid_id} parity file has no chip_index")
    rerun_ll = rerun.to_crs(chips.crs)
    n = min(20, len(rerun_ll))
    chosen = sorted(random.Random(20260923).sample(list(rerun_ll.index), n))
    by_index = chips.set_index("chip_index")
    rows = []
    for position in chosen:
        poly = rerun_ll.loc[position]
        chip_index = int(poly["chip_index"])
        if chip_index not in by_index.index:
            rows.append({"grid_id": grid_id, "chip_index": chip_index, "ok": False, "reason": "missing"})
            continue
        chip = by_index.loc[chip_index]
        stem_ok = Path(str(chip["source_tile"])).stem == str(poly["source_tile"])
        px = (float(chip["maxx"]) - float(chip["minx"])) / float(chip["valid_w"])
        py = (float(chip["maxy"]) - float(chip["miny"])) / float(chip["valid_h"])
        buffered = chip.geometry.buffer(2.0 * max(px, py))
        hits = float(buffered.intersection(poly.geometry).area) > 0
        rows.append(
            {
                "grid_id": grid_id,
                "chip_index": chip_index,
                "source_tile": str(poly["source_tile"]),
                "stem_ok": stem_ok,
                "intersects_within_2px": hits,
                "ok": bool(stem_ok and hits),
                "reason": "" if stem_ok and hits else "stem" if not stem_ok else "no_intersection",
            }
        )
    frame = pd.DataFrame(rows)
    failed = frame[~frame["ok"]]
    log.info("%s chip_index check %d/%d ok", grid_id, int(frame["ok"].sum()), len(frame))
    if not failed.empty:
        raise RuntimeError(
            f"{grid_id} chip_index does not match the parity rerun on {len(failed)} of {len(frame)} chips"
        )
    return frame


def _write_images(
    grid_id: str,
    paths: list[Path],
    n_expected: int,
    log,
    chip_root: Path | None = None,
) -> None:
    image_dir = (CHIP_ROOT if chip_root is None else chip_root) / grid_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    dataset = SlidingWindowDataset(
        paths, chip_size=CHIP_SIZE, overlap=OVERLAP, window_origin_mode="anchored",
    )
    try:
        for index in range(len(dataset)):
            destination = image_dir / f"{index:05d}.png"
            if destination.exists() and destination.stat().st_size > 0:
                continue
            tensor, meta = dataset[index]
            if meta.chip_index != index:
                raise RuntimeError(f"{grid_id} image index drifted at {index}")
            array = np.rint(tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(np.transpose(array, (1, 2, 0)), mode="RGB").save(destination)
            if index % 400 == 0:
                log.info("%s wrote image %d/%d", grid_id, index, len(dataset))
    finally:
        dataset.close()
    n_png = len(list(image_dir.glob("*.png")))
    if n_png != n_expected:
        raise RuntimeError(f"{grid_id} png count {n_png}, expected {n_expected}")


def _write_yolo(
    grid_id: str,
    chips: gpd.GeoDataFrame,
    pre_hits: pd.DataFrame,
    chip_root: Path | None = None,
) -> None:
    label_dir = (CHIP_ROOT if chip_root is None else chip_root) / grid_id / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    grouped = {int(k): g for k, g in pre_hits.groupby("chip_index")} if len(pre_hits) else {}
    for row in chips.itertuples(index=False):
        lines = []
        for hit in grouped.get(int(row.chip_index), pd.DataFrame()).itertuples(index=False):
            box = yolo_box_from_geo_bounds(
                float(hit.minx), float(hit.miny), float(hit.maxx), float(hit.maxy),
                chip_minx=float(row.minx), chip_miny=float(row.miny),
                chip_maxx=float(row.maxx), chip_maxy=float(row.maxy),
                valid_w=int(row.valid_w), valid_h=int(row.valid_h),
                chip_size=CHIP_SIZE,
            )
            if box is None:
                continue
            lines.append("0 {:.6f} {:.6f} {:.6f} {:.6f}".format(*box))
        (label_dir / f"{int(row.chip_index):05d}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "",
            encoding="utf-8",
        )


def _manifest(grid_id: str, chips: gpd.GeoDataFrame, pre_hits: pd.DataFrame, post_hits: pd.DataFrame, post_small: set[int]) -> pd.DataFrame:
    def ids_for(frame: pd.DataFrame, only: set[int] | None = None) -> dict[int, str]:
        if frame.empty:
            return {}
        use = frame if only is None else frame[frame["poly_id"].isin(only)]
        if use.empty:
            return {}
        grouped = use.groupby("chip_index")["poly_id"].apply(lambda s: ",".join(str(int(v)) for v in s))
        return {int(k): v for k, v in grouped.items()}

    pre_ids = ids_for(pre_hits)
    post_ids = ids_for(post_hits)
    small_ids = ids_for(post_hits, post_small)
    out = chips.drop(columns="geometry").copy()
    out["source_tile"] = out["source_tile"].map(lambda name: Path(str(name)).stem)
    out["y_pre"] = out["chip_index"].map(lambda i: int(i) in pre_ids)
    out["y_post"] = out["chip_index"].map(lambda i: int(i) in post_ids)
    out["y_post_small"] = out["chip_index"].map(lambda i: int(i) in small_ids)
    out["pre_poly_ids"] = out["chip_index"].map(lambda i: pre_ids.get(int(i), ""))
    out["post_poly_ids"] = out["chip_index"].map(lambda i: post_ids.get(int(i), ""))
    out["post_small_poly_ids"] = out["chip_index"].map(lambda i: small_ids.get(int(i), ""))
    return out


def main() -> int:
    log = setup_logging("step2_chips_and_labels")
    feasibility = _require_step1()
    selection = feasibility["selection"]
    train_ids = set(TRAIN_PV_GRIDS) | {selection["e_train"]}
    grids = [
        *TRAIN_PV_GRIDS, selection["e_train"], HOLDOUT_PV_GRID, selection["e_test"],
    ]
    pv_ids = set(TRAIN_PV_GRIDS) | {HOLDOUT_PV_GRID}
    STEP2_DIR.mkdir(parents=True, exist_ok=True)
    checks = []
    subset_violations = []
    for grid_id in grids:
        log.info("labels for %s", grid_id)
        paths = _tile_paths(grid_id)
        chips = chip_windows_from_tifs(paths, grid_id)
        if grid_id in pv_ids:
            checks.append(_check_chip_index(grid_id, chips, log))
        pre = _read_layer(VECTOR_DIR / grid_id / "predictions_metric.gpkg")
        post_path = VECTOR_DIR / grid_id / "predictions_metric_cls_filtered.gpkg"
        post = _read_layer(post_path) if post_path.exists() else pre.iloc[0:0].copy()
        pre_hits = _pair_table(chips, pre, "pre")
        post_hits = _pair_table(chips, post, "post")
        small_ids = set(post.loc[post["area_m2"] < SMALL_AREA_M2, "poly_id"].astype(int))
        manifest = _manifest(grid_id, chips, pre_hits, post_hits, small_ids)
        violations = int((manifest["y_post"] & ~manifest["y_pre"]).sum())
        subset_violations.append({"grid_id": grid_id, "post_not_in_pre": violations})
        log.info("%s y_post outside y_pre: %d", grid_id, violations)
        out_dir = CHIP_ROOT / grid_id
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out_dir / "manifest.csv", index=False)
        hits = pd.concat([pre_hits, post_hits], ignore_index=True)
        hits.to_csv(STEP2_DIR / f"intersections_{grid_id}.csv", index=False)
        if grid_id in train_ids:
            _write_yolo(grid_id, chips, pre_hits)
        else:
            label_dir = out_dir / "labels"
            if label_dir.exists():
                raise RuntimeError(f"holdout {grid_id} already has a labels directory; refusing to leave training boxes there")
        _write_images(grid_id, paths, len(chips), log)
    if checks:
        pd.concat(checks, ignore_index=True).to_csv(STEP2_DIR / "chip_index_check.csv", index=False)
    pd.DataFrame(subset_violations).to_csv(STEP2_DIR / "subset_check.csv", index=False)
    (STEP2_DIR / "step2.done").write_text("ok\n", encoding="utf-8")
    log.info("step 2 wrote chips under %s", CHIP_ROOT)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import logging
        logging.getLogger("step2_chips_and_labels").exception("step 2 failed")
        raise
