"""Patch CT seg trainset v2b -> v2c.

v2b stores mask value 2 as the union of confirmed solar-thermal collectors and
generic hard negatives (roof, skylight, other, unknown false positives). v2c
keeps value 2 only for confirmed collectors. Every other value-2 pixel becomes
background. PV and ignore pixels are not edited, and no new 2s are painted.

Final mask:
  0   background, including every non-collector hard negative
  1   PV
  2   swh (collectors.gpkg rows with mask_value == 2: solar water heaters
      and pool heaters)
  255 ignore, unchanged

Every output file is written under --out. Changed masks are rewritten with the
source rasterio profile. Unchanged masks, collectors, and hard_negatives are
shutil.copy2'd. Nothing is hardlinked: v2 and v2b already share inodes, so a
link would let a later in-place edit destroy the previous version.

swh_labelled is true only when this grid was actually checked for collectors
(an FP-review gpkg, or any collectors.csv row). A false flag means "no value 2"
is not evidence that the grid has no solar water heater.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Repo root is on sys.path so this file matches the other training scripts.
# The patch does not import project modules.

MASK_VALUES = {
    "0": "background (incl. non-collector hard negatives)",
    "1": "pv",
    "2": "swh (solar water heater + pool heater)",
    "255": "ignore",
}
SWH_LABELLED_RULE = (
    "True iff the grid has an FP-review gpkg on --fp-review-glob "
    "(grid id = parent directory of 'review', not the gpkg filename) "
    "or at least one collectors.csv row of any mask_value. "
    "False means solar-thermal collectors were not labelled there, so "
    "absence of mask value 2 is not evidence of absence."
)
DEFAULT_FP_REVIEW_GLOB = (
    "/home/gao/projects/ZAsolar/results/cape_town/v3c_targeted_hn_aerial_2025/"
    "G*/review/*_reviewed.gpkg"
)
WORKERS = 8
REQUIRED_SRC = (
    "manifest.csv",
    "chips_512px_index.csv",
    "collectors.csv",
    "collectors.gpkg",
    "hard_negatives.csv",
    "summary.json",
)
_GEOMS = {}
_SRC_ROOT = None
_OUT_ROOT = None


def _grid_from_fp_review_path(path) -> str | None:
    """Grid id is the parent of the ``review`` directory closest to the file."""
    parts = Path(path).parts
    indexes = [i for i, part in enumerate(parts) if part == "review"]
    if not indexes:
        return None
    i = indexes[-1]
    if i == 0:
        return None
    grid = parts[i - 1]
    if not grid or grid in (os.sep, "/", "."):
        return None
    return grid


def swh_labelled_grids(fp_review_paths, collector_grids, all_grids):
    """Flag each trainset grid that was labelled for solar-thermal collectors.

    Parameters are plain iterables. ``fp_review_paths`` are review gpkg paths;
    the grid id is the parent of the ``review`` directory. ``collector_grids``
    is every grid with at least one collectors.csv row, any mask_value.
    ``all_grids`` is the universe written to the output (one entry per grid,
    first-seen order). Other grids are ignored.

    Returns ``{grid: {"swh_labelled": bool, "reason": str}}``. ``reason`` is
    ``fp_review+collectors``, ``fp_review``, ``collectors``, or ``unlabelled``.
    """
    reviewed = set()
    for path in fp_review_paths:
        grid = _grid_from_fp_review_path(path)
        if grid:
            reviewed.add(grid)
    collectors = {str(g) for g in collector_grids if g is not None and str(g) != "nan"}
    out = {}
    for grid in all_grids:
        key = str(grid)
        if key in out:
            continue
        in_review = key in reviewed
        in_collectors = key in collectors
        if in_review and in_collectors:
            reason = "fp_review+collectors"
        elif in_review:
            reason = "fp_review"
        elif in_collectors:
            reason = "collectors"
        else:
            reason = "unlabelled"
        out[key] = {"swh_labelled": in_review or in_collectors, "reason": reason}
    return out


def remap_mask(mask, collector_bool):
    """Turn value-2 pixels that are outside a collector into background.

    ``collector_bool`` is true on pixels whose center falls in a confirmed
    collector polygon (the same burn v2b used). Pixels that are already 2 and
    inside a collector stay 2. Values other than 2 are never written, so this
    cannot create a new 2 and cannot edit PV (1) or ignore (255).

    Returns ``(new_mask, n_remapped)``. ``n_remapped`` is how many pixels went
    from 2 to 0. ``mask`` is not modified.
    """
    mask_arr = np.asarray(mask)
    collector = np.asarray(collector_bool, dtype=bool)
    if mask_arr.shape != collector.shape:
        raise ValueError(
            f"shape mismatch: mask {mask_arr.shape} vs collector_bool {collector.shape}"
        )
    new = np.array(mask_arr, copy=True)
    selected = (new == 2) & ~collector
    n_remapped = int(selected.sum())
    new[selected] = 0
    return new, n_remapped


def refuse(message: str) -> None:
    print(f"refuse: {message}", file=sys.stderr)
    raise SystemExit(2)


def validate_out(src: Path, out: Path) -> None:
    """--out must be a new or empty directory, and must not be inside --src."""
    src_r, out_r = src.resolve(), out.resolve()
    if src_r == out_r:
        refuse("--out is the same directory as --src")
    if src_r in out_r.parents:
        refuse(f"--out is inside --src ({out})")
    if out.exists() and not out.is_dir():
        refuse(f"--out exists and is not a directory ({out})")
    if out.exists() and any(out.iterdir()):
        refuse(f"--out exists and is not empty ({out})")


def rewrite_mask_path(path: str, src: Path, out: Path) -> str:
    """Replace the --src prefix of a mask path with --out. Image paths stay."""
    src_r, out_r = src.resolve(), out.resolve()
    try:
        rel = Path(path).resolve().relative_to(src_r)
    except ValueError:
        refuse(f"mask path is not under --src {src_r}: {path}")
    dest = out_r / rel
    if dest.resolve() == src_r or src_r in dest.resolve().parents:
        refuse(f"refusing to write a mask into --src: {dest}")
    if out_r != dest.resolve() and out_r not in dest.resolve().parents:
        refuse(f"mask destination escaped --out: {dest}")
    return str(dest)


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _sum_by_split(splits, values) -> dict:
    frame = pd.DataFrame({"split": splits, "v": values})
    if frame.empty:
        return {}
    return {str(k): int(v) for k, v in frame.groupby("split", sort=False)["v"].sum().items()}


def _counts_by_split(frame: pd.DataFrame, column: str) -> dict:
    if frame.empty:
        return {}
    return {str(k): int(v) for k, v in frame.groupby("split")[column].sum().items()}


def _swh_by_subtype(collectors: pd.DataFrame) -> dict:
    """Confirmed collectors only (mask_value == 2), same shape as hn_by_subtype."""
    if collectors.empty or "mask_value" not in collectors.columns:
        return {}
    confirmed = collectors.loc[pd.to_numeric(collectors["mask_value"], errors="coerce") == 2]
    if confirmed.empty:
        return {}
    subtype = confirmed["subtype"].fillna("unknown").astype(str)
    split = confirmed["split"].astype(str)
    tab = pd.DataFrame({"split": split, "subtype": subtype}).groupby(
        ["split", "subtype"]
    ).size().unstack(fill_value=0)
    return {
        str(col): {str(idx): int(tab.loc[idx, col]) for idx in tab.index}
        for col in tab.columns
    }


def _rasterize_collectors(geoms, crs, transform, height, width):
    """Burn EPSG:4326 collector polygons onto this tile.

    Same rule as patch_v2b and the v2 builder: reproject to the tile CRS,
    fill=0, all_touched=False. A pixel is inside only when its center is
    covered. Empty input is an all-false mask, not a skip of the remap.
    """
    import geopandas as gpd
    from rasterio.features import rasterize

    if not geoms:
        return np.zeros((height, width), dtype=bool)
    projected = gpd.GeoSeries(list(geoms), crs=4326).to_crs(crs)
    projected = projected[projected.notna() & ~projected.is_empty]
    if projected.empty:
        return np.zeros((height, width), dtype=bool)
    burned = rasterize(
        [(geom, 1) for geom in projected],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
    )
    return burned.astype(bool)


def _init_worker(wkt_by_grid, src_root: str, out_root: str) -> None:
    global _GEOMS, _SRC_ROOT, _OUT_ROOT
    _GEOMS = wkt_by_grid
    _SRC_ROOT = Path(src_root)
    _OUT_ROOT = Path(out_root)


def _geoms_for(grid: str):
    from shapely import wkt

    cached = _GEOMS.get(grid)
    if cached is None:
        return []
    if cached and isinstance(cached[0], str):
        parsed = [wkt.loads(text) for text in cached]
        _GEOMS[grid] = parsed
    return _GEOMS[grid]


def _assert_fresh_file(src: str, dst: Path) -> None:
    if os.path.samestat(os.stat(src), os.stat(dst)):
        raise AssertionError(f"output is a hardlink of the source mask: {dst}")


def _process_manifest_row(task):
    import rasterio

    src = task["src_mask"]
    dst = Path(task["dst_mask"])
    dst_r = dst.resolve()
    if dst_r == _SRC_ROOT or _SRC_ROOT in dst_r.parents:
        raise AssertionError(f"refusing to write into --src: {dst}")
    if _OUT_ROOT not in dst_r.parents:
        raise AssertionError(f"destination escaped --out: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src) as dataset:
        mask = dataset.read(1)
        profile = dataset.profile.copy()
        crs = dataset.crs
        transform = dataset.transform
        height, width = dataset.height, dataset.width
    collector = _rasterize_collectors(_geoms_for(task["grid"]), crs, transform, height, width)
    new, n_remapped = remap_mask(mask, collector)
    changed = new != mask
    # The collector burn is the definition of "inside". Those pixels must be
    # byte-identical to the source, including PV and ignore under the polygon.
    if np.any(changed & collector):
        raise AssertionError(f"pixel inside a collector polygon changed: {src}")
    if changed.any() and not (np.all(mask[changed] == 2) and np.all(new[changed] == 0)):
        raise AssertionError(f"a pixel other than value 2 changed: {src}")
    n2_before = int((mask == 2).sum())
    n_pv = int((mask == 1).sum())
    n_ignore = int((mask == 255).sum())
    if (n2_before, n_pv, n_ignore) != (task["n_hn"], task["n_pv"], task["n_ignore"]):
        raise AssertionError(
            f"manifest counts do not match the mask {src}: "
            f"file (2,1,255)=({n2_before},{n_pv},{n_ignore}) "
            f"manifest=({task['n_hn']},{task['n_pv']},{task['n_ignore']})"
        )
    counts = np.bincount(new.ravel(), minlength=256)
    if n_remapped == 0:
        shutil.copy2(src, dst)
        wrote = False
    else:
        with rasterio.open(dst, "w", **profile) as out_ds:
            out_ds.write(new, 1)
        wrote = True
    _assert_fresh_file(src, dst)
    chips = []
    for index, x, y, size in task["chips"]:
        if y < 0 or x < 0 or y + size > height or x + size > width:
            raise AssertionError(f"chip {(x, y, size)} does not fit {src}")
        window = new[y:y + size, x:x + size]
        chips.append((index, int((window == 2).sum())))
    return {
        "idx": task["idx"],
        "split": task["split"],
        "n_bg": int(counts[0]),
        "n_pv": int(counts[1]),
        "n_swh": int(counts[2]),
        "n_ignore": int(counts[255]),
        "n2_before": n2_before,
        "n_remapped": n_remapped,
        "wrote": wrote,
        "collector_px": int(collector.sum()),
        "chips": chips,
    }


def process_manifest_row(task):
    try:
        return _process_manifest_row(task)
    except Exception as exc:
        raise RuntimeError(f"mask failed: {task.get('src_mask')}") from exc


def _confirmed_wkt_by_grid(collectors_gpkg: Path) -> dict:
    import geopandas as gpd

    table = gpd.read_file(collectors_gpkg)
    if table.crs is None:
        refuse(f"{collectors_gpkg} has no CRS")
    table = table.to_crs(4326)
    if "mask_value" not in table.columns:
        refuse(f"{collectors_gpkg} has no mask_value column")
    confirmed = table.loc[pd.to_numeric(table["mask_value"], errors="coerce") == 2]
    grouped = {}
    if len(confirmed):
        for grid, subset in confirmed.groupby(confirmed["grid"].astype(str)):
            texts = []
            for geom in subset.geometry:
                if geom is not None and not geom.is_empty:
                    texts.append(geom.wkt)
            if texts:
                grouped[str(grid)] = texts
    return grouped


def _split_order(keys) -> list:
    preferred = ["train", "val", "eval"]
    present = set(keys)
    return [name for name in preferred if name in present] + sorted(present - set(preferred))


def _print_verification(before, after, remapped, collector_px) -> None:
    print("value 2 pixels by split, before (v2b) -> after (v2c); assert after <= before")
    for split in _split_order(set(before) | set(after) | set(remapped)):
        left = int(before.get(split, 0))
        right = int(after.get(split, 0))
        moved = int(remapped.get(split, 0))
        print(f"  {split}: {left} -> {right}  remapped_hn_to_bg={moved}")
        if right > left:
            raise AssertionError(f"{split}: value 2 increased from {left} to {right}")
        if left - right != moved:
            raise AssertionError(
                f"{split}: remapped {moved} != before-after {left - right}"
            )
    print(f"collector pixels unchanged: ok (collector burn covered {int(collector_px)} px)")


def _build_summary(src_summary, manifest, chips, grids, collectors, stats, src, out, fp_glob):
    summary = json.loads(Path(src_summary).read_text())
    stale_hn = summary.pop("hn_px", None)
    stale_collector_neg = summary.pop("collector_neg_px", None)
    stale_collector_ignore = summary.pop("collector_ignore_px", None)
    heldout = summary.get("heldout")
    args_block = summary.get("args")
    v2b_block = summary.get("v2b")

    splits = manifest["split"].astype(str)
    px = (manifest["width"] * manifest["height"]).groupby(splits).sum()
    ignore = manifest.groupby(splits)["n_ignore"].sum()
    summary["pv_px"] = _counts_by_split(manifest, "n_pv")
    summary["tiles"] = {str(k): int(v) for k, v in manifest.groupby("split").size().items()}
    summary["grids"] = {
        str(k): int(v) for k, v in manifest.groupby("split")["grid"].nunique().items()
    }
    summary["ignore_frac"] = {
        str(k): float(round(ignore.loc[k] / px.loc[k], 4)) for k in px.index
    } if len(px) else {}
    summary["mask_values"] = dict(MASK_VALUES)
    summary["swh_px"] = _counts_by_split(manifest, "n_swh")
    summary["swh_ignore_px"] = _counts_by_split(manifest, "n_swh_ignore_px")
    summary["remapped_hn_to_bg_px"] = {str(k): int(v) for k, v in stats["remapped"].items()}
    summary["swh_by_subtype"] = _swh_by_subtype(collectors)
    labelled = grids.loc[grids["swh_labelled"].astype(bool)]
    labelled_n = {split: 0 for split in _split_order(manifest["split"].astype(str).unique())}
    if len(labelled):
        for split, count in labelled.groupby("split").size().items():
            labelled_n[str(split)] = int(count)
    summary["swh_labelled_grids"] = labelled_n
    summary["swh_labelled_rule"] = SWH_LABELLED_RULE
    summary["chips"] = {str(k): int(v) for k, v in chips.groupby("split").size().items()} if len(chips) else {}

    for key in ("heldout", "args", "v2b"):
        summary.pop(key, None)
    summary["heldout"] = heldout
    summary["args"] = args_block
    summary["v2b"] = v2b_block
    summary["v2c"] = {
        "source": "v2b",
        "src": str(src),
        "out": str(out),
        "fp_review_glob": fp_glob,
        "n_fp_review_gpkg": int(stats["n_fp_review_gpkg"]),
        "masks_rewritten": int(stats["masks_rewritten"]),
        "masks_copied": int(stats["masks_copied"]),
        "collector_pixels_unchanged": True,
        "collector_burn_px": int(stats["collector_px"]),
        "value2_before": {str(k): int(v) for k, v in stats["before"].items()},
        "value2_after": {str(k): int(v) for k, v in stats["after"].items()},
        "swh_ignore_px_source": (
            "sum of v2b manifest n_collector_ignore_px by split. "
            "v2b summary.json hn_px and collector_ignore_px were copied from v2 "
            "and not refreshed when the patch moved 255 -> 2."
        ),
        "v2b_summary_hn_px": stale_hn,
        "v2b_summary_collector_neg_px": stale_collector_neg,
        "v2b_summary_collector_ignore_px": stale_collector_ignore,
    }
    return _jsonify(summary)


def _load_inputs(src: Path):
    missing = [name for name in REQUIRED_SRC if not (src / name).is_file()]
    if missing:
        refuse(f"--src is missing {missing}")
    manifest = pd.read_csv(src / "manifest.csv")
    chips = pd.read_csv(src / "chips_512px_index.csv")
    collectors = pd.read_csv(src / "collectors.csv")
    for column in (
        "grid", "split", "batch", "image", "mask", "width", "height", "px_m",
        "n_bg", "n_pv", "n_hn", "n_ignore", "n_collector_hn_px", "n_collector_ignore_px",
    ):
        if column not in manifest.columns:
            refuse(f"manifest.csv has no column {column}")
    for column in ("grid", "split", "image", "mask", "x", "y", "size", "n_pv", "n_hn"):
        if column not in chips.columns:
            refuse(f"chips_512px_index.csv has no column {column}")
    if "grid" not in collectors.columns:
        refuse("collectors.csv has no grid column")
    if manifest["mask"].duplicated().any():
        refuse("manifest has duplicate mask paths")
    unknown = set(chips["mask"]) - set(manifest["mask"])
    if unknown:
        refuse(f"{len(unknown)} chip masks are not in the manifest")
    split_n = manifest.groupby("grid")["split"].nunique()
    if (split_n > 1).any():
        bad = split_n[split_n > 1].index.astype(str).tolist()
        refuse(f"grid belongs to more than one split: {bad[:8]}")
    return manifest, chips, collectors


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, required=True, help="v2b data directory")
    parser.add_argument("--out", type=Path, required=True,
                        help="v2c data directory; must not exist or must be empty")
    parser.add_argument("--fp-review-glob", default=DEFAULT_FP_REVIEW_GLOB,
                        help="glob of *_reviewed.gpkg files; grid id is the parent of review/")
    args = parser.parse_args(argv)
    src = args.src
    out = args.out
    if not src.is_dir():
        refuse(f"--src is not a directory: {src}")
    validate_out(src, out)
    manifest, chips, collectors = _load_inputs(src)
    manifest = manifest.reset_index(drop=True)
    chips = chips.reset_index(drop=True)

    review_paths = sorted(glob.glob(args.fp_review_glob))
    all_grids = list(dict.fromkeys(manifest["grid"].astype(str)))
    collector_grids = collectors["grid"].astype(str).tolist() if len(collectors) else []
    flags = swh_labelled_grids(review_paths, collector_grids, all_grids)
    wkt_by_grid = _confirmed_wkt_by_grid(src / "collectors.gpkg")

    chips_for = {}
    for index, mask, x, y, size in zip(
        chips.index, chips["mask"], chips["x"], chips["y"], chips["size"]
    ):
        chips_for.setdefault(mask, []).append((int(index), int(x), int(y), int(size)))
    tasks = []
    for index, row in manifest.iterrows():
        tasks.append({
            "idx": int(index),
            "grid": str(row["grid"]),
            "split": str(row["split"]),
            "src_mask": row["mask"],
            "dst_mask": rewrite_mask_path(row["mask"], src, out),
            "chips": chips_for.get(row["mask"], []),
            "n_hn": int(row["n_hn"]),
            "n_pv": int(row["n_pv"]),
            "n_ignore": int(row["n_ignore"]),
        })

    out.mkdir(parents=True, exist_ok=True)
    print(
        f"v2c patch {src.resolve()} -> {out.resolve()}  "
        f"tiles={len(tasks)} workers={WORKERS}",
        flush=True,
    )
    results = []
    # spawn, not fork: the parent has already opened the collector GeoPackage
    # through GDAL, and a forked child that opens masks deadlocks on that lock.
    # This is still a multiprocessing Pool of 8 workers over manifest rows.
    with get_context("spawn").Pool(
        WORKERS,
        initializer=_init_worker,
        initargs=(wkt_by_grid, str(src.resolve()), str(out.resolve())),
    ) as pool:
        total = len(tasks)
        for done, result in enumerate(pool.imap_unordered(process_manifest_row, tasks, chunksize=16), start=1):
            results.append(result)
            if done % 1000 == 0 or done == total:
                print(f"masks {done}/{total}", flush=True)
    if len(results) != len(manifest):
        raise AssertionError(f"expected {len(manifest)} tile results, got {len(results)}")
    by_index = {result["idx"]: result for result in results}

    n_bg = np.empty(len(manifest), dtype=np.int64)
    n_pv = np.empty(len(manifest), dtype=np.int64)
    n_swh = np.empty(len(manifest), dtype=np.int64)
    n_ignore = np.empty(len(manifest), dtype=np.int64)
    n2_before = np.empty(len(manifest), dtype=np.int64)
    n_remapped = np.empty(len(manifest), dtype=np.int64)
    wrote = np.zeros(len(manifest), dtype=bool)
    for index, result in by_index.items():
        n_bg[index] = result["n_bg"]
        n_pv[index] = result["n_pv"]
        n_swh[index] = result["n_swh"]
        n_ignore[index] = result["n_ignore"]
        n2_before[index] = result["n2_before"]
        n_remapped[index] = result["n_remapped"]
        wrote[index] = result["wrote"]
    if np.any(n_swh > n2_before):
        raise AssertionError("a tile gained value-2 pixels")

    manifest = manifest.rename(columns={
        "n_hn": "n_swh",
        "n_collector_ignore_px": "n_swh_ignore_px",
    }).drop(columns=["n_collector_hn_px"])
    manifest["n_bg"] = n_bg
    manifest["n_pv"] = n_pv
    manifest["n_swh"] = n_swh
    manifest["n_ignore"] = n_ignore
    manifest["n_swh_px"] = n_swh
    manifest["mask"] = [rewrite_mask_path(path, src, out) for path in manifest["mask"]]
    manifest["swh_labelled"] = manifest["grid"].astype(str).map(
        lambda grid: bool(flags[grid]["swh_labelled"])
    )
    manifest_front = [
        "grid", "split", "batch", "image", "mask", "width", "height", "px_m",
        "n_bg", "n_pv", "n_swh", "n_ignore", "n_swh_px", "n_swh_ignore_px",
        "swh_labelled",
    ]
    manifest_extra = [column for column in manifest.columns if column not in manifest_front]
    manifest = manifest[manifest_front + manifest_extra]
    if not bool((manifest["n_swh_px"] == manifest["n_swh"]).all()):
        raise AssertionError("n_swh_px must equal n_swh")

    chip_n = np.empty(len(chips), dtype=np.int64)
    seen = np.zeros(len(chips), dtype=bool)
    for result in results:
        for index, count in result["chips"]:
            chip_n[index] = count
            if seen[index]:
                raise AssertionError(f"chip {index} recomputed twice")
            seen[index] = True
    if len(chips) and not bool(seen.all()):
        raise AssertionError(f"{int((~seen).sum())} chips were not recomputed")
    chips = chips.rename(columns={"n_hn": "n_swh"})
    if len(chips):
        chips["n_swh"] = chip_n
    chips["mask"] = [rewrite_mask_path(path, src, out) for path in chips["mask"]]
    chips["swh_labelled"] = chips["grid"].astype(str).map(
        lambda grid: bool(flags[grid]["swh_labelled"])
    )
    chip_front = ["grid", "split", "image", "mask", "x", "y", "size", "n_pv", "n_swh", "swh_labelled"]
    chip_extra = [column for column in chips.columns if column not in chip_front]
    chips = chips[chip_front + chip_extra]

    grid_rows = []
    grid_split = manifest.groupby("grid", sort=False)["split"].first()
    for grid, split in grid_split.items():
        info = flags[str(grid)]
        grid_rows.append({
            "grid": str(grid),
            "split": str(split),
            "swh_labelled": bool(info["swh_labelled"]),
            "reason": info["reason"],
        })
    grids = pd.DataFrame(grid_rows, columns=["grid", "split", "swh_labelled", "reason"])
    grids = grids.sort_values(["split", "grid"]).reset_index(drop=True)

    splits = manifest["split"].astype(str).tolist()
    before = _sum_by_split(splits, n2_before)
    after = _sum_by_split(splits, n_swh)
    remapped = _sum_by_split(splits, n_remapped)
    collector_px = int(sum(result["collector_px"] for result in results))
    _print_verification(before, after, remapped, collector_px)
    n_labelled = int(sum(1 for info in flags.values() if info["swh_labelled"]))
    print(
        f"fp-review gpkg: {len(review_paths)}  "
        f"swh_labelled grids: {n_labelled} / {len(flags)}",
        flush=True,
    )

    stats = {
        "before": before,
        "after": after,
        "remapped": remapped,
        "collector_px": collector_px,
        "n_fp_review_gpkg": len(review_paths),
        "masks_rewritten": int(wrote.sum()),
        "masks_copied": int((~wrote).sum()),
    }
    summary = _build_summary(
        src / "summary.json", manifest, chips, grids, collectors, stats,
        src.resolve(), out.resolve(), args.fp_review_glob,
    )
    manifest.to_csv(out / "manifest.csv", index=False)
    chips.to_csv(out / "chips_512px_index.csv", index=False)
    grids.to_csv(out / "grids_swh_labelled.csv", index=False)
    for name in ("collectors.csv", "collectors.gpkg", "hard_negatives.csv"):
        destination = out / name
        shutil.copy2(src / name, destination)
        if os.path.samestat(os.stat(src / name), os.stat(destination)):
            raise AssertionError(f"output is a hardlink of {name}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["v2c"], indent=1))


if __name__ == "__main__":
    main()
