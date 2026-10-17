#!/usr/bin/env python3
"""Build the Cape Town seg trainset v2e from a real copy of v2d2.

v2e keeps v2d2 and puts the v2c masks and table rows back on three train
grids (G1971, G1687, G1240). Val and eval are not edited. Inputs are read
only. Outputs are refused when the destination already exists and is not empty.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

V2C = Path("/home/gao/data/ZAsolar/geid_temporal/ct_seg_trainset_v2c_20261015")
V2D2 = Path("/home/gao/data/ZAsolar/geid_temporal/ct_seg_trainset_v2d2_20261017")
LABELS_SRC = Path("/home/gao/zasolar_data/geid_temporal/ct_labels_corrected_v2d1_20261016")
ORIGINALS = Path("/home/gao/projects/ZAsolar/data/annotations/Capetown")
GRID_MANIFEST_SRC = Path("/home/gao/zasolar_data/capetown_manifest_v3_2_20261017/manifest.csv")

OUT = Path("/home/gao/data/ZAsolar/geid_temporal/ct_seg_trainset_v2e_20261017")
LABELS_OUT = Path("/home/gao/zasolar_data/geid_temporal/ct_labels_corrected_v2e_20261017")

PLAN_DIR = Path("/home/gao/projects/zasolar_plans_20261017/v2e")
MD5_BEFORE = PLAN_DIR / "md5_inputs_before.txt"
MD5_AFTER = PLAN_DIR / "md5_inputs_after.txt"

REVERT_GRIDS = ("G1971", "G1687", "G1240")
REVERT_SET = set(REVERT_GRIDS)
LABEL_OVERWRITE = {
    "G1971": "G1971_SAM2_260403.gpkg",
    "G1687": "G1687_SAM2_260403.gpkg",
    "G1240": "G1240_SAM2_260322.gpkg",
}
PATH_V2D2 = "ct_seg_trainset_v2d2_20261017"
PATH_V2C = "ct_seg_trainset_v2c_20261015"
PATH_V2E = "ct_seg_trainset_v2e_20261017"
LABELS_V2D1_TOKEN = "ct_labels_corrected_v2d1_20261016"
LABELS_V2E_TOKEN = "ct_labels_corrected_v2e_20261017"
GRID_MANIFEST_SPLITS = ("train", "val", "eval_orig10")
ROW_TABLES = (
    "manifest.csv",
    "chips_512px_index.csv",
    "collectors.csv",
    "hard_negatives.csv",
    "grids_swh_labelled.csv",
)

PROTECTED = (
    V2C,
    Path("/home/gao/data/ZAsolar/geid_temporal/ct_seg_trainset_v2d_20261016"),
    Path("/home/gao/data/ZAsolar/geid_temporal/ct_seg_trainset_v2d1_20261016"),
    V2D2,
    Path("/home/gao/data/ZAsolar/geid_temporal/ct_labels_corrected_v2d_20261016"),
    Path("/home/gao/data/ZAsolar/geid_temporal/ct_labels_corrected_v2d1_20261016"),
    ORIGINALS,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def assert_safe_output(path: Path) -> None:
    resolved = path.resolve()
    for protected in PROTECTED:
        root = protected.resolve()
        if resolved == root or root in resolved.parents:
            die(f"refuse to write {path} inside protected tree {protected}")


def refuse_if_nonempty(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        die(f"refuse: output path exists and is a file: {path}")
    if any(path.iterdir()):
        die(f"refuse: output dir exists and is non-empty: {path}")
    path.rmdir()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def list_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        kept = []
        for name in dirnames:
            child = Path(dirpath) / name
            if child.is_symlink():
                die(f"unexpected directory symlink under {root}: {child}")
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                die(f"unexpected symlink under {root}: {path}")
            if path.is_file():
                files.append(path)
    files.sort(key=lambda item: item.as_posix())
    return files


def write_md5_list(files: list[Path], dest: Path) -> dict[str, str]:
    assert_safe_output(dest)
    digests: dict[str, str] = {}
    lines: list[str] = []
    for index, path in enumerate(files, start=1):
        digest = md5_file(path)
        digests[path.as_posix()] = digest
        lines.append(f"{digest}  {path.as_posix()}\n")
        if index % 2000 == 0:
            log(f"  hashed {index}/{len(files)}")
    dest.write_text("".join(lines))
    return digests


def input_files() -> list[Path]:
    roots = (V2C / "data", V2D2 / "data", LABELS_SRC)
    for root in roots:
        if not root.is_dir():
            die(f"missing input directory: {root}")
    files: list[Path] = []
    for root in roots:
        files.extend(list_files(root))
    return files


def copy_file(src: Path, dst: Path) -> None:
    assert_safe_output(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    shutil.copystat(src, dst)
    if dst.is_symlink() or os.path.samestat(src.stat(), dst.stat()):
        die(f"copy is a link, not a real copy: {src} -> {dst}")


def copytree_real(src: Path, dst: Path) -> None:
    assert_safe_output(dst)
    if dst.exists():
        die(f"copy destination already exists: {dst}")

    def _copy(source: str, destination: str) -> None:
        copy_file(Path(source), Path(destination))

    shutil.copytree(src, dst, symlinks=False, copy_function=_copy)


def relative_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in list_files(root)}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            die(f"csv has no header: {path}")
        fields = list(reader.fieldnames)
        rows = list(reader)
    return fields, rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    assert_safe_output(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)


def remap_dataset_path(value: str) -> str:
    return value.replace(PATH_V2C, PATH_V2E).replace(PATH_V2D2, PATH_V2E)


def remap_row(row: dict[str, str]) -> dict[str, str]:
    return {key: remap_dataset_path(value) for key, value in row.items()}


def row_key(table: str, row: dict[str, str]) -> tuple[str, ...]:
    if table == "manifest.csv":
        return (row["grid"], row["image"])
    if table == "chips_512px_index.csv":
        return (row["grid"], row["image"], row["x"], row["y"], row["size"])
    if table == "collectors.csv":
        return (row["grid"], row["ref"], row["origin"], row["mask_value"], row["subtype"])
    if table == "hard_negatives.csv":
        return (row["grid"], row["split"], row["subtype"])
    if table == "grids_swh_labelled.csv":
        return (row["grid"], row["split"])
    die(f"no key for {table}")
    return ()


def replace_reverted_rows(table: str) -> dict[str, int]:
    """Keep v2d2 row positions. Fill reverted-grid slots from v2c by key."""
    fields, rows = read_csv(OUT / "data" / table)
    src_fields, src_rows = read_csv(V2C / "data" / table)
    if fields != src_fields:
        die(f"{table}: v2c columns {src_fields} differ from v2e columns {fields}")

    buckets: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    src_counts: Counter[str] = Counter()
    for row in src_rows:
        if row["grid"] in REVERT_SET:
            src_counts[row["grid"]] += 1
            buckets[row_key(table, row)].append(remap_row(row))

    replaced = 0
    unmatched = 0
    for index, row in enumerate(rows):
        if row["grid"] not in REVERT_SET:
            continue
        key = row_key(table, row)
        pool = buckets.get(key)
        if not pool:
            unmatched += 1
            continue
        rows[index] = pool.pop(0)
        replaced += 1

    leftovers = sum(len(pool) for pool in buckets.values())
    if unmatched or leftovers:
        die(
            f"{table}: key match failed for {REVERT_GRIDS} "
            f"(unmatched v2d2 slots {unmatched}, leftover v2c rows {leftovers})"
        )
    dst_counts = Counter(row["grid"] for row in rows if row["grid"] in REVERT_SET)
    if dict(dst_counts) != dict(src_counts):
        die(f"{table}: reverted row counts {dict(dst_counts)} != v2c {dict(src_counts)}")

    write_csv(OUT / "data" / table, fields, rows)
    return {"replaced": replaced, **{f"rows_{grid}": src_counts[grid] for grid in REVERT_GRIDS}}


def drop_tile_diff_rows() -> tuple[int, int]:
    path = OUT / "data" / "v2d_tile_diff.csv"
    fields, rows = read_csv(path)
    kept = [row for row in rows if row["grid"] not in REVERT_SET]
    dropped = len(rows) - len(kept)
    if any(row["grid"] in REVERT_SET for row in kept):
        die("v2d_tile_diff.csv still contains a reverted grid")
    write_csv(path, fields, kept)
    return dropped, len(kept)


def rewrite_csv_paths() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted((OUT / "data").rglob("*.csv")):
        text = path.read_text()
        found = text.count(PATH_V2D2)
        if found:
            assert_safe_output(path)
            path.write_text(text.replace(PATH_V2D2, PATH_V2E))
        counts[path.name] = found
    return counts


def replace_masks() -> dict[str, int]:
    counts: dict[str, int] = {}
    for grid in REVERT_GRIDS:
        src = V2C / "data" / "train" / grid
        dst = OUT / "data" / "train" / grid
        if not src.is_dir() or not dst.is_dir():
            die(f"missing train mask dir for {grid}: src={src.is_dir()} dst={dst.is_dir()}")
        src_names = relative_files(src)
        dst_names = relative_files(dst)
        if src_names != dst_names:
            die(
                f"{grid}: mask filename set differs "
                f"(only v2c {sorted(src_names - dst_names)[:5]}, "
                f"only v2d2 {sorted(dst_names - src_names)[:5]})"
            )
        for rel in dst_names:
            target = dst / rel
            assert_safe_output(target)
            target.unlink()
        for rel in sorted(src_names):
            copy_file(src / rel, dst / rel)
        counts[grid] = len(src_names)
    return counts


def nonempty_geometries(path: Path) -> int:
    import fiona
    from shapely.geometry import shape

    count = 0
    for layer in fiona.listlayers(path):
        with fiona.open(path, layer=layer) as src:
            for feat in src:
                geom = feat["geometry"]
                if not geom:
                    continue
                if shape(geom).is_empty:
                    continue
                count += 1
    return count


def insert_v2e_summary(pv_px: dict[str, int], chips: dict[str, int], tiles: dict[str, int]) -> None:
    path = OUT / "data" / "summary.json"
    original = path.read_text()
    if not original.endswith("\n"):
        original += "\n"
    body = original.rstrip("\n")
    if not body.endswith("}"):
        die("summary.json does not end with an object")
    close = body.rfind("}")
    previous = body.rfind("}", 0, close)
    if previous < 0:
        die("summary.json has no nested object to append after")
    payload = {
        "from": (
            str(V2D2)
            + " with train masks and rows of G1971, G1687, G1240 reverted to "
            + str(V2C)
        ),
        "reverted_grids": list(REVERT_GRIDS),
        "pv_px": {split: pv_px[split] for split in ("eval", "train", "val")},
        "chips": {split: chips[split] for split in ("eval", "train", "val")},
        "tiles": {split: tiles[split] for split in ("eval", "train", "val")},
    }
    encoded = json.dumps(payload, indent=1)
    inner = [" " + line if line else line for line in encoded.splitlines()[1:-1]]
    block = ' "v2e": {\n' + "\n".join(inner) + "\n }"
    updated = body[: previous + 1] + "," + body[previous + 1 : close] + block + "\n}\n"
    parsed = json.loads(updated)
    if set(parsed) - set(json.loads(original)) != {"v2e"}:
        die("summary.json update changed keys other than adding v2e")
    for key in ("from", "reverted_grids", "pv_px", "chips", "tiles"):
        if key not in parsed["v2e"]:
            die(f"v2e summary key missing {key}")
    assert_safe_output(path)
    path.write_text(updated)


def write_readme(stats: dict) -> None:
    pv = stats["pv_px"]
    chips = stats["chips"]
    tiles = stats["tiles"]
    grids = stats["grids"]
    polygons = stats["polygons"]
    text = f"""# CT seg trainset v2e (2026-10-17 NZDT)

Phase-1 training set. v2e is a real copy of `ct_seg_trainset_v2d2_20261017` with three train grids put back on the v2c human labels: G1971, G1687 and G1240. The v2d SAM3 thin-strip replacements on those three grids are not in this set.

Kept from v2d2: train grids G1189, G1190 and G1238; the G1630 seg130 fix; the G1972 row-113 tightening; the G1682 zero-area drops. v2d sliver-to-255 edits in train G1749, G1972 and G2029 are still here. They are not SAM3 thin-strip replacements.

Val and eval are the v2d2 files, byte for byte, apart from the dataset name inside CSV paths (`ct_seg_trainset_v2d2_20261017` rewritten to `ct_seg_trainset_v2e_20261017`). G2028 is val, so its SAM3 thin-strip edit stays. G1630 is val and keeps both the seg130 fix and its SAM3 thin-strip adds. Do not read v2e val as v2c labels.

## How it was built

`build_v2e.py` copied this directory from v2d2 (no links), then:

- replaced the train mask files of G1971, G1687 and G1240 with the v2c masks (same filenames)
- replaced those grids' rows in `manifest.csv`, `chips_512px_index.csv`, `collectors.csv`, `hard_negatives.csv` and `grids_swh_labelled.csv` with the v2c rows, paths remapped to v2e, in the original v2d2 row positions
- dropped those three grids from `v2d_tile_diff.csv`

Labels live in `ct_labels_corrected_v2e_20261017`, a copy of `ct_labels_corrected_v2d1_20261016`. Only these three files were overwritten, from `projects/ZAsolar/data/annotations/Capetown`:

- `Capetown/G1971_SAM2_260403.gpkg` ({polygons["G1971"]} non-empty geometries)
- `Capetown/G1687_SAM2_260403.gpkg` ({polygons["G1687"]} non-empty geometries)
- `Capetown/G1240_SAM2_260322.gpkg` ({polygons["G1240"]} non-empty geometries)

`data/grid_manifest_v2e.csv` is the v3.2 manifest restricted to splits train, val and eval_orig10. `label_file` points at this labels copy. `polygon_count` and `md5` were recomputed for the three reverted grids. Other columns, including `status` and `notes`, are unchanged from v3.2, so those three rows still say `frozen_v2d2`.

## Counts

| split | grids | tiles | chips | pv pixels |
|---|---:|---:|---:|---:|
| train | {grids["train"]} | {tiles["train"]} | {chips["train"]} | {pv["train"]} |
| val | {grids["val"]} | {tiles["val"]} | {chips["val"]} | {pv["val"]} |
| eval | {grids["eval"]} | {tiles["eval"]} | {chips["eval"]} | {pv["eval"]} |

Train polygon_count in `grid_manifest_v2e.csv` (non-empty geometries, v3.2 definition): {stats["train_polygons"]}.

`summary.json` keeps the headline `grids` / `tiles` / `pv_px` / `chips` objects it inherited from the v2d2 file. Those headline numbers are the older pre-v2d totals. Use the `v2e` key, or the table above.

## Not done

The hard-negative coordinate cache (`hn_clean`) has not been rebuilt. Masks changed on G1971, G1687 and G1240, so do not train from this directory until that cache and `train2_v2e.py` exist. This build did not use a GPU and did not train.

Mask scheme is unchanged: 0 background, 1 PV, 2 solar-thermal collector, 255 ignore.
"""
    path = OUT / "README.md"
    assert_safe_output(path)
    path.write_text(text)


def write_grid_manifest() -> dict:
    fields, rows = read_csv(GRID_MANIFEST_SRC)
    selected = [dict(row) for row in rows if row["split"] in GRID_MANIFEST_SPLITS]
    if not selected:
        die("v3.2 manifest has no train/val/eval_orig10 rows")
    by_source = {row["source_grid_id"]: row for row in selected}
    missing = [grid for grid in REVERT_GRIDS if grid not in by_source]
    if missing:
        die(f"v3.2 manifest is missing reverted grids {missing}")

    polygon_counts: dict[str, int] = {}
    for grid, filename in LABEL_OVERWRITE.items():
        label_path = LABELS_OUT / "Capetown" / filename
        original = ORIGINALS / filename
        if md5_file(label_path) != md5_file(original):
            die(f"{filename} in the v2e labels copy is not the annotation original")
        count = nonempty_geometries(label_path)
        digest = md5_file(label_path)
        row = by_source[grid]
        row["label_file"] = row["label_file"].replace(LABELS_V2D1_TOKEN, LABELS_V2E_TOKEN)
        if LABELS_V2E_TOKEN not in row["label_file"]:
            die(f"{grid} label_file was not repointed: {row['label_file']}")
        if Path(row["label_file"]).resolve() != label_path.resolve():
            die(f"{grid} label_file {row['label_file']} does not resolve to {label_path}")
        row["polygon_count"] = str(count)
        row["md5"] = digest
        polygon_counts[grid] = count

    for row in selected:
        if row["source_grid_id"] in REVERT_SET:
            continue
        if row["label_file"]:
            row["label_file"] = row["label_file"].replace(LABELS_V2D1_TOKEN, LABELS_V2E_TOKEN)
            if not Path(row["label_file"]).is_file():
                die(f"label_file missing after repoint: {row['label_file']}")
        elif row["polygon_count"] not in ("0", ""):
            die(f"{row['grid_id']} has no label_file but polygon_count {row['polygon_count']}")

    dest = OUT / "data" / "grid_manifest_v2e.csv"
    write_csv(dest, fields, selected)
    train_polygons = sum(int(row["polygon_count"] or 0) for row in selected if row["split"] == "train")
    return {
        "rows": len(selected),
        "by_split": dict(Counter(row["split"] for row in selected)),
        "polygons": polygon_counts,
        "train_polygons": train_polygons,
    }


def split_counts() -> dict:
    _, manifest = read_csv(OUT / "data" / "manifest.csv")
    _, chips = read_csv(OUT / "data" / "chips_512px_index.csv")
    grids: dict[str, set[str]] = defaultdict(set)
    pv_px: Counter[str] = Counter()
    tiles: Counter[str] = Counter()
    for row in manifest:
        grids[row["split"]].add(row["grid"])
        tiles[row["split"]] += 1
        pv_px[row["split"]] += int(row["n_pv"])
    chip_counts: Counter[str] = Counter(row["split"] for row in chips)
    return {
        "grids": {split: len(items) for split, items in grids.items()},
        "tiles": dict(tiles),
        "chips": dict(chip_counts),
        "pv_px": dict(pv_px),
        "manifest_rows": len(manifest),
        "chip_rows": len(chips),
    }


def assert_masks_match_sources() -> None:
    log("checking mask bytes against v2c and v2d2")
    data = OUT / "data"
    for split in ("train", "val", "eval"):
        for grid_dir in sorted((data / split).iterdir()):
            if not grid_dir.is_dir():
                continue
            source_root = V2C if grid_dir.name in REVERT_SET else V2D2
            if split != "train":
                source_root = V2D2
                if grid_dir.name in REVERT_SET:
                    die(f"reverted grid {grid_dir.name} is in {split}, not train")
            src = source_root / "data" / split / grid_dir.name
            if relative_files(grid_dir) != relative_files(src):
                die(f"filename set drifted for {split}/{grid_dir.name}")
            for path in list_files(grid_dir):
                rel = path.relative_to(data)
                src_path = source_root / "data" / rel
                if md5_file(path) != md5_file(src_path):
                    die(f"mask bytes differ from {source_root.name}: {rel}")


def assert_unchanged_rows() -> None:
    """Non-reverted rows stay on the v2d2 values, with only the dataset token changed."""
    for table in ROW_TABLES + ("v2d_tile_diff.csv",):
        out_fields, out_rows = read_csv(OUT / "data" / table)
        src_fields, src_rows = read_csv(V2D2 / "data" / table)
        if out_fields != src_fields:
            die(f"{table}: columns changed")
        if table == "v2d_tile_diff.csv":
            src_rows = [row for row in src_rows if row["grid"] not in REVERT_SET]
        else:
            src_rows = [remap_row(row) for row in src_rows if row["grid"] not in REVERT_SET]
            out_kept = [row for row in out_rows if row["grid"] not in REVERT_SET]
            if out_kept != src_rows:
                die(f"{table}: a non-reverted row changed")
            continue
        if out_rows != [remap_row(row) for row in src_rows]:
            die("v2d_tile_diff.csv kept rows changed beyond the dataset token")


def assert_reverted_rows_match_v2c() -> None:
    for table in ROW_TABLES:
        _, out_rows = read_csv(OUT / "data" / table)
        _, src_rows = read_csv(V2C / "data" / table)
        src_by_key: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in src_rows:
            if row["grid"] in REVERT_SET:
                src_by_key[row_key(table, row)].append(remap_row(row))
        for row in out_rows:
            if row["grid"] not in REVERT_SET:
                continue
            pool = src_by_key.get(row_key(table, row))
            if not pool or pool.pop(0) != row:
                die(f"{table}: reverted row is not the remapped v2c row")
        if any(src_by_key.values()):
            die(f"{table}: v2c reverted rows were not all used")


def assert_no_old_dataset_token() -> None:
    for path in (OUT / "data").rglob("*.csv"):
        text = path.read_text()
        if PATH_V2D2 in text or PATH_V2C in text:
            die(f"{path.name} still contains a v2c or v2d2 dataset path")


def assert_no_links(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                die(f"symlink in output: {path}")


def assert_label_copy() -> None:
    src_files = list_files(LABELS_SRC)
    dst_files = list_files(LABELS_OUT)
    src_rel = {path.relative_to(LABELS_SRC).as_posix() for path in src_files}
    dst_rel = {path.relative_to(LABELS_OUT).as_posix() for path in dst_files}
    if src_rel != dst_rel:
        die("v2e labels copy does not have the same filenames as v2d1 labels")
    overwritten = {f"Capetown/{name}" for name in LABEL_OVERWRITE.values()}
    for rel in sorted(src_rel):
        src = LABELS_SRC / rel
        dst = LABELS_OUT / rel
        same = md5_file(src) == md5_file(dst)
        if rel in overwritten and same:
            die(f"reverted label was not overwritten: {rel}")
        if rel not in overwritten and not same:
            die(f"label file changed unexpectedly: {rel}")
        original = ORIGINALS / Path(rel).name
        if rel in overwritten and md5_file(dst) != md5_file(original):
            die(f"reverted label does not match the annotation original: {rel}")


def preflight() -> None:
    """Read-only checks. A failure here writes nothing except the later md5 list."""
    for grid in REVERT_GRIDS:
        src = V2C / "data" / "train" / grid
        dst = V2D2 / "data" / "train" / grid
        if relative_files(src) != relative_files(dst):
            die(f"preflight: {grid} mask filenames differ between v2c and v2d2")
    for table in ROW_TABLES:
        _, left = read_csv(V2D2 / "data" / table)
        _, right = read_csv(V2C / "data" / table)
        left_keys = [row_key(table, row) for row in left if row["grid"] in REVERT_SET]
        right_keys = [row_key(table, row) for row in right if row["grid"] in REVERT_SET]
        if Counter(left_keys) != Counter(right_keys):
            die(f"preflight: {table} key multiset differs for {REVERT_GRIDS}")
    _, manifest = read_csv(V2D2 / "data" / "manifest.csv")
    splits = {row["grid"]: row["split"] for row in manifest}
    for grid in REVERT_GRIDS:
        if splits.get(grid) != "train":
            die(f"preflight: {grid} split is {splits.get(grid)}, expected train")
    if splits.get("G2028") != "val":
        die(f"preflight: G2028 split is {splits.get('G2028')}, expected val")
    expected = {"G1971": 354, "G1687": 200, "G1240": 7}
    found = {
        grid: nonempty_geometries(ORIGINALS / filename)
        for grid, filename in LABEL_OVERWRITE.items()
    }
    if found != expected:
        die(f"preflight: original non-empty polygon counts {found} != {expected}")
    log(f"preflight ok; original polygons {found}")


def main() -> None:
    for path in (V2C / "data", V2D2 / "data", LABELS_SRC, ORIGINALS, GRID_MANIFEST_SRC):
        if not path.exists():
            die(f"missing input: {path}")
    for filename in LABEL_OVERWRITE.values():
        if not (ORIGINALS / filename).is_file():
            die(f"missing annotation original: {filename}")
        if not (LABELS_SRC / "Capetown" / filename).is_file():
            die(f"missing v2d1 label to overwrite: {filename}")

    assert_safe_output(OUT)
    assert_safe_output(LABELS_OUT)
    refuse_if_nonempty(OUT)
    refuse_if_nonempty(LABELS_OUT)
    preflight()

    files = input_files()
    log(f"hashing {len(files)} input files -> {MD5_BEFORE.name}")
    before = write_md5_list(files, MD5_BEFORE)

    log(f"copying {V2D2} -> {OUT}")
    copytree_real(V2D2, OUT)
    log(f"copying labels {LABELS_SRC} -> {LABELS_OUT}")
    copytree_real(LABELS_SRC, LABELS_OUT)

    rewrites = rewrite_csv_paths()
    log(f"csv path rewrites: {rewrites}")
    mask_counts = replace_masks()
    log(f"masks replaced: {mask_counts}")
    row_counts = {table: replace_reverted_rows(table) for table in ROW_TABLES}
    log(f"rows replaced: {row_counts}")
    dropped, kept = drop_tile_diff_rows()
    log(f"v2d_tile_diff.csv dropped {dropped}, kept {kept}")

    for grid, filename in LABEL_OVERWRITE.items():
        copy_file(ORIGINALS / filename, LABELS_OUT / "Capetown" / filename)
        log(f"label overwrite {filename}")

    stats = split_counts()
    manifest_info = write_grid_manifest()
    stats["polygons"] = manifest_info["polygons"]
    stats["train_polygons"] = manifest_info["train_polygons"]
    stats["grid_manifest_rows"] = manifest_info["rows"]
    stats["grid_manifest_by_split"] = manifest_info["by_split"]
    insert_v2e_summary(stats["pv_px"], stats["chips"], stats["tiles"])
    write_readme(stats)

    log("integrity checks")
    assert_no_links(OUT)
    assert_no_links(LABELS_OUT)
    assert_masks_match_sources()
    assert_unchanged_rows()
    assert_reverted_rows_match_v2c()
    assert_no_old_dataset_token()
    assert_label_copy()
    for split, expected in (("train", 79), ("val", 14), ("eval", 5)):
        if stats["grids"].get(split) != expected:
            die(f"{split} grid count {stats['grids'].get(split)} != {expected}")
    if stats["polygons"] != {"G1971": 354, "G1687": 200, "G1240": 7}:
        die(f"reverted polygon counts {stats['polygons']} differ from the annotation originals")
    if stats["train_polygons"] != 5645:
        die(f"train polygon_count sum {stats['train_polygons']} != 5645")

    log(f"hashing inputs again -> {MD5_AFTER.name}")
    after = write_md5_list(files, MD5_AFTER)
    if before != after or MD5_BEFORE.read_text() != MD5_AFTER.read_text():
        die("input md5 changed during the build")

    checksum_path = OUT / "MD5SUMS"
    output_files = [path for path in list_files(OUT) if path != checksum_path]
    output_files.extend(list_files(LABELS_OUT))
    output_files.sort(key=lambda item: item.as_posix())
    log(f"hashing {len(output_files)} v2e outputs -> MD5SUMS")
    write_md5_list(output_files, checksum_path)
    listed = [line.split("  ", 1)[1].strip() for line in checksum_path.read_text().splitlines()]
    if checksum_path.as_posix() in listed:
        die("MD5SUMS lists itself")
    if len(listed) != len(output_files):
        die("MD5SUMS row count does not match the output file list")

    stats["masks_replaced"] = mask_counts
    stats["rows_replaced"] = row_counts
    stats["tile_diff_dropped"] = dropped
    stats["tile_diff_kept"] = kept
    stats["csv_path_rewrites"] = rewrites
    stats["input_files"] = len(files)
    stats["output_files_in_MD5SUMS"] = len(output_files)
    log("COUNTS " + json.dumps(stats, sort_keys=True))
    log("v2e build ok")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
