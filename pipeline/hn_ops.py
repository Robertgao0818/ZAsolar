"""Hard-negative operations — stable public API for the dataset builder.

Extracts reusable HN functions from the CLI scripts in
``scripts/training/``.  The builder calls these functions directly;
the CLI scripts become thin wrappers.

Public API
----------
- ``extract_reviewed_fp_hn(grids, output_dir, ...) -> HNResult``
- ``extract_small_fp_hn(shortlist_csv, output_dir, ...) -> HNResult``
- ``merge_hn_into_coco(base_dir, hn_images_list, output_dir) -> MergeResult``
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HNResult:
    """Result of an HN extraction step."""
    images: list[dict] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    source_type: str = ""
    n_grids: int = 0
    n_chips: int = 0


@dataclass
class MergeResult:
    """Result of merging HN chips into a base COCO dataset."""
    total_train_images: int = 0
    total_val_images: int = 0
    total_annotations: int = 0
    n_base_positive: int = 0
    n_base_easy_neg: int = 0
    n_hn_chips: int = 0
    hn_ratio: float = 0.0


def extract_reviewed_fp_hn(
    grids: list[str],
    output_dir: Path,
    chip_size: int = 400,
    tiles_root: Path | None = None,
    img_id_start: int = 900000,
) -> HNResult:
    """Extract HN chips from reviewed FP predictions.

    Wraps the logic from ``scripts/training/export_targeted_hn.py``.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.training.export_targeted_hn import (
        load_fp_locations, extract_fp_chips,
    )

    fp_by_grid = load_fp_locations(grids)
    total_fp = sum(len(gdf) for gdf in fp_by_grid.values())

    if total_fp == 0:
        return HNResult(source_type="reviewed_fp_hn")

    images, provenance = extract_fp_chips(
        fp_by_grid, output_dir,
        chip_size=chip_size,
        tiles_root=tiles_root,
    )

    # Remap IDs if needed
    if img_id_start != 900000:
        offset = img_id_start - 900000
        for img in images:
            img["id"] += offset
        for p in provenance:
            p["image_id"] += offset

    return HNResult(
        images=images,
        provenance=provenance,
        source_type="reviewed_fp_hn",
        n_grids=len(fp_by_grid),
        n_chips=len(images),
    )


def extract_small_fp_hn(
    shortlist_csv: Path,
    output_dir: Path,
    chip_size: int = 400,
    sample_rate: float = 0.5,
    tiles_root: Path | None = None,
    seed: int = 42,
    img_id_start: int = 950000,
) -> HNResult:
    """Extract HN chips from curated small-FP shortlist.

    Wraps the logic from ``scripts/training/export_v4_hn.py``.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.training.export_v4_hn import (
        load_shortlist, stratified_sample,
        load_fp_geometries, extract_hn_chips,
    )

    shortlist = load_shortlist(shortlist_csv)
    if len(shortlist) == 0:
        return HNResult(source_type="small_fp_hn")

    if sample_rate < 1.0:
        sampled = stratified_sample(shortlist, sample_rate, seed=seed)
    else:
        sampled = shortlist

    fp_by_grid = load_fp_geometries(sampled)
    if not fp_by_grid:
        return HNResult(source_type="small_fp_hn")

    images, provenance = extract_hn_chips(
        fp_by_grid, output_dir,
        chip_size=chip_size,
        tiles_root=tiles_root,
    )

    # Remap IDs to target segment
    offset = img_id_start - 900000
    for img in images:
        img["id"] += offset
    for p in provenance:
        p["image_id"] += offset

    return HNResult(
        images=images,
        provenance=provenance,
        source_type="small_fp_hn",
        n_grids=len(fp_by_grid),
        n_chips=len(images),
    )


def extract_negative_pool_hn(
    archetypes: list[str],
    output_dir: Path,
    chip_size: int = 400,
    min_confidence: str | None = None,
    regions: list[str] | None = None,
    tiles_root: Path | None = None,
    img_id_start: int = 960000,
    manifest_csv: Path | None = None,
    require_training_eligible: bool = True,
) -> HNResult:
    """Extract HN chips from the project-level negative pool.

    Reads ``data/negative_pool/manifest.csv`` (the monotonically-accumulating
    archetype catalog), filters by ``archetypes`` / ``regions`` /
    ``min_confidence`` (A1 >= A2 >= A3 floor), and crops a ``chip_size`` chip
    centred on each row's ``bbox_geo_wkt`` from the row's imagery_layer tiles.

    Rows tagged ``actually_pv_mislabeled`` are always excluded (per the pool
    README: deprecation trail, never trained on).

    ``require_training_eligible`` (default True) honours the manifest's
    ``training_eligible`` column (written by ``backfill_geometry.py``): rows
    flagged ``false`` are provenance-only (e.g. geid_2024_02 chips gated out by
    the imagery-layer balance rule — see the F1-gap plan C-1) and are skipped
    so the GEID appearance domain cannot silently monopolise the HN stream.
    Pass ``False`` only for diagnostics / breadth audits over the full pool.

    NOTE: the pool is a provenance manifest — chips are derived on demand
    (rule 08-runpod-large-files).  Rows without a populated ``bbox_geo_wkt``
    cannot be cropped and are skipped with a logged count; if no row carries
    geometry this returns ``n_chips=0``.
    """
    import csv as _csv
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parent.parent
    if manifest_csv is None:
        manifest_csv = repo_root / "data" / "negative_pool" / "manifest.csv"
    if not manifest_csv.exists():
        print(f"[negative_pool] manifest not found: {manifest_csv}")
        return HNResult(source_type="negative_pool")

    conf_rank = {"A1": 3, "A2": 2, "A3": 1}
    floor = conf_rank.get(min_confidence, 0) if min_confidence else 0
    arch_set = set(archetypes)
    region_set = set(regions) if regions else None

    rows: list[dict] = []
    skipped_no_geom = 0
    skipped_not_eligible = 0
    with open(manifest_csv, newline="") as f:
        for row in _csv.DictReader(f):
            if row.get("archetype") == "actually_pv_mislabeled":
                continue
            if arch_set and row.get("archetype") not in arch_set:
                continue
            if region_set and row.get("region") not in region_set:
                continue
            if floor and conf_rank.get(row.get("archetype_confidence"), 0) < floor:
                continue
            if require_training_eligible and not _is_training_eligible(row):
                skipped_not_eligible += 1
                continue
            wkt = (row.get("bbox_geo_wkt") or "").strip()
            if not wkt:
                skipped_no_geom += 1
                continue
            rows.append(row)

    if skipped_not_eligible:
        print(f"[negative_pool] {skipped_not_eligible} matching rows are "
              f"provenance-only (training_eligible=false) — skipped "
              f"(imagery-layer balance gate)")
    if skipped_no_geom:
        print(f"[negative_pool] {skipped_no_geom} matching rows lack "
              f"bbox_geo_wkt — cannot crop, skipped (backfill geometry to "
              f"enable this HN stream)")
    if not rows:
        print(f"[negative_pool] 0 croppable chips for archetypes={sorted(arch_set)} "
              f"regions={regions} min_confidence={min_confidence}")
        return HNResult(source_type="negative_pool", n_grids=0, n_chips=0)

    # Crop a chip per row, centred on the bbox centroid.
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    from shapely import wkt as _wkt

    chip_dir = output_dir / "train"
    chip_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    provenance: list[dict] = []
    img_id = img_id_start
    grids_seen: set[str] = set()
    for row in rows:
        region = row["region"]
        grid_id = row["grid_id"]
        layer = row.get("imagery_layer") or None
        try:
            geom = _wkt.loads(row["bbox_geo_wkt"])
        except Exception:  # noqa: BLE001
            continue

        tile_path = _resolve_tile_for_geom(
            geom, grid_id, region, layer, tiles_root
        )
        if tile_path is None:
            print(f"[negative_pool] {grid_id}: no tile for chip "
                  f"{row.get('chip_id')}; skip")
            continue

        with rasterio.open(tile_path) as src:
            # bbox_geo_wkt is stored in EPSG:4326 (backfill_geometry.py); the
            # tile may be in any native CRS (e.g. vexcel_2024 / aerial_legacy
            # are EPSG:3857). Reproject the centroid into the tile's CRS before
            # indexing pixels — never assume lon/lat == raster units
            # (rule 06-multi-city: branch/look up CRS, do not assume EPSG).
            x_native, y_native = _geom_xy_in_crs(geom, src.crs)
            py, px = src.index(x_native, y_native)
            x0 = max(0, int(px - chip_size // 2))
            y0 = max(0, int(py - chip_size // 2))
            x0 = min(x0, max(0, src.width - chip_size))
            y0 = min(y0, max(0, src.height - chip_size))
            w = min(chip_size, src.width - x0)
            h = min(chip_size, src.height - y0)
            if w < chip_size * 0.5 or h < chip_size * 0.5:
                continue
            window = Window(x0, y0, w, h)
            data = src.read(window=window)
            if w < chip_size or h < chip_size:
                padded = np.zeros(
                    (data.shape[0], chip_size, chip_size), dtype=data.dtype
                )
                padded[:, :h, :w] = data
                data = padded
            if np.all(data >= 245):
                continue  # blank tile margin

            chip_name = (
                f"np_{region}_{grid_id}_{row.get('archetype', 'na')}"
                f"_{img_id}.tif"
            )
            chip_path = chip_dir / chip_name
            profile = src.profile.copy()
            for key in ("photometric", "compress", "jpeg_quality",
                        "jpegtablesmode"):
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

        images.append({
            "id": img_id,
            "file_name": f"train/{chip_name}",
            "width": chip_size,
            "height": chip_size,
            "positive": False,
            "fp_source": grid_id,
        })
        provenance.append({
            "image_id": img_id,
            "chip_file": chip_name,
            "source_tile": tile_path.stem,
            "x0": x0,
            "y0": y0,
            "width": w,
            "height": h,
            "n_annotations": 0,
            "split": "train",
            "source_type": "negative_pool",
            "archetype": row.get("archetype", ""),
            "region": region,
            "imagery_layer": layer or "",
            "chip_id": row.get("chip_id", ""),
        })
        img_id += 1
        grids_seen.add(grid_id)

    return HNResult(
        images=images,
        provenance=provenance,
        source_type="negative_pool",
        n_grids=len(grids_seen),
        n_chips=len(images),
    )


def _is_training_eligible(row: dict) -> bool:
    """True if the manifest row may enter a training bundle.

    Honours the ``training_eligible`` column when present (``true``/``false``);
    rows that predate that column (no value) default to eligible so the gate is
    purely additive.  See ``backfill_geometry.py`` for who writes it.
    """
    val = (row.get("training_eligible") or "").strip().lower()
    if val == "false":
        return False
    return True  # "true" or absent (legacy rows)


def _geom_xy_in_crs(geom, dst_crs):
    """Return the geometry centroid (x, y) reprojected into ``dst_crs``.

    ``bbox_geo_wkt`` rows are stored in EPSG:4326 (see ``backfill_geometry.py``).
    Tiles may be in any native CRS (vexcel_2024 / aerial_legacy are EPSG:3857),
    so the centroid must be reprojected before it can be compared against tile
    bounds or fed to ``src.index`` — comparing lon/lat against metre-scale
    bounds silently resolves nothing (rule 06-multi-city: never assume EPSG).

    When ``dst_crs`` is already geographic / EPSG:4326 (CT aerial layers) the
    transform is a no-op and the original lon/lat is returned unchanged.
    """
    centroid = geom.centroid
    lon, lat = centroid.x, centroid.y
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


def _resolve_tile_for_geom(geom, grid_id, region, layer, tiles_root):
    """Return the GeoTIFF that contains ``geom`` for a negative-pool row.

    Branches on ``chunked`` vs ``mosaic`` file layout (rule 06-multi-city): a
    mosaic resolves to a single file; a chunked layer is a directory of geo
    chips and we pick the chunk whose bounds contain the geometry centroid.

    The geometry is stored in EPSG:4326; each candidate tile's bounds are in
    that tile's native CRS, so the centroid is reprojected per-tile via
    ``src.crs`` before the contains-check (do not assume lon/lat == tile units).
    """
    from pathlib import Path as _Path

    import rasterio

    from core.grid_utils import resolve_tiles_dir

    try:
        base = (tiles_root if tiles_root is not None
                else resolve_tiles_dir(grid_id, region=region,
                                       imagery_layer=layer))
    except Exception as exc:  # noqa: BLE001
        print(f"[negative_pool] {grid_id}: tile resolve failed ({exc})")
        return None

    base = _Path(base)
    if base.is_file():
        return base  # mosaic layout

    if tiles_root is not None and base.is_dir():
        # tiles_root override may point one level above the grid subdir
        cand = base / grid_id
        if cand.is_dir():
            base = cand

    if not base.is_dir():
        return None

    for tif in sorted(base.glob("*.tif")):
        try:
            with rasterio.open(tif) as src:
                left, bottom, right, top = src.bounds
                x_native, y_native = _geom_xy_in_crs(geom, src.crs)
                if left <= x_native <= right and bottom <= y_native <= top:
                    return tif
        except Exception:  # noqa: BLE001
            continue
    return None


def merge_hn_into_coco(
    base_dir: Path,
    hn_results: list[HNResult],
    output_dir: Path,
) -> MergeResult:
    """Merge HN chips into a base COCO dataset.

    Hard-links or copies base images and appends HN images to train.json.
    Val split is unchanged.
    """
    with open(base_dir / "train.json") as f:
        base_train = json.load(f)
    with open(base_dir / "val.json") as f:
        base_val = json.load(f)

    # Hard-link base images to output
    for split in ("train", "val"):
        src_split = base_dir / split
        dst_split = output_dir / split
        dst_split.mkdir(parents=True, exist_ok=True)
        if src_split.exists():
            for img_file in src_split.iterdir():
                dst_file = dst_split / img_file.name
                if not dst_file.exists():
                    try:
                        dst_file.hardlink_to(img_file)
                    except OSError:
                        shutil.copy2(img_file, dst_file)

    # Collect all HN images
    all_hn_images: list[dict] = []
    hn_descriptions: list[str] = []
    for hn in hn_results:
        all_hn_images.extend(hn.images)
        if hn.n_chips > 0:
            hn_descriptions.append(f"{hn.source_type}({hn.n_chips})")

    # Merge into train
    merged_images = base_train["images"] + all_hn_images
    merged_annots = base_train["annotations"]  # HN chips have no annotations

    merged = {
        "info": {
            **base_train["info"],
            "description": (
                base_train["info"].get("description", "")
                + " + " + ", ".join(hn_descriptions)
            ),
        },
        "licenses": base_train.get("licenses", []),
        "categories": base_train["categories"],
        "images": merged_images,
        "annotations": merged_annots,
    }

    with open(output_dir / "train.json", "w") as f:
        json.dump(merged, f)
    with open(output_dir / "val.json", "w") as f:
        json.dump(base_val, f)

    # Copy base provenance files to output
    for prov_name in ("train_provenance.csv", "val_provenance.csv"):
        src_prov = base_dir / prov_name
        if src_prov.exists():
            dst_prov = output_dir / prov_name
            if not dst_prov.exists():
                shutil.copy2(src_prov, dst_prov)

    # Write combined HN provenance
    import csv
    all_prov = []
    for hn in hn_results:
        all_prov.extend(hn.provenance)
    if all_prov:
        prov_path = output_dir / "hn_provenance.csv"
        with open(prov_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_prov[0].keys())
            writer.writeheader()
            writer.writerows(all_prov)

    n_base_pos = sum(1 for img in base_train["images"] if img.get("positive", True))
    n_base_neg = len(base_train["images"]) - n_base_pos
    n_hn = len(all_hn_images)
    total = len(merged_images)

    return MergeResult(
        total_train_images=total,
        total_val_images=len(base_val["images"]),
        total_annotations=len(merged_annots),
        n_base_positive=n_base_pos,
        n_base_easy_neg=n_base_neg,
        n_hn_chips=n_hn,
        hn_ratio=n_hn / total if total else 0.0,
    )
