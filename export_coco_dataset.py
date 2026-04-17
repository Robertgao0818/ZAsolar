"""
Export solar panel annotations → COCO instance segmentation dataset.

Supports multi-region export (Cape Town, Johannesburg, or both).
Produces 400×400 chips with 0.25 overlap from GeoTIFF tiles, matching
annotations to chips via spatial intersection. Outputs:
  - COCO JSON (train / val)
  - Chip images directory
  - Chip provenance manifest CSV

Usage:
    python export_coco_dataset.py [--output-dir data/coco] [--chip-size 400]
                                  [--overlap 0.25] [--seed 42]
                                  [--regions cape_town johannesburg]
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.affinity import affine_transform
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from core.grid_utils import get_grid_paths, normalize_grid_id, TILES_ROOT, resolve_tiles_dir
from core.annotation_loader import discover_annotations, load_annotation_gdf, AnnotationEntry

BASE_DIR = Path(__file__).parent


# ════════════════════════════════════════════════════════════════════════
# Annotation loading (registry-based, multi-region)
# ════════════════════════════════════════════════════════════════════════

def load_annotations(
    regions: list[str] | None = None,
    exclude_grids: set[str] | None = None,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, str]]:
    """Load per-grid annotation GeoDataFrames via the annotation registry.

    Returns:
        (grid_annotations, grid_regions) where:
        - grid_annotations: dict grid_id → GeoDataFrame (EPSG:4326)
        - grid_regions: dict grid_id → region_key (for tile resolution)
    """
    entries = discover_annotations(regions=regions, exclude_grids=exclude_grids)

    grid_annotations: dict[str, gpd.GeoDataFrame] = {}
    grid_regions: dict[str, str] = {}

    for grid_id, entry in entries.items():
        # Skip grids without tiles
        tiles_dir = resolve_tiles_dir(grid_id, region=entry.region_key)
        if not tiles_dir.exists():
            continue

        gdf = load_annotation_gdf(entry)
        if len(gdf) == 0:
            continue

        grid_annotations[grid_id] = gdf
        grid_regions[grid_id] = entry.region_key
        print(f"[ANNOT] {grid_id} ({entry.region_key}): {len(gdf)} polygons "
              f"from {entry.path.name}")

    return grid_annotations, grid_regions


def get_geo_tiles(grid_id: str, *, region: str | None = None) -> list[Path]:
    """Return sorted list of *_geo.tif for a grid (region-aware)."""
    tiles_dir = resolve_tiles_dir(grid_id, region=region)
    tiles = sorted(tiles_dir.glob(f"{grid_id}_*_*_geo.tif"))
    if not tiles:
        tiles = sorted([
            f for f in tiles_dir.glob(f"{grid_id}_*_*.tif")
            if "_geo" not in f.stem and "mosaic" not in f.stem
        ])
    return tiles


# ════════════════════════════════════════════════════════════════════════
# Tile-level split
# ════════════════════════════════════════════════════════════════════════
def assign_annotations_to_tiles(
    annotations: gpd.GeoDataFrame, tiles: list[Path]
) -> dict[str, list[int]]:
    """Map tile stem → list of annotation indices whose centroid falls in the tile."""
    tile_to_annots: dict[str, list[int]] = {}
    tile_bounds = {}
    for t in tiles:
        with rasterio.open(t) as src:
            b = src.bounds
            tile_bounds[t.stem] = box(b.left, b.bottom, b.right, b.top)

    for idx, row in annotations.iterrows():
        centroid = row.geometry.centroid
        for stem, bbox in tile_bounds.items():
            if bbox.contains(centroid):
                tile_to_annots.setdefault(stem, []).append(idx)
                break  # centroid in exactly one tile

    # Also add tiles with no annotations (empty tiles)
    for t in tiles:
        if t.stem not in tile_to_annots:
            tile_to_annots[t.stem] = []

    return tile_to_annots


def split_tiles(
    tile_to_annots: dict[str, list[int]],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """80/20 tile-level split stratified by positive-instance count.

    Tiles are sorted by annotation count (descending) then distributed
    greedily to balance the val set toward the target fraction.
    """
    rng = random.Random(seed)

    # separate positive and empty tiles
    positive = [(stem, annots) for stem, annots in tile_to_annots.items() if annots]
    empty = [stem for stem, annots in tile_to_annots.items() if not annots]

    # shuffle then sort by count descending for greedy allocation
    rng.shuffle(positive)
    positive.sort(key=lambda x: len(x[1]), reverse=True)

    total_annots = sum(len(a) for _, a in positive)
    target_val = int(total_annots * val_fraction)

    val_stems, train_stems = [], []
    val_count = 0
    for stem, annots in positive:
        if val_count < target_val:
            val_stems.append(stem)
            val_count += len(annots)
        else:
            train_stems.append(stem)

    # distribute empty tiles proportionally
    rng.shuffle(empty)
    n_val_empty = max(1, int(len(empty) * val_fraction))
    val_stems.extend(empty[:n_val_empty])
    train_stems.extend(empty[n_val_empty:])

    return train_stems, val_stems


# ════════════════════════════════════════════════════════════════════════
# Chip extraction
# ════════════════════════════════════════════════════════════════════════
def polygon_to_pixel_coords(geom, transform):
    """Convert a Shapely polygon from geo coords to pixel coords using rasterio inverse transform."""
    inv = ~transform
    # affine_transform expects [a, b, d, e, xoff, yoff] from Affine(a, b, c, d, e, f)
    a, b, c, d, e, f = inv.a, inv.b, inv.c, inv.d, inv.e, inv.f
    return affine_transform(geom, [a, b, d, e, c, f])


def polygon_to_coco_segmentation(pixel_poly) -> list[list[float]]:
    """Convert a Shapely polygon (pixel coords) to COCO segmentation format."""
    segments = []
    # exterior ring
    coords = list(pixel_poly.exterior.coords)
    flat = []
    for coord in coords:
        x, y = coord[:2]
        flat.extend([round(float(x), 2), round(float(y), 2)])
    if len(flat) >= 6:
        segments.append(flat)
    # interior rings (holes) — rare but handle anyway
    for interior in pixel_poly.interiors:
        coords = list(interior.coords)
        flat = []
        for coord in coords:
            x, y = coord[:2]
            flat.extend([round(float(x), 2), round(float(y), 2)])
        if len(flat) >= 6:
            segments.append(flat)
    return segments


def scan_chips_from_tile(
    tile_path: Path,
    annotations: gpd.GeoDataFrame,
    annot_indices: list[int],
    chip_size: int,
    overlap: float,
    split_name: str,
    image_id_start: int,
    annot_id_start: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Scan chips from one tile — metadata only, no disk writes.

    Returns (images, annotations, provenance) for COCO JSON.
    Chip annotation geometries are stored in images[i]["_chip_annots"]
    for deferred writing by write_chip().
    """
    images = []
    coco_annots = []
    provenance = []

    stride = int(chip_size * (1.0 - overlap))

    with rasterio.open(tile_path) as src:
        tile_transform = src.transform
        tile_w, tile_h = src.width, src.height

        # Pre-compute annotation geometries in pixel space
        annot_pixel_geoms = {}
        for aidx in annot_indices:
            geom = annotations.loc[aidx, "geometry"]
            pgeom = polygon_to_pixel_coords(geom, tile_transform)
            if not pgeom.is_empty and pgeom.is_valid:
                annot_pixel_geoms[aidx] = pgeom

        img_id = image_id_start
        ann_id = annot_id_start

        for y0 in range(0, tile_h, stride):
            for x0 in range(0, tile_w, stride):
                # clip to tile bounds
                x1 = min(x0 + chip_size, tile_w)
                y1 = min(y0 + chip_size, tile_h)
                w = x1 - x0
                h = y1 - y0
                if w < chip_size // 2 or h < chip_size // 2:
                    continue  # skip tiny edge chips

                chip_box = box(x0, y0, x1, y1)

                # Find annotations that intersect this chip
                chip_annots = []
                for aidx, pgeom in annot_pixel_geoms.items():
                    inter = pgeom.intersection(chip_box)
                    if inter.is_empty:
                        continue
                    # clip to chip bounds & shift to chip-local coords
                    clipped = inter
                    # shift coords so chip top-left = (0, 0)
                    shifted = affine_transform(clipped, [1, 0, 0, 1, -x0, -y0])
                    if shifted.is_empty or shifted.area < 4:  # < 4 sq px
                        continue
                    chip_annots.append((aidx, shifted))

                is_positive = len(chip_annots) > 0
                chip_name = f"{tile_path.stem}__{x0}_{y0}.tif"

                # COCO image entry (with deferred write info)
                images.append({
                    "id": img_id,
                    "file_name": f"{split_name}/{chip_name}",
                    "width": chip_size,
                    "height": chip_size,
                    "positive": is_positive,
                    "_tile_path": str(tile_path),
                    "_x0": x0, "_y0": y0, "_w": w, "_h": h,
                    "_chip_annots": chip_annots,
                })

                provenance.append({
                    "image_id": img_id,
                    "chip_file": chip_name,
                    "source_tile": tile_path.stem,
                    "x0": x0,
                    "y0": y0,
                    "width": w,
                    "height": h,
                    "n_annotations": len(chip_annots),
                    "split": split_name,
                })

                # COCO annotations
                for aidx, shifted_geom in chip_annots:
                    polys = [shifted_geom] if shifted_geom.geom_type == "Polygon" else list(shifted_geom.geoms)
                    for poly in polys:
                        if poly.is_empty or poly.area < 4:
                            continue
                        seg = polygon_to_coco_segmentation(poly)
                        if not seg:
                            continue
                        bx, by, bx2, by2 = poly.bounds
                        coco_annots.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": 1,
                            "segmentation": seg,
                            "bbox": [round(bx, 2), round(by, 2),
                                     round(bx2 - bx, 2), round(by2 - by, 2)],
                            "area": round(poly.area, 2),
                            "iscrowd": 0,
                            "source_annotation_idx": int(aidx),
                        })
                        ann_id += 1

                img_id += 1

    return images, coco_annots, provenance


def write_selected_chips(
    images: list[dict],
    output_dir: Path,
    chip_size: int,
) -> None:
    """Write only selected chip images to disk.

    Reads pixel data from source tiles and writes GeoTIFF chips.
    Cleans up internal fields (_tile_path, _x0, etc.) from image dicts.
    """
    # Group by tile to minimize file opens
    from collections import defaultdict
    by_tile: dict[str, list[dict]] = defaultdict(list)
    for img in images:
        by_tile[img["_tile_path"]].append(img)

    for tile_path_str, tile_images in by_tile.items():
        tile_path = Path(tile_path_str)
        with rasterio.open(tile_path) as src:
            for img in tile_images:
                x0, y0 = img["_x0"], img["_y0"]
                w, h = img["_w"], img["_h"]
                window = Window(x0, y0, w, h)
                data = src.read(window=window)

                if w < chip_size or h < chip_size:
                    padded = np.zeros((data.shape[0], chip_size, chip_size), dtype=data.dtype)
                    padded[:, :h, :w] = data
                    data = padded

                chip_path = output_dir / img["file_name"]
                chip_path.parent.mkdir(parents=True, exist_ok=True)

                profile = src.profile.copy()
                for key in ("photometric", "compress", "jpeg_quality", "jpegtablesmode"):
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

    # Clean up internal fields from image dicts
    for img in images:
        for key in ("_tile_path", "_x0", "_y0", "_w", "_h", "_chip_annots"):
            img.pop(key, None)


def balance_chips(
    images: list[dict],
    annotations: list[dict],
    provenance: list[dict],
    seed: int = 42,
    neg_ratio: float = 1.0,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Subsample empty chips. neg_ratio controls negative:positive ratio (default 1:1)."""
    pos_ids = {img["id"] for img in images if img["positive"]}
    neg_imgs = [img for img in images if not img["positive"]]

    target_neg = int(len(pos_ids) * neg_ratio)
    rng = random.Random(seed)

    if len(neg_imgs) > target_neg:
        neg_imgs = rng.sample(neg_imgs, target_neg)

    neg_ids = {img["id"] for img in neg_imgs}
    keep_ids = pos_ids | neg_ids

    images_out = [img for img in images if img["id"] in keep_ids]
    annots_out = [a for a in annotations if a["image_id"] in keep_ids]
    prov_out = [p for p in provenance if p["image_id"] in keep_ids]

    return images_out, annots_out, prov_out


def build_coco_json(images: list[dict], annotations: list[dict], split: str,
                    category_name: str = "solar_panel",
                    regions: list[str] | None = None) -> dict:
    """Build COCO-format JSON dict."""
    region_desc = ", ".join(regions) if regions else "Cape Town"
    return {
        "info": {
            "description": f"{region_desc} Solar Panel Detection - {split}",
            "version": "1.0",
            "year": 2026,
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": category_name, "supercategory": "object"}
        ],
        "images": images,
        "annotations": annotations,
    }


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════
def build_base_coco(params: dict) -> dict:
    """Build a base COCO dataset from parameters.

    This is the public API for programmatic use (called by the dataset
    builder).  The CLI ``main()`` is a thin wrapper around this function.

    Args:
        params: Dict with keys matching CLI args:
            regions, output_dir, chip_size, overlap, val_fraction, seed,
            no_balance, manifest, tier_filter, category_name, neg_ratio,
            exclude_grids, audit_csv, exclude_audit_labels.

    Returns:
        Dict with build results: output_dir, train_images, val_images,
        train_annotations, val_annotations, grid_data, grid_regions.
    """
    regions = params.get("regions", ["cape_town"])
    output_dir = Path(params.get("output_dir", "data/coco"))
    chip_size = params.get("chip_size", 400)
    overlap = params.get("overlap", 0.25)
    val_fraction = params.get("val_fraction", 0.2)
    seed = params.get("seed", 42)
    no_balance = params.get("no_balance", False)
    manifest_path_str = params.get("manifest")
    tier_filter = params.get("tier_filter", "T1+T2")
    category_name = params.get("category_name", "solar_panel")
    neg_ratio = params.get("neg_ratio", 1.0)
    exclude_grids_list = params.get("exclude_grids")
    audit_csv_str = params.get("audit_csv")
    exclude_audit_labels = params.get("exclude_audit_labels",
                                      ["heater_or_non_pv", "uncertain"])

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load annotations ──────────────────────────────────────────────
    exclude_set = set(exclude_grids_list) if exclude_grids_list else None
    grid_annotations, grid_regions = load_annotations(
        regions=regions, exclude_grids=exclude_set,
    )

    # ── Manifest-based tier filtering ──────────────────────────────────
    manifest_path = Path(manifest_path_str) if manifest_path_str else None
    if manifest_path and manifest_path.exists() and tier_filter != "T1+T2":
        import csv as csv_mod
        with open(manifest_path, encoding="utf-8") as f:
            manifest_rows = list(csv_mod.DictReader(f))

        # Build set of (grid_id, row_index) to keep
        allowed_tiers = set(tier_filter.split("+"))
        keep_set: dict[str, set[int]] = {}
        for row in manifest_rows:
            if row["quality_tier"] in allowed_tiers:
                gid = row["grid_id"]
                # annotation_id format: {grid_id}_{idx:03d}
                idx = int(row["annotation_id"].split("_")[-1])
                keep_set.setdefault(gid, set()).add(idx)

        for gid in list(grid_annotations.keys()):
            if gid in keep_set:
                mask = grid_annotations[gid].index.isin(keep_set[gid])
                before = len(grid_annotations[gid])
                grid_annotations[gid] = grid_annotations[gid][mask].reset_index(drop=True)
                after = len(grid_annotations[gid])
                print(f"[TIER] {gid}: {before} → {after} (tier={tier_filter})")
                if after == 0:
                    print(f"[WARN] {gid} has 0 annotations after tier filter, skipping")
                    del grid_annotations[gid]
            else:
                print(f"[WARN] {gid} not in manifest, removing")
                del grid_annotations[gid]
    elif manifest_path and manifest_path.exists():
        print(f"[TIER] Using all tiers (T1+T2), manifest loaded: {manifest_path}")

    # ── Audit-based heater filtering ──────────────────────────────────
    if audit_csv_str:
        audit_path = Path(audit_csv_str)
        if not audit_path.exists():
            print(f"[WARN] --audit-csv not found: {audit_path}")
        else:
            import csv as csv_mod
            with open(audit_path, encoding="utf-8") as f:
                audit_rows = list(csv_mod.DictReader(f))

            exclude_labels_set = set(exclude_audit_labels)
            exclude_set_audit: dict[str, set[int]] = {}
            for row in audit_rows:
                if row.get("audit_label", "") in exclude_labels_set:
                    gid = row["grid_id"]
                    ridx = int(row["row_index"])
                    exclude_set_audit.setdefault(gid, set()).add(ridx)

            total_excluded = 0
            for gid in list(grid_annotations.keys()):
                if gid in exclude_set_audit:
                    before = len(grid_annotations[gid])
                    mask = ~grid_annotations[gid].index.isin(exclude_set_audit[gid])
                    grid_annotations[gid] = grid_annotations[gid][mask].reset_index(drop=True)
                    after = len(grid_annotations[gid])
                    removed = before - after
                    total_excluded += removed
                    if removed > 0:
                        print(f"[AUDIT] {gid}: {before} → {after} ({removed} heater/uncertain removed)")
                    if after == 0:
                        print(f"[WARN] {gid} has 0 annotations after audit filter, skipping")
                        del grid_annotations[gid]

            print(f"[AUDIT] Total excluded: {total_excluded} annotations "
                  f"(labels: {', '.join(sorted(exclude_labels_set))})")

    # ── Per-grid tile split ───────────────────────────────────────────
    all_train_stems = set()
    all_val_stems = set()
    # Collect per-grid data for processing
    grid_data = {}

    for grid_id, annots in grid_annotations.items():
        region = grid_regions.get(grid_id)
        tiles = get_geo_tiles(grid_id, region=region)
        if not tiles:
            print(f"[WARN] No tiles found for {grid_id}, skipping")
            continue

        tile_map = {t.stem: t for t in tiles}
        tile_to_annots = assign_annotations_to_tiles(annots, tiles)
        train_stems, val_stems = split_tiles(
            tile_to_annots, val_fraction=val_fraction, seed=seed
        )

        # Verify no overlap
        overlap_check = set(train_stems) & set(val_stems)
        assert not overlap_check, f"Tile overlap in splits: {overlap_check}"

        n_train_annots = sum(len(tile_to_annots[s]) for s in train_stems if s in tile_to_annots)
        n_val_annots = sum(len(tile_to_annots[s]) for s in val_stems if s in tile_to_annots)
        print(f"[SPLIT] {grid_id}: train={len(train_stems)} tiles ({n_train_annots} annots), "
              f"val={len(val_stems)} tiles ({n_val_annots} annots)")

        all_train_stems.update(train_stems)
        all_val_stems.update(val_stems)
        grid_data[grid_id] = {
            "annotations": annots,
            "tile_map": tile_map,
            "tile_to_annots": tile_to_annots,
            "train_stems": train_stems,
            "val_stems": val_stems,
            "region": region,
        }

    # ── Scan chips (metadata only, no disk writes) ──────────────────
    for split_name, stem_attr in [("train", "train_stems"), ("val", "val_stems")]:
        all_images = []
        all_annots = []
        all_prov = []
        img_id_counter = 1
        ann_id_counter = 1

        for grid_id, gd in grid_data.items():
            stems = gd[stem_attr]
            for stem in stems:
                tile_path = gd["tile_map"].get(stem)
                if tile_path is None:
                    continue
                annot_indices = gd["tile_to_annots"].get(stem, [])
                imgs, anns, prov = scan_chips_from_tile(
                    tile_path=tile_path,
                    annotations=gd["annotations"],
                    annot_indices=annot_indices,
                    chip_size=chip_size,
                    overlap=overlap,
                    split_name=split_name,
                    image_id_start=img_id_counter,
                    annot_id_start=ann_id_counter,
                )
                # Add region to provenance rows
                grid_region = gd.get("region", "")
                for p in prov:
                    p["region"] = grid_region or ""
                img_id_counter += len(imgs)
                ann_id_counter += len(anns)
                all_images.extend(imgs)
                all_annots.extend(anns)
                all_prov.extend(prov)

        n_pos = sum(1 for img in all_images if img["positive"])
        n_neg = len(all_images) - n_pos
        print(f"[SCAN] {split_name}: {len(all_images)} chips "
              f"({n_pos} positive, {n_neg} negative), {len(all_annots)} instances")

        # Balance positive:negative (before writing anything to disk)
        if not no_balance:
            all_images, all_annots, all_prov = balance_chips(
                all_images, all_annots, all_prov, seed=seed,
                neg_ratio=neg_ratio,
            )
            n_pos2 = sum(1 for img in all_images if img["positive"])
            n_neg2 = len(all_images) - n_pos2
            print(f"[BALANCE] {split_name}: {len(all_images)} chips after balancing "
                  f"({n_pos2} positive, {n_neg2} negative, ratio={neg_ratio})")

        # Write only selected chips to disk
        print(f"[WRITE] {split_name}: writing {len(all_images)} chips to disk...")
        write_selected_chips(all_images, output_dir, chip_size)

        # Write COCO JSON
        coco = build_coco_json(all_images, all_annots, split_name,
                              category_name=category_name,
                              regions=regions)
        json_path = output_dir / f"{split_name}.json"
        json_path.write_text(
            json.dumps(coco, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[SAVE] {json_path} ({len(all_images)} images, {len(all_annots)} annotations)")

        # Write provenance manifest
        import csv
        prov_path = output_dir / f"{split_name}_provenance.csv"
        if all_prov:
            keys = all_prov[0].keys()
            with open(prov_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(all_prov)
            print(f"[SAVE] {prov_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Export complete! Regions: {', '.join(regions)}")
    print(f"  Output: {output_dir}")
    for grid_id, gd in grid_data.items():
        print(f"  {grid_id}: train tiles = {gd['train_stems']}")
        print(f"  {grid_id}: val tiles   = {gd['val_stems']}")

    return {
        "output_dir": output_dir,
        "grid_data": grid_data,
        "grid_regions": grid_regions,
        "regions": regions,
    }


# ════════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Export solar panel annotations to COCO instance segmentation dataset"
    )
    parser.add_argument(
        "--output-dir", default="data/coco",
        help="Output directory for COCO dataset (default: data/coco)",
    )
    parser.add_argument("--chip-size", type=int, default=400)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-balance", action="store_true",
        help="Skip 1:1 positive:negative balancing",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to annotation_manifest.csv for quality tier filtering",
    )
    parser.add_argument(
        "--tier-filter", choices=["T1", "T2", "T1+T2"], default="T1+T2",
        help="Quality tier filter: T1, T2, or T1+T2 (default: T1+T2, use all)",
    )
    parser.add_argument(
        "--category-name", default="solar_panel",
        help="COCO category name (default: solar_panel)",
    )
    parser.add_argument(
        "--neg-ratio", type=float, default=1.0,
        help="Negative:positive chip ratio (default: 1.0 = 1:1). Use 0.15 to reduce easy negatives.",
    )
    parser.add_argument(
        "--exclude-grids", nargs="+", default=None,
        help="Grid IDs to exclude from export (e.g. benchmark holdout grids)",
    )
    parser.add_argument(
        "--audit-csv", type=str, default=None,
        help="Path to GT heater audit CSV (from build_gt_heater_audit.py). "
             "Annotations matching --exclude-audit-labels are removed from training export.",
    )
    parser.add_argument(
        "--exclude-audit-labels", nargs="+",
        default=["heater_or_non_pv", "uncertain"],
        help="Audit labels to exclude (default: heater_or_non_pv uncertain)",
    )
    parser.add_argument(
        "--regions", nargs="+", default=["cape_town"],
        help="Region(s) to include (default: cape_town). "
             "Use 'cape_town johannesburg' for multi-region export.",
    )
    args = parser.parse_args()

    build_base_coco({
        "regions": args.regions,
        "output_dir": args.output_dir,
        "chip_size": args.chip_size,
        "overlap": args.overlap,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "no_balance": args.no_balance,
        "manifest": args.manifest,
        "tier_filter": args.tier_filter,
        "category_name": args.category_name,
        "neg_ratio": args.neg_ratio,
        "exclude_grids": args.exclude_grids,
        "audit_csv": args.audit_csv,
        "exclude_audit_labels": args.exclude_audit_labels,
    })


if __name__ == "__main__":
    main()
