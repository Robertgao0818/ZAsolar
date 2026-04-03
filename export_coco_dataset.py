"""
Export Cape Town solar panel annotations → COCO instance segmentation dataset.

Produces 400×400 chips with 0.25 overlap from GeoTIFF tiles, matching
annotations to chips via spatial intersection. Outputs:
  - COCO JSON (train / val)
  - Chip images directory
  - Chip provenance manifest CSV

Usage:
    python export_coco_dataset.py [--output-dir data/coco] [--chip-size 400]
                                  [--overlap 0.25] [--seed 42]
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

from core.grid_utils import get_grid_paths, normalize_grid_id

# ════════════════════════════════════════════════════════════════════════
# Annotation sources
# ════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"

CLEANED_DIR = ANNOTATIONS_DIR / "cleaned"


def _discover_cleaned_sources() -> dict[str, dict]:
    """Auto-discover SAM2 annotation files from cleaned/ directory.

    Each *_SAM2_*.gpkg is matched to a grid ID and its first layer is used.
    """
    import fiona

    sources = {}
    for f in sorted(CLEANED_DIR.glob("*_SAM2_*.gpkg")):
        grid_id = f.name.split("_SAM2_")[0]
        layers = fiona.listlayers(str(f))
        if layers:
            sources[grid_id] = {"file": f, "layer": layers[0]}
    return sources


def load_annotations() -> dict[str, gpd.GeoDataFrame]:
    """Load per-grid annotation GeoDataFrames (EPSG:4326) from cleaned/ dir."""
    sources = _discover_cleaned_sources()
    result = {}
    for grid_id, src in sources.items():
        # Skip grids without tiles
        tiles_dir = BASE_DIR / "tiles" / grid_id
        if not tiles_dir.exists():
            continue
        gdf = gpd.read_file(str(src["file"]), layer=src["layer"])
        # Drop invalid / empty geometries
        gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid & ~gdf.geometry.is_empty]
        gdf = gdf.reset_index(drop=True)
        result[grid_id] = gdf
        print(f"[ANNOT] {grid_id}: {len(gdf)} polygons from {src['file'].name}")
    return result


def get_geo_tiles(grid_id: str) -> list[Path]:
    """Return sorted list of *_geo.tif for a grid."""
    tiles_dir = BASE_DIR / "tiles" / grid_id
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
) -> tuple[list[dict], list[dict], list[dict]]:
    """Subsample empty chips to match positive chip count (1:1 ratio)."""
    pos_ids = {img["id"] for img in images if img["positive"]}
    neg_imgs = [img for img in images if not img["positive"]]

    n_pos = len(pos_ids)
    rng = random.Random(seed)

    if len(neg_imgs) > n_pos:
        neg_imgs = rng.sample(neg_imgs, n_pos)

    neg_ids = {img["id"] for img in neg_imgs}
    keep_ids = pos_ids | neg_ids

    images_out = [img for img in images if img["id"] in keep_ids]
    annots_out = [a for a in annotations if a["image_id"] in keep_ids]
    prov_out = [p for p in provenance if p["image_id"] in keep_ids]

    return images_out, annots_out, prov_out


def build_coco_json(images: list[dict], annotations: list[dict], split: str,
                    category_name: str = "solar_panel") -> dict:
    """Build COCO-format JSON dict."""
    return {
        "info": {
            "description": f"Cape Town Solar Panel Detection - {split}",
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
def main():
    parser = argparse.ArgumentParser(
        description="Export Cape Town annotations to COCO instance segmentation dataset"
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load annotations ──────────────────────────────────────────────
    grid_annotations = load_annotations()

    # ── Manifest-based tier filtering ──────────────────────────────────
    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path and manifest_path.exists() and args.tier_filter != "T1+T2":
        import csv as csv_mod
        with open(manifest_path, encoding="utf-8") as f:
            manifest_rows = list(csv_mod.DictReader(f))

        # Build set of (grid_id, row_index) to keep
        allowed_tiers = set(args.tier_filter.split("+"))
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
                print(f"[TIER] {gid}: {before} → {after} (tier={args.tier_filter})")
                if after == 0:
                    print(f"[WARN] {gid} has 0 annotations after tier filter, skipping")
                    del grid_annotations[gid]
            else:
                print(f"[WARN] {gid} not in manifest, removing")
                del grid_annotations[gid]
    elif manifest_path and manifest_path.exists():
        print(f"[TIER] Using all tiers (T1+T2), manifest loaded: {manifest_path}")

    # ── Per-grid tile split ───────────────────────────────────────────
    all_train_stems = set()
    all_val_stems = set()
    # Collect per-grid data for processing
    grid_data = {}

    for grid_id, annots in grid_annotations.items():
        tiles = get_geo_tiles(grid_id)
        if not tiles:
            print(f"[WARN] No tiles found for {grid_id}, skipping")
            continue

        tile_map = {t.stem: t for t in tiles}
        tile_to_annots = assign_annotations_to_tiles(annots, tiles)
        train_stems, val_stems = split_tiles(
            tile_to_annots, val_fraction=args.val_fraction, seed=args.seed
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
                    chip_size=args.chip_size,
                    overlap=args.overlap,
                    split_name=split_name,
                    image_id_start=img_id_counter,
                    annot_id_start=ann_id_counter,
                )
                img_id_counter += len(imgs)
                ann_id_counter += len(anns)
                all_images.extend(imgs)
                all_annots.extend(anns)
                all_prov.extend(prov)

        n_pos = sum(1 for img in all_images if img["positive"])
        n_neg = len(all_images) - n_pos
        print(f"[SCAN] {split_name}: {len(all_images)} chips "
              f"({n_pos} positive, {n_neg} negative), {len(all_annots)} instances")

        # Balance positive:negative 1:1 (before writing anything to disk)
        if not args.no_balance:
            all_images, all_annots, all_prov = balance_chips(
                all_images, all_annots, all_prov, seed=args.seed
            )
            n_pos2 = sum(1 for img in all_images if img["positive"])
            n_neg2 = len(all_images) - n_pos2
            print(f"[BALANCE] {split_name}: {len(all_images)} chips after balancing "
                  f"({n_pos2} positive, {n_neg2} negative)")

        # Write only selected chips to disk
        print(f"[WRITE] {split_name}: writing {len(all_images)} chips to disk...")
        write_selected_chips(all_images, output_dir, args.chip_size)

        # Write COCO JSON
        coco = build_coco_json(all_images, all_annots, split_name,
                              category_name=args.category_name)
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
    print("Export complete!")
    print(f"  Output: {output_dir}")
    for grid_id, gd in grid_data.items():
        print(f"  {grid_id}: train tiles = {gd['train_stems']}")
        print(f"  {grid_id}: val tiles   = {gd['val_stems']}")


if __name__ == "__main__":
    main()
