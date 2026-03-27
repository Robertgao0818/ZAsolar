#!/usr/bin/env python3
"""
批量完成 review 后续处理：
1. SAM 点提示补标 FN markers
2. 导出 correct 多边形到 cleaned/ 目录
3. 生成 FN 缩略图到 D 盘供人工检查

用法：
    python scripts/annotations/batch_finalize_reviews.py G1689 G1690 ...
    python scripts/annotations/batch_finalize_reviews.py --all-pending
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[2]
TILES_ROOT = Path("/mnt/d/ZAsolar/tiles")
RESULTS_DIR = BASE_DIR / "results"
CLEANED_DIR = BASE_DIR / "data" / "annotations" / "cleaned"
THUMBNAILS_ROOT = Path("/mnt/d/ZAsolar/review_thumbnails")

# Date suffix for cleaned filenames
DATE_SUFFIX = datetime.now().strftime("%y%m%d")


def find_pending_grids() -> list[str]:
    """Find grids with review files but no cleaned output."""
    pending = []
    for review_dir in sorted(RESULTS_DIR.glob("G*/review")):
        grid_id = review_dir.parent.name
        reviewed_gpkg = review_dir / f"{grid_id}_reviewed.gpkg"
        if not reviewed_gpkg.exists():
            continue
        # Check if already cleaned today
        cleaned = CLEANED_DIR / f"{grid_id}_SAM2_{DATE_SUFFIX}.gpkg"
        if not cleaned.exists():
            pending.append(grid_id)
    return pending


def sam_fill_fn(grid_id: str) -> int:
    """Use SAM point-prompt to segment FN markers. Returns number of new polygons."""
    review_dir = RESULTS_DIR / grid_id / "review"
    fn_csv = review_dir / "fn_markers.csv"
    reviewed_path = review_dir / f"{grid_id}_reviewed.gpkg"

    if not fn_csv.exists() or not reviewed_path.exists():
        return 0

    markers = []
    with open(fn_csv) as f:
        for row in csv.DictReader(f):
            markers.append(row)

    if not markers:
        return 0

    print(f"  [{grid_id}] {len(markers)} FN markers → SAM segmentation...")

    # Lazy-load SAM (only when needed)
    import torch
    from transformers import SamModel, SamProcessor
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape
    import rasterio

    processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
    model = SamModel.from_pretrained("facebook/sam-vit-large")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    gdf = gpd.read_file(str(reviewed_path))
    crs = gdf.crs
    # Remove old SAM FN polygons if re-running
    if "source" in gdf.columns:
        original = gdf[gdf["source"] != "sam_fn_marker"].copy()
    else:
        original = gdf.copy()

    tiles_dir = TILES_ROOT / grid_id
    new_polys = []

    for m in markers:
        tile_key = m["tile_key"]
        px, py = float(m.get("px", m.get("x"))), float(m.get("py", m.get("y")))

        geo_path = tiles_dir / f"{tile_key}_geo.tif"
        if not geo_path.exists():
            geo_path = tiles_dir / f"{tile_key}.tif"
        if not geo_path.exists():
            print(f"    [SKIP] tile not found: {tile_key}")
            continue

        with rasterio.open(geo_path) as src:
            img_array = src.read()  # (C, H, W)
            transform = src.transform

        img_rgb = np.moveaxis(img_array[:3], 0, -1)  # (H, W, 3)
        pil_img = Image.fromarray(img_rgb)

        inputs = processor(pil_img, input_points=[[[px, py]]], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        masks_list = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        scores = outputs.iou_scores.cpu().numpy()[0][0]

        # Use mask index 1 (medium granularity)
        mask = masks_list[0][0][1].numpy().astype(np.uint8)
        print(f"    [{tile_key}] score={scores[1]:.3f} pixels={mask.sum()}")

        # Collect all mask polygons, keep the largest one
        candidates = []
        for geom, val in rio_shapes(mask, transform=transform):
            if val == 1:
                poly = shape(geom)
                if poly.is_valid and poly.area > 0:
                    candidates.append(poly)
        if candidates:
            best = max(candidates, key=lambda p: p.area)
            new_polys.append({
                "geometry": best, "source": "sam_fn_marker",
                "tile_key": tile_key, "confidence": float(scores[1]),
                "review_status": "correct",
            })

    if new_polys:
        fn_gdf = gpd.GeoDataFrame(new_polys, crs=crs)
        for col in original.columns:
            if col not in fn_gdf.columns and col != "geometry":
                fn_gdf[col] = None
        for col in fn_gdf.columns:
            if col not in original.columns and col != "geometry":
                original[col] = None
        merged = gpd.GeoDataFrame(
            pd.concat([original, fn_gdf], ignore_index=True), crs=crs
        )
        merged.to_file(str(reviewed_path), driver="GPKG")
        print(f"    Saved {len(merged)} polygons ({len(original)} det + {len(new_polys)} SAM)")

    return len(new_polys)


def export_cleaned(grid_id: str) -> int:
    """Export correct polygons to cleaned/ directory. Returns count."""
    reviewed_path = RESULTS_DIR / grid_id / "review" / f"{grid_id}_reviewed.gpkg"
    if not reviewed_path.exists():
        return 0

    gdf = gpd.read_file(str(reviewed_path))
    gt = gdf[gdf["review_status"] == "correct"].copy()

    if len(gt) == 0:
        print(f"  [{grid_id}] No correct polygons, skipping cleaned export")
        return 0

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLEANED_DIR / f"{grid_id}_SAM2_{DATE_SUFFIX}.gpkg"
    gt.to_file(str(out_path), driver="GPKG")
    print(f"  [{grid_id}] Exported {len(gt)} GT polygons → {out_path.name}")
    return len(gt)


def generate_fn_thumbnails(grid_id: str) -> int:
    """Generate FN thumbnails showing SAM-segmented polygons."""
    review_dir = RESULTS_DIR / grid_id / "review"
    reviewed_path = review_dir / f"{grid_id}_reviewed.gpkg"

    if not reviewed_path.exists():
        return 0

    import rasterio
    from rasterio.windows import Window
    from PIL import ImageDraw

    gdf = gpd.read_file(str(reviewed_path))
    # Get SAM FN polygons
    sam_polys = gdf[gdf.get("source", pd.Series()) == "sam_fn_marker"]
    if len(sam_polys) == 0:
        return 0

    thumb_dir = THUMBNAILS_ROOT / grid_id
    thumb_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = TILES_ROOT / grid_id
    margin = 40  # extra pixels around polygon bbox

    count = 0
    tile_counts = {}  # track per-tile numbering
    for idx, row in sam_polys.iterrows():
        tile_key = row.get("tile_key", row.get("source_tile", ""))
        if not tile_key:
            continue

        geo_path = tiles_dir / f"{tile_key}_geo.tif"
        if not geo_path.exists():
            geo_path = tiles_dir / f"{tile_key}.tif"
        if not geo_path.exists():
            continue

        with rasterio.open(geo_path) as src:
            transform = src.transform
            h, w = src.height, src.width

            # Convert polygon geo coords to pixel coords
            from rasterio.transform import rowcol
            geom = row.geometry
            if geom.geom_type == "MultiPolygon":
                coords_list = [list(p.exterior.coords) for p in geom.geoms]
            else:
                coords_list = [list(geom.exterior.coords)]

            all_px = []
            for coords in coords_list:
                px_coords = []
                for x, y in coords:
                    r, c = rowcol(transform, x, y)
                    px_coords.append((int(c), int(r)))
                    all_px.append((int(c), int(r)))

            # Bounding box in pixel space
            xs = [p[0] for p in all_px]
            ys = [p[1] for p in all_px]
            x0 = max(0, min(xs) - margin)
            y0 = max(0, min(ys) - margin)
            x1 = min(w, max(xs) + margin)
            y1 = min(h, max(ys) + margin)

            window = Window(x0, y0, x1 - x0, y1 - y0)
            img = src.read(window=window)

        img_rgb = np.moveaxis(img[:3], 0, -1)
        pil = Image.fromarray(img_rgb).convert("RGBA")
        overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for coords in coords_list:
            px_shifted = []
            for x, y in coords:
                r, c = rowcol(transform, x, y)
                px_shifted.append((int(c) - x0, int(r) - y0))
            draw.polygon(px_shifted, fill=(0, 255, 0, 80), outline=(0, 255, 0, 255))

        pil = Image.alpha_composite(pil, overlay).convert("RGB")

        tile_counts[tile_key] = tile_counts.get(tile_key, 0) + 1
        tc = tile_counts[tile_key]
        count += 1
        out_path = thumb_dir / f"FN_{count:02d}_{tile_key}_{tc}.png"
        pil.save(str(out_path))

    print(f"  [{grid_id}] Generated {count} FN thumbnails → {thumb_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Batch finalize reviewed grids")
    parser.add_argument("grids", nargs="*", help="Grid IDs to process")
    parser.add_argument("--all-pending", action="store_true",
                        help="Process all grids with reviews but no cleaned output")
    parser.add_argument("--skip-sam", action="store_true",
                        help="Skip SAM FN segmentation")
    parser.add_argument("--skip-thumbnails", action="store_true",
                        help="Skip thumbnail generation")
    args = parser.parse_args()

    if args.all_pending:
        grids = find_pending_grids()
        print(f"Found {len(grids)} pending grids: {', '.join(grids)}")
    elif args.grids:
        grids = args.grids
    else:
        parser.print_help()
        sys.exit(1)

    if not grids:
        print("No grids to process.")
        return

    total_sam, total_gt, total_thumb = 0, 0, 0

    for gid in grids:
        print(f"\n{'='*50}")
        print(f"  {gid}")
        print(f"{'='*50}")

        # Step 1: SAM fill FN
        if not args.skip_sam:
            total_sam += sam_fill_fn(gid)

        # Step 2: Export cleaned
        total_gt += export_cleaned(gid)

        # Step 3: FN thumbnails
        if not args.skip_thumbnails:
            total_thumb += generate_fn_thumbnails(gid)

    print(f"\n{'='*50}")
    print(f"DONE: {len(grids)} grids processed")
    print(f"  SAM FN polygons: {total_sam}")
    print(f"  GT exported: {total_gt}")
    print(f"  Thumbnails: {total_thumb}")


if __name__ == "__main__":
    main()
