"""Export the train20_val5 frozen COCO dataset.

Combines:
  - JHB CBD Vexcel 2024 (Ch2 clean GT, 23 grids: 20 train + 3 val)
  - CT aerial_2025 (SAM2 native annotations, 22 grids: 20 train + 2 val)

The split, grid-list, and per-grid annotation source are pinned in
``configs/datasets/train20_val5.yaml``. This wrapper reuses the chip
scanning / writing helpers from ``export_coco_dataset.py`` but bypasses
the registry-based annotation discovery so we can:
  1. Use the Ch2 clean GT for JHB (not the V4-reviewed registry entries).
  2. Force JHB tiles from ``vexcel_2024`` (not the default ``aerial_2023``).
  3. Reproject annotations to each tile's native CRS (Vexcel is 3857, not 4326).

Usage:
    python scripts/training/export_train20_val5.py \\
        --output-dir ~/zasolar_data/coco/coco_train20_val5 \\
        --neg-ratio 0.15

    # dry-run (no chip writes, just summary counts)
    python scripts/training/export_train20_val5.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import yaml
from shapely.geometry import box as shapely_box

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.grid_utils import resolve_tiles_dir  # noqa: E402

from export_coco_dataset import (  # noqa: E402
    balance_chips,
    build_coco_json,
    scan_chips_from_tile,
    write_selected_chips,
)


def assign_annotations_to_tiles_intersecting(
    annotations: gpd.GeoDataFrame,
    tiles: list[Path],
) -> dict[str, list[int]]:
    """Map tile stem → indices of every polygon whose geometry intersects the tile bbox.

    Differs from ``export_coco_dataset.assign_annotations_to_tiles`` (which assigns
    by centroid only). Centroid-only assignment drops polygon segments visible in
    neighbouring tiles, creating false-negative training noise on cross-tile
    arrays — rare in count (~1% of polygons) but disproportionately the large
    installations that sit on tile seams.

    Whole-grid splits in this wrapper avoid the train/val leakage that would
    otherwise concern this change: for any grid, all its tiles go to one split.
    """
    tile_bounds = {}
    for t in tiles:
        with rasterio.open(t) as src:
            b = src.bounds
            tile_bounds[t.stem] = shapely_box(b.left, b.bottom, b.right, b.top)

    tile_to_annots: dict[str, list[int]] = {stem: [] for stem in tile_bounds}
    for idx, row in annotations.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        for stem, bbox in tile_bounds.items():
            if geom.intersects(bbox):
                tile_to_annots[stem].append(idx)
    return tile_to_annots


SPEC_PATH = PROJECT_ROOT / "configs/datasets/train20_val5.yaml"


def parse_grid_entry(entry):
    if isinstance(entry, dict):
        return entry["grid_id"], entry.get("file")
    return entry, None


def load_spec_grids(spec):
    """Yield (split, region_key, grid_id, annotation_path, imagery_layer)."""
    for split_name, regions in spec["splits"].items():
        for region_key, cfg in regions.items():
            ann_root = PROJECT_ROOT / cfg["annotation_root"]
            imagery_layer = cfg["imagery_layer"]
            for entry in cfg.get("grids", []):
                grid_id, fname = parse_grid_entry(entry)
                if fname:
                    path = ann_root / fname
                else:
                    pattern = cfg["annotation_pattern"]
                    path = ann_root / pattern.format(grid_id=grid_id)
                yield split_name, region_key, grid_id, path, imagery_layer


# JHB clean_gt 'source' column → label_source enum (consumed by
# train.py --per-source-mask-weight). Strict rule: any V3-C-tainted
# polygon gets boundary weight 0; only pure-human gets 1.
_JHB_SOURCE_TO_LABEL_SOURCE = {
    "V3C_TP":           "reviewed_prediction",   # pure V3-C accepted
    "SAM_supp+V3C_TP":  "sam_refined_review",    # V3-C dissolved with SAM supplement
    "Li_marked":        "human_manual",          # human-drawn (no V3-C)
}

# CT 'source' column → label_source enum. CRITICAL: only the very first
# CT batch (G1189/G1190/G1238, drawn by hand in QGIS+GeoSAM before the
# Pred Review GUI existed) is true H-tier supervision. EVERY other CT
# file — regardless of the `source` column value or the date suffix —
# was produced by V3-C inference + human review (with optional SAM-based
# FN补标), so the polygons are V3-C-derived and must get boundary weight 0
# under the strict rule.
_CT_MANUAL_GRIDS = frozenset({"G1189", "G1190", "G1238"})

def _ct_source_to_label_source(value) -> str:
    """Map CT 'source' column values for non-manual grids (V3-C-derived path)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "reviewed_prediction"        # Pred Correct = V3-C accepted via review GUI
    if value == "sam2":
        return "reviewed_prediction"        # SAM2 polygon prompted by model candidate, then reviewed
    if value in ("sam_fn_review", "sam_fn_marker"):
        return "legacy_weak_supervision"    # pre-2026-04-13 SAM补标 tools, unreliable
    raise ValueError(f"unknown CT 'source' value {value!r} — extend _ct_source_to_label_source")


def attach_label_source(g: gpd.GeoDataFrame, region_key: str, src_path: Path,
                        grid_id: str) -> gpd.GeoDataFrame:
    """Tag each polygon with label_source for downstream per-source mask loss weighting."""
    g = g.copy()
    if region_key == "johannesburg":
        if "source" not in g.columns:
            raise ValueError(
                f"JHB annotation file missing 'source' column for label_source mapping: {src_path}"
            )
        unknown = set(g["source"].dropna().unique()) - set(_JHB_SOURCE_TO_LABEL_SOURCE)
        if unknown:
            raise ValueError(
                f"JHB clean_gt {src_path} has unmapped 'source' values: {sorted(unknown)}. "
                f"Update _JHB_SOURCE_TO_LABEL_SOURCE."
            )
        g["label_source"] = g["source"].map(_JHB_SOURCE_TO_LABEL_SOURCE)
    elif region_key == "cape_town":
        if grid_id in _CT_MANUAL_GRIDS:
            # Original QGIS+GeoSAM manual batch (predates Pred Review GUI).
            g["label_source"] = "human_manual_sam_assisted"
        elif "source" not in g.columns:
            # Early single-mask schema (e.g. G1635: method='single', no source col).
            # Per the rule that only G1189/G1190/G1238 are true manual, every other
            # CT file is V3-C-derived → reviewed_prediction (boundary weight 0).
            g["label_source"] = "reviewed_prediction"
        else:
            g["label_source"] = g["source"].map(_ct_source_to_label_source)
    else:
        raise ValueError(f"unknown region_key {region_key!r} for label_source mapping")
    return g


def load_annotations_for_grid(path: Path, tile_crs, region_key: str,
                              grid_id: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    g = g.to_crs(tile_crs)
    return attach_label_source(g, region_key, path, grid_id)


def get_tiles(grid_id: str, region_key: str, imagery_layer: str) -> list[Path]:
    tiles_dir = resolve_tiles_dir(grid_id, region=region_key, imagery_layer=imagery_layer)
    if tiles_dir.is_file():
        return [tiles_dir]
    tiles = sorted(tiles_dir.glob(f"{grid_id}_*_*_geo.tif"))
    if not tiles:
        tiles = sorted(p for p in tiles_dir.glob(f"{grid_id}_*.tif")
                       if "mosaic" not in p.stem)
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=None,
                    help="Output dir. Default: ~/zasolar_data/coco/coco_train20_val5")
    ap.add_argument("--chip-size", type=int, default=400)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--neg-ratio", type=float, default=0.15,
                    help="Negative:positive chip ratio (default 0.15)")
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Scan chips and report counts; do not write chips/JSON to disk.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else \
        Path.home() / "zasolar_data/coco/coco_train20_val5"
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    spec = yaml.safe_load(SPEC_PATH.read_text())
    print(f"[SPEC] {SPEC_PATH}")
    print(f"[OUT ] {output_dir}{' (DRY-RUN)' if args.dry_run else ''}")
    print()

    # ── Per-grid: load annotations + tiles, build tile_to_annots ───────
    grid_records = []
    for split_name, region_key, grid_id, ann_path, imagery_layer in load_spec_grids(spec):
        if not ann_path.exists():
            print(f"[MISS] {grid_id}: annotation not found {ann_path}", file=sys.stderr)
            continue
        tiles = get_tiles(grid_id, region_key, imagery_layer)
        if not tiles:
            print(f"[MISS] {grid_id}: no tiles in {region_key}/{imagery_layer}", file=sys.stderr)
            continue

        import rasterio
        with rasterio.open(tiles[0]) as src:
            tile_crs = src.crs

        annots = load_annotations_for_grid(ann_path, tile_crs, region_key, grid_id)
        tile_to_annots = assign_annotations_to_tiles_intersecting(annots, tiles)

        n_pos = sum(len(v) for v in tile_to_annots.values())
        n_tiles = len(tile_to_annots)
        n_pos_tiles = sum(1 for v in tile_to_annots.values() if v)
        print(f"[GRID] {split_name:5s} {region_key:12s} {grid_id} ({imagery_layer}): "
              f"{n_tiles} tiles ({n_pos_tiles} pos), {n_pos} annots, {len(annots)} polygons "
              f"-> tile CRS {tile_crs}")

        grid_records.append({
            "split": split_name,
            "region": region_key,
            "grid_id": grid_id,
            "imagery_layer": imagery_layer,
            "annotation_path": ann_path,
            "tiles": tiles,
            "annots": annots,
            "tile_to_annots": tile_to_annots,
            "tile_map": {t.stem: t for t in tiles},
        })

    # ── Scan chips per split ───────────────────────────────────────────
    summary = {"output_dir": str(output_dir), "splits": {}}
    all_provenance_rows = []
    for split_name in ("train", "val"):
        all_images = []
        all_annots = []
        all_prov = []
        img_id_counter = 1
        ann_id_counter = 1

        for rec in grid_records:
            if rec["split"] != split_name:
                continue
            for stem, annot_indices in rec["tile_to_annots"].items():
                tile_path = rec["tile_map"][stem]
                imgs, anns, prov = scan_chips_from_tile(
                    tile_path=tile_path,
                    annotations=rec["annots"],
                    annot_indices=annot_indices,
                    chip_size=args.chip_size,
                    overlap=args.overlap,
                    split_name=split_name,
                    image_id_start=img_id_counter,
                    annot_id_start=ann_id_counter,
                )
                for img in imgs:
                    img["region"] = rec["region"]
                    img["grid_id"] = rec["grid_id"]
                    img["imagery_layer"] = rec["imagery_layer"]
                for p in prov:
                    p["region"] = rec["region"]
                    p["grid_id"] = rec["grid_id"]
                    p["imagery_layer"] = rec["imagery_layer"]
                img_id_counter += len(imgs)
                ann_id_counter += len(anns)
                all_images.extend(imgs)
                all_annots.extend(anns)
                all_prov.extend(prov)

        n_pos = sum(1 for img in all_images if img["positive"])
        n_neg = len(all_images) - n_pos
        print(f"\n[SCAN] {split_name}: {len(all_images)} chips "
              f"({n_pos} positive, {n_neg} negative), {len(all_annots)} instances")

        if not args.no_balance:
            all_images, all_annots, all_prov = balance_chips(
                all_images, all_annots, all_prov,
                seed=args.seed, neg_ratio=args.neg_ratio,
            )
            n_pos2 = sum(1 for img in all_images if img["positive"])
            n_neg2 = len(all_images) - n_pos2
            print(f"[BALANCE] {split_name}: {len(all_images)} chips after balancing "
                  f"({n_pos2} positive, {n_neg2} negative, ratio={args.neg_ratio})")

        summary["splits"][split_name] = {
            "images": len(all_images),
            "annotations": len(all_annots),
            "positive": sum(1 for img in all_images if img["positive"]),
        }

        if args.dry_run:
            continue

        print(f"[WRITE] {split_name}: writing {len(all_images)} chips")
        write_selected_chips(all_images, output_dir, args.chip_size)

        coco = build_coco_json(
            all_images, all_annots, split_name,
            category_name="solar_panel",
            regions=["cape_town", "johannesburg"],
        )
        json_path = output_dir / f"{split_name}.json"
        json_path.write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
        print(f"[SAVE] {json_path} ({len(all_images)} images, {len(all_annots)} annots)")

        all_provenance_rows.extend(all_prov)

    # ── Manifest + provenance ──────────────────────────────────────────
    if not args.dry_run:
        import csv
        if all_provenance_rows:
            keys = list(all_provenance_rows[0].keys())
            prov_path = output_dir / "provenance.csv"
            with open(prov_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(all_provenance_rows)
            print(f"[SAVE] {prov_path}")

        manifest = {
            "spec_path": str(SPEC_PATH.relative_to(PROJECT_ROOT)),
            "spec_name": spec.get("spec_name"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chip_size": args.chip_size,
            "overlap": args.overlap,
            "neg_ratio": args.neg_ratio,
            "seed": args.seed,
            "summary": summary,
            "grids": [
                {
                    "split": r["split"],
                    "region": r["region"],
                    "grid_id": r["grid_id"],
                    "imagery_layer": r["imagery_layer"],
                    "annotation_path": str(r["annotation_path"].relative_to(PROJECT_ROOT)),
                    "n_polygons": int(len(r["annots"])),
                }
                for r in grid_records
            ],
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[SAVE] {output_dir / 'manifest.json'}")

    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
