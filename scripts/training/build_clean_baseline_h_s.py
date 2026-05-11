#!/usr/bin/env python3
"""Build the clean H+S baseline COCO dataset.

Pool:
  - Cape Town 260320 T1 SAM2 manual anchors: G1189/G1190/G1238.
  - Johannesburg Vexcel 2024 true-FN SAM additions:
    results/johannesburg/v3c_vexcel_2024_ch1_sample/G*/review/G*_sam_added.gpkg.

The build intentionally leaves imagery/GSD untouched. It writes standard
``train.json`` / ``val.json`` plus chip GeoTIFFs consumable by ``train.py``.
The validation split is an internal training monitor only; model selection
still belongs to the V1.4 grid-level validation harness.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box as shapely_box

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.grid_utils import resolve_tiles_dir  # noqa: E402
from export_coco_dataset import (  # noqa: E402
    balance_chips,
    build_coco_json,
    scan_chips_from_tile,
    write_selected_chips,
)

CT_GRIDS = ("G1189", "G1190", "G1238")
JHB_GRIDS = (
    "G0772", "G0773", "G0774", "G0775", "G0776",
    "G0814", "G0815", "G0816", "G0817", "G0818",
    "G0853", "G0854", "G0855", "G0856", "G0857",
    "G0888", "G0889", "G0890", "G0891", "G0892",
    "G0922", "G0923", "G0924", "G0925", "G0926",
)
DEFAULT_VAL_GRIDS = ("G1238", "G0816", "G0925")

EXPECTED_CT_POLYGONS = 475
EXPECTED_JHB_POLYGONS = 1699


def _load_gdf(path: Path, tile_crs) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    return gdf.to_crs(tile_crs).reset_index(drop=True)


def _tiles_for(grid_id: str, region: str, imagery_layer: str) -> list[Path]:
    tiles_dir = resolve_tiles_dir(grid_id, region=region, imagery_layer=imagery_layer)
    if tiles_dir.is_file():
        return [tiles_dir]
    tiles = sorted(tiles_dir.glob(f"{grid_id}_*_*_geo.tif"))
    if not tiles:
        tiles = sorted(p for p in tiles_dir.glob(f"{grid_id}_*.tif") if "mosaic" not in p.stem)
    return tiles


def _assign_intersections(annotations: gpd.GeoDataFrame, tiles: list[Path]) -> dict[str, list[int]]:
    tile_bounds = {}
    for tile in tiles:
        with rasterio.open(tile) as src:
            b = src.bounds
            tile_bounds[tile.stem] = shapely_box(b.left, b.bottom, b.right, b.top)

    out: dict[str, list[int]] = {stem: [] for stem in tile_bounds}
    for idx, row in annotations.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        for stem, bbox in tile_bounds.items():
            if geom.intersects(bbox):
                out[stem].append(idx)
    return out


def _annotation_records() -> list[dict]:
    records: list[dict] = []
    for grid_id in CT_GRIDS:
        records.append({
            "region": "cape_town",
            "imagery_layer": "aerial_2025",
            "grid_id": grid_id,
            "label_source": "human_manual_sam_assisted",
            "source_group": "H",
            "annotation_path": PROJECT_ROOT / "data" / "annotations" / "Capetown" / f"{grid_id}_SAM2_260320.gpkg",
        })
    for grid_id in JHB_GRIDS:
        records.append({
            "region": "johannesburg",
            "imagery_layer": "vexcel_2024",
            "grid_id": grid_id,
            "label_source": "sam_added_true_fn",
            "source_group": "S",
            "annotation_path": (
                PROJECT_ROOT
                / "results"
                / "johannesburg"
                / "v3c_vexcel_2024_ch1_sample"
                / grid_id
                / "review"
                / f"{grid_id}_sam_added.gpkg"
            ),
        })
    return records


def _load_grid_records() -> list[dict]:
    records = []
    for rec in _annotation_records():
        ann_path = Path(rec["annotation_path"])
        if not ann_path.exists():
            raise FileNotFoundError(f"annotation file missing: {ann_path}")

        tiles = _tiles_for(rec["grid_id"], rec["region"], rec["imagery_layer"])
        if not tiles:
            raise FileNotFoundError(
                f"tiles missing for {rec['region']}/{rec['imagery_layer']}/{rec['grid_id']}"
            )

        with rasterio.open(tiles[0]) as src:
            tile_crs = src.crs
        annots = _load_gdf(ann_path, tile_crs)
        tile_to_annots = _assign_intersections(annots, tiles)

        records.append({
            **rec,
            "annotation_path": ann_path,
            "tiles": tiles,
            "tile_map": {t.stem: t for t in tiles},
            "annots": annots,
            "tile_to_annots": tile_to_annots,
            "tile_crs": str(tile_crs),
        })
    return records


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).expanduser()
    val_grids = set(args.val_grids)

    records = _load_grid_records()
    ct_total = sum(len(r["annots"]) for r in records if r["source_group"] == "H")
    jhb_total = sum(len(r["annots"]) for r in records if r["source_group"] == "S")
    if args.enforce_counts:
        if ct_total != EXPECTED_CT_POLYGONS or jhb_total != EXPECTED_JHB_POLYGONS:
            raise RuntimeError(
                f"source count mismatch: CT={ct_total} expected={EXPECTED_CT_POLYGONS}, "
                f"JHB={jhb_total} expected={EXPECTED_JHB_POLYGONS}"
            )

    print(f"[POOL] grids={len(records)} polygons={ct_total + jhb_total} "
          f"(CT_H={ct_total}, JHB_S={jhb_total})")
    print(f"[SPLIT] val_grids={','.join(sorted(val_grids)) or '(none)'}")

    source_rows = []
    for rec in records:
        n_tile_refs = sum(len(v) for v in rec["tile_to_annots"].values())
        n_pos_tiles = sum(1 for v in rec["tile_to_annots"].values() if v)
        split = "val" if rec["grid_id"] in val_grids else "train"
        source_rows.append({
            "split": split,
            "region": rec["region"],
            "imagery_layer": rec["imagery_layer"],
            "grid_id": rec["grid_id"],
            "source_group": rec["source_group"],
            "label_source": rec["label_source"],
            "polygons": len(rec["annots"]),
            "tiles": len(rec["tiles"]),
            "positive_tiles": n_pos_tiles,
            "tile_annotation_refs": n_tile_refs,
            "tile_crs": rec["tile_crs"],
            "annotation_path": str(rec["annotation_path"].relative_to(PROJECT_ROOT)),
        })
        print(f"[GRID] {split:5s} {rec['region']:12s} {rec['grid_id']} "
              f"{rec['source_group']} polygons={len(rec['annots'])} "
              f"tiles={len(rec['tiles'])} pos_tiles={n_pos_tiles}")

    if args.dry_run:
        return {
            "output_dir": str(output_dir),
            "dry_run": True,
            "source_rows": source_rows,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "source_manifest.csv", source_rows)

    summary: dict = {
        "dataset_id": args.dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "source_policy": {
            "cape_town": "Only G1189/G1190/G1238 *_SAM2_260320.gpkg T1 manual SAM2 anchors.",
            "johannesburg": "Only v3c_vexcel_2024_ch1_sample per-grid *_sam_added.gpkg true-FN SAM additions.",
            "excluded": "Reviewed predictions, V3C_TP-only polygons, CT 260322-260403 weak pre-realtime SAM2 batches, and imagery/GSD reprojection.",
        },
        "expected_polygons": {
            "ct_h": EXPECTED_CT_POLYGONS,
            "jhb_s": EXPECTED_JHB_POLYGONS,
            "total": EXPECTED_CT_POLYGONS + EXPECTED_JHB_POLYGONS,
        },
        "actual_polygons": {
            "ct_h": ct_total,
            "jhb_s": jhb_total,
            "total": ct_total + jhb_total,
        },
        "chip_size": args.chip_size,
        "overlap": args.overlap,
        "neg_ratio": args.neg_ratio,
        "val_grids": sorted(val_grids),
        "splits": {},
    }

    all_prov_rows = []
    for split_name in ("train", "val"):
        all_images = []
        all_annots = []
        all_prov = []
        img_id_counter = 1
        ann_id_counter = 1

        for rec in records:
            rec_split = "val" if rec["grid_id"] in val_grids else "train"
            if rec_split != split_name:
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
                    img["source_group"] = rec["source_group"]
                    img["label_source"] = rec["label_source"]
                for ann in anns:
                    ann["source_group"] = rec["source_group"]
                    ann["label_source"] = rec["label_source"]
                    ann["grid_id"] = rec["grid_id"]
                    ann["region"] = rec["region"]
                for row in prov:
                    row["region"] = rec["region"]
                    row["grid_id"] = rec["grid_id"]
                    row["imagery_layer"] = rec["imagery_layer"]
                    row["source_group"] = rec["source_group"]
                    row["label_source"] = rec["label_source"]
                img_id_counter += len(imgs)
                ann_id_counter += len(anns)
                all_images.extend(imgs)
                all_annots.extend(anns)
                all_prov.extend(prov)

        n_pos = sum(1 for img in all_images if img["positive"])
        n_neg = len(all_images) - n_pos
        print(f"[SCAN] {split_name}: {len(all_images)} chips "
              f"({n_pos} positive, {n_neg} negative), {len(all_annots)} instances")

        if args.neg_ratio >= 0:
            all_images, all_annots, all_prov = balance_chips(
                all_images, all_annots, all_prov, seed=args.seed, neg_ratio=args.neg_ratio
            )
            n_pos = sum(1 for img in all_images if img["positive"])
            n_neg = len(all_images) - n_pos
            print(f"[BALANCE] {split_name}: {len(all_images)} chips "
                  f"({n_pos} positive, {n_neg} negative)")

        print(f"[WRITE] {split_name}: writing {len(all_images)} chips")
        write_selected_chips(all_images, output_dir, args.chip_size)

        coco = build_coco_json(
            all_images,
            all_annots,
            split_name,
            category_name="solar_panel",
            regions=["cape_town", "johannesburg"],
        )
        coco["info"]["dataset_id"] = args.dataset_id
        coco["info"]["source_policy"] = summary["source_policy"]
        json_path = output_dir / f"{split_name}.json"
        json_path.write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")

        _write_csv(output_dir / f"{split_name}_provenance.csv", all_prov)
        all_prov_rows.extend(all_prov)
        source_counts = {}
        for img in all_images:
            key = img.get("source_group", "")
            source_counts[key] = source_counts.get(key, 0) + 1
        summary["splits"][split_name] = {
            "images": len(all_images),
            "annotations": len(all_annots),
            "positive_images": n_pos,
            "negative_images": n_neg,
            "image_source_counts": source_counts,
        }
        print(f"[SAVE] {json_path} images={len(all_images)} annots={len(all_annots)}")

    _write_csv(output_dir / "provenance.csv", all_prov_rows)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[SAVE] {output_dir / 'dataset_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "zasolar_data" / "coco" / "coco_clean_hs_ct260320_jhb_sam_added_20260510"),
    )
    parser.add_argument("--dataset-id", default="clean_hs_ct260320_jhb_sam_added_20260510")
    parser.add_argument("--chip-size", type=int, default=400)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--neg-ratio", type=float, default=0.15,
                        help="Negative:positive chip ratio. Use -1 to skip balancing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-grids", nargs="*", default=list(DEFAULT_VAL_GRIDS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-enforce-counts", dest="enforce_counts", action="store_false")
    parser.set_defaults(enforce_counts=True)
    return parser.parse_args()


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
