"""Build train20_val5 + hard-negative variant.

HN sources:
  - JHB CBD train-20 V3-C FPs on vexcel_2024 (computed fresh from V3-C
    predictions at results/johannesburg/v3c_vexcel_2024/<grid>/predictions_metric.gpkg
    minus clean GT, IoU < 0.1, post_conf >= 0.85 already applied at inference time).
  - CT V4.1 small-FP shortlist filtered to train grids
    (results/analysis/small_fp/taxonomy_run/hn_small_fp_shortlist.csv).

Audit: any HN chip whose 400×400 window in metric CRS intersects a clean GT
polygon with IoU >= 0.1 is dropped. Prevents teaching the detector to suppress
real PV pixels.

Output: ``~/zasolar_data/coco/coco_train20_val5_hn/`` — same layout as the base
COCO; train.json grows, val.json unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.windows import Window
from shapely.geometry import box as shapely_box

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.grid_utils import resolve_tiles_dir  # noqa: E402

BASE_COCO = Path.home() / "zasolar_data/coco/coco_train20_val5"
OUT_COCO = Path.home() / "zasolar_data/coco/coco_train20_val5_hn"
SPEC_PATH = PROJECT_ROOT / "configs/datasets/train20_val5.yaml"

TRAIN_JHB = ["G0772","G0773","G0774","G0775","G0814","G0815","G0818",
             "G0853","G0854","G0855","G0856","G0857","G0888","G0889",
             "G0890","G0892","G0922","G0923","G0924","G0926"]
TRAIN_CT = ["G1238","G1300","G1411","G1570","G1572","G1634","G1635",
            "G1743","G1800","G1806","G1862","G1911","G1919","G1920",
            "G1972","G1973","G1975","G1976","G2027","G2029"]

CT_SHORTLIST_CSV = PROJECT_ROOT / "results/analysis/small_fp/taxonomy_run/hn_small_fp_shortlist.csv"
JHB_PRED_ROOT = PROJECT_ROOT / "results/johannesburg/v3c_vexcel_2024"
JHB_GT_ROOT = PROJECT_ROOT / "data/annotations_channel2_clean"
CRS_BY_REGION = {"johannesburg": "EPSG:32735", "cape_town": "EPSG:32734"}

CHIP_SIZE = 400
HN_IMG_ID_OFFSET = 10_000_000  # avoid collision with base ids


def build_jhb_fp_pool(iou_thresh: float = 0.1) -> gpd.GeoDataFrame:
    """V3-C Vexcel preds minus clean GT (max-IoU < iou_thresh), per train grid."""
    rows = []
    for grid in TRAIN_JHB:
        pred_p = JHB_PRED_ROOT / grid / "predictions_metric.gpkg"
        gt_p = JHB_GT_ROOT / grid / f"{grid}_clean_gt.gpkg"
        if not pred_p.exists() or not gt_p.exists():
            print(f"  [MISS] {grid}: missing predictions or clean_gt")
            continue
        pred = gpd.read_file(pred_p)
        gt = gpd.read_file(gt_p).to_crs(pred.crs)
        # Spatial index for fast IoU computation
        gt_sindex = gt.sindex
        for i, p in pred.iterrows():
            cands = list(gt_sindex.intersection(p.geometry.bounds))
            max_iou = 0.0
            for j in cands:
                gt_geom = gt.geometry.iloc[j]
                inter = p.geometry.intersection(gt_geom)
                if inter.is_empty:
                    continue
                u = p.geometry.union(gt_geom).area
                if u > 0:
                    iou = inter.area / u
                    if iou > max_iou:
                        max_iou = iou
            if max_iou < iou_thresh:
                rows.append({
                    "grid_id": grid,
                    "region": "johannesburg",
                    "imagery_layer": "vexcel_2024",
                    "pred_id": int(p["value"]) if "value" in p else i,
                    "confidence": float(p["confidence"]),
                    "area_m2": float(p["area_m2"]),
                    "geometry": p.geometry,
                })
    print(f"  JHB Vexcel FPs: {len(rows)}")
    return gpd.GeoDataFrame(rows, crs="EPSG:32735")


def build_ct_fp_pool(annotation_paths: dict[str, Path]) -> gpd.GeoDataFrame:
    """Filter CT shortlist to train grids, attach polygon geometry from per-grid prediction gpkg."""
    df = pd.read_csv(CT_SHORTLIST_CSV)
    df = df[df.grid_id.isin(TRAIN_CT)].reset_index(drop=True)
    print(f"  CT shortlist filtered to train grids: {len(df)}")
    # The shortlist has grid_id + pred_id; the per-grid V4.1 prediction GPKG has the geometry.
    # Look for a matching predictions_metric.gpkg.
    rows = []
    for grid, sub in df.groupby("grid_id"):
        # Try V4.1 first, fall back to V3-C results path
        candidates = [
            PROJECT_ROOT / f"results/cape_town/v3c_targeted_hn_aerial_2025/{grid}/predictions_metric.gpkg",
            PROJECT_ROOT / f"results/{grid}/predictions_metric.gpkg",
        ]
        pred_p = next((c for c in candidates if c.exists()), None)
        if pred_p is None:
            print(f"    [MISS] {grid}: no predictions_metric.gpkg")
            continue
        pred = gpd.read_file(pred_p).to_crs("EPSG:32734")
        # pred_id in the shortlist is the row index into the GPKG (verified empirically)
        for _, r in sub.iterrows():
            pid = int(r["pred_id"])
            if pid < 0 or pid >= len(pred):
                continue
            geom = pred.geometry.iloc[pid]
            rows.append({
                "grid_id": grid,
                "region": "cape_town",
                "imagery_layer": "aerial_2025",
                "pred_id": int(r["pred_id"]),
                "confidence": float(r["confidence"]),
                "area_m2": float(r["area_m2"]),
                "geometry": geom,
            })
    print(f"  CT FPs attached with geometry: {len(rows)}")
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:32734")
    return gpd.GeoDataFrame(rows, crs="EPSG:32734")


def chip_overlaps_gt(chip_bounds_4326, gt_4326):
    """True if chip 4326 bbox overlaps any GT polygon with IoU >= 0.1."""
    if gt_4326 is None or len(gt_4326) == 0:
        return False
    cb = shapely_box(*chip_bounds_4326)
    cb_area = cb.area
    sidx = gt_4326.sindex
    for j in sidx.intersection(cb.bounds):
        g = gt_4326.geometry.iloc[j]
        inter = cb.intersection(g)
        if inter.is_empty:
            continue
        u = cb.union(g).area
        if u > 0 and inter.area / u >= 0.1:
            return True
    return False


def find_tile_for_point(grid_id: str, lon: float, lat: float,
                        region: str, imagery_layer: str) -> Path | None:
    d = resolve_tiles_dir(grid_id, region=region, imagery_layer=imagery_layer)
    if d.is_file():
        return d
    for tif in d.glob(f"{grid_id}_*_*_geo.tif"):
        with rasterio.open(tif) as src:
            from pyproj import Transformer
            t_to_tile = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            tx, ty = t_to_tile.transform(lon, lat)
            l, b, r, top = src.bounds
            if l <= tx <= r and b <= ty <= top:
                return tif
    return None


def extract_hn_chips(fp_gdf: gpd.GeoDataFrame, gt_by_grid: dict[str, gpd.GeoDataFrame],
                    chip_dir: Path, prefix: str) -> tuple[list[dict], list[dict]]:
    """Extract chip TIFs centered on FP centroids; return (image_records, provenance_rows)."""
    chip_dir.mkdir(parents=True, exist_ok=True)
    images = []
    provenance = []
    seen = set()  # dedup by (tile_stem, x0, y0)

    fp_4326 = fp_gdf.to_crs("EPSG:4326")
    for i, row in fp_4326.iterrows():
        grid = row["grid_id"]
        region = row["region"]
        layer = row["imagery_layer"]
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y  # lon, lat
        tile_path = find_tile_for_point(grid, cx, cy, region, layer)
        if tile_path is None:
            continue
        with rasterio.open(tile_path) as src:
            from pyproj import Transformer
            t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            tx, ty = t.transform(cx, cy)
            inv = ~src.transform
            col, ro = inv * (tx, ty)
            x0 = int(round(col)) - CHIP_SIZE // 2
            y0 = int(round(ro)) - CHIP_SIZE // 2
            x0 = max(0, min(x0, src.width - CHIP_SIZE))
            y0 = max(0, min(y0, src.height - CHIP_SIZE))
            key = (tile_path.stem, x0, y0)
            if key in seen:
                continue
            seen.add(key)
            window = Window(x0, y0, CHIP_SIZE, CHIP_SIZE)
            data = src.read(window=window)
            chip_transform = src.window_transform(window)
            win_bounds_native = rasterio.windows.bounds(window, src.transform)
            # convert to 4326 for GT audit
            t_back = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            l4, b4 = t_back.transform(win_bounds_native[0], win_bounds_native[1])
            r4, t4 = t_back.transform(win_bounds_native[2], win_bounds_native[3])
            chip_bounds_4326 = (min(l4, r4), min(b4, t4), max(l4, r4), max(b4, t4))

            # GT audit
            gt = gt_by_grid.get(grid)
            if chip_overlaps_gt(chip_bounds_4326, gt):
                continue

            chip_name = f"{prefix}_{grid}_{tile_path.stem}_{x0}_{y0}.tif"
            chip_path = chip_dir / chip_name
            profile = src.profile.copy()
            for k in ("photometric", "compress", "jpeg_quality", "jpegtablesmode"):
                profile.pop(k, None)
            profile.update(driver="GTiff", width=CHIP_SIZE, height=CHIP_SIZE,
                          transform=chip_transform, compress="lzw")
            with rasterio.open(str(chip_path), "w", **profile) as dst:
                dst.write(data)

            images.append({
                "file_name": f"train/{chip_name}",
                "width": CHIP_SIZE,
                "height": CHIP_SIZE,
                "positive": False,
                "region": region,
                "grid_id": grid,
                "imagery_layer": layer,
            })
            provenance.append({
                "chip_file": chip_name,
                "source_tile": tile_path.stem,
                "x0": x0, "y0": y0,
                "width": CHIP_SIZE, "height": CHIP_SIZE,
                "n_annotations": 0,
                "split": "train",
                "region": region,
                "grid_id": grid,
                "imagery_layer": layer,
                "hn_source": prefix,
                "fp_pred_id": int(row["pred_id"]),
                "fp_confidence": float(row["confidence"]),
                "fp_area_m2": float(row["area_m2"]),
            })
    return images, provenance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_COCO))
    ap.add_argument("--iou-thresh", type=float, default=0.1)
    ap.add_argument("--max-jhb-fps-per-grid", type=int, default=None,
                    help="Cap JHB FPs per grid to balance distribution")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train").mkdir(exist_ok=True)
    (out_dir / "val").mkdir(exist_ok=True)

    print("[STEP 1] Build JHB FP pool from V3-C Vexcel predictions")
    jhb_fp = build_jhb_fp_pool(iou_thresh=args.iou_thresh)
    if args.max_jhb_fps_per_grid:
        jhb_fp = (jhb_fp.groupby("grid_id", group_keys=False)
                  .apply(lambda d: d.head(args.max_jhb_fps_per_grid)))
    print(f"  JHB FP pool: {len(jhb_fp)}")
    jhb_fp.to_file(out_dir / "jhb_fp_pool.gpkg", driver="GPKG")

    print("\n[STEP 2] Build CT FP pool from V4.1 small-FP shortlist")
    spec = yaml.safe_load(SPEC_PATH.read_text())
    ann_paths = {}
    for region_key, regs in spec["splits"].items():
        if region_key != "train": continue
    ct_fp = build_ct_fp_pool(ann_paths)
    print(f"  CT FP pool: {len(ct_fp)}")
    ct_fp.to_file(out_dir / "ct_fp_pool.gpkg", driver="GPKG")

    # Load clean GT per grid for audit
    print("\n[STEP 3] Load clean GT per grid for audit")
    gt_by_grid = {}
    for grid in TRAIN_JHB:
        gt_p = JHB_GT_ROOT / grid / f"{grid}_clean_gt.gpkg"
        if gt_p.exists():
            gt_by_grid[grid] = gpd.read_file(gt_p).to_crs("EPSG:4326")
    # CT GT from spec
    for entry in spec["splits"]["train"]["cape_town"].get("grids", []):
        gid = entry["grid_id"]
        fname = entry["file"]
        ct_gt_p = PROJECT_ROOT / spec["splits"]["train"]["cape_town"]["annotation_root"] / fname
        if ct_gt_p.exists():
            gdf = gpd.read_file(ct_gt_p)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            gt_by_grid[gid] = gdf.to_crs("EPSG:4326")

    print(f"  GT loaded for {len(gt_by_grid)} grids")

    print("\n[STEP 4] Extract JHB HN chips")
    jhb_imgs, jhb_prov = extract_hn_chips(jhb_fp, gt_by_grid, out_dir / "train", prefix="hn_jhb")
    print(f"  JHB HN chips written: {len(jhb_imgs)}")

    print("\n[STEP 5] Extract CT HN chips")
    ct_imgs, ct_prov = extract_hn_chips(ct_fp, gt_by_grid, out_dir / "train", prefix="hn_ct")
    print(f"  CT HN chips written: {len(ct_imgs)}")

    print("\n[STEP 6] Build new train.json (base + HN)")
    base_train = json.loads((BASE_COCO / "train.json").read_text())
    base_val = json.loads((BASE_COCO / "val.json").read_text())

    # Copy base train chips into new train dir
    print("  Copying base train chips...")
    for img in base_train["images"]:
        src = BASE_COCO / img["file_name"]
        dst = out_dir / img["file_name"]
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    print("  Copying base val chips...")
    for img in base_val["images"]:
        src = BASE_COCO / img["file_name"]
        dst = out_dir / img["file_name"]
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # Append HN images with offset IDs
    next_id = HN_IMG_ID_OFFSET
    new_imgs = list(base_train["images"])
    for hn_imgs in (jhb_imgs, ct_imgs):
        for img in hn_imgs:
            img["id"] = next_id
            next_id += 1
            new_imgs.append(img)

    new_train = {
        "info": {**base_train["info"],
                 "description": base_train["info"]["description"] + " + HN"},
        "licenses": base_train["licenses"],
        "categories": base_train["categories"],
        "images": new_imgs,
        "annotations": base_train["annotations"],  # HN chips have no annots
    }
    (out_dir / "train.json").write_text(json.dumps(new_train, indent=2) + "\n", encoding="utf-8")
    (out_dir / "val.json").write_text(json.dumps(base_val, indent=2) + "\n", encoding="utf-8")

    # Provenance
    import csv
    base_prov_path = BASE_COCO / "provenance.csv"
    if base_prov_path.exists():
        out_prov = out_dir / "provenance.csv"
        all_rows = []
        with open(base_prov_path) as f:
            base_rows = list(csv.DictReader(f))
        keys = list(base_rows[0].keys()) if base_rows else []
        for r in base_rows:
            r["hn_source"] = ""
            all_rows.append(r)
        all_keys = list(keys)
        for k in ("hn_source","fp_pred_id","fp_confidence","fp_area_m2"):
            if k not in all_keys:
                all_keys.append(k)
        for r in jhb_prov + ct_prov:
            r["image_id"] = ""
            for k in all_keys:
                if k not in r:
                    r[k] = ""
            all_rows.append(r)
        with open(out_prov, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(set(all_keys + ["hn_source","fp_pred_id","fp_confidence","fp_area_m2"])))
            w.writeheader()
            w.writerows(all_rows)

    # Manifest
    base_manifest = json.loads((BASE_COCO / "manifest.json").read_text())
    base_manifest["spec_name"] = base_manifest.get("spec_name", "") + "_hn"
    base_manifest["hn_summary"] = {
        "jhb_fp_pool": len(jhb_fp),
        "ct_fp_pool": len(ct_fp),
        "jhb_chips_written": len(jhb_imgs),
        "ct_chips_written": len(ct_imgs),
        "iou_audit_thresh": args.iou_thresh,
    }
    base_manifest["summary"]["splits"]["train"] = {
        "images": len(new_imgs),
        "annotations": len(base_train["annotations"]),
        "positive": sum(1 for i in new_imgs if i.get("positive")),
    }
    (out_dir / "manifest.json").write_text(json.dumps(base_manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\n[DONE] {out_dir}")
    pos = base_manifest["summary"]["splits"]["train"]["positive"]
    print(f"  train: {len(new_imgs)} chips ({pos} pos, {len(new_imgs) - pos} neg)")
    print(f"  val:   {len(base_val['images'])} chips")
    print(f"  HN: JHB {len(jhb_imgs)} + CT {len(ct_imgs)} = {len(jhb_imgs) + len(ct_imgs)}")


if __name__ == "__main__":
    main()
