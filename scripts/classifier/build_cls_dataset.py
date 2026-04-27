"""
Build PV vs non-PV classification dataset from reviewed predictions.

Registry-driven: enumerates every `(region, model_run, grid)` from
`configs/datasets/regions.yaml` via `core.region_registry`, discovers
`review/{grid}_reviewed.gpkg` under each model_run's results_path, and uses
reviewed `review_status` ({correct, edit} → PV; {delete} → non-PV) as labels.
The `reviewed.gpkg` is the source of truth for both area_m2 and status —
`predictions_metric.gpkg` is only consulted when joining auxiliary label
sources (GT heater audit, small-FP taxonomy) that reference pred_id.

Split policy: region-stratified whole-grid holdout. Each source bucket
(`cape_town:legacy_flat_batch003`, `cape_town:v3c_targeted_hn_aerial_2025`,
`johannesburg:v4_aerial_2023`) gets its own 80/20 grid split, then results
are concatenated. Prevents one region dominating either split.

Usage:
    python scripts/classifier/build_cls_dataset.py \
        --output-dir data/cls_pv_thermal_v1 \
        --area-cutoff 30 \
        --val-fraction 0.2 \
        --include-taxonomy \
        --include-gt-audit \
        --aug-profile current
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core import region_registry  # noqa: E402
from core.grid_utils import TILES_ROOT, resolve_tiles_dir  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

CHIP_SIZE = 400       # extraction chip size (pixels)
IMG_SIZE = 224        # output image size (pixels)

LABEL_PV = "pv"
LABEL_NONPV = "non_pv"

TAXONOMY_PV_LABELS = {"correct_detection"}
GT_AUDIT_PV_LABELS = {"pv"}
GT_AUDIT_NONPV_LABELS = {"heater_or_non_pv"}


@dataclass(frozen=True)
class GridSource:
    """One discovered `(region, model_run, grid)` bucket with review data."""

    region: str
    model_run: str
    grid_id: str
    results_path: Path           # absolute dir under which grid_id/review/ lives
    reviewed_gpkg: Path | None   # absolute path, may be None for csv-only
    review_csv: Path | None
    deprecated: bool
    source_bucket: str           # "<region>:<model_run>" — stratum key


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _read_deprecated_flags() -> dict[tuple[str, str], bool]:
    import yaml
    flags: dict[tuple[str, str], bool] = {}
    with open(region_registry.REGIONS_YAML) as f:
        raw = yaml.safe_load(f)
    for region_key, region_data in raw.get("regions", {}).items():
        for run_id, run_data in (region_data.get("model_runs") or {}).items():
            flags[(region_key, run_id)] = bool(run_data.get("deprecated", False))
    return flags


def discover_grid_sources(
    *,
    include_deprecated: bool = False,
    include_legacy_flat: bool = True,
) -> list[GridSource]:
    """Enumerate every grid with reviewed data via the region registry.

    Skips model_runs marked `deprecated: true` in regions.yaml unless
    `include_deprecated=True`. Includes legacy flat `results/G*/review/`
    as a `cape_town:legacy_flat_batch003` pseudo-bucket unless disabled.
    """
    deprecated_flags = _read_deprecated_flags()
    sources: list[GridSource] = []

    for region in region_registry.list_regions():
        for run_id in region_registry.list_model_runs(region):
            if deprecated_flags.get((region, run_id), False) and not include_deprecated:
                continue
            try:
                results_path = region_registry.get_model_run_path(region, run_id)
            except KeyError:
                continue
            if not results_path.exists():
                continue
            for grid_dir in sorted(results_path.glob("G*")):
                review_dir = grid_dir / "review"
                if not review_dir.is_dir():
                    continue
                gpkg_candidates = list(review_dir.glob("*_reviewed.gpkg"))
                csv_path = review_dir / "detection_review_decisions.csv"
                if not gpkg_candidates and not csv_path.exists():
                    continue
                sources.append(
                    GridSource(
                        region=region,
                        model_run=run_id,
                        grid_id=grid_dir.name,
                        results_path=results_path,
                        reviewed_gpkg=gpkg_candidates[0] if gpkg_candidates else None,
                        review_csv=csv_path if csv_path.exists() else None,
                        deprecated=deprecated_flags.get((region, run_id), False),
                        source_bucket=f"{region}:{run_id}",
                    )
                )

    if include_legacy_flat:
        for grid_dir in sorted(RESULTS_DIR.glob("G*")):
            if not grid_dir.is_dir():
                continue
            review_dir = grid_dir / "review"
            if not review_dir.is_dir():
                continue
            gpkg_candidates = list(review_dir.glob("*_reviewed.gpkg"))
            csv_path = review_dir / "detection_review_decisions.csv"
            if not gpkg_candidates and not csv_path.exists():
                continue
            sources.append(
                GridSource(
                    region="cape_town",
                    model_run="legacy_flat_batch003",
                    grid_id=grid_dir.name,
                    results_path=RESULTS_DIR,
                    reviewed_gpkg=gpkg_candidates[0] if gpkg_candidates else None,
                    review_csv=csv_path if csv_path.exists() else None,
                    deprecated=False,
                    source_bucket="cape_town:legacy_flat_batch003",
                )
            )

    return sources


# ---------------------------------------------------------------------------
# Reviewed-gpkg loading
# ---------------------------------------------------------------------------

def load_reviewed_predictions(
    sources: list[GridSource],
    area_cutoff: float,
) -> pd.DataFrame:
    """Load reviewed predictions from discovered sources.

    Authoritative fields come from `*_reviewed.gpkg`: `review_status`,
    `area_m2`, and geometry (for centroid). Internal row index within each
    gpkg serves as a stable per-grid pred_id for downstream joins.

    Returns DataFrame with columns:
        region, model_run, source_bucket, grid_id, pred_id, label,
        area_m2, confidence, source_tile, centroid_lon, centroid_lat,
        results_path, source
    """
    records: list[dict] = []
    missing_gpkg: list[str] = []

    for src in sources:
        if src.reviewed_gpkg is None:
            missing_gpkg.append(f"{src.source_bucket}/{src.grid_id}")
            continue

        try:
            preds = gpd.read_file(src.reviewed_gpkg)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: failed to read {src.reviewed_gpkg}: {e}")
            continue

        if preds.crs and preds.crs.to_epsg() != 4326:
            preds_4326 = preds.to_crs(epsg=4326)
        else:
            preds_4326 = preds

        for pred_id, (row, row_4326) in enumerate(
            zip(preds.itertuples(index=False), preds_4326.itertuples(index=False))
        ):
            status = getattr(row, "review_status", None) or getattr(row, "status", None)
            if status not in ("correct", "edit", "delete"):
                continue
            area = getattr(row, "area_m2", None)
            if area is None or area >= area_cutoff:
                continue

            geom_4326 = getattr(row_4326, "geometry", None)
            if geom_4326 is None:
                continue
            centroid = geom_4326.centroid

            records.append({
                "region": src.region,
                "model_run": src.model_run,
                "source_bucket": src.source_bucket,
                "grid_id": src.grid_id,
                "pred_id": pred_id,
                "label": LABEL_PV if status in ("correct", "edit") else LABEL_NONPV,
                "area_m2": float(area),
                "confidence": float(getattr(row, "confidence", 0) or 0),
                "source_tile": getattr(row, "source_tile", "") or "",
                "centroid_lon": centroid.x,
                "centroid_lat": centroid.y,
                "results_path": str(src.results_path.relative_to(PROJECT_ROOT)),
                "source": "reviewed",
            })

    df = pd.DataFrame(records)
    if missing_gpkg:
        print(f"  INFO: skipped {len(missing_gpkg)} grids with csv-only review "
              f"(no reviewed.gpkg → area unknown): {missing_gpkg[:5]}"
              f"{'...' if len(missing_gpkg) > 5 else ''}")
    print(f"  Loaded {len(df)} reviewed predictions from "
          f"{df['grid_id'].nunique() if len(df) else 0} grids / "
          f"{df['source_bucket'].nunique() if len(df) else 0} buckets")
    if len(df):
        print(f"  PV: {(df['label'] == LABEL_PV).sum()}, "
              f"non-PV: {(df['label'] == LABEL_NONPV).sum()}")
    return df


# ---------------------------------------------------------------------------
# External label sources (taxonomy + GT heater audit)
# ---------------------------------------------------------------------------

def _centroid_from_predictions_metric(
    grid_id: str, pred_id: int, source_lookup: dict[str, GridSource],
) -> tuple[float, float, str, float, float] | None:
    """Resolve (lon, lat, source_tile, area_m2, confidence) for a pred_id by
    reading `predictions_metric.gpkg` under the grid's registered results_path.
    """
    src = source_lookup.get(grid_id)
    if src is None:
        return None
    pred_path = src.results_path / grid_id / "predictions_metric.gpkg"
    if not pred_path.exists():
        return None
    try:
        preds = gpd.read_file(pred_path)
    except Exception:  # noqa: BLE001
        return None
    if pred_id >= len(preds):
        return None
    if preds.crs and preds.crs.to_epsg() != 4326:
        preds_4326 = preds.to_crs(epsg=4326)
    else:
        preds_4326 = preds
    row = preds.iloc[pred_id]
    centroid = preds_4326.iloc[pred_id].geometry.centroid
    return (
        centroid.x,
        centroid.y,
        str(row.get("source_tile", "") or ""),
        float(row.get("area_m2", 0) or 0),
        float(row.get("confidence", 0) or 0),
    )


def load_taxonomy_chips(
    taxonomy_csv: Path,
    area_cutoff: float,
    source_lookup: dict[str, GridSource],
) -> pd.DataFrame:
    """Load small-FP taxonomy labels (77 thermal + other HN subclasses)."""
    df = pd.read_csv(taxonomy_csv)
    df = df[df["area_m2"] < area_cutoff].copy()

    records: list[dict] = []
    for _, row in df.iterrows():
        grid_id = row["grid_id"]
        pred_id = int(row["pred_id"])
        human = row.get("human_label", "")
        label = LABEL_PV if human in TAXONOMY_PV_LABELS else LABEL_NONPV
        src = source_lookup.get(grid_id)
        if src is None:
            continue
        geo = _centroid_from_predictions_metric(grid_id, pred_id, source_lookup)
        if geo is None:
            continue
        lon, lat, source_tile, area_m2, confidence = geo
        records.append({
            "region": src.region,
            "model_run": src.model_run,
            "source_bucket": src.source_bucket,
            "grid_id": grid_id,
            "pred_id": pred_id,
            "label": label,
            "area_m2": area_m2 or float(row["area_m2"]),
            "confidence": confidence,
            "source_tile": source_tile,
            "centroid_lon": lon,
            "centroid_lat": lat,
            "results_path": str(src.results_path.relative_to(PROJECT_ROOT)),
            "source": "taxonomy",
            "taxonomy_subtype": human,
        })

    result = pd.DataFrame(records)
    print(f"  Taxonomy: {len(result)} chips "
          f"(PV: {(result['label'] == LABEL_PV).sum() if len(result) else 0}, "
          f"non-PV: {(result['label'] == LABEL_NONPV).sum() if len(result) else 0})")
    return result


def _centroid_from_annotation_gpkg(
    source_file: str, row_index: int, region: str,
) -> tuple[float, float] | None:
    """Resolve (lon, lat) from an annotation gpkg under the region's annotations_dir."""
    try:
        config = region_registry.get_region_config(region)
    except KeyError:
        return None
    ann_path = PROJECT_ROOT / config.paths.annotations_dir / source_file
    if not ann_path.exists():
        return None
    try:
        gdf = gpd.read_file(ann_path)
    except Exception:  # noqa: BLE001
        return None
    if row_index >= len(gdf):
        return None
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    centroid = gdf.iloc[row_index].geometry.centroid
    return centroid.x, centroid.y


def load_gt_heater_audit(
    audit_csv: Path,
    area_cutoff: float,
    source_lookup: dict[str, GridSource],
) -> pd.DataFrame:
    """Load GT heater audit labels (585 pv / 80 heater / 6 uncertain).

    Centroids are resolved from annotation gpkgs (`<region>.annotations_dir/
    <source_file>`, row_index), not from predictions_metric.gpkg — GT audit
    labels annotate ground-truth polygons, not model predictions.
    """
    df = pd.read_csv(audit_csv)
    df = df[df["area_m2"] < area_cutoff].copy()
    df = df[df["audit_label"].isin(GT_AUDIT_PV_LABELS | GT_AUDIT_NONPV_LABELS)].copy()

    records: list[dict] = []
    skipped_no_ann = 0
    for _, row in df.iterrows():
        grid_id = row["grid_id"]
        label = LABEL_PV if row["audit_label"] in GT_AUDIT_PV_LABELS else LABEL_NONPV
        src = source_lookup.get(grid_id)
        if src is None:
            continue
        try:
            row_index = int(row["row_index"])
        except (KeyError, ValueError):
            continue
        source_file = row.get("source_file", "")
        geo = _centroid_from_annotation_gpkg(source_file, row_index, src.region)
        if geo is None:
            skipped_no_ann += 1
            continue
        lon, lat = geo
        records.append({
            "region": src.region,
            "model_run": src.model_run,
            "source_bucket": src.source_bucket,
            "grid_id": grid_id,
            "pred_id": -1,  # GT polygon, not a prediction
            "gt_row_index": row_index,
            "gt_source_file": source_file,
            "label": label,
            "area_m2": float(row["area_m2"]),
            "confidence": float(row.get("confidence", 0) or 0),
            "source_tile": "",
            "centroid_lon": lon,
            "centroid_lat": lat,
            "results_path": str(src.results_path.relative_to(PROJECT_ROOT)),
            "source": "gt_audit",
        })

    result = pd.DataFrame(records)
    if skipped_no_ann:
        print(f"  WARN: GT audit skipped {skipped_no_ann} rows (annotation gpkg lookup failed)")
    print(f"  GT audit: {len(result)} chips "
          f"(PV: {(result['label'] == LABEL_PV).sum() if len(result) else 0}, "
          f"non-PV: {(result['label'] == LABEL_NONPV).sum() if len(result) else 0})")
    return result


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def region_stratified_whole_grid_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-source-bucket grid-level 80/20 split, concatenated.

    Each stratum ({cape_town:legacy_flat_batch003,
    cape_town:v3c_targeted_hn_aerial_2025, johannesburg:v4_aerial_2023}) gets
    its own GroupShuffleSplit so all buckets appear in both train and val.
    """
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []

    for stratum in sorted(df["source_bucket"].unique()):
        part = df[df["source_bucket"] == stratum]
        if part["grid_id"].nunique() < 2:
            train_parts.append(part)
            continue
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=seed
        )
        tr_idx, va_idx = next(splitter.split(part, groups=part["grid_id"]))
        train_parts.append(part.iloc[tr_idx])
        val_parts.append(part.iloc[va_idx])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[:0]
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[:0]

    train_grids = set(train_df["grid_id"].unique())
    val_grids = set(val_df["grid_id"].unique())
    overlap = train_grids & val_grids
    if overlap:
        # Possible if the same grid_id appears in two source_buckets (CT and
        # JHB both have e.g. G1189). Those are physically distinct grids;
        # we need to separate them by (source_bucket, grid_id) instead.
        train_keys = set(zip(train_df["source_bucket"], train_df["grid_id"]))
        val_keys = set(zip(val_df["source_bucket"], val_df["grid_id"]))
        key_overlap = train_keys & val_keys
        if key_overlap:
            raise AssertionError(f"Bucket+grid leakage: {key_overlap}")
        # Pure grid_id overlap across regions is expected; harmless.

    print(f"  Train: {len(train_df)} chips from {train_df['grid_id'].nunique()} "
          f"distinct grid IDs across {len(train_parts)} buckets")
    print(f"  Val:   {len(val_df)} chips from {val_df['grid_id'].nunique()} "
          f"distinct grid IDs across {len(val_parts)} buckets")
    for bucket in sorted(df["source_bucket"].unique()):
        tr_n = (train_df["source_bucket"] == bucket).sum()
        va_n = (val_df["source_bucket"] == bucket).sum()
        print(f"    {bucket}: train={tr_n} val={va_n}")
    return train_df, val_df


# ---------------------------------------------------------------------------
# Tile lookup + chip extraction
# ---------------------------------------------------------------------------

def _find_tile(
    lon: float,
    lat: float,
    grid_id: str,
    region: str,
    tiles_root_override: Path | None,
) -> Path | None:
    """Find tile GeoTIFF containing a lon/lat point.

    Resolves per-grid tiles via `core.grid_utils.resolve_tiles_dir(grid_id,
    region=)` unless `tiles_root_override` is provided.
    """
    if tiles_root_override is not None:
        grid_dir = tiles_root_override / grid_id
    else:
        try:
            grid_dir = resolve_tiles_dir(grid_id, region=region)
        except Exception:  # noqa: BLE001
            return None
    if not grid_dir.exists():
        return None
    # Support both chunked directory and single-file mosaic layouts
    if grid_dir.is_file():
        return grid_dir if _bounds_contains(grid_dir, lon, lat) else None
    for tif in grid_dir.glob(f"{grid_id}_*_*_geo.tif"):
        if _bounds_contains(tif, lon, lat):
            return tif
    return None


def _bounds_contains(tif: Path, lon: float, lat: float) -> bool:
    with rasterio.open(tif) as src:
        left, bottom, right, top = src.bounds
        return left <= lon <= right and bottom <= lat <= top


def extract_chip(
    lon: float, lat: float, grid_id: str, region: str,
    tiles_root_override: Path | None,
    tile_cache: dict,
    chip_size: int = CHIP_SIZE,
) -> np.ndarray | None:
    """Extract a chip centered on (lon, lat). Returns HWC uint8 array or None."""
    tile_path = _find_tile(lon, lat, grid_id, region, tiles_root_override)
    if tile_path is None:
        return None

    tile_key = str(tile_path)
    if tile_key not in tile_cache:
        tile_cache[tile_key] = rasterio.open(tile_path)

    src = tile_cache[tile_key]
    py, px = src.index(lon, lat)

    x0 = max(0, int(px - chip_size // 2))
    y0 = max(0, int(py - chip_size // 2))
    x0 = min(x0, max(0, src.width - chip_size))
    y0 = min(y0, max(0, src.height - chip_size))

    w = min(chip_size, src.width - x0)
    h = min(chip_size, src.height - y0)

    if w < chip_size * 0.5 or h < chip_size * 0.5:
        return None

    window = Window(x0, y0, w, h)
    data = src.read(window=window)

    if w < chip_size or h < chip_size:
        padded = np.zeros((data.shape[0], chip_size, chip_size), dtype=data.dtype)
        padded[:, :h, :w] = data
        data = padded

    if np.all(data >= 245):
        return None

    img = data[:3].transpose(1, 2, 0)
    return img


def extract_and_save_chips(
    df: pd.DataFrame,
    output_dir: Path,
    split: str,
    tiles_root_override: Path | None,
    img_size: int = IMG_SIZE,
) -> tuple[int, int]:
    """Extract chips for a dataframe split, save as PNG.

    Returns (saved, skipped)."""
    saved = 0
    skipped = 0
    tile_cache: dict[str, rasterio.DatasetReader] = {}

    for label in (LABEL_PV, LABEL_NONPV):
        (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    try:
        for _, row in df.iterrows():
            chip = extract_chip(
                row["centroid_lon"], row["centroid_lat"],
                row["grid_id"], row["region"],
                tiles_root_override, tile_cache,
            )
            if chip is None:
                skipped += 1
                continue
            resized = cv2.resize(chip, (img_size, img_size), interpolation=cv2.INTER_AREA)
            source_tag = row.get("source", "reviewed")
            pid = int(row["pred_id"])
            if pid < 0:
                # Auxiliary sources (gt_audit) have no pred_id; disambiguate by
                # gt_row_index so multiple chips per grid don't collide.
                row_idx = int(row.get("gt_row_index", 0))
                fname = f"{row['region']}_{row['grid_id']}_aux{row_idx}_{source_tag}.png"
            else:
                fname = f"{row['region']}_{row['grid_id']}_pred{pid}_{source_tag}.png"
            out_path = output_dir / split / row["label"] / fname
            cv2.imwrite(str(out_path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            saved += 1
    finally:
        for handle in tile_cache.values():
            handle.close()

    if skipped > 0:
        print(f"  {split}: saved {saved}, skipped {skipped} (no tile / blank)")
    else:
        print(f"  {split}: saved {saved}")
    return saved, skipped


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _count_breakdown(df: pd.DataFrame) -> dict:
    """Per-bucket / per-source / per-class counts for a split."""
    if df.empty:
        return {"total": 0}
    per_bucket = (
        df.groupby(["source_bucket", "label"]).size().unstack(fill_value=0).to_dict("index")
    )
    per_source = (
        df.groupby(["source", "label"]).size().unstack(fill_value=0).to_dict("index")
    )
    per_region = (
        df.groupby(["region", "label"]).size().unstack(fill_value=0).to_dict("index")
    )
    return {
        "total": int(len(df)),
        "pv": int((df["label"] == LABEL_PV).sum()),
        "non_pv": int((df["label"] == LABEL_NONPV).sum()),
        "per_region": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in per_region.items()},
        "per_source_bucket": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in per_bucket.items()},
        "per_source": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in per_source.items()},
        "grids": sorted(df["grid_id"].unique().tolist()),
    }


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    reviewed_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_saved: int,
    val_saved: int,
    train_skipped: int,
    val_skipped: int,
    taxonomy_added: int,
    gt_audit_added: int,
) -> Path:
    manifest = {
        "description": "PV vs non-PV binary classification dataset (registry-driven)",
        "built_by": "scripts/classifier/build_cls_dataset.py",
        "seed": args.seed,
        "area_cutoff_m2": args.area_cutoff,
        "val_fraction": args.val_fraction,
        "img_size": args.img_size,
        "chip_extraction_size": CHIP_SIZE,
        "aug_profile": args.aug_profile,
        "include_taxonomy": args.include_taxonomy,
        "include_gt_audit": args.include_gt_audit,
        "include_deprecated_model_runs": args.include_deprecated,
        "include_legacy_flat_bucket": args.include_legacy_flat,
        "label_mapping": {
            "pv": "correct | edit (reviewed) | correct_detection (taxonomy) | pv (gt_audit)",
            "non_pv": "delete (reviewed) | thermal/skylight/shadow/pergola (taxonomy) | heater_or_non_pv (gt_audit)",
        },
        "counts": {
            "reviewed_pool": _count_breakdown(reviewed_df),
            "train_selected": _count_breakdown(train_df),
            "val_selected": _count_breakdown(val_df),
        },
        "extraction_stats": {
            "train_saved": int(train_saved),
            "train_skipped": int(train_skipped),
            "val_saved": int(val_saved),
            "val_skipped": int(val_skipped),
            "taxonomy_added_to_train": int(taxonomy_added),
            "gt_audit_added_to_train": int(gt_audit_added),
        },
    }
    path = output_dir / "dataset_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "cls_pv_thermal_v1")
    parser.add_argument("--area-cutoff", type=float, default=30.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument(
        "--aug-profile",
        default="current",
        choices=["current", "flip_only"],
        help="Augmentation profile to record in manifest (no build-time effect). See exp_cls_augmentation_ablation.md.",
    )
    parser.add_argument(
        "--include-taxonomy", action="store_true",
        help="Add taxonomy_labeled.csv chips (77 thermal + others) to train set",
    )
    parser.add_argument(
        "--taxonomy-csv", type=Path,
        default=PROJECT_ROOT / "results" / "analysis" / "small_fp" / "taxonomy_run" / "small_fp_taxonomy_labeled.csv",
    )
    parser.add_argument(
        "--include-gt-audit", action="store_true",
        help="Add gt_heater_audit phase1 labels (585 pv + 80 heater) to train set",
    )
    parser.add_argument(
        "--gt-audit-csv", type=Path,
        default=PROJECT_ROOT / "results" / "analysis" / "gt_heater_audit" / "heater_20260405_2141" / "audit_labels_phase1.csv",
    )
    parser.add_argument(
        "--include-deprecated", action="store_true",
        help="Include model_runs marked deprecated in regions.yaml (e.g. v3c_geid_2024_02)",
    )
    parser.add_argument(
        "--include-legacy-flat", action=argparse.BooleanOptionalAction, default=True,
        help="Include legacy flat results/G*/review/ bucket (pre-PR3 CT batch 003)",
    )
    parser.add_argument(
        "--tiles-root", type=Path, default=None,
        help="Optional: override tile root (else per-grid resolution via region registry)",
    )
    args = parser.parse_args()

    # --- Step 1: discover reviewed sources ---
    print("[1/5] Discovering reviewed grids via region_registry...")
    sources = discover_grid_sources(
        include_deprecated=args.include_deprecated,
        include_legacy_flat=args.include_legacy_flat,
    )
    by_grid: dict[str, GridSource] = {}
    for s in sources:
        by_grid.setdefault(s.grid_id, s)
    print(f"  Found {len(sources)} (region, model_run, grid) tuples "
          f"across {len({s.source_bucket for s in sources})} source buckets")

    # --- Step 2: load reviewed predictions ---
    print(f"\n[2/5] Loading reviewed predictions (area < {args.area_cutoff:.0f} m²)...")
    reviewed_df = load_reviewed_predictions(sources, area_cutoff=args.area_cutoff)
    if reviewed_df.empty:
        print("ERROR: no reviewed predictions found.")
        return 1

    # --- Step 3: region-stratified whole-grid split ---
    print(f"\n[3/5] Region-stratified whole-grid split "
          f"(val_fraction={args.val_fraction}, seed={args.seed})...")
    train_df, val_df = region_stratified_whole_grid_split(
        reviewed_df, args.val_fraction, args.seed
    )

    # --- Step 4: auxiliary label sources (train-only) ---
    taxonomy_added = 0
    gt_audit_added = 0
    train_grid_keys = set(zip(train_df["source_bucket"], train_df["grid_id"]))

    if args.include_taxonomy and args.taxonomy_csv.exists():
        print(f"\n  Loading taxonomy chips from {args.taxonomy_csv.name}...")
        tax_df = load_taxonomy_chips(args.taxonomy_csv, args.area_cutoff, by_grid)
        if len(tax_df):
            # Restrict to train buckets + dedup vs reviewed
            tax_df = tax_df[
                tax_df.apply(lambda r: (r["source_bucket"], r["grid_id"]) in train_grid_keys, axis=1)
            ]
            existing = set(zip(train_df["source_bucket"], train_df["grid_id"], train_df["pred_id"]))
            tax_df = tax_df[
                ~tax_df.apply(lambda r: (r["source_bucket"], r["grid_id"], r["pred_id"]) in existing, axis=1)
            ]
            taxonomy_added = len(tax_df)
            if taxonomy_added:
                train_df = pd.concat([train_df, tax_df], ignore_index=True)
                print(f"  Added {taxonomy_added} taxonomy chips to train set")

    if args.include_gt_audit and args.gt_audit_csv.exists():
        print(f"\n  Loading GT heater audit from {args.gt_audit_csv.name}...")
        audit_df = load_gt_heater_audit(args.gt_audit_csv, args.area_cutoff, by_grid)
        if len(audit_df):
            # GT audit rows live at GT polygon centroids, not prediction
            # centroids — they cannot collide with reviewed predictions under
            # (bucket, grid, pred_id). Restrict to train buckets only.
            audit_df = audit_df[
                audit_df.apply(lambda r: (r["source_bucket"], r["grid_id"]) in train_grid_keys, axis=1)
            ]
            gt_audit_added = len(audit_df)
            if gt_audit_added:
                train_df = pd.concat([train_df, audit_df], ignore_index=True)
                print(f"  Added {gt_audit_added} GT audit chips to train set")

    # --- Step 5: extract and save chips ---
    print(f"\n[4/5] Extracting chips (tiles_root_override={args.tiles_root})...")
    for split in ("train", "val"):
        for label in (LABEL_PV, LABEL_NONPV):
            label_dir = args.output_dir / split / label
            if label_dir.exists():
                shutil.rmtree(label_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_saved, train_skipped = extract_and_save_chips(
        train_df, args.output_dir, "train", args.tiles_root, args.img_size
    )
    val_saved, val_skipped = extract_and_save_chips(
        val_df, args.output_dir, "val", args.tiles_root, args.img_size
    )

    # --- Step 6: manifest ---
    print("\n[5/5] Writing manifest...")
    manifest_path = write_manifest(
        args.output_dir, args, reviewed_df, train_df, val_df,
        train_saved, val_saved, train_skipped, val_skipped,
        taxonomy_added, gt_audit_added,
    )

    print("\n=== Dataset Summary ===")
    print(f"  Train: {train_saved} saved")
    print(f"  Val:   {val_saved} saved")
    print(f"  Manifest: {manifest_path}")
    print(f"  Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
