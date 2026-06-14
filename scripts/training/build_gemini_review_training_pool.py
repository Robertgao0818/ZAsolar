#!/usr/bin/env python3
"""Build a weak training pool from Gemini visual review outputs.

Gemini review is used here as an automation filter, not as gold annotation.
High-confidence PV-present decisions become untrusted weak positives with
``label_source=gemini_reviewed_prediction``. High-confidence PV-absent decisions
become hard-negative candidates after a conservative GT-overlap audit.

Expected candidate manifest columns:
  - grid_id
  - pred_id
  - region_key is strongly recommended for cross-city/national grids where
    grid_id follows the unified JNB grid numbering and no longer identifies a
    city by itself.
  - image_path, candidate_id, predictions_path, region, imagery_layer,
    model_run, results_root are optional but used when present.  ``region`` is
    accepted as an alias for ``region_key``.

Gemini JSONL can be either the single-image output from
``solar_backdating/scripts/validation/gemini_solar_image_review.py`` or a
flattened JSONL with top-level ``pv_present``, ``confidence`` and
``quality_flag`` fields. Rows are joined by candidate_id, image_path, or
(grid_id, pred_id), in that order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core import region_registry  # noqa: E402
from core.annotation_loader import resolve_gt_path  # noqa: E402
from core.grid_utils import get_results_root, resolve_tiles_dir  # noqa: E402


LABEL_SOURCE = "gemini_reviewed_prediction"
TP_OVERLAP_THRESHOLD = 0.05
REGION_COLUMN_CANDIDATES = ("region_key", "region", "source_region", "city")


def _normalize_region(value: Any) -> str:
    text = _norm_text(value)
    if text.lower() == "jnb":
        text = "johannesburg"
    return region_registry.normalize_region_key(text) or ""


@dataclass(frozen=True)
class GeminiDecision:
    pv_present: bool | None
    confidence: float | None
    quality_flag: str
    evidence: str
    notes: str
    decision_source: str
    raw: dict[str, Any]


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _norm_path(value: Any) -> str:
    text = _norm_text(value)
    if not text:
        return ""
    return str(Path(text).expanduser())


def _as_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "pv", "present"}:
        return True
    if text in {"false", "0", "no", "non_pv", "absent"}:
        return False
    return None


def _as_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_region_key(row: pd.Series | dict[str, Any], grid_id: str) -> str:
    for col in REGION_COLUMN_CANDIDATES:
        if col in row:
            region = _normalize_region(row.get(col))
            if region:
                return region

    # Last-resort compatibility path for older manifests.  Unified JNB/national
    # grid IDs should carry region_key explicitly because a bare grid_id is not
    # a reliable city key.
    #
    # ADR-0002 / CPT regrid: after CT retired its G\d{4} namespace, a bare
    # overlap G-ID (G1189 etc.) has NO active owner and lookup_regions() returns
    # BOTH retired claimants (cape_town first, by regions.yaml order). Treat that
    # tie the same way lookup_region() (singular) does — take the first hit —
    # instead of dropping to "" and losing the row's region key. CT census is the
    # only flow that feeds bare G-IDs here; JHB-historical flows carry region_key
    # in the manifest (handled by the loop above) or pass --region.
    hits = region_registry.lookup_regions(grid_id)
    if hits:
        return hits[0]
    return ""


def _flatten_gemini_record(record: dict[str, Any]) -> GeminiDecision:
    parsed = record.get("parsed")
    if not isinstance(parsed, dict):
        parsed = record

    return GeminiDecision(
        pv_present=_as_bool_or_none(parsed.get("pv_present")),
        confidence=_as_float_or_none(parsed.get("confidence")),
        quality_flag=_norm_text(parsed.get("quality_flag") or record.get("quality_flag")).lower(),
        evidence=_norm_text(parsed.get("evidence") or record.get("evidence")),
        notes=_norm_text(parsed.get("notes") or record.get("notes")),
        decision_source=_norm_text(
            parsed.get("decision_source")
            or record.get("decision_source")
            or ("gemini_cli" if "parsed" in record else "gemini_jsonl")
        ),
        raw=record,
    )


def load_gemini_jsonl(paths: list[Path]) -> tuple[dict[str, GeminiDecision], dict[str, GeminiDecision], dict[tuple[str, int], GeminiDecision]]:
    by_candidate: dict[str, GeminiDecision] = {}
    by_image: dict[str, GeminiDecision] = {}
    by_grid_pred: dict[tuple[str, int], GeminiDecision] = {}

    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
                if not isinstance(record, dict):
                    continue
                decision = _flatten_gemini_record(record)

                candidate_id = _norm_text(record.get("candidate_id") or record.get("anchor_id"))
                if candidate_id:
                    by_candidate[candidate_id] = decision

                image_path = _norm_path(record.get("image_path") or record.get("chip_path"))
                if image_path:
                    by_image[image_path] = decision

                grid_id = _norm_text(record.get("grid_id")).upper()
                pred_id = record.get("pred_id")
                if grid_id and pred_id not in (None, ""):
                    try:
                        by_grid_pred[(grid_id, int(pred_id))] = decision
                    except (TypeError, ValueError):
                        pass

    return by_candidate, by_image, by_grid_pred


def find_decision(
    row: pd.Series,
    *,
    by_candidate: dict[str, GeminiDecision],
    by_image: dict[str, GeminiDecision],
    by_grid_pred: dict[tuple[str, int], GeminiDecision],
) -> GeminiDecision | None:
    candidate_id = _norm_text(row.get("candidate_id") or row.get("anchor_id"))
    if candidate_id and candidate_id in by_candidate:
        return by_candidate[candidate_id]

    image_path = _norm_path(row.get("image_path") or row.get("chip_path"))
    if image_path and image_path in by_image:
        return by_image[image_path]

    grid_id = _norm_text(row.get("grid_id")).upper()
    pred_id = row.get("pred_id")
    if grid_id and pred_id not in (None, ""):
        try:
            return by_grid_pred.get((grid_id, int(pred_id)))
        except (TypeError, ValueError):
            return None
    return None


def classify_decision(
    decision: GeminiDecision | None,
    *,
    positive_threshold: float,
    negative_threshold: float,
) -> str:
    if decision is None:
        return "missing_gemini_decision"
    if decision.quality_flag != "usable":
        return f"skip_quality_{decision.quality_flag or 'unknown'}"
    if decision.confidence is None:
        return "skip_missing_confidence"
    if decision.pv_present is True:
        return "weak_positive" if decision.confidence >= positive_threshold else "skip_low_conf_positive"
    if decision.pv_present is False:
        return "hard_negative_candidate" if decision.confidence >= negative_threshold else "skip_low_conf_negative"
    return "skip_null_presence"


def resolve_predictions_path(row: pd.Series, results_root: Path | None) -> Path:
    explicit = _norm_path(row.get("predictions_path"))
    if explicit:
        return Path(explicit)

    grid_id = _norm_text(row.get("grid_id")).upper()
    if not grid_id:
        raise ValueError("candidate row missing grid_id")

    row_results_root = _norm_path(row.get("results_root"))
    if row_results_root:
        return Path(row_results_root) / grid_id / "predictions_metric.gpkg"

    if results_root is not None:
        return results_root / grid_id / "predictions_metric.gpkg"

    region = _row_region_key(row, grid_id)
    model_run = _norm_text(row.get("model_run"))
    return get_results_root(region=region or None, model_run=model_run or None) / grid_id / "predictions_metric.gpkg"


def load_prediction_geometry(
    row: pd.Series,
    *,
    results_root: Path | None,
    cache: dict[Path, gpd.GeoDataFrame],
) -> tuple[Any, str, Path]:
    path = resolve_predictions_path(row, results_root)
    if path not in cache:
        if not path.exists():
            raise FileNotFoundError(f"predictions_metric.gpkg not found: {path}")
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            # Most legacy prediction exports are already lon/lat; make the
            # assumption explicit for downstream GeoPackage consumers.
            gdf = gdf.set_crs(epsg=4326)
        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        cache[path] = gdf

    pred_id = int(row["pred_id"])
    gdf = cache[path]
    if pred_id < 0 or pred_id >= len(gdf):
        raise IndexError(f"pred_id {pred_id} out of range for {path} ({len(gdf)} rows)")
    pred_row = gdf.iloc[pred_id]
    return pred_row.geometry, str(gdf.crs), path


def load_gt_for_grid(grid_id: str, region: str | None, cache: dict[tuple[str, str], gpd.GeoDataFrame | None]) -> gpd.GeoDataFrame | None:
    key = (region or "", grid_id)
    if key in cache:
        return cache[key]
    try:
        gt_path = resolve_gt_path(grid_id, region=region)
    except Exception:
        cache[key] = None
        return None
    if not gt_path.exists():
        cache[key] = None
        return None
    gt = gpd.read_file(gt_path)
    if gt.empty:
        cache[key] = None
        return None
    if gt.crs is None:
        gt = gt.set_crs(epsg=4326)
    if gt.crs.to_epsg() != 4326:
        gt = gt.to_crs(epsg=4326)
    cache[key] = gt
    return gt


def gt_overlap_status(
    geom,
    *,
    grid_id: str,
    region: str | None,
    gt_cache: dict[tuple[str, str], gpd.GeoDataFrame | None],
    threshold: float,
) -> tuple[str, float]:
    gt = load_gt_for_grid(grid_id, region, gt_cache)
    if gt is None or gt.empty:
        return "no_gt_available", 0.0

    max_frac = 0.0
    candidates = gt[gt.intersects(geom)]
    for gt_geom in candidates.geometry:
        inter = geom.intersection(gt_geom)
        if inter.is_empty or geom.area == 0:
            continue
        max_frac = max(max_frac, float(inter.area / geom.area))
    if max_frac >= threshold:
        return "drop_gt_overlap", max_frac
    return "no_gt_overlap", max_frac


def row_output_base(
    row: pd.Series,
    decision: GeminiDecision | None,
    triage_label: str,
    *,
    geometry_source_path: Path | None = None,
    audit_status: str = "",
    gt_overlap_frac: float = 0.0,
) -> dict[str, Any]:
    out = {k: _norm_text(v) for k, v in row.to_dict().items() if k != "geometry"}
    out.update({
        "triage_label": triage_label,
        "gemini_pv_present": decision.pv_present if decision else "",
        "gemini_confidence": decision.confidence if decision else "",
        "gemini_quality_flag": decision.quality_flag if decision else "",
        "gemini_decision_source": decision.decision_source if decision else "",
        "gemini_evidence": decision.evidence if decision else "",
        "gemini_notes": decision.notes if decision else "",
        "geometry_source_path": str(geometry_source_path) if geometry_source_path else "",
        "hn_audit_status": audit_status,
        "gt_overlap_frac": round(gt_overlap_frac, 6),
    })
    return out


def write_gpkg(path: Path, rows: list[dict[str, Any]], crs: str = "EPSG:4326") -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    gdf.to_file(path, driver="GPKG")


def write_review_compatible(root: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    for grid_id, group in pd.DataFrame(rows).groupby("grid_id"):
        grid_rows = [r for r in rows if r.get("grid_id") == grid_id]
        out_dir = root / str(grid_id) / "review"
        out_dir.mkdir(parents=True, exist_ok=True)
        gdf = gpd.GeoDataFrame(grid_rows, geometry="geometry", crs="EPSG:4326")
        gdf.to_file(out_dir / f"{grid_id}_reviewed.gpkg", driver="GPKG")


def _tiles_for(grid_id: str, region: str | None, imagery_layer: str | None) -> list[Path]:
    tiles_dir = resolve_tiles_dir(grid_id, region=region, imagery_layer=imagery_layer)
    if tiles_dir.is_file():
        return [tiles_dir]
    tiles = sorted(tiles_dir.glob(f"{grid_id}_*_*_geo.tif"))
    if not tiles:
        tiles = sorted(p for p in tiles_dir.glob(f"{grid_id}_*.tif") if "mosaic" not in p.stem)
    return tiles


def _hardlink_or_copy_tree(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.exists():
        return
    for src in src_dir.iterdir():
        dst = dst_dir / src.name
        if dst.exists():
            continue
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def build_weak_positive_coco(
    *,
    weak_positive_rows: list[dict[str, Any]],
    base_coco: Path,
    output_coco: Path,
    chip_size: int,
    overlap: float,
    neg_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """Append Gemini weak positives as train chips to an existing COCO dataset."""
    import rasterio
    from export_coco_dataset import (
        assign_annotations_to_tiles,
        balance_chips,
        scan_chips_from_tile,
        write_selected_chips,
    )

    if not base_coco.exists():
        raise FileNotFoundError(f"--base-coco not found: {base_coco}")
    if output_coco.resolve() == base_coco.resolve():
        raise ValueError("--output-coco must differ from --base-coco")

    output_coco.mkdir(parents=True, exist_ok=True)
    _hardlink_or_copy_tree(base_coco / "train", output_coco / "train")
    _hardlink_or_copy_tree(base_coco / "val", output_coco / "val")

    with (base_coco / "train.json").open("r", encoding="utf-8") as fh:
        base_train = json.load(fh)
    with (base_coco / "val.json").open("r", encoding="utf-8") as fh:
        base_val = json.load(fh)

    img_id_counter = max((int(img["id"]) for img in base_train.get("images", [])), default=0) + 1
    ann_id_counter = max((int(ann["id"]) for ann in base_train.get("annotations", [])), default=0) + 1
    addon_images: list[dict[str, Any]] = []
    addon_annots: list[dict[str, Any]] = []
    addon_prov: list[dict[str, Any]] = []

    if weak_positive_rows:
        weak_gdf = gpd.GeoDataFrame(weak_positive_rows, geometry="geometry", crs="EPSG:4326")
        group_cols = ["region_key", "imagery_layer", "grid_id"]
        for col in group_cols:
            if col not in weak_gdf.columns:
                weak_gdf[col] = ""

        for (region, imagery_layer, grid_id), group in weak_gdf.groupby(group_cols, dropna=False):
            region = _normalize_region(region)
            imagery_layer = _norm_text(imagery_layer) or None
            grid_id = _norm_text(grid_id).upper()
            if not region:
                print(f"[COCO-WARN] {grid_id}: missing region_key, skipping weak positives")
                continue
            tiles = _tiles_for(grid_id, region, imagery_layer)
            if not tiles:
                print(f"[COCO-WARN] {grid_id}: no tiles found, skipping weak positives")
                continue
            with rasterio.open(tiles[0]) as src:
                tile_crs = src.crs
            annots = group.to_crs(tile_crs).reset_index(drop=True)
            tile_to_annots = assign_annotations_to_tiles(annots, tiles)

            tile_map = {t.stem: t for t in tiles}
            for stem, annot_indices in tile_to_annots.items():
                if not annot_indices:
                    continue
                tile_path = tile_map[stem]
                imgs, anns, prov = scan_chips_from_tile(
                    tile_path=tile_path,
                    annotations=annots,
                    annot_indices=annot_indices,
                    chip_size=chip_size,
                    overlap=overlap,
                    split_name="train",
                    image_id_start=img_id_counter,
                    annot_id_start=ann_id_counter,
                )
                for img in imgs:
                    raw_name = Path(img["file_name"]).name
                    img["file_name"] = f"train/gemini_weak_{grid_id}_{raw_name}"
                    img["region"] = region or ""
                    img["grid_id"] = grid_id
                    img["imagery_layer"] = imagery_layer or ""
                    img["weak_source"] = LABEL_SOURCE
                for row in prov:
                    row["chip_file"] = f"gemini_weak_{grid_id}_{row['chip_file']}"
                    row["region"] = region or ""
                    row["grid_id"] = grid_id
                    row["imagery_layer"] = imagery_layer or ""
                    row["source_type"] = LABEL_SOURCE
                img_id_counter += len(imgs)
                ann_id_counter += len(anns)
                addon_images.extend(imgs)
                addon_annots.extend(anns)
                addon_prov.extend(prov)

    if neg_ratio >= 0:
        addon_images, addon_annots, addon_prov = balance_chips(
            addon_images,
            addon_annots,
            addon_prov,
            seed=seed,
            neg_ratio=neg_ratio,
        )

    write_selected_chips(addon_images, output_coco, chip_size)

    merged_train = {
        **base_train,
        "info": {
            **base_train.get("info", {}),
            "description": (
                base_train.get("info", {}).get("description", "COCO train")
                + " + Gemini weak positives"
            ),
        },
        "images": base_train.get("images", []) + addon_images,
        "annotations": base_train.get("annotations", []) + addon_annots,
    }
    (output_coco / "train.json").write_text(json.dumps(merged_train) + "\n", encoding="utf-8")
    (output_coco / "val.json").write_text(json.dumps(base_val) + "\n", encoding="utf-8")

    prov_path = output_coco / "gemini_weak_positive_provenance.csv"
    if addon_prov:
        with prov_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(addon_prov[0].keys()))
            writer.writeheader()
            writer.writerows(addon_prov)

    return {
        "output_coco": str(output_coco),
        "addon_images": len(addon_images),
        "addon_annotations": len(addon_annots),
        "provenance": str(prov_path) if addon_prov else "",
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    candidates = pd.read_csv(args.candidate_manifest)
    required = {"grid_id", "pred_id"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate manifest missing required columns: {sorted(missing)}")

    by_candidate, by_image, by_grid_pred = load_gemini_jsonl(args.gemini_jsonl)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_cache: dict[Path, gpd.GeoDataFrame] = {}
    gt_cache: dict[tuple[str, str], gpd.GeoDataFrame | None] = {}
    triage_rows: list[dict[str, Any]] = []
    weak_positive_rows: list[dict[str, Any]] = []
    hn_rows: list[dict[str, Any]] = []
    dropped_hn_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    results_root = args.results_root.expanduser() if args.results_root else None

    for _, row in candidates.iterrows():
        grid_id = _norm_text(row.get("grid_id")).upper()
        row = row.copy()
        row["grid_id"] = grid_id
        region = _row_region_key(row, grid_id)
        row["region_key"] = region
        # Keep legacy downstream consumers working while making region_key the
        # explicit authority for unified cross-city grid IDs.
        row["region"] = region
        decision = find_decision(
            row,
            by_candidate=by_candidate,
            by_image=by_image,
            by_grid_pred=by_grid_pred,
        )
        triage_label = classify_decision(
            decision,
            positive_threshold=args.positive_threshold,
            negative_threshold=args.negative_threshold,
        )
        geometry = None
        geometry_source_path: Path | None = None
        audit_status = ""
        overlap_frac = 0.0

        if triage_label in {"weak_positive", "hard_negative_candidate"}:
            geometry, _, geometry_source_path = load_prediction_geometry(
                row, results_root=results_root, cache=pred_cache
            )

        if triage_label == "hard_negative_candidate":
            audit_status, overlap_frac = gt_overlap_status(
                geometry,
                grid_id=grid_id,
                region=region,
                gt_cache=gt_cache,
                threshold=args.gt_overlap_threshold,
            )
            if audit_status == "drop_gt_overlap":
                triage_label = "drop_hn_gt_overlap"
            elif audit_status == "no_gt_available" and not args.include_unaudited_hn:
                triage_label = "skip_hn_no_gt_available"

        base = row_output_base(
            row,
            decision,
            triage_label,
            geometry_source_path=geometry_source_path,
            audit_status=audit_status,
            gt_overlap_frac=overlap_frac,
        )
        triage_rows.append(base)

        if geometry is None:
            continue

        if triage_label == "weak_positive":
            out = {
                **base,
                "geometry": geometry,
                "review_status": "correct",
                "source": LABEL_SOURCE,
                "label_source": LABEL_SOURCE,
                "quality_tier": "T2",
                "semantic_confidence": "A2",
                "mask_trusted": False,
            }
            weak_positive_rows.append(out)
            review_rows.append(out)
        elif triage_label == "hard_negative_candidate":
            out = {
                **base,
                "geometry": geometry,
                "review_status": "delete",
                "source": "gemini_hard_negative",
                "label_source": "",
                "quality_tier": "",
                "semantic_confidence": "",
                "mask_trusted": "",
            }
            hn_rows.append(out)
            review_rows.append(out)
        elif triage_label == "drop_hn_gt_overlap":
            dropped_hn_rows.append({**base, "geometry": geometry})

    triage_path = args.output_dir / "gemini_review_triage_manifest.csv"
    with triage_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = sorted({k for row in triage_rows for k in row.keys()})
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(triage_rows)

    weak_path = args.output_dir / "gemini_weak_positives.gpkg"
    hn_path = args.output_dir / "gemini_hard_negative_candidates.gpkg"
    dropped_path = args.output_dir / "gemini_hn_dropped_gt_overlap.gpkg"
    write_gpkg(weak_path, weak_positive_rows)
    write_gpkg(hn_path, hn_rows)
    write_gpkg(dropped_path, dropped_hn_rows)

    review_root = args.output_dir / "review_compatible"
    if args.write_review_compatible:
        write_review_compatible(review_root, review_rows)

    coco_summary: dict[str, Any] | None = None
    if args.base_coco or args.output_coco:
        if not args.base_coco or not args.output_coco:
            raise ValueError("--base-coco and --output-coco must be provided together")
        coco_summary = build_weak_positive_coco(
            weak_positive_rows=weak_positive_rows,
            base_coco=args.base_coco.expanduser(),
            output_coco=args.output_coco.expanduser(),
            chip_size=args.chip_size,
            overlap=args.overlap,
            neg_ratio=args.coco_neg_ratio,
            seed=args.seed,
        )

    counts = pd.Series([r["triage_label"] for r in triage_rows]).value_counts().to_dict()
    summary = {
        "candidate_manifest": str(args.candidate_manifest),
        "gemini_jsonl": [str(p) for p in args.gemini_jsonl],
        "positive_threshold": args.positive_threshold,
        "negative_threshold": args.negative_threshold,
        "gt_overlap_threshold": args.gt_overlap_threshold,
        "include_unaudited_hn": args.include_unaudited_hn,
        "counts": {str(k): int(v) for k, v in counts.items()},
        "outputs": {
            "triage_manifest": str(triage_path),
            "weak_positives_gpkg": str(weak_path) if weak_positive_rows else "",
            "hard_negative_candidates_gpkg": str(hn_path) if hn_rows else "",
            "dropped_hn_gt_overlap_gpkg": str(dropped_path) if dropped_hn_rows else "",
            "review_compatible_root": str(review_root) if args.write_review_compatible else "",
            "merged_coco": coco_summary or {},
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--gemini-jsonl",
        type=Path,
        nargs="+",
        required=True,
        help="One or more Gemini JSONL outputs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Fallback root with <grid>/predictions_metric.gpkg.",
    )
    parser.add_argument("--positive-threshold", type=float, default=0.90)
    parser.add_argument("--negative-threshold", type=float, default=0.90)
    parser.add_argument("--gt-overlap-threshold", type=float, default=TP_OVERLAP_THRESHOLD)
    parser.add_argument("--chip-size", type=int, default=400)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--coco-neg-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-coco", type=Path, default=None)
    parser.add_argument(
        "--output-coco",
        type=Path,
        default=None,
        help="If set with --base-coco, append Gemini weak-positive train chips to this merged COCO dir.",
    )
    parser.add_argument(
        "--include-unaudited-hn",
        action="store_true",
        help="Keep Gemini-negative candidates even when no GT source is available for overlap audit.",
    )
    parser.add_argument(
        "--write-review-compatible",
        action="store_true",
        help="Also write <output>/review_compatible/<grid>/review/<grid>_reviewed.gpkg.",
    )
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
