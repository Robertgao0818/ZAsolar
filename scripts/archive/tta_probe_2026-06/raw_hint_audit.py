#!/usr/bin/env python3
"""Audit whether clean-GT polygons had raw detector hints.

For each grid/source GT polygon, this script compares:

- raw detector boxes from ``raw_detections.pkl``
- raw detector masks vectorized at the artifact mask threshold
- old pixel-or finalized polygons
- per-detection finalized polygons
- mutual-IoU clusters built from per-detection polygons
- optional SAM maskbox reference polygons

The goal is to separate true no-hint misses from cases where the detector had
some proposal support but finalization merged, split, or filtered it poorly.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from shapely.strtree import STRtree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.inference.raw_artifact import read_artifact  # noqa: E402
from core.postproc import vectorize_chip_mask  # noqa: E402


DEFAULT_GRIDS = ["G0816", "G0817", "G0925"]


@dataclass(frozen=True)
class GeometrySet:
    name: str
    geoms: list[BaseGeometry]
    scores: np.ndarray


def _safe_iou(a: BaseGeometry, b: BaseGeometry) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    denom = a.area + b.area - inter
    return float(inter / denom) if denom > 0 else 0.0


def _unary_safe(geoms: list[BaseGeometry]) -> BaseGeometry | None:
    if not geoms:
        return None
    if len(geoms) == 1:
        return geoms[0]
    return unary_union(geoms)


def _reproject_geoms(
    geoms: list[BaseGeometry],
    src_crs: str,
    dst_crs,
) -> list[BaseGeometry]:
    if not geoms or str(src_crs) == str(dst_crs):
        return geoms
    gs = gpd.GeoSeries(geoms, crs=src_crs).to_crs(dst_crs)
    return list(gs.values)


def _load_gpkg_as_set(path: Path, name: str, dst_crs) -> GeometrySet | None:
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    if gdf.empty:
        return GeometrySet(name=name, geoms=[], scores=np.zeros(0, dtype=float))
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS")
    if str(gdf.crs) != str(dst_crs):
        gdf = gdf.to_crs(dst_crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    score_col = None
    for candidate in ("confidence", "score", "sam_score"):
        if candidate in gdf.columns:
            score_col = candidate
            break
    scores = (
        pd.to_numeric(gdf[score_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        if score_col is not None
        else np.ones(len(gdf), dtype=float)
    )
    return GeometrySet(name=name, geoms=list(gdf.geometry.values), scores=scores)


def _choose_per_detection_path(grid_root: Path) -> Path | None:
    for rel in (
        "per_detection/predictions_metric.gpkg",
        "instance_only/predictions_metric.gpkg",
        "new_instance_adaptive_hysteresis/predictions_metric.gpkg",
        "instance_pre_nms/predictions_metric.gpkg",
    ):
        candidate = grid_root / rel
        if candidate.exists():
            return candidate
    return None


def _raw_sets_from_artifact(
    raw_path: Path,
    dst_crs,
    *,
    mask_threshold: float | None = None,
) -> tuple[GeometrySet, GeometrySet, float]:
    artifact = read_artifact(raw_path)
    threshold = (
        float(artifact.mask_threshold_used)
        if mask_threshold is None
        else float(mask_threshold)
    )

    box_geoms: list[BaseGeometry] = []
    box_scores: list[float] = []
    mask_geoms: list[BaseGeometry] = []
    mask_scores: list[float] = []
    raw_crs: str | None = None

    for chip in artifact.chips:
        if raw_crs is None:
            raw_crs = chip.source_crs
        elif raw_crs != chip.source_crs:
            raise ValueError(
                f"mixed raw CRS in {raw_path}: {raw_crs!r} and {chip.source_crs!r}"
            )
        win_tr = Affine(*chip.window_transform)
        src_tr = Affine(*chip.source_transform)
        for det in chip.detections:
            x1, y1, x2, y2 = det.box_source_xyxy
            corners = [
                src_tr * (x1, y1),
                src_tr * (x2, y1),
                src_tr * (x2, y2),
                src_tr * (x1, y2),
            ]
            bx = Polygon(corners)
            if not bx.is_empty:
                box_geoms.append(bx)
                box_scores.append(float(det.score))

            result = vectorize_chip_mask(
                det.mask_crop_uint8,
                tuple(det.mask_crop_offset),
                threshold=threshold,
                window_transform=win_tr,
                source_crs=chip.source_crs,
                multi_component="union",
            )
            merged = _unary_safe([g for g in result.geoms if g is not None and not g.is_empty])
            if merged is not None and not merged.is_empty:
                mask_geoms.append(merged)
                mask_scores.append(float(det.score))

    if raw_crs is None:
        raw_crs = str(dst_crs)
    box_geoms = _reproject_geoms(box_geoms, raw_crs, dst_crs)
    mask_geoms = _reproject_geoms(mask_geoms, raw_crs, dst_crs)
    return (
        GeometrySet("raw_box", box_geoms, np.array(box_scores, dtype=float)),
        GeometrySet("raw_mask", mask_geoms, np.array(mask_scores, dtype=float)),
        threshold,
    )


def _mutual_iou_set(base: GeometrySet, threshold: float = 0.3) -> GeometrySet:
    geoms = base.geoms
    n = len(geoms)
    if n == 0:
        return GeometrySet(f"mutual_iou_{threshold}", [], np.zeros(0, dtype=float))
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    tree = STRtree(geoms)
    for i, geom in enumerate(geoms):
        for j in tree.query(geom):
            j = int(j)
            if j <= i:
                continue
            if _safe_iou(geom, geoms[j]) >= threshold:
                union(i, j)

    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)

    out_geoms: list[BaseGeometry] = []
    out_scores: list[float] = []
    for idxs in members.values():
        merged = _unary_safe([geoms[i] for i in idxs])
        if merged is None or merged.is_empty:
            continue
        out_geoms.append(merged)
        out_scores.append(float(base.scores[idxs].max()) if len(base.scores) else 0.0)
    return GeometrySet(f"mutual_iou_{threshold}", out_geoms, np.array(out_scores, dtype=float))


def _per_gt_stats(gt: gpd.GeoDataFrame, geom_set: GeometrySet) -> pd.DataFrame:
    rows: list[dict] = []
    geoms = geom_set.geoms
    scores = geom_set.scores
    tree = STRtree(geoms) if geoms else None
    for gt_pos, gt_geom in enumerate(gt.geometry.values):
        hit_geoms: list[BaseGeometry] = []
        hit_scores: list[float] = []
        best_iou = 0.0
        best_coverage = 0.0
        best_score = 0.0
        if tree is not None:
            for raw_idx in tree.query(gt_geom):
                raw_idx = int(raw_idx)
                geom = geoms[raw_idx]
                if not geom.intersects(gt_geom):
                    continue
                inter = geom.intersection(gt_geom).area
                if inter <= 0:
                    continue
                score = float(scores[raw_idx]) if raw_idx < len(scores) else 0.0
                hit_geoms.append(geom)
                hit_scores.append(score)
                union = geom.area + gt_geom.area - inter
                iou = inter / union if union > 0 else 0.0
                coverage = inter / gt_geom.area if gt_geom.area > 0 else 0.0
                if iou > best_iou:
                    best_iou = float(iou)
                if coverage > best_coverage:
                    best_coverage = float(coverage)
                if score > best_score:
                    best_score = score
        union_coverage = 0.0
        if hit_geoms:
            merged = _unary_safe(hit_geoms)
            if merged is not None and not merged.is_empty and gt_geom.area > 0:
                union_coverage = min(float(merged.intersection(gt_geom).area / gt_geom.area), 1.0)
        rows.append({
            f"{geom_set.name}_n_hits": len(hit_geoms),
            f"{geom_set.name}_best_iou": best_iou,
            f"{geom_set.name}_best_coverage": best_coverage,
            f"{geom_set.name}_union_coverage": union_coverage,
            f"{geom_set.name}_max_score": best_score,
        })
    return pd.DataFrame(rows)


def _grid_paths(root: Path, grid: str) -> dict[str, Path | None]:
    grid_root = root / grid
    return {
        "grid_root": grid_root,
        "raw": grid_root / "raw/raw_detections.pkl",
        "clean_gt": Path("data/annotations_channel2_clean") / grid / f"{grid}_clean_gt.gpkg",
        "old_pixel_or": grid_root / "old_pixel_or/predictions_metric.gpkg",
        "per_detection": _choose_per_detection_path(grid_root),
        "ref_sam_maskbox": Path("results/johannesburg/v3c_sam_maskbox_vexcel_2024")
        / grid
        / "predictions_metric.gpkg",
    }


def audit_grid(
    grid: str,
    *,
    root: Path,
    raw_mask_threshold: float | None,
    mutual_iou: float,
) -> tuple[pd.DataFrame, dict]:
    paths = _grid_paths(root, grid)
    for key in ("raw", "clean_gt", "old_pixel_or", "per_detection"):
        path = paths[key]
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"{grid}: missing {key}: {path}")

    gt = gpd.read_file(paths["clean_gt"])
    if gt.crs is None:
        raise ValueError(f"{paths['clean_gt']} has no CRS")
    gt = gt[gt.geometry.notna() & ~gt.geometry.is_empty].reset_index(drop=True)
    gt["gt_area_m2"] = gt.geometry.area

    raw_box, raw_mask, effective_mask_threshold = _raw_sets_from_artifact(
        paths["raw"],
        gt.crs,
        mask_threshold=raw_mask_threshold,
    )
    old = _load_gpkg_as_set(paths["old_pixel_or"], "old_pixel_or", gt.crs)
    per_det = _load_gpkg_as_set(paths["per_detection"], "per_detection", gt.crs)
    if old is None or per_det is None:
        raise RuntimeError(f"{grid}: failed to load finalized predictions")
    mutual = _mutual_iou_set(per_det, threshold=mutual_iou)
    ref_sam = _load_gpkg_as_set(paths["ref_sam_maskbox"], "ref_sam_maskbox", gt.crs)

    detail = gt.drop(columns="geometry").copy()
    if "grid" in detail.columns:
        detail["grid"] = grid
        detail.insert(0, "gt_idx", np.arange(len(detail), dtype=int))
        cols = ["grid", "gt_idx"] + [
            c for c in detail.columns if c not in {"grid", "gt_idx"}
        ]
        detail = detail[cols]
    else:
        detail.insert(0, "grid", grid)
        detail.insert(1, "gt_idx", np.arange(len(detail), dtype=int))
    for geom_set in (raw_box, raw_mask, old, per_det, mutual):
        detail = pd.concat([detail, _per_gt_stats(gt, geom_set)], axis=1)
    if ref_sam is not None:
        detail = pd.concat([detail, _per_gt_stats(gt, ref_sam)], axis=1)

    meta = {
        "grid": grid,
        "raw_detections": len(raw_box.geoms),
        "raw_masks_vectorized": len(raw_mask.geoms),
        "raw_mask_threshold": effective_mask_threshold,
        "old_pixel_or_n": len(old.geoms),
        "per_detection_n": len(per_det.geoms),
        "per_detection_path": str(paths["per_detection"]),
        "mutual_iou_n": len(mutual.geoms),
        "ref_sam_maskbox_n": None if ref_sam is None else len(ref_sam.geoms),
    }
    return detail, meta


def _summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    groups = detail.groupby(["grid", "source"], dropna=False)
    for (grid, source), df in groups:
        n = len(df)
        raw_box_hint = df["raw_box_n_hits"] > 0
        raw_mask_hint = df["raw_mask_n_hits"] > 0
        old05 = df["old_pixel_or_best_iou"] >= 0.5
        per05 = df["per_detection_best_iou"] >= 0.5
        mutual05 = df["mutual_iou_0.3_best_iou"] >= 0.5
        old03 = df["old_pixel_or_best_iou"] >= 0.3
        per03 = df["per_detection_best_iou"] >= 0.3
        mutual03 = df["mutual_iou_0.3_best_iou"] >= 0.3
        row = {
            "grid": grid,
            "source": source,
            "n_gt": n,
            "raw_box_hint_rate": round(float(raw_box_hint.mean()), 3),
            "raw_mask_hint_rate": round(float(raw_mask_hint.mean()), 3),
            "no_raw_box_hint": int((~raw_box_hint).sum()),
            "box_hint_no_mask_hint": int((raw_box_hint & ~raw_mask_hint).sum()),
            "old_iou03_recall": round(float(old03.mean()), 3),
            "perdet_iou03_recall": round(float(per03.mean()), 3),
            "mutual_iou03_recall": round(float(mutual03.mean()), 3),
            "old_iou05_recall": round(float(old05.mean()), 3),
            "perdet_iou05_recall": round(float(per05.mean()), 3),
            "mutual_iou05_recall": round(float(mutual05.mean()), 3),
            "perdet_gain_old_fail_iou05": int((~old05 & per05).sum()),
            "perdet_gain_old_fail_iou05_share": round(float((~old05 & per05).mean()), 3),
            "old_pass_perdet_fail_iou05": int((old05 & ~per05).sum()),
            "raw_box_hint_old_fail_perdet_pass_iou05": int((raw_box_hint & ~old05 & per05).sum()),
            "no_raw_box_hint_perdet_pass_iou05": int((~raw_box_hint & per05).sum()),
            "median_raw_box_best_iou": round(float(df["raw_box_best_iou"].median()), 3),
            "median_raw_mask_best_iou": round(float(df["raw_mask_best_iou"].median()), 3),
            "median_old_iou": round(float(df["old_pixel_or_best_iou"].median()), 3),
            "median_perdet_iou": round(float(df["per_detection_best_iou"].median()), 3),
            "median_mutual_iou": round(float(df["mutual_iou_0.3_best_iou"].median()), 3),
        }
        if "ref_sam_maskbox_best_iou" in df.columns:
            ref05 = df["ref_sam_maskbox_best_iou"] >= 0.5
            ref03 = df["ref_sam_maskbox_best_iou"] >= 0.3
            row["ref_sam_iou03_recall"] = round(float(ref03.mean()), 3)
            row["ref_sam_iou05_recall"] = round(float(ref05.mean()), 3)
            row["median_ref_sam_iou"] = round(float(df["ref_sam_maskbox_best_iou"].median()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _bucket_sam_supp(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    sub = detail[detail["source"] == "SAM_supp+V3C_TP"].copy()
    for grid, df in sub.groupby("grid"):
        raw_box_hint = df["raw_box_n_hits"] > 0
        raw_mask_hint = df["raw_mask_n_hits"] > 0
        old05 = df["old_pixel_or_best_iou"] >= 0.5
        per05 = df["per_detection_best_iou"] >= 0.5
        buckets = {
            "no_raw_box_hint": ~raw_box_hint,
            "box_hint_no_mask_hint": raw_box_hint & ~raw_mask_hint,
            "mask_hint_old_fail_perdet_pass": raw_mask_hint & ~old05 & per05,
            "mask_hint_old_fail_perdet_fail": raw_mask_hint & ~old05 & ~per05,
            "mask_hint_old_pass_perdet_fail": raw_mask_hint & old05 & ~per05,
            "mask_hint_old_pass_perdet_pass": raw_mask_hint & old05 & per05,
        }
        row = {"grid": grid, "n_sam_supp": len(df)}
        for name, mask in buckets.items():
            row[name] = int(mask.sum())
            row[f"{name}_share"] = round(float(mask.mean()), 3) if len(df) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown(
    path: Path,
    summary: pd.DataFrame,
    buckets: pd.DataFrame,
    metas: list[dict],
) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_empty_"
        rows = df.fillna("").astype(str).values.tolist()
        headers = list(df.columns)
        out = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            out.append("| " + " | ".join(row) + " |")
        return "\n".join(out)

    lines: list[str] = []
    lines.append("# Raw Detector Hint Audit")
    lines.append("")
    lines.append("This audit asks whether each clean GT polygon had any raw detector proposal before finalization.")
    lines.append("")
    lines.append("## Raw Artifacts")
    lines.append("")
    meta_df = pd.DataFrame(metas)
    lines.append(md_table(meta_df))
    lines.append("")

    sam = summary[summary["source"] == "SAM_supp+V3C_TP"].copy()
    cols = [
        "grid", "n_gt", "raw_box_hint_rate", "raw_mask_hint_rate",
        "old_iou05_recall", "perdet_iou05_recall", "mutual_iou05_recall",
        "perdet_gain_old_fail_iou05", "no_raw_box_hint",
        "median_old_iou", "median_perdet_iou",
    ]
    optional_cols = ["ref_sam_iou05_recall", "median_ref_sam_iou"]
    cols.extend([c for c in optional_cols if c in sam.columns])
    lines.append("## SAM_supp+V3C_TP Summary")
    lines.append("")
    lines.append(md_table(sam[cols]))
    lines.append("")

    lines.append("## SAM_supp+V3C_TP Buckets at IoU@0.5")
    lines.append("")
    lines.append(md_table(buckets))
    lines.append("")

    v3c = summary[summary["source"] == "V3C_TP"].copy()
    if not v3c.empty:
        lines.append("## V3C_TP Summary")
        lines.append("")
        lines.append(md_table(v3c[cols]))
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `raw_box_hint_rate` is the share of GT polygons touched by at least one raw detector box.")
    lines.append("- `raw_mask_hint_rate` is the share touched by at least one raw mask vectorized at the artifact mask threshold.")
    lines.append("- `perdet_gain_old_fail_iou05` counts GT where old pixel-or fails IoU@0.5 but per-detection passes.")
    lines.append("- `no_raw_box_hint` is the strict no-proposal bucket; post-processing cannot recover these without new detector/SAM proposals.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", default=",".join(DEFAULT_GRIDS))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/analysis/finalizer_mask_shaping_ablation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis/finalizer_mask_shaping_ablation/raw_hint_audit"),
    )
    parser.add_argument("--raw-mask-threshold", type=float, default=None)
    parser.add_argument("--mutual-iou", type=float, default=0.3)
    args = parser.parse_args()

    grids = [g.strip() for g in args.grids.split(",") if g.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    details: list[pd.DataFrame] = []
    metas: list[dict] = []
    for grid in grids:
        print(f"[audit] {grid}")
        detail, meta = audit_grid(
            grid,
            root=args.root,
            raw_mask_threshold=args.raw_mask_threshold,
            mutual_iou=args.mutual_iou,
        )
        detail.to_csv(args.output_dir / f"{grid}_raw_hint_detail.csv", index=False)
        details.append(detail)
        metas.append(meta)
        print(
            f"[audit] {grid}: raw={meta['raw_detections']} "
            f"raw_masks={meta['raw_masks_vectorized']} perdet={meta['per_detection_n']}"
        )

    all_detail = pd.concat(details, ignore_index=True)
    summary = _summarize(all_detail)
    buckets = _bucket_sam_supp(all_detail)

    all_detail.to_csv(args.output_dir / "raw_hint_detail.csv", index=False)
    summary.to_csv(args.output_dir / "raw_hint_summary.csv", index=False)
    buckets.to_csv(args.output_dir / "sam_supp_buckets.csv", index=False)
    pd.DataFrame(metas).to_csv(args.output_dir / "raw_hint_inputs.csv", index=False)
    _write_markdown(args.output_dir / "summary.md", summary, buckets, metas)

    print(f"[audit] wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
