#!/usr/bin/env python3
"""Build Channel 2 clean GT for the 25 JHB CBD Vexcel grids.

Composition rule (per grid):

    clean_gt = dissolve_cluster(
        V3C_pred[review_status=correct]   # edit / delete dropped
        ∪ SAM_added                       # RA-drawn FN polygons
        ∪ Li[include=true]                # micro 3 grids only
    )

`dissolve_cluster` does a connected-component union of polygons that
intersect (or come within `--touch-tol` metres). Each output cluster
becomes one installation polygon with provenance tracking.

Inputs (per grid):
  results/johannesburg/v3c_vexcel_2024/<grid>/predictions_metric.gpkg
  results/johannesburg/v3c_vexcel_2024_ch1_sample/<grid>/review/
      detection_review_decisions.csv
      <grid>_sam_added.gpkg
  data/annotations_channel2_micro/<grid>/<grid>_li_gt.gpkg  (micro only,
      may also live in /mnt/d/ZAsolar/qgis_handoff/ch1_fn_review_2026-05-03/)

Outputs (per grid):
  data/annotations_channel2_clean/<grid>/<grid>_clean_gt.gpkg
  data/annotations_channel2_clean/clean_gt_summary_25grid.csv
  data/annotations_channel2_clean/edit_uncovered_audit.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_ROOT = PROJECT_ROOT / "results" / "johannesburg" / "v3c_vexcel_2024"
REVIEW_ROOT = PROJECT_ROOT / "results" / "johannesburg" / "v3c_vexcel_2024_ch1_sample"
MICRO_ROOT = PROJECT_ROOT / "data" / "annotations_channel2_micro"
LI_HANDOFF = Path("/mnt/d/ZAsolar/qgis_handoff/ch1_fn_review_2026-05-03")
DEFAULT_OUT = PROJECT_ROOT / "data" / "annotations_channel2_clean"
TARGET_CRS = "EPSG:32735"

MICRO_GRIDS = {"G0774", "G0816", "G0922"}


def load_reviewed(grid: str) -> gpd.GeoDataFrame:
    """Canonical source: <grid>_reviewed.gpkg with review_status attached."""
    rev_path = REVIEW_ROOT / grid / "review" / f"{grid}_reviewed.gpkg"
    if not rev_path.exists():
        return gpd.GeoDataFrame(columns=["review_status", "geometry"], geometry="geometry", crs=TARGET_CRS)
    rev = gpd.read_file(rev_path)
    if rev.crs is None or str(rev.crs) != TARGET_CRS:
        rev = rev.to_crs(TARGET_CRS)
    return rev.reset_index(drop=True)


def split_by_status(rev: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, list[int], list[int], list[int]]:
    if rev.empty or "review_status" not in rev.columns:
        empty = gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
        return empty, [], [], []
    correct = rev[rev.review_status == "correct"]
    edit_ids = rev.index[rev.review_status == "edit"].tolist()
    delete_ids = rev.index[rev.review_status == "delete"].tolist()
    correct_ids = correct.index.tolist()
    out = correct[["geometry"]].copy()
    out["source"] = "V3C_TP"
    out["src_id"] = [f"v3c_{i}" for i in correct_ids]
    return out.reset_index(drop=True), correct_ids, edit_ids, delete_ids


def load_sam_added(grid: str) -> gpd.GeoDataFrame:
    sam_path = REVIEW_ROOT / grid / "review" / f"{grid}_sam_added.gpkg"
    if not sam_path.exists():
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    sam = gpd.read_file(sam_path)
    if sam.crs is None or str(sam.crs) != TARGET_CRS:
        sam = sam.to_crs(TARGET_CRS)
    out = sam[["geometry"]].copy()
    out["source"] = "SAM_supp"
    out["src_id"] = [f"sam_{i}" for i in range(len(sam))]
    return out.reset_index(drop=True)


def load_li_marked(grid: str) -> gpd.GeoDataFrame:
    """Only for micro grids — Li polygons with include=true."""
    if grid not in MICRO_GRIDS:
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    candidates = [
        LI_HANDOFF / grid / f"{grid}_li_gt.gpkg",
        MICRO_ROOT / grid / f"{grid}_li_gt.gpkg",
    ]
    li_path = next((p for p in candidates if p.exists()), None)
    if li_path is None:
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    li = gpd.read_file(li_path)
    if li.crs is None or str(li.crs) != TARGET_CRS:
        li = li.to_crs(TARGET_CRS)
    if "include" not in li.columns:
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    sub = li[li["include"].astype(str).str.lower().isin({"true", "1", "yes"})]
    if sub.empty:
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    out = sub[["geometry"]].copy()
    li_ids = sub["li_id"].astype(str).tolist() if "li_id" in sub.columns else [str(i) for i in range(len(sub))]
    out["source"] = "Li_marked"
    out["src_id"] = [f"li_{i}" for i in li_ids]
    return out.reset_index(drop=True)


def audit_edits_vs_sam(grid: str, rev: gpd.GeoDataFrame, edit_ids: list[int],
                       sam: gpd.GeoDataFrame) -> list[dict]:
    """For each edit pred, classify the SAM replacement pattern:
      - case_A_outer  : at least one sam polygon contains/overlaps and is LARGER than V3C edit
                        (V3C cut too small, SAM redrew bigger)
      - case_B_inner  : at least one sam polygon overlaps and is SMALLER than V3C edit
                        (V3C cut too big, SAM redrew the real PV inside)
      - mixed         : both patterns present
      - uncovered     : no sam polygon overlaps -> RA never re-cut
    Coverage threshold: any non-zero area overlap (>0.01 m²)."""
    if not edit_ids:
        return []
    edits = rev.iloc[edit_ids]
    rows = []
    sidx = sam.sindex if not sam.empty else None
    for pid, geom in zip(edit_ids, edits.geometry):
        v3c_area = float(geom.area)
        n_outer = n_inner = 0  # outer = sam larger; inner = sam smaller
        if sidx is not None:
            for j in sidx.intersection(geom.bounds):
                sg = sam.geometry.iloc[j]
                inter = sg.intersection(geom)
                if inter.is_empty or inter.area <= 0.01:
                    continue
                if sg.area >= v3c_area:
                    n_outer += 1
                else:
                    n_inner += 1
        if n_outer == 0 and n_inner == 0:
            kind = "uncovered"
        elif n_outer > 0 and n_inner > 0:
            kind = "mixed"
        elif n_outer > 0:
            kind = "case_A_outer"
        else:
            kind = "case_B_inner"
        rows.append({"grid": grid, "pred_id": int(pid), "edit_area_m2": v3c_area,
                     "n_sam_outer": n_outer, "n_sam_inner": n_inner,
                     "kind": kind, "covered": kind != "uncovered"})
    return rows


def dissolve_cluster(parts: gpd.GeoDataFrame, min_overlap_m2: float) -> gpd.GeoDataFrame:
    """Connected-component union: any two parts whose intersection area
    exceeds `min_overlap_m2` collapse to one polygon. Edge-touching
    (shared boundary, zero area) does NOT merge."""
    if parts.empty:
        return gpd.GeoDataFrame(
            columns=["clean_id", "source", "n_components", "components", "area_m2", "geometry"],
            geometry="geometry", crs=TARGET_CRS,
        )
    parts = parts.reset_index(drop=True)
    sidx = parts.geometry.sindex
    n = len(parts)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, gi in enumerate(parts.geometry):
        for j in sidx.intersection(gi.bounds):
            if j <= i:
                continue
            gj = parts.geometry.iloc[j]
            if not gi.intersects(gj):
                continue
            inter = gi.intersection(gj)
            if (not inter.is_empty) and inter.area > min_overlap_m2:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out_rows = []
    for cid, idxs in enumerate(groups.values()):
        geoms = [parts.geometry.iloc[i] for i in idxs]
        merged = unary_union(geoms)
        if merged.is_empty:
            continue
        if merged.geom_type == "Polygon":
            geom = merged
        elif merged.geom_type == "MultiPolygon":
            # Pick largest polygon? Keep multipolygon for fidelity but
            # downstream eval expects single polygon. Use convex hull only
            # if buffer trick failed. Here we keep MultiPolygon and let
            # consumer handle — most clusters resolve to single polygon.
            geom = merged
        else:
            continue
        srcs = [parts.source.iloc[i] for i in idxs]
        ids = [parts.src_id.iloc[i] for i in idxs]
        src_summary = "+".join(sorted(set(srcs)))
        out_rows.append({
            "clean_id": cid,
            "source": src_summary,
            "n_components": len(idxs),
            "components": ";".join(ids),
            "area_m2": float(geom.area),
            "geometry": geom,
        })
    return gpd.GeoDataFrame(out_rows, geometry="geometry", crs=TARGET_CRS)


def build_grid(grid: str, out_root: Path, min_overlap_m2: float) -> tuple[dict, list[dict]]:
    rev = load_reviewed(grid)
    v3c, correct_ids, edit_ids, delete_ids = split_by_status(rev)
    sam = load_sam_added(grid)
    li = load_li_marked(grid)
    parts = pd.concat([v3c, sam, li], ignore_index=True) if any(len(x) for x in (v3c, sam, li)) else v3c
    parts = gpd.GeoDataFrame(parts, geometry="geometry", crs=TARGET_CRS)

    edit_audit = audit_edits_vs_sam(grid, rev, edit_ids, sam)

    clean = dissolve_cluster(parts, min_overlap_m2=min_overlap_m2)
    clean["grid"] = grid
    clean["annotation_id"] = [f"{grid}_T1_{i}" for i in range(len(clean))]
    clean["label"] = "pv"
    clean["axis_a"] = "A1"
    clean["axis_b"] = "H"

    grid_out = out_root / grid
    grid_out.mkdir(parents=True, exist_ok=True)
    clean_path = grid_out / f"{grid}_clean_gt.gpkg"
    clean.to_file(clean_path, layer="T1", driver="GPKG")

    summary = {
        "grid": grid,
        "v3c_total_pred": int(len(rev)),
        "v3c_correct": len(correct_ids),
        "v3c_edit": len(edit_ids),
        "v3c_delete": len(delete_ids),
        "sam_added": len(sam),
        "li_marked": len(li),
        "edit_covered_by_sam": int(sum(1 for r in edit_audit if r["covered"])),
        "edit_uncovered": int(sum(1 for r in edit_audit if not r["covered"])),
        "input_parts": len(parts),
        "clean_total": len(clean),
        "merged_pairs": int(len(parts) - len(clean)),
        "is_micro": grid in MICRO_GRIDS,
    }
    return summary, edit_audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="*", default=None,
                    help="Subset of grids to build (default: all 25)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-overlap-m2", type=float, default=0.01,
                    help="Minimum area overlap in m² required to merge two polygons. "
                         "Edge-touching (zero area) does NOT merge. Default 0.01 m² "
                         "= effectively any non-zero overlap, deduplicates floating-point dust.")
    args = ap.parse_args()

    if args.grids:
        grids = args.grids
    else:
        grids = sorted(d.name for d in REVIEW_ROOT.glob("G*") if d.is_dir())
    if not grids:
        print("no grids found", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    edit_audit_rows = []
    for g in grids:
        try:
            s, ea = build_grid(g, args.out, args.min_overlap_m2)
        except Exception as e:
            print(f"[ERR] {g}: {e}", file=sys.stderr)
            continue
        summaries.append(s)
        edit_audit_rows.extend(ea)
        print(f"{g} clean={s['clean_total']:>4} (parts={s['input_parts']}, merged={s['merged_pairs']}, "
              f"V3C_TP={s['v3c_correct']}, SAM={s['sam_added']}, Li={s['li_marked']}, "
              f"edit={s['v3c_edit']} cov={s['edit_covered_by_sam']}/uncov={s['edit_uncovered']})")

    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(args.out / "clean_gt_summary_25grid.csv", index=False)
    audit_df = pd.DataFrame(edit_audit_rows)
    audit_df.to_csv(args.out / "edit_uncovered_audit.csv", index=False)
    print(f"\ntotals: clean={sum_df['clean_total'].sum()}, "
          f"merged_pairs={sum_df['merged_pairs'].sum()}, "
          f"edit_uncov={sum_df['edit_uncovered'].sum()}")
    print(f"summary -> {args.out / 'clean_gt_summary_25grid.csv'}")
    print(f"edit audit -> {args.out / 'edit_uncovered_audit.csv'}")


if __name__ == "__main__":
    main()
