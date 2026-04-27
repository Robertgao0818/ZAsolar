"""
Cross-grid deduplication for the cascade classifier pool.

GEID mosaics for adjacent CBD grids overlap by ~35-70 m, so the same
physical PV/non-PV object can be detected and labeled twice in two
different grids. This script:

1. Builds a union-find over polygons within the same (detector, label)
   bucket whose geometries overlap by IoU >= 0.3 OR 50% containment on
   either side.
2. Adds three columns to manifest.{csv,gpkg}: dup_group_id, dup_size,
   is_canonical (canonical = largest area_m2 within group; tie broken
   by chip_id for determinism).
3. If a labeled subset CSV is given, aggregates human_label by
   dup_group_id, flags label disagreements, prints a corrected subtype
   distribution, and writes a deduplicated CSV.

Single-row groups get dup_group_id = "" (no duplicate).

Usage:
    python scripts/classifier/dedup_cls_pool.py \\
        --pool-dir data/cls_pv_nonpv_v3c_v42_cascade \\
        --labeled-csv data/cls_pv_nonpv_v3c_v42_cascade/labeler/v3c__both/nonpv_subtype_labeled.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def find_overlap_groups(
    gdf: gpd.GeoDataFrame, iou_threshold: float, contain_threshold: float
) -> dict[int, int]:
    """Return {row_index → group_root_index} for rows in different grids
    whose polygons overlap by the given thresholds.

    Self-grid pairs are skipped (same-grid duplicates within one detector
    do not occur in our prediction pipeline).
    """
    if len(gdf) == 0:
        return {}
    gdf = gdf.reset_index(drop=True)
    n = len(gdf)
    # Pull the positional index for both sides via reset_index, so the
    # column name doesn't collide with sjoin's internal index_right.
    left = gdf.reset_index().rename(columns={"index": "i_left"})
    right = gdf.reset_index()[["index", "geometry"]].rename(
        columns={"index": "i_right"}
    )
    sj = gpd.sjoin(left, right, how="inner", predicate="intersects")
    # sjoin uses the right df's index for index_right; we kept i_right as a
    # column and prefer it explicitly in case index_right was duplicated.
    j_col = "i_right" if "i_right" in sj.columns else "index_right"
    sj = sj[sj["i_left"] < sj[j_col]]

    uf = UnionFind(n)
    geoms = list(gdf.geometry)
    grids = gdf["grid_id"].tolist()
    for _, row in sj.iterrows():
        i = int(row["i_left"])
        j = int(row[j_col])
        if grids[i] == grids[j]:
            continue
        a, b = geoms[i], geoms[j]
        if not a.intersects(b):
            continue
        inter = a.intersection(b).area
        if inter <= 0:
            continue
        union_area = a.area + b.area - inter
        iou = inter / union_area if union_area > 0 else 0.0
        if (
            iou >= iou_threshold
            or inter / a.area >= contain_threshold
            or inter / b.area >= contain_threshold
        ):
            uf.union(i, j)

    return {i: uf.find(i) for i in range(n)}


def assign_dup_columns(
    gdf: gpd.GeoDataFrame, group_map: dict[int, int]
) -> gpd.GeoDataFrame:
    gdf = gdf.reset_index(drop=True)
    roots = pd.Series([group_map[i] for i in range(len(gdf))], index=gdf.index)
    sizes = roots.value_counts()

    dup_group_id = pd.Series("", index=gdf.index, dtype="string")
    is_canonical = pd.Series(True, index=gdf.index)
    dup_size = pd.Series(1, index=gdf.index)

    for root, size in sizes.items():
        if size < 2:
            continue
        members = roots[roots == root].index.tolist()
        canonical = max(
            members, key=lambda idx: (gdf.at[idx, "area_m2"], gdf.at[idx, "chip_id"])
        )
        gid = f"g{int(root):05d}"
        for m in members:
            dup_group_id.at[m] = gid
            dup_size.at[m] = size
            is_canonical.at[m] = m == canonical

    gdf["dup_group_id"] = dup_group_id
    gdf["dup_size"] = dup_size
    gdf["is_canonical"] = is_canonical
    return gdf


def dedup_one_bucket(
    bucket: gpd.GeoDataFrame, iou_threshold: float, contain_threshold: float
) -> gpd.GeoDataFrame:
    if len(bucket) == 0:
        return bucket
    group_map = find_overlap_groups(bucket, iou_threshold, contain_threshold)
    return assign_dup_columns(bucket, group_map)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool-dir", type=Path, required=True)
    p.add_argument("--labeled-csv", type=Path, default=None)
    p.add_argument("--metric-crs", default="EPSG:32735")
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--contain-threshold", type=float, default=0.5)
    args = p.parse_args()

    pool = args.pool_dir
    print(f"[1/3] Loading {pool/'manifest.gpkg'} ...")
    gdf = gpd.read_file(pool / "manifest.gpkg")
    if str(gdf.crs) != args.metric_crs:
        gdf = gdf.to_crs(args.metric_crs)

    print(f"  {len(gdf)} rows total")
    print("\n[2/3] Dedup within each (detector, label) bucket ...")
    parts = []
    for (det, lab), bucket in gdf.groupby(["detector", "label"], sort=False):
        bucket = bucket.copy()
        n_before = len(bucket)
        bucket = dedup_one_bucket(
            bucket, args.iou_threshold, args.contain_threshold
        )
        n_groups = (bucket["dup_size"] > 1).sum()
        n_canonical = bucket["is_canonical"].sum()
        n_dropped_if_dedup = n_before - n_canonical
        n_unique_groups = bucket.loc[bucket["dup_size"] > 1, "dup_group_id"].nunique()
        print(
            f"  {det:<6}{lab:<7} | rows={n_before:>4} | "
            f"in_dup_groups={n_groups:>3} (across {n_unique_groups} groups) | "
            f"canonical={n_canonical:>4} | redundant={n_dropped_if_dedup:>3}"
        )
        parts.append(bucket)

    out = pd.concat(parts, ignore_index=False).sort_index()
    # Re-attach existing chip_path / split etc. by index
    out_gdf = gpd.GeoDataFrame(out, geometry="geometry", crs=args.metric_crs)

    csv_cols = [c for c in out_gdf.columns if c != "geometry"]
    out_gdf.drop(columns=["geometry"]).to_csv(pool / "manifest.csv", index=False)
    out_gdf.to_file(pool / "manifest.gpkg", driver="GPKG", layer="cls_pool")
    print(f"\n  Wrote {pool/'manifest.csv'} and {pool/'manifest.gpkg'}")
    print(f"  New columns: dup_group_id, dup_size, is_canonical")
    print(
        f"  Pool unique objects: {out_gdf['is_canonical'].sum()} "
        f"(was {len(out_gdf)} rows; dropping non-canonical removes "
        f"{len(out_gdf) - out_gdf['is_canonical'].sum()} duplicate detections)"
    )

    if args.labeled_csv is None:
        return 0

    print(f"\n[3/3] Apply dedup to labeled CSV {args.labeled_csv} ...")
    lab = pd.read_csv(args.labeled_csv)
    print(f"  {len(lab)} labeled rows")

    merged = lab.merge(
        out_gdf[["chip_id", "dup_group_id", "dup_size", "is_canonical"]],
        on="chip_id", how="left",
    )

    # Label-consistency check across dup groups
    multi = merged[merged["dup_size"] > 1]
    inconsistencies = []
    for gid, grp in multi.groupby("dup_group_id"):
        labels = sorted(grp["human_label"].dropna().unique().tolist())
        if len(labels) > 1:
            inconsistencies.append(
                {"dup_group_id": gid, "labels": labels, "chips": grp["chip_id"].tolist()}
            )
    if inconsistencies:
        print(f"\n  ⚠ {len(inconsistencies)} dup groups have inconsistent labels:")
        for inc in inconsistencies:
            print(
                f"    {inc['dup_group_id']}: {inc['labels']} | "
                f"chips={inc['chips']}"
            )
    else:
        print(f"  ✓ all dup groups have a unique human_label (no audit conflicts)")

    # Canonical-only dedup view
    dedup = merged[(merged["dup_size"] == 1) | (merged["is_canonical"])].copy()
    print(
        f"\n  Labeled rows: {len(lab)} → "
        f"unique physical objects: {len(dedup)} "
        f"(removed {len(lab) - len(dedup)} duplicate detections)"
    )

    # Subtype distribution before/after
    print("\n=== Subtype distribution: raw labels (n={}) ===".format(len(lab)))
    print(lab["human_label"].value_counts(dropna=False).to_string())
    print("\n=== Subtype distribution: dedup canonical (n={}) ===".format(len(dedup)))
    print(dedup["human_label"].value_counts(dropna=False).to_string())

    # Persist deduplicated labeled CSV (canonical rows + dup_group_id columns)
    out_csv = args.labeled_csv.parent / (
        args.labeled_csv.stem + "_dedup.csv"
    )
    cols_to_write = [
        "chip_id", "detector", "source_detector", "grid_id", "pred_idx",
        "iou_to_gt", "area_m2", "human_label",
        "dup_group_id", "dup_size", "is_canonical",
    ]
    cols_to_write = [c for c in cols_to_write if c in dedup.columns]
    dedup[cols_to_write].to_csv(out_csv, index=False)
    print(f"\n  Wrote {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
