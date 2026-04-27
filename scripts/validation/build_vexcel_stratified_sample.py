#!/usr/bin/env python3
"""Build an OSM-informed stratified Vexcel evaluation sample.

The script uses OSM only as a candidate-generation signal. Missing OSM
buildings are not treated as proof that a grid is truly empty.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets" / "vexcel_urban_coverage.yaml"
DEFAULT_GRID_DIR = PROJECT_ROOT / "data" / "vexcel_task_grids"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "vexcel_eval_samples"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache" / "vexcel_osm"
WGS84 = "EPSG:4326"
STRATA = ("cbd", "residential", "industrial_logistics")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_region_grids(grid_dir: Path, region_key: str) -> gpd.GeoDataFrame:
    path = grid_dir / f"{region_key}_task_grid.gpkg"
    gdf = gpd.read_file(path).to_crs(WGS84)
    if "gridcell_id" not in gdf.columns:
        raise ValueError(f"{path} is missing gridcell_id")
    return gdf


def fetch_osm_features(region_key: str, bounds: tuple[float, float, float, float], cache_dir: Path) -> gpd.GeoDataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{region_key}_osm_features.gpkg"
    if cache_path.exists():
        return gpd.read_file(cache_path)

    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))
    import osmnx as ox  # noqa: PLC0415

    west, south, east, north = bounds
    tags = {
        "building": True,
        "landuse": ["industrial", "commercial", "retail"],
        "industrial": True,
        "man_made": ["works"],
        "shop": ["mall", "supermarket", "department_store"],
    }
    features = ox.features.features_from_bbox((west, south, east, north), tags=tags)
    features = features.reset_index()
    features = gpd.GeoDataFrame(features, geometry="geometry", crs=WGS84)
    features.to_file(cache_path, driver="GPKG")
    return features


def _nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str) != "")


def split_osm_features(features: gpd.GeoDataFrame, metric_crs: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if features.empty:
        empty = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=metric_crs)
        return empty.copy(), empty.copy()

    features = features.to_crs(metric_crs).copy()
    if "building" not in features.columns:
        features["building"] = pd.NA
    if "landuse" not in features.columns:
        features["landuse"] = pd.NA
    for col in ("industrial", "man_made", "shop"):
        if col not in features.columns:
            features[col] = pd.NA

    is_building = _nonempty(features["building"])
    landuse = features["landuse"].astype(str)
    is_industrial_landuse = landuse.isin({"industrial", "commercial", "retail"})
    has_industrial_tag = _nonempty(features["industrial"])
    is_works = features["man_made"].astype(str).isin({"works"})
    is_major_shop = features["shop"].astype(str).isin({"mall", "supermarket", "department_store"})

    buildings = features.loc[is_building].copy()
    landuse_features = features.loc[is_industrial_landuse | has_industrial_tag | is_works | is_major_shop].copy()

    polygon_types = {"Polygon", "MultiPolygon"}
    buildings["building_area_sqm"] = buildings.geometry.area.where(
        buildings.geometry.geom_type.isin(polygon_types),
        0.0,
    )
    building_tag = buildings["building"].astype(str)
    buildings["is_large_building"] = buildings["building_area_sqm"] >= 1500
    buildings["is_industrial_building"] = (
        building_tag.isin({"industrial", "warehouse", "commercial", "retail", "factory", "manufacture"})
        | _nonempty(buildings["industrial"])
    )
    buildings["geometry"] = buildings.geometry.representative_point()

    landuse_features = landuse_features.loc[landuse_features.geometry.geom_type.isin(polygon_types)].copy()
    landuse_features["landuse_area_sqm"] = landuse_features.geometry.area
    return buildings, landuse_features


def compute_region_metrics(
    region_key: str,
    grids_wgs: gpd.GeoDataFrame,
    osm_features: gpd.GeoDataFrame,
    metric_crs: str,
) -> gpd.GeoDataFrame:
    grids = grids_wgs.to_crs(metric_crs).copy()
    buildings, landuse_features = split_osm_features(osm_features, metric_crs)

    metrics = grids[["gridcell_id", "geometry"]].copy()
    metrics["region_key"] = region_key
    metrics["n_buildings"] = 0
    metrics["building_area_sqm"] = 0.0
    metrics["n_large_buildings"] = 0
    metrics["large_building_area_sqm"] = 0.0
    metrics["n_industrial_buildings"] = 0
    metrics["industrial_landuse_area_sqm"] = 0.0

    if not buildings.empty:
        joined = gpd.sjoin(
            buildings[[
                "building_area_sqm",
                "is_large_building",
                "is_industrial_building",
                "geometry",
            ]],
            grids[["gridcell_id", "geometry"]],
            how="inner",
            predicate="within",
        )
        if not joined.empty:
            agg = joined.groupby("gridcell_id").agg(
                n_buildings=("geometry", "size"),
                building_area_sqm=("building_area_sqm", "sum"),
                n_large_buildings=("is_large_building", "sum"),
                large_building_area_sqm=("building_area_sqm", lambda s: s[joined.loc[s.index, "is_large_building"]].sum()),
                n_industrial_buildings=("is_industrial_building", "sum"),
            )
            metrics = metrics.drop(columns=[
                "n_buildings",
                "building_area_sqm",
                "n_large_buildings",
                "large_building_area_sqm",
                "n_industrial_buildings",
            ]).merge(agg, on="gridcell_id", how="left")

    if not landuse_features.empty:
        clipped = gpd.overlay(
            landuse_features[["geometry"]],
            grids[["gridcell_id", "geometry"]],
            how="intersection",
            keep_geom_type=True,
        )
        if not clipped.empty:
            clipped["industrial_landuse_area_sqm"] = clipped.geometry.area
            landuse_agg = clipped.groupby("gridcell_id")["industrial_landuse_area_sqm"].sum()
            metrics = metrics.drop(columns=["industrial_landuse_area_sqm"]).merge(
                landuse_agg,
                on="gridcell_id",
                how="left",
            )

    fill_zero = [
        "n_buildings",
        "building_area_sqm",
        "n_large_buildings",
        "large_building_area_sqm",
        "n_industrial_buildings",
        "industrial_landuse_area_sqm",
    ]
    for col in fill_zero:
        metrics[col] = metrics[col].fillna(0)

    grid_area = metrics.geometry.area.replace(0, np.nan)
    metrics["building_density_km2"] = metrics["n_buildings"] / (grid_area / 1_000_000.0)
    metrics["building_area_share"] = metrics["building_area_sqm"] / grid_area
    metrics["large_building_area_share"] = metrics["large_building_area_sqm"] / grid_area
    metrics["industrial_landuse_share"] = metrics["industrial_landuse_area_sqm"] / grid_area
    for share_col in ("building_area_share", "large_building_area_share", "industrial_landuse_share"):
        metrics[share_col] = metrics[share_col].clip(lower=0.0, upper=1.0)
    metrics["industrial_building_share"] = np.where(
        metrics["n_buildings"] > 0,
        metrics["n_industrial_buildings"] / metrics["n_buildings"],
        0.0,
    )

    dense = metrics.loc[metrics["n_buildings"] > 0].copy()
    if dense.empty:
        center = metrics.geometry.centroid.unary_union.centroid
    else:
        center_row = dense.sort_values(["building_density_km2", "building_area_share"], ascending=False).iloc[0]
        center = center_row.geometry.centroid
    metrics["distance_to_density_core_m"] = metrics.geometry.centroid.distance(center)

    metrics["industrial_score"] = (
        metrics["industrial_landuse_share"].clip(0, 1) * 2.0
        + metrics["large_building_area_share"].clip(0, 1)
        + metrics["industrial_building_share"].clip(0, 1) * 0.75
    )
    return metrics.to_crs(WGS84)


def assign_strata(metrics: gpd.GeoDataFrame, min_buildings: int) -> gpd.GeoDataFrame:
    df = metrics.copy()
    nonempty = df.loc[df["n_buildings"] >= min_buildings]
    if nonempty.empty:
        df["suggested_stratum"] = "osm_sparse_or_empty"
        df["sampling_score"] = 0.0
        return df

    density_q75 = float(nonempty["building_density_km2"].quantile(0.75))
    industrial_q75 = float(nonempty["industrial_score"].quantile(0.75))
    industrial_cut = max(0.08, industrial_q75)

    df["suggested_stratum"] = "osm_sparse_or_empty"
    industrial = (
        (df["n_buildings"] >= min_buildings)
        & (
            (df["industrial_score"] >= industrial_cut)
            | ((df["n_large_buildings"] >= 2) & (df["large_building_area_share"] >= 0.08))
        )
    )
    cbd = (
        (df["n_buildings"] >= min_buildings)
        & (df["building_density_km2"] >= density_q75)
        & (df["distance_to_density_core_m"] <= 2500)
        & ~industrial
    )
    residential = (df["n_buildings"] >= min_buildings) & ~(industrial | cbd)

    df.loc[industrial, "suggested_stratum"] = "industrial_logistics"
    df.loc[cbd, "suggested_stratum"] = "cbd"
    df.loc[residential, "suggested_stratum"] = "residential"

    density_norm = df["building_density_km2"] / max(float(df["building_density_km2"].max()), 1.0)
    center_norm = 1.0 - (df["distance_to_density_core_m"] / max(float(df["distance_to_density_core_m"].max()), 1.0))
    industrial_norm = df["industrial_score"] / max(float(df["industrial_score"].max()), 1.0)

    df["sampling_score"] = 0.0
    df.loc[df["suggested_stratum"] == "cbd", "sampling_score"] = (
        0.7 * density_norm + 0.3 * center_norm
    )
    df.loc[df["suggested_stratum"] == "residential", "sampling_score"] = (
        0.8 * density_norm + 0.2 * (1.0 - industrial_norm)
    )
    df.loc[df["suggested_stratum"] == "industrial_logistics", "sampling_score"] = (
        0.7 * industrial_norm + 0.3 * density_norm
    )
    return df


def ensure_min_cbd(df: gpd.GeoDataFrame, min_cbd: int, min_buildings: int) -> gpd.GeoDataFrame:
    """Promote dense central non-industrial grids so each city has CBD options."""
    out = df.copy()
    current = int((out["suggested_stratum"] == "cbd").sum())
    if current >= min_cbd:
        return out

    pool = out.loc[
        (out["n_buildings"] >= min_buildings)
        & (out["suggested_stratum"] != "industrial_logistics")
        & (out["suggested_stratum"] != "cbd")
    ].copy()
    if pool.empty:
        return out

    pool["cbd_promotion_score"] = (
        pool["building_density_km2"].rank(pct=True)
        + (1.0 - pool["distance_to_density_core_m"].rank(pct=True))
    )
    promote_ids = pool.sort_values("cbd_promotion_score", ascending=False).head(min_cbd - current)["gridcell_id"]
    out.loc[out["gridcell_id"].isin(promote_ids), "suggested_stratum"] = "cbd"

    density_norm = out["building_density_km2"] / max(float(out["building_density_km2"].max()), 1.0)
    center_norm = 1.0 - (out["distance_to_density_core_m"] / max(float(out["distance_to_density_core_m"].max()), 1.0))
    promoted = out["gridcell_id"].isin(promote_ids)
    out.loc[promoted, "sampling_score"] = 0.7 * density_norm[promoted] + 0.3 * center_norm[promoted]
    return out


def choose_with_spacing(part: pd.DataFrame, n: int, metric_crs: str, min_spacing_m: float) -> list[str]:
    ordered = gpd.GeoDataFrame(part.copy(), geometry="geometry", crs=WGS84).to_crs(metric_crs)
    ordered = ordered.sort_values("sampling_score", ascending=False)
    chosen: list[str] = []
    chosen_points = []
    for _, row in ordered.iterrows():
        point = row.geometry.centroid
        if chosen_points and min(point.distance(p) for p in chosen_points) < min_spacing_m:
            continue
        chosen.append(row["gridcell_id"])
        chosen_points.append(point)
        if len(chosen) >= n:
            return chosen
    for _, row in ordered.iterrows():
        gid = row["gridcell_id"]
        if gid not in chosen:
            chosen.append(gid)
            if len(chosen) >= n:
                break
    return chosen


def select_region_sample(
    candidates: gpd.GeoDataFrame,
    *,
    per_region: int,
    quotas: dict[str, int],
    metric_crs: str,
    min_spacing_m: float,
) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    selected_ids: set[str] = set()

    for stratum in STRATA:
        n = quotas.get(stratum, 0)
        part = candidates.loc[candidates["suggested_stratum"] == stratum]
        picks = choose_with_spacing(part, n, metric_crs, min_spacing_m)
        for gid in picks:
            selected.append((gid, stratum))
            selected_ids.add(gid)

    if len(selected) < per_region:
        needed = per_region - len(selected)
        fallback = candidates.loc[
            candidates["suggested_stratum"].isin(STRATA)
            & ~candidates["gridcell_id"].isin(selected_ids)
        ].sort_values("sampling_score", ascending=False)
        for _, row in fallback.head(needed).iterrows():
            selected.append((row["gridcell_id"], f"fill_{row['suggested_stratum']}"))

    if len(selected) < per_region:
        needed = per_region - len(selected)
        sparse = candidates.loc[~candidates["gridcell_id"].isin(selected_ids)].sort_values(
            "n_buildings",
            ascending=False,
        )
        for _, row in sparse.head(needed).iterrows():
            selected.append((row["gridcell_id"], f"fill_{row['suggested_stratum']}"))

    return selected[:per_region]


def write_outputs(
    sample: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    output_dir: Path,
    stem: str,
    *,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        output_dir / f"{stem}.csv": sample.drop(columns="geometry"),
        output_dir / f"{stem}_candidates.csv": candidates.drop(columns="geometry"),
    }
    gpkg_path = output_dir / f"{stem}.gpkg"
    candidates_gpkg_path = output_dir / f"{stem}_candidates.gpkg"
    geojson_path = output_dir / f"{stem}.geojson"
    for path in [*outputs, gpkg_path, candidates_gpkg_path, geojson_path]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        if path.exists():
            path.unlink()
    for path, df in outputs.items():
        df.to_csv(path, index=False)
    sample.to_file(gpkg_path, driver="GPKG")
    sample.to_file(geojson_path, driver="GeoJSON")
    candidates.to_file(candidates_gpkg_path, driver="GPKG")

    print(f"sample:     {display_path(output_dir / f'{stem}.csv')}")
    print(f"candidates: {display_path(output_dir / f'{stem}_candidates.csv')}")
    print(sample.groupby(["region_key", "grid_type"]).size().to_string())


def parse_quotas(value: str) -> dict[str, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError("--quotas must be cbd,residential,industrial_logistics")
    nums = [int(v) for v in parts]
    return dict(zip(STRATA, nums, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--regions", nargs="*")
    parser.add_argument("--per-region", type=int, default=10)
    parser.add_argument("--quotas", default="3,4,3", help="cbd,residential,industrial_logistics counts per region")
    parser.add_argument("--min-coverage-fraction", type=float, default=0.75)
    parser.add_argument("--min-buildings", type=int, default=5)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--min-spacing-m", type=float, default=750.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    selected_regions = args.regions or list(config["regions"].keys())
    quotas = parse_quotas(args.quotas)
    if sum(quotas.values()) != args.per_region:
        raise ValueError("--quotas must sum to --per-region")

    samples: list[gpd.GeoDataFrame] = []
    candidate_frames: list[gpd.GeoDataFrame] = []
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    for region_key in selected_regions:
        region = config["regions"][region_key]
        grids = load_region_grids(args.grid_dir, region_key)
        grids = grids.loc[grids["coverage_fraction"].astype(float) >= args.min_coverage_fraction].copy()
        metric_crs = str(grids["crs_metric"].dropna().iloc[0])
        bounds = tuple(float(v) for v in grids.total_bounds)
        print(f"[{region_key}] fetching/loading OSM features for bounds {bounds}", flush=True)
        osm = fetch_osm_features(region_key, bounds, args.cache_dir)
        metrics = compute_region_metrics(region_key, grids, osm, metric_crs)
        metrics = metrics.merge(
            grids.drop(columns="geometry"),
            on=["gridcell_id", "region_key"],
            how="left",
        )
        metrics = assign_strata(metrics, args.min_buildings)
        metrics = ensure_min_cbd(metrics, quotas["cbd"], args.min_buildings)
        metrics["candidate_rank"] = (
            metrics.groupby("suggested_stratum")["sampling_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        metrics["candidate_pool"] = metrics["candidate_rank"] <= (args.per_region * args.candidate_multiplier)
        candidate_frames.append(metrics)

        region_candidates = metrics.loc[metrics["candidate_pool"]].copy()
        picks = select_region_sample(
            region_candidates,
            per_region=args.per_region,
            quotas=quotas,
            metric_crs=metric_crs,
            min_spacing_m=args.min_spacing_m,
        )
        pick_df = pd.DataFrame(picks, columns=["gridcell_id", "grid_type"])
        region_sample = metrics.merge(pick_df, on="gridcell_id", how="inner")
        region_sample["selected_reason"] = region_sample["grid_type"].map(
            lambda s: f"osm_stratified_{s}"
        )
        samples.append(region_sample)

        print(
            f"[{region_key}] selected {len(region_sample)}; "
            f"strata={region_sample['grid_type'].value_counts().to_dict()}",
            flush=True,
        )

    sample = gpd.GeoDataFrame(pd.concat(samples, ignore_index=True), geometry="geometry", crs=WGS84)
    sample = sample.sort_values(["region_key", "grid_type", "gridcell_id"]).reset_index(drop=True)
    sample.insert(0, "sample_id", [f"VXEV{i:04d}" for i in range(1, len(sample) + 1)])
    sample["sample_seed"] = args.seed
    sample["selected_at_utc"] = now
    sample["human_annotation_status"] = "pending"
    sample["machine_detection_status"] = "pending"
    sample["reviewed_prediction_status"] = "pending"

    candidates = gpd.GeoDataFrame(pd.concat(candidate_frames, ignore_index=True), geometry="geometry", crs=WGS84)
    stem = f"vexcel_eval_grids_osm_stratified_seed{args.seed}_per_region{args.per_region}"
    write_outputs(sample, candidates, args.output_dir, stem, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
