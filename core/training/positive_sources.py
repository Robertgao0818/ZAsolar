"""Positive-source loaders for the unified_reviewall training pool.

Extracted (2026-06-12, architecture review step 8) from
``scripts/training/build_unified_reviewall.py`` so the declarative dataset
builder (``pipeline.dataset_builder`` v2 path) can drive the SAME CT/JHB
selection + label_source derivation through public functions instead of
importing the DEPRECATED bespoke script's privates and monkeypatching its
module-level ``JHB_REVIEW_ROOT`` global.

Behavioural contract (byte-equivalence): the function bodies are moved
**verbatim** from the bespoke builder. Two non-behavioural changes:

  1. ``review_root`` is an **explicit parameter** of
     ``_load_jhb_grid_annotations`` (was a module-level global the caller
     monkeypatched + restored). The default preserves the bespoke constant.
  2. The CT-entries cache is **keyed by the discovery region tuple** (was a
     single nullable module global). Same output, but concurrency-safe and no
     mutable-global dependency.

``scripts/training/build_unified_reviewall.py`` now re-imports these names
(thin CLI shell); ``pipeline.dataset_builder`` imports them directly and
passes ``review_root`` explicitly. See the build_unified_reviewall header for
the DEPRECATED status + byte-diff verdict.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box as shapely_box

from core.grid_utils import resolve_tiles_dir
from core.annotation_loader import (
    discover_annotations, load_annotation_gdf,
)
from export_coco_dataset import _MASK_TRUSTED


# Repo root inferred from this module's location (core/training/ → parents[2]).
# Matches the bespoke builder's ``PROJECT_ROOT = parents[2]`` (which was
# scripts/training/, also two levels below repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default JHB review-product root. Callers may override via the ``review_root``
# parameter of ``_load_jhb_grid_annotations`` (the v2 builder passes the spec's
# ``review_root``); the bespoke CLI passes this constant.
JHB_REVIEW_ROOT = (
    PROJECT_ROOT / "results" / "johannesburg" / "v3c_vexcel_2024_ch1_sample"
)

UNTRUSTED_SOURCES = {k for k, v in _MASK_TRUSTED.items() if not v}
TRUSTED_SOURCES = {k for k, v in _MASK_TRUSTED.items() if v}


def _tiles_for(grid_id: str, region: str, imagery_layer: str | None = None) -> list[Path]:
    tiles_dir = resolve_tiles_dir(grid_id, region=region, imagery_layer=imagery_layer)
    if tiles_dir.is_file():
        return [tiles_dir]
    tiles = sorted(tiles_dir.glob(f"{grid_id}_*_*_geo.tif"))
    if not tiles:
        tiles = sorted(p for p in tiles_dir.glob(f"{grid_id}_*.tif")
                       if "mosaic" not in p.stem)
    return tiles


def _assign_intersections(annotations: gpd.GeoDataFrame,
                          tiles: list[Path]) -> dict[str, list[int]]:
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


def _load_jhb_grid_annotations(grid_id: str, tile_crs,
                               review_root: Path | None = None) -> gpd.GeoDataFrame:
    """Combine V3C-correct reviewed predictions + browser SAM_added FN into
    one GDF with label_source tagged per row.

    ``review_root`` is the JHB review-product root containing
    ``<grid>/review/<grid>_{reviewed,sam_added}.gpkg``. Defaults to the
    module-level ``JHB_REVIEW_ROOT`` constant when omitted (preserves the
    bespoke builder's behaviour). The v2 dataset builder passes the spec's
    ``review_root`` explicitly instead of monkeypatching a global.
    """
    if review_root is None:
        review_root = JHB_REVIEW_ROOT
    review_dir = review_root / grid_id / "review"
    reviewed_path = review_dir / f"{grid_id}_reviewed.gpkg"
    sam_added_path = review_dir / f"{grid_id}_sam_added.gpkg"

    parts = []
    if reviewed_path.exists():
        g_rev = gpd.read_file(reviewed_path)
        if "review_status" in g_rev.columns:
            g_rev = g_rev[g_rev["review_status"] == "correct"].copy()
        else:
            print(f"[WARN] {reviewed_path.name} missing review_status column; keeping all rows")
        g_rev["label_source"] = "reviewed_prediction"
        parts.append(g_rev)
    else:
        print(f"[WARN] missing {reviewed_path}")

    if sam_added_path.exists():
        g_sam = gpd.read_file(sam_added_path)
        g_sam["label_source"] = "sam_added_browser"
        parts.append(g_sam)
    else:
        print(f"[WARN] missing {sam_added_path}")

    if not parts:
        return gpd.GeoDataFrame(columns=["geometry", "label_source"], crs="EPSG:4326")

    common_cols = set.intersection(*(set(p.columns) for p in parts))
    common_cols = list(common_cols | {"label_source", "geometry"})
    parts = [p[[c for c in common_cols if c in p.columns]].copy() for p in parts]
    gdf = pd.concat(parts, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=parts[0].crs)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(tile_crs).reset_index(drop=True)


def _ct_source_to_label_source(src):
    """Map CT-batch GeoPackage 'source' value → label_source enum.

    CT batch003/004 schemas use NaN for V3C-reviewed-accepted and
    'sam_fn_marker' for non-interactive FN catches.  CT early SAM2
    (G1189/G1190/G1238) uses 'sam2'.  CT batch001/002/002b have no
    'source' column at all (default to human_manual_sam_assisted).
    """
    if src is None or (isinstance(src, float) and pd.isna(src)):
        return "reviewed_prediction"     # V3-C accepted; halo-prone → untrusted
    s = str(src).lower().strip()
    # CT FN-补切 family: all non-interactive batch SAM cut (pre-browser-tool
    # 2026-04-13). marker = clicked marker that triggers a SAM cut at the
    # marker location; review = sam_fn_review.py CLI batch tool. Both produce
    # boundary noise without per-instance human refine → untrusted.
    if s in ("sam_fn_marker", "sam_fn_review"):
        return "sam_added_true_fn"
    if s == "sam2":
        return "human_manual_sam_assisted"
    if s == "reviewed_prediction":
        return "reviewed_prediction"
    if s == "human_manual_sam_assisted":
        return "human_manual_sam_assisted"
    # Unknown provenance marker → fail fast. Conservative default to
    # human_manual_sam_assisted (trusted) was wrong: it silently marked
    # halo-prone batch-SAM outputs as trusted, defeating the mask_trusted
    # gate. Add new sources here explicitly with their correct
    # trusted/untrusted classification.
    raise ValueError(
        f"unknown CT 'source' value {s!r}; map it explicitly above with "
        f"a trusted/untrusted classification (see export_coco_dataset."
        f"_MASK_TRUSTED for the enum)"
    )


@lru_cache(maxsize=None)
def _ct_entries_for(regions: tuple[str, ...]):
    """Discover annotation entries for ``regions``, cached by the region key.

    Replaces the bespoke builder's single nullable ``_CT_ENTRIES_CACHE``
    module global with a per-key cache (concurrency-safe; multiple region
    sets coexist). ``regions`` must be a hashable tuple.
    """
    return discover_annotations(regions=list(regions))


def _ct_entries():
    """Discover cape_town annotation entries (cached). Byte-equivalent to the
    bespoke ``_ct_entries`` — same single-region discovery, just routed
    through the per-key cache."""
    return _ct_entries_for(("cape_town",))


def _load_ct_grid_annotations(grid_id: str, tile_crs) -> gpd.GeoDataFrame:
    """Load CT annotations + tag label_source per row."""
    entries = _ct_entries()
    if grid_id not in entries:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    gdf = load_annotation_gdf(entries[grid_id])
    if "label_source" in gdf.columns:
        pass  # already tagged
    elif "source" in gdf.columns:
        gdf["label_source"] = gdf["source"].apply(_ct_source_to_label_source)
    else:
        # batch001/002/002b SAM2-QGIS manual schema → all human_manual_sam_assisted
        gdf["label_source"] = "human_manual_sam_assisted"
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(tile_crs).reset_index(drop=True)


def _per_record_summary(records, kind="train"):
    rows = []
    for rec in records:
        n_trusted = (
            rec["annots"]["label_source"].isin(TRUSTED_SOURCES).sum()
            if "label_source" in rec["annots"].columns else 0
        )
        n_untrusted = (
            rec["annots"]["label_source"].isin(UNTRUSTED_SOURCES).sum()
            if "label_source" in rec["annots"].columns else 0
        )
        rows.append({
            "split": rec["split"],
            "region": rec["region"],
            "grid_id": rec["grid_id"],
            "n_polygons": len(rec["annots"]),
            "n_trusted": int(n_trusted),
            "n_untrusted": int(n_untrusted),
            "n_tiles": len(rec["tiles"]),
        })
    return rows


def _selected_annotations_from_records(records: list[dict]) -> list[dict]:
    """Flatten per-record annotation GeoDataFrames into a per-polygon
    provenance list for the build manifest.

    ``source_id`` is the positional row index within the record's annotation
    GeoDataFrame (which has been re-indexed via ``reset_index(drop=True)`` in
    the loaders), matching the row-index-as-id convention used elsewhere in
    the pipeline (e.g. predictions_metric.gpkg). ``source_file`` is resolved
    from ``label_source`` via the record's ``source_files`` map.
    """
    selected: list[dict] = []
    for rec in records:
        annots = rec["annots"]
        src_map = rec.get("source_files", {})
        has_label_source = "label_source" in annots.columns
        # CT records carry a single source file keyed by None.
        default_src = src_map.get(None)
        for pos, (_, row) in enumerate(annots.iterrows()):
            label_source = (
                str(row["label_source"]) if has_label_source else None
            )
            src = src_map.get(label_source, default_src)
            selected.append({
                "region": rec["region"],
                "grid_id": rec["grid_id"],
                "imagery_layer": rec["imagery_layer"],
                "split": rec["split"],
                "label_source": label_source,
                "source_file": (
                    _src_rel(src) if src is not None else None
                ),
                "source_id": pos,
            })
    return selected


def _src_rel(path: Path) -> str:
    """Repo-relative path string when possible, else absolute."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
