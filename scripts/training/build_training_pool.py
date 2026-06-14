#!/usr/bin/env python3
"""Materialize a provenance-only positive-sample pool manifest.

Phase 1 of the training-set normalization effort (2026-05-29). This script
enumerates every positive annotation polygon registered across all regions
in ``configs/datasets/regions.yaml`` and writes a **provenance manifest** —
NOT chips. Chips are extracted on demand at COCO-export time per rule
``08-runpod-large-files.md``; storing them here would duplicate the tile
pool for no benefit (same design as ``data/negative_pool/manifest.csv``).

Two buckets, split solely on ``mask_trusted`` from the single-source YAML
``data/training_pool/boundary_trust_rules.yaml`` (loaded via
``core.boundary_trust``):

  - ``positive_trusted_manifest.csv``   : mask_trusted == True
  - ``positive_untrusted_manifest.csv`` : mask_trusted == False

The 2-bucket trusted/untrusted view is DERIVED from ``mask_trusted``; it is
layered on top of the A1/A2/A3 × H/R/S/G × T1/T2 model (see
``.claude/rules/07-annotation-semantics.md``), it does not replace them.
``T1 ⊂ trusted``. Tier is NEVER auto-promoted from pool fields.

Fail-closed policy (rule 07): unknown / missing / None ``label_source`` →
``untrusted``. We do NOT call ``export_coco_dataset.mask_trusted_for`` (which
raises on None/unknown); the fail-closed default is implemented here.

label_source derivation
-----------------------
The discovered annotation gpkgs do not carry a ``label_source`` column; they
carry a per-polygon ``source`` column (or none). We derive ``label_source``
from ``source`` using the same enum mapping as the production builder
``core.training.positive_sources._ct_source_to_label_source`` (extracted
2026-06-12 from build_unified_reviewall), **extended and intentionally
DIVERGENT** to cover all observed values across both regions:

  DIVERGENCE (do NOT unify — verified 2026-06-12 architecture review step 8):
  this pool builder is fail-closed (unknown / empty / google_earth source →
  ``None`` or ``legacy_weak_supervision`` → untrusted, keeping the pool a
  complete provenance record), whereas the detector-train loader
  ``_ct_source_to_label_source`` is fail-fast (raises ValueError on unknown).
  This builder also takes ``(schema_type, has_source_column)`` to apply
  schema-aware column-absent defaults, a responsibility the train-loader
  partitions into ``_load_ct_grid_annotations`` instead. The two derivations
  have different output domains by design and are kept separate.

extended to cover all observed values across both regions:

  source value      -> label_source                 -> bucket
  --------------------------------------------------------------
  None / NaN        -> reviewed_prediction          -> untrusted
  'sam_fn_marker'   -> sam_added_true_fn             -> untrusted
  'sam_fn_review'   -> sam_added_true_fn             -> untrusted
  'sam2'            -> human_manual_sam_assisted     -> trusted
  'google_earth'    -> legacy_weak_supervision       -> untrusted
  'reviewed_prediction'        -> reviewed_prediction        -> untrusted
  'human_manual_sam_assisted'  -> human_manual_sam_assisted  -> trusted
  <no source column>-> schema-aware default (see below)

Schema-aware default for files with NO ``source`` column:
  - sam2 schema (CT batch001/002/002b QGIS+GeoSAM manual): the production
    builder defaults these to ``human_manual_sam_assisted`` (trusted) — we
    match that for the sam2 / legacy_ct schema.
  - v4_reviewed schema (JHB ``G*_V4_*`` with no source col): these are
    V3-C/V4 reviewed-prediction outputs → ``reviewed_prediction`` (untrusted).
  - anything else → None → fail-closed untrusted.

Usage:
    python scripts/training/build_training_pool.py \\
        --as-of-date 2026-05-29 \\
        --output-dir data/training_pool
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core import region_registry  # noqa: E402
from core.annotation_loader import discover_annotations, load_annotation_gdf  # noqa: E402
from core.boundary_trust import mask_trusted_map  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────

# Reproducibility: the runtime has no reliable wall-clock, so added_date is
# NOT datetime.now(). The driver passes --as-of-date; this literal is the
# documented fallback used only when the flag is omitted.
DEFAULT_AS_OF_DATE = "2026-05-29"

ARCHIVE_LAYER = "aerial_2023"
ARCHIVE_REASON = (
    "aerial_2023 vintage/GSD mismatch vs vexcel_2024 base (2026-05-29 decision)"
)

# CBD vs Sandton JHB batches, identified by annotation filename suffix
# (confirmed against regions.yaml annotation_source declarations).
CBD_SUFFIX = "_V4_260407"      # G0772–G0926 (== vexcel_2024 coverage 25-grid)
SANDTON_SUFFIX = "_V4_260421"  # G1110–G1254 (aerial_2023 only, no vexcel)

MANIFEST_COLUMNS = [
    "poly_id",
    "region",
    "grid_id",
    "imagery_layer",
    "source_file",
    "source_layer",
    "source_id",
    "label_source",
    "quality_tier",
    "semantic_confidence",
    "boundary_trust",
    "mask_trusted",
    "added_date",
    "archived",
    "archived_reason",
    "notes",
]


# ──────────────────────────────────────────────────────────────────────────
# label_source derivation — a deliberate FORK of
# core.training.positive_sources._ct_source_to_label_source (do NOT unify; see
# the module docstring DIVERGENCE note). This variant is fail-closed (returns
# None on unknown) + schema-aware; the train-loader variant is fail-fast.
# ──────────────────────────────────────────────────────────────────────────

def _source_to_label_source(src, schema_type: str, has_source_column: bool) -> str | None:
    """Map a per-polygon ``source`` value → label_source enum.

    Returns None when provenance cannot be determined (→ fail-closed untrusted).

    Critical distinction (matches the known ``source`` values of
    core.training.positive_sources._ct_source_to_label_source, but diverges on
    the unknown/empty/google_earth tail — see module docstring):
      - ``source`` column PRESENT but null  → V3-C accepted reviewed prediction
        → ``reviewed_prediction`` (untrusted). Many CT batch003/004 ``_SAM2_``
        files carry a null ``source`` column for this reason.
      - ``source`` column ABSENT entirely   → fall back to the schema default:
          * sam2 / legacy_ct manual schema → ``human_manual_sam_assisted`` (trusted)
            (the production builder's `else` branch for files with no source col).
          * v4_reviewed schema (JHB G*_V4 with no source col) → ``reviewed_prediction``.
          * anything else → None → fail-closed untrusted.
    """
    is_null = src is None or (isinstance(src, float) and pd.isna(src)) or str(src).strip() == ""
    if is_null:
        if not has_source_column:
            # Column absent: schema-default provenance.
            if schema_type in ("sam2", "legacy_ct"):
                return "human_manual_sam_assisted"
            if schema_type == "v4_reviewed":
                return "reviewed_prediction"
            return None
        # Column present but this row is null → V3-C accepted reviewed prediction.
        return "reviewed_prediction"
    s = str(src).lower().strip()
    if s in ("sam_fn_marker", "sam_fn_review"):
        return "sam_added_true_fn"
    if s == "sam2":
        return "human_manual_sam_assisted"
    if s == "reviewed_prediction":
        return "reviewed_prediction"
    if s == "human_manual_sam_assisted":
        return "human_manual_sam_assisted"
    if s == "google_earth":
        # Legacy Google Earth weak supervision (JHB01-06, a few CT singletons).
        # Never trusted — fail-closed to the legacy enum.
        return "legacy_weak_supervision"
    # Unknown provenance marker → fail-closed (untrusted) rather than raise,
    # so the pool stays a complete provenance record. Add new values above.
    return None


def _resolve_imagery_layer(grid_id: str, region_key: str) -> str | None:
    """Resolve the imagery layer a grid's annotation_source corresponds to.

    Uses region_registry.resolve_imagery_layer_for_grid, which returns the
    region default layer when it covers the grid, else the first covering
    layer. For JHB this yields ``aerial_2023`` (the default, covers all 100
    census grids) for every G* grid — which is exactly the layer the
    ``G*_V4_*`` reviewed-prediction annotations were produced on, and the
    layer flagged for archival. JHB01-06 → ``aerial_legacy``. CT → ``aerial_2025``.
    """
    try:
        return region_registry.resolve_imagery_layer_for_grid(grid_id, region_key)
    except KeyError:
        return None


def _load_manifest_lookup(manifest_csv: Path) -> dict:
    """Build a (grid_id, annotation_id) -> (quality_tier, semantic_confidence) map.

    annotation_manifest.csv is currently CT-only (3 grids). The join is
    best-effort on (grid_id, annotation_id); rows without a manifest match
    get NA tier/confidence (never fabricated).
    """
    lookup: dict[tuple[str, str], tuple] = {}
    if not manifest_csv.exists():
        print(f"[MANIFEST] {manifest_csv} not found; tier/confidence left NA")
        return lookup
    m = pd.read_csv(manifest_csv, dtype=str)
    for _, row in m.iterrows():
        key = (str(row.get("grid_id", "")), str(row.get("annotation_id", "")))
        lookup[key] = (row.get("quality_tier"), row.get("semantic_confidence"))
    print(
        f"[MANIFEST] loaded {len(lookup)} rows from {manifest_csv.name} "
        f"(grids: {sorted(m['grid_id'].dropna().unique())})"
    )
    return lookup


# ──────────────────────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────────────────────

def build_pool(
    regions: list[str] | None,
    as_of_date: str,
    manifest_csv: Path,
) -> pd.DataFrame:
    mask_trusted = mask_trusted_map()  # label_source -> bool, from single-source YAML
    manifest_lookup = _load_manifest_lookup(manifest_csv)

    entries = discover_annotations(regions=regions)
    rows: list[dict] = []
    missing_label_source = 0
    unknown_source_values: dict[str, int] = {}

    for grid_id, entry in sorted(entries.items()):
        region_key = entry.region_key
        imagery_layer = _resolve_imagery_layer(grid_id, region_key)
        source_file = entry.path.name
        source_layer = entry.annotation_layer or entry.path.stem

        gdf = load_annotation_gdf(entry)
        if len(gdf) == 0:
            continue

        has_source = "source" in gdf.columns
        # Stable per-row id within the file: use the file's own annotation_id
        # column if present, else the 0-based positional index.
        has_ann_id = "annotation_id" in gdf.columns

        for pos_idx, (_, prow) in enumerate(gdf.iterrows()):
            src_val = prow["source"] if has_source else None
            label_source = _source_to_label_source(src_val, entry.schema_type, has_source)

            if label_source is None:
                missing_label_source += 1
                if has_source and src_val is not None and not (
                    isinstance(src_val, float) and pd.isna(src_val)
                ):
                    sv = str(src_val)
                    unknown_source_values[sv] = unknown_source_values.get(sv, 0) + 1

            # Fail-closed: unknown/missing/None label_source → untrusted.
            is_trusted = bool(mask_trusted.get(label_source, False)) if label_source else False
            boundary_trust = "trusted" if is_trusted else "untrusted"

            # source_id = stable in-file row identifier
            if has_ann_id and pd.notna(prow.get("annotation_id")):
                source_id = str(prow["annotation_id"])
            else:
                source_id = str(pos_idx)

            # poly_id scheme: {region}:{grid_id}:{source_layer}:{source_id}
            poly_id = f"{region_key}:{grid_id}:{source_layer}:{source_id}"

            # Manifest join (best-effort, CT-only currently)
            qtier, sconf = manifest_lookup.get(
                (grid_id, source_id), (None, None)
            )

            archived = (imagery_layer == ARCHIVE_LAYER)
            rows.append({
                "poly_id": poly_id,
                "region": region_key,
                "grid_id": grid_id,
                "imagery_layer": imagery_layer if imagery_layer is not None else "",
                "source_file": source_file,
                "source_layer": source_layer,
                "source_id": source_id,
                "label_source": label_source if label_source is not None else "",
                "quality_tier": qtier if qtier is not None else "",
                "semantic_confidence": sconf if sconf is not None else "",
                "boundary_trust": boundary_trust,
                "mask_trusted": is_trusted,
                "added_date": as_of_date,
                "archived": archived,
                "archived_reason": ARCHIVE_REASON if archived else "",
                "notes": "",
            })

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if missing_label_source:
        print(
            f"[FAIL-CLOSED] {missing_label_source} polygons had no resolvable "
            f"label_source → untrusted. Unknown source values: "
            f"{unknown_source_values or '(all were column-absent/null already handled)'}"
        )
    return df


def _print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    n_trusted = int(df["mask_trusted"].sum())
    n_untrusted = total - n_trusted
    print(f"\n[POOL] total positives = {total}")
    print(f"[POOL] trusted   = {n_trusted}")
    print(f"[POOL] untrusted = {n_untrusted}")

    print("\n[POOL] by region (trusted / untrusted):")
    for region, grp in df.groupby("region"):
        t = int(grp["mask_trusted"].sum())
        print(f"  {region:16s}: total={len(grp):5d}  trusted={t:5d}  untrusted={len(grp) - t:5d}")

    print("\n[POOL] by label_source:")
    for ls, grp in df.groupby(df["label_source"].replace("", "<none>")):
        print(f"  {ls:28s}: {len(grp):5d}  (mask_trusted={bool(grp['mask_trusted'].iloc[0])})")

    # Archived (aerial_2023) split CBD vs Sandton
    arch = df[df["archived"]]
    cbd = arch[arch["source_file"].str.contains(CBD_SUFFIX, na=False)]
    sand = arch[arch["source_file"].str.contains(SANDTON_SUFFIX, na=False)]
    other = arch[
        ~arch["source_file"].str.contains(CBD_SUFFIX, na=False)
        & ~arch["source_file"].str.contains(SANDTON_SUFFIX, na=False)
    ]
    print(f"\n[ARCHIVE] aerial_2023 flagged positives = {len(arch)}")
    print(f"  CBD     ({CBD_SUFFIX}, G0772-G0926): {len(cbd)} polygons, "
          f"{cbd['grid_id'].nunique()} grids")
    print(f"  Sandton ({SANDTON_SUFFIX}, G1110-G1254): {len(sand)} polygons, "
          f"{sand['grid_id'].nunique()} grids")
    if len(other):
        print(f"  other aerial_2023: {len(other)} polygons, "
              f"grids={sorted(other['grid_id'].unique())}")
    print(f"  by region: "
          f"{ {r: int(len(g)) for r, g in arch.groupby('region')} }")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regions", nargs="+", default=None,
                        help="Region keys (regions.yaml). Default = all registered regions. "
                             "Region NEVER inferred from grid_id (rule 06).")
    parser.add_argument("--as-of-date", default=DEFAULT_AS_OF_DATE,
                        help=f"ISO date stamped as added_date (default {DEFAULT_AS_OF_DATE}). "
                             "No wall-clock used (reproducibility).")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "training_pool"),
                        help="Output dir for the two manifest CSVs.")
    parser.add_argument("--manifest-csv",
                        default=str(PROJECT_ROOT / "data" / "annotations" / "annotation_manifest.csv"),
                        help="annotation_manifest.csv for tier/confidence join (CT-only currently).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_pool(
        regions=args.regions,
        as_of_date=args.as_of_date,
        manifest_csv=Path(args.manifest_csv).expanduser(),
    )

    trusted_df = df[df["mask_trusted"]].reset_index(drop=True)
    untrusted_df = df[~df["mask_trusted"]].reset_index(drop=True)

    trusted_path = out_dir / "positive_trusted_manifest.csv"
    untrusted_path = out_dir / "positive_untrusted_manifest.csv"
    trusted_df.to_csv(trusted_path, index=False)
    untrusted_df.to_csv(untrusted_path, index=False)

    _print_summary(df)
    print(f"\n[WRITE] {trusted_path}  ({len(trusted_df)} rows)")
    print(f"[WRITE] {untrusted_path}  ({len(untrusted_df)} rows)")


if __name__ == "__main__":
    main()
