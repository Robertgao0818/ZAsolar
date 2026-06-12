"""Ingest fresh FP-review sources into data/negative_pool/manifest.csv.

This is the second intended ingest path named in the pool README (the first is
``bootstrap_from_cls_v2.py``).  It unifies new FP sources behind a single
**human/cls-agreement filter** so the monotonic pool only ever accretes
verified non-PV lookalikes — never a model's unilateral guess that might be a
GT gap (a real, unlabelled PV).  The pool is append-only and corrections are
irreversible, so the bar for entry is deliberately high.

Two source adapters (F1-gap plan C-1, docs/plans/2026-06-10-rcnn-f1-gap-review.md):

1. ``gemini_fpcut`` — Gemini FP-review drops (e.g. the JHB full382 2026-06-01
   sweep).  A row is admitted ONLY if BOTH a Gemini verdict says non-PV AND a
   solar_cls subtype label agrees it is non-PV (or an explicit human-review
   record is present).  Gemini alone is not enough — it has mis-cut 107-119
   real PV in CT cross-domain; cls/human agreement is the guard.

2. ``empty_grid_probe`` — cross-domain *verified-non-PV* FPs from
   confirmed-zero-PV empty-grid probes (xdomain60: 10 grids, 33 polygons).  The
   whole grid is verified to contain zero real PV, so every prediction there is
   a verified FP and no per-polygon agreement evidence is needed — the grid-level
   verification IS the agreement.

Hard block (irreversible-pollution guard)
-----------------------------------------
``BFN0126`` and ``DBN0044`` over-paint polygons are NEVER ingested as pure FP:
the xdomain60 write-up warns they may be Li-under-annotated *real* PV.  Any
attempt to ingest geometry from these grids is refused, regardless of source.

Usage::

    # Gemini FP-cut drops, agreement-filtered against a cls subtype CSV:
    python scripts/training/negative_pool/ingest_fp_audit.py gemini_fpcut \\
        --verdict-jsonl <merged_hi.jsonl> \\
        --cls-subtype-csv <subtype_labels.csv> \\
        [--dry-run]

    # cross-domain empty-grid probe FPs (verified non-PV):
    python scripts/training/negative_pool/ingest_fp_audit.py empty_grid_probe \\
        --empty-grid-csv results/analysis/xdomain60/empty_grid_fp.csv \\
        --results-root results/vexcel \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
POOL_ROOT = PROJECT_ROOT / "data" / "negative_pool"
MANIFEST_CSV = POOL_ROOT / "manifest.csv"
TAXONOMY_YAML = POOL_ROOT / "archetype_taxonomy.yaml"

# manifest column order — append-only, mirrors backfill_geometry.py output.
MANIFEST_COLUMNS = [
    "chip_id",
    "archetype",
    "archetype_confidence",
    "region",
    "imagery_layer",
    "grid_id",
    "detector",
    "source_run",
    "source_pred_id",
    "bbox_geo_wkt",
    "preview_path",
    "added_date",
    "notes",
    "training_eligible",
]

# Grids that must NEVER be ingested as pure FP (possible Li-under-annotated PV).
# Pool pollution is irreversible — block at ingest, covered by a unit test.
BLOCKED_GRIDS = {"BFN0126", "DBN0044"}

# Gemini lookalike_type -> negative-pool archetype (controlled vocabulary).
LOOKALIKE_TO_ARCHETYPE = {
    "skylight": "skylight_roof_window",
    "skylight_roof_window": "skylight_roof_window",
    "water_heater": "solar_thermal_water_heater",
    "solar_thermal": "solar_thermal_water_heater",
    "solar_thermal_water_heater": "solar_thermal_water_heater",
    "hvac": "hvac_rooftop_equipment",
    "hvac_rooftop_equipment": "hvac_rooftop_equipment",
    "corrugated_metal": "corrugated_metal_roof",
    "metal_roof": "corrugated_metal_roof",
    "shadow": "roof_shadow_dark_fixture",
    "road_marking": "ground_road_marking",
    "pergola": "pergola_carport_shadow",
    "carport": "pergola_carport_shadow",
}
DEFAULT_ARCHETYPE = "other_unknown"


# ── shared helpers ──────────────────────────────────────────────────────────

def _valid_archetypes() -> set[str]:
    import yaml

    if not TAXONOMY_YAML.exists():
        return set(LOOKALIKE_TO_ARCHETYPE.values()) | {DEFAULT_ARCHETYPE}
    d = yaml.safe_load(TAXONOMY_YAML.read_text())
    arch = d.get("archetypes", {})
    return set(arch.keys()) if isinstance(arch, dict) else set(arch)


def existing_chip_ids() -> set[str]:
    if not MANIFEST_CSV.exists():
        return set()
    with MANIFEST_CSV.open(newline="") as f:
        return {row["chip_id"] for row in csv.DictReader(f) if row.get("chip_id")}


def assert_not_blocked(grid_id: str) -> None:
    """Raise if a grid is on the irreversible-pollution block list.

    This is the *hard* guard for programmatic/external callers that ingest a
    single known grid and want a loud failure (e.g. a one-off script handing in
    an explicit grid list). The bulk adapters here deliberately do NOT call it:
    they iterate over many verdict records / grids and skip blocked grids with a
    counted ``continue`` (``n_blocked``) so one blocked grid in a batch does not
    abort the whole ingest. The block list itself is enforced in both paths via
    ``BLOCKED_GRIDS`` (``test_gemini_fpcut_skips_blocked_grid`` covers the soft
    path; ``test_assert_not_blocked_*`` covers this hard guard).
    """
    if grid_id in BLOCKED_GRIDS:
        raise ValueError(
            f"refusing to ingest grid {grid_id}: on BLOCKED_GRIDS "
            f"(possible Li-under-annotated real PV; pool pollution is "
            f"irreversible). See F1-gap plan C-1."
        )


def append_rows(new_rows: list[dict], *, dry_run: bool) -> None:
    """Append rows to the manifest (idempotent on chip_id), unless dry-run."""
    if not new_rows:
        print("No new rows to append.")
        return
    if dry_run:
        print(f"(dry-run) would append {len(new_rows)} rows")
        return
    write_header = not MANIFEST_CSV.exists() or MANIFEST_CSV.stat().st_size == 0
    if MANIFEST_CSV.exists() and MANIFEST_CSV.stat().st_size > 0:
        with MANIFEST_CSV.open() as f:
            write_header = not f.readline().strip().startswith("chip_id")
    with MANIFEST_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in new_rows:
            writer.writerow({k: r.get(k, "") for k in MANIFEST_COLUMNS})
    print(f"Appended {len(new_rows)} rows to "
          f"{MANIFEST_CSV.relative_to(PROJECT_ROOT)}")


# ── agreement filter ────────────────────────────────────────────────────────

def load_cls_nonpv_index(cls_subtype_csv: Path) -> dict[tuple[str, str, str], str]:
    """Index {(grid_id, detector, pred_idx): subtype} for non-PV cls chips.

    The cls subtype CSV (solar_cls cls_pv_thermal_v2) carries a ``subtype`` per
    chip; ``actually_pv_mislabeled`` means cls says it IS PV, so we exclude it
    from the non-PV index (its presence blocks agreement).  Returns only the
    chips cls calls non-PV, keyed so the Gemini verdict can be joined.
    """
    idx: dict[tuple[str, str, str], str] = {}
    import re

    fname_re = re.compile(
        r"^[a-z0-9]+_(?P<det>v3c|v4_2)_(?P<grid>G\d+)_p(?P<pred>\d+)__"
    )
    with cls_subtype_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            subtype = row.get("subtype", "")
            m = fname_re.match(row.get("chip_filename", ""))
            if not m:
                continue
            key = (m.group("grid"), m.group("det"), str(int(m.group("pred"))))
            if subtype == "actually_pv_mislabeled":
                # cls disagrees (says PV) -> record as a veto sentinel
                idx[key] = "__pv__"
            else:
                idx.setdefault(key, subtype)
    return idx


def gemini_says_nonpv(rec: dict) -> bool:
    """True iff the Gemini verdict record is a confident non-PV drop."""
    action = rec.get("production_action")
    label = rec.get("label")
    pv = rec.get("pv_present")
    # explicit non-PV drop; require_human_review must be cleared
    return (
        action == "drop"
        and label in ("not_pv", "non_pv")
        and pv is False
        and not rec.get("requires_human_review", False)
    )


# ── adapter: Gemini FP-cut drops ────────────────────────────────────────────

def ingest_gemini_fpcut(args) -> int:
    verdict_path = Path(args.verdict_jsonl)
    if not verdict_path.exists():
        print(f"MISSING: verdict jsonl not found: {verdict_path}", file=sys.stderr)
        return 2

    cls_idx: dict[tuple[str, str, str], str] | None = None
    if args.cls_subtype_csv:
        cls_csv = Path(args.cls_subtype_csv)
        if not cls_csv.exists():
            print(f"MISSING: cls subtype csv not found: {cls_csv}", file=sys.stderr)
            return 2
        cls_idx = load_cls_nonpv_index(cls_csv)
        print(f"Loaded cls non-PV index: {len(cls_idx)} chips")

    valid_arch = _valid_archetypes()
    seen = existing_chip_ids()
    today = date.today().isoformat()

    n_total = 0
    n_gemini_nonpv = 0
    n_blocked = 0
    n_no_cls_agreement = 0
    n_dup = 0
    new_rows: list[dict] = []

    with verdict_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_total += 1
            if not gemini_says_nonpv(rec):
                continue
            n_gemini_nonpv += 1

            grid_id = rec.get("grid_id", "")
            if grid_id in BLOCKED_GRIDS:
                n_blocked += 1
                continue

            region = rec.get("region_key") or rec.get("region") or ""
            detector = rec.get("detector", "unifiedA")
            pred_id = str(rec.get("pred_id", ""))

            # ── human/cls-agreement filter ──
            # admit only if cls also says non-PV (or an explicit human record).
            has_human = bool(rec.get("human_reviewed") or rec.get("human_label"))
            agrees = has_human
            if not agrees and cls_idx is not None:
                key = (grid_id, detector, pred_id)
                sub = cls_idx.get(key)
                # also try v3c/v4_2 fallbacks if the verdict detector differs
                if sub is None:
                    for det in ("v3c", "v4_2"):
                        sub = cls_idx.get((grid_id, det, pred_id))
                        if sub is not None:
                            break
                agrees = sub is not None and sub != "__pv__"
            if not agrees:
                n_no_cls_agreement += 1
                continue

            lookalike = (rec.get("lookalike_type") or "").lower()
            archetype = LOOKALIKE_TO_ARCHETYPE.get(lookalike, DEFAULT_ARCHETYPE)
            if archetype not in valid_arch:
                archetype = DEFAULT_ARCHETYPE

            chip_id = f"{region}_{grid_id}_{detector}_p{pred_id}"
            if chip_id in seen:
                n_dup += 1
                continue
            seen.add(chip_id)

            new_rows.append({
                "chip_id": chip_id,
                "archetype": archetype,
                "archetype_confidence": "A2",
                "region": region,
                "imagery_layer": args.imagery_layer or "",
                "grid_id": grid_id,
                "detector": detector,
                "source_run": args.source_run,
                "source_pred_id": pred_id,
                "bbox_geo_wkt": "",  # geometry backfilled separately
                "preview_path": "",
                "added_date": today,
                "notes": f"gemini_fpcut+cls_agreement; lookalike={lookalike}",
                "training_eligible": "true",
            })

    print("--- gemini_fpcut ingest summary ---")
    print(f"  verdict records:            {n_total}")
    print(f"  gemini non-PV drops:        {n_gemini_nonpv}")
    print(f"  blocked grids (BFN/DBN):    {n_blocked}")
    print(f"  no cls/human agreement:     {n_no_cls_agreement}")
    print(f"  duplicates (already in pool): {n_dup}")
    print(f"  admitted (new rows):        {len(new_rows)}")
    if new_rows:
        c = Counter(r["archetype"] for r in new_rows)
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<32s} {v}")
    append_rows(new_rows, dry_run=args.dry_run)
    return 0


# ── adapter: cross-domain empty-grid probe FPs ──────────────────────────────

def ingest_empty_grid_probe(args) -> int:
    csv_path = Path(args.empty_grid_csv)
    if not csv_path.exists():
        print(f"MISSING: empty-grid csv not found: {csv_path}", file=sys.stderr)
        return 2

    import geopandas as gpd

    from core import region_registry as rr
    from core.grid_utils import normalize_region

    results_root = Path(args.results_root)
    valid_arch = _valid_archetypes()
    seen = existing_chip_ids()
    today = date.today().isoformat()

    n_grids = 0
    n_blocked = 0
    n_polys = 0
    n_missing_gpkg = 0
    n_dup = 0
    new_rows: list[dict] = []

    with csv_path.open(newline="") as f:
        grid_rows = list(csv.DictReader(f))

    for grow in grid_rows:
        region = normalize_region(grow.get("region")) or grow.get("region", "")
        grid_id = grow.get("grid_id", "")
        if int(grow.get("n_pred", "0") or 0) == 0:
            continue
        if grid_id in BLOCKED_GRIDS:
            n_blocked += 1
            continue
        n_grids += 1

        # resolve the grid's predictions gpkg (whole-grid verified non-PV)
        gpkg = _find_grid_gpkg(results_root, grid_id)
        if gpkg is None:
            print(f"[empty_grid_probe] {grid_id}: predictions gpkg not found "
                  f"under {results_root}; skip")
            n_missing_gpkg += 1
            continue

        # imagery layer for this region (single vexcel layer per xdomain region)
        try:
            layers = list(rr.get_region_config(region).imagery_layers.keys())
            imagery_layer = layers[0] if len(layers) == 1 else ""
        except Exception:  # noqa: BLE001
            imagery_layer = ""

        gdf = gpd.read_file(gpkg)
        gdf_4326 = gdf.to_crs("EPSG:4326")
        from shapely.geometry import box

        for i, geom in enumerate(gdf_4326.geometry):
            if geom is None or geom.is_empty:
                continue
            n_polys += 1
            minx, miny, maxx, maxy = geom.bounds
            wkt = box(minx, miny, maxx, maxy).wkt
            chip_id = f"{region}_{grid_id}_xdomainA_p{i}"
            if chip_id in seen:
                n_dup += 1
                continue
            seen.add(chip_id)
            new_rows.append({
                "chip_id": chip_id,
                "archetype": DEFAULT_ARCHETYPE,
                # A3, not A2: these rows carry only a GRID-level zero-PV
                # verification (whole grid confirmed to contain no PV), not a
                # per-polygon archetype review — every prediction is a verified
                # FP but its specific lookalike archetype is uncharacterised.
                # A3 = weak/un-per-chip-reviewed (ANNOTATION_SPEC Two-Axis,
                # rule 07: do not overstate archetype confidence). It also keeps
                # them below the A2 floor most HN specs use, so they cannot
                # silently enter a bundle even if flipped training_eligible.
                "archetype_confidence": "A3",
                "region": region,
                "imagery_layer": imagery_layer,
                "grid_id": grid_id,
                "detector": "unifiedA",
                "source_run": args.source_run,
                "source_pred_id": str(i),
                "bbox_geo_wkt": wkt,
                "preview_path": "",
                "added_date": today,
                "notes": "xdomain60 confirmed-zero-PV empty-grid probe FP "
                         "(grid-level verification only; no per-polygon "
                         "archetype review)",
                # cross-domain verified non-PV in a NEW appearance domain; the
                # imagery-layer balance gate (C-1) still governs whether it
                # enters a training bundle, so flag provenance-only by default.
                "training_eligible": "false",
            })

    print("--- empty_grid_probe ingest summary ---")
    print(f"  grids processed:            {n_grids}")
    print(f"  blocked grids (BFN/DBN):    {n_blocked}")
    print(f"  missing gpkg:               {n_missing_gpkg}")
    print(f"  polygons (verified non-PV): {n_polys}")
    print(f"  duplicates:                 {n_dup}")
    print(f"  admitted (new rows):        {len(new_rows)}")
    append_rows(new_rows, dry_run=args.dry_run)
    return 0


def _find_grid_gpkg(results_root: Path, grid_id: str) -> Path | None:
    """Find <grid>/predictions_metric.gpkg under results_root (recursive)."""
    if not results_root.exists():
        return None
    hits = list(results_root.glob(f"**/{grid_id}/predictions_metric.gpkg"))
    return hits[0] if hits else None


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="report counts without writing the manifest")
    sub = parser.add_subparsers(dest="adapter", required=True)

    g = sub.add_parser("gemini_fpcut", help="Gemini FP-review drops")
    g.add_argument("--verdict-jsonl", required=True)
    g.add_argument("--cls-subtype-csv", default=None,
                   help="solar_cls subtype CSV for the agreement filter")
    g.add_argument("--imagery-layer", default="",
                   help="imagery layer of the drops (e.g. vexcel_2024)")
    g.add_argument("--source-run", default="gemini_fpcut")
    g.set_defaults(func=ingest_gemini_fpcut)

    e = sub.add_parser("empty_grid_probe", help="cross-domain verified non-PV")
    e.add_argument("--empty-grid-csv", required=True)
    e.add_argument("--results-root", default="results/vexcel")
    e.add_argument("--source-run", default="xdomain60_empty_probe")
    e.set_defaults(func=ingest_empty_grid_probe)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
