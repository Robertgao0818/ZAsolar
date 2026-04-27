"""Audit classifier data sources — registry-driven inventory of reviewed
decisions, GT heater audits, and small-FP taxonomy chips.

Enumerates every registered `(region, model_run)` from `configs/datasets/regions.yaml`
via `core.region_registry`, locates `review/detection_review_decisions.csv` and
`review/{grid}_reviewed.gpkg` under each model_run's results_path, and reports
per-region / per-model_run / per-grid / per-class / area-bucketed counts.

Also includes the legacy flat `results/G*/review/` bucket (CT batch 003,
pre-PR3 layout) and external label sources:

  - `results/analysis/gt_heater_audit/*/audit_labels_phase1.csv`
  - `results/analysis/small_fp/*/small_fp_taxonomy_labeled.csv`

Produces `results/analysis/classifier_data_inventory/<run_id>/summary.{json,md}`.

Run:

    python scripts/classifier/audit_cls_sources.py --run-id 2026-04-22
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import fiona

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core import region_registry  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
INVENTORY_ROOT = RESULTS_DIR / "analysis" / "classifier_data_inventory"

AREA_CUTOFF_M2 = 30.0  # matches classifier area_cutoff


@dataclass
class BucketCounts:
    """Reviewed-decision counts for one (region, model_run, grid) bucket."""

    region: str
    model_run: str
    grid_id: str
    results_path: str
    has_gpkg: bool
    has_csv: bool
    small: Counter = field(default_factory=Counter)  # area < cutoff
    all_: Counter = field(default_factory=Counter)
    deprecated: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["small"] = dict(self.small)
        d["all_"] = dict(self.all_)
        return d


def _read_reviewed_gpkg(gpkg: Path) -> tuple[Counter, Counter]:
    """Return (small_counts, all_counts) of review_status values, keyed by
    `area_m2 < AREA_CUTOFF_M2`.
    """
    small: Counter = Counter()
    all_: Counter = Counter()
    with fiona.open(gpkg) as src:
        for feat in src:
            props = feat["properties"]
            status = props.get("review_status") or props.get("status")
            area = props.get("area_m2")
            if not status or area is None:
                continue
            all_[status] += 1
            if area < AREA_CUTOFF_M2:
                small[status] += 1
    return small, all_


def _read_review_csv_only(csv_path: Path) -> tuple[Counter, Counter]:
    """Fallback when no `_reviewed.gpkg` exists — decisions without areas.

    Returns (small=empty Counter, all=full Counter). Small bucket is
    empty-by-construction because area is unknown.
    """
    all_: Counter = Counter()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            status = row.get("status", "")
            if status:
                all_[status] += 1
    return Counter(), all_


def _scan_model_run(region: str, run_id: str, deprecated: bool) -> list[BucketCounts]:
    """Scan `<results_path>/G*/review/` for reviewed data in one model_run."""
    try:
        results_path = region_registry.get_model_run_path(region, run_id)
    except KeyError:
        return []
    if not results_path.exists():
        return []

    buckets: list[BucketCounts] = []
    for grid_dir in sorted(results_path.glob("G*")):
        review_dir = grid_dir / "review"
        if not review_dir.is_dir():
            continue
        csv_path = review_dir / "detection_review_decisions.csv"
        gpkg_candidates = list(review_dir.glob("*_reviewed.gpkg"))
        has_csv = csv_path.exists()
        has_gpkg = bool(gpkg_candidates)
        if not (has_csv or has_gpkg):
            continue

        small: Counter = Counter()
        all_: Counter = Counter()
        notes = ""
        if has_gpkg:
            try:
                small, all_ = _read_reviewed_gpkg(gpkg_candidates[0])
            except Exception as e:  # noqa: BLE001
                notes = f"gpkg read failed: {e}"
                if has_csv:
                    small, all_ = _read_review_csv_only(csv_path)
                    notes += " (fell back to csv; areas unknown)"
        elif has_csv:
            small, all_ = _read_review_csv_only(csv_path)
            notes = "csv-only (no reviewed.gpkg; areas unknown)"

        buckets.append(
            BucketCounts(
                region=region,
                model_run=run_id,
                grid_id=grid_dir.name,
                results_path=str(results_path.relative_to(PROJECT_ROOT)),
                has_gpkg=has_gpkg,
                has_csv=has_csv,
                small=small,
                all_=all_,
                deprecated=deprecated,
                notes=notes,
            )
        )
    return buckets


def _scan_legacy_flat() -> list[BucketCounts]:
    """Legacy CT batch 003 layout: `results/G*/review/*_reviewed.gpkg`.

    Not registered as a model_run in regions.yaml.
    """
    buckets: list[BucketCounts] = []
    for grid_dir in sorted(RESULTS_DIR.glob("G*")):
        if not grid_dir.is_dir():
            continue
        review_dir = grid_dir / "review"
        if not review_dir.is_dir():
            continue
        csv_path = review_dir / "detection_review_decisions.csv"
        gpkg_candidates = list(review_dir.glob("*_reviewed.gpkg"))
        has_csv = csv_path.exists()
        has_gpkg = bool(gpkg_candidates)
        if not (has_csv or has_gpkg):
            continue

        small: Counter = Counter()
        all_: Counter = Counter()
        if has_gpkg:
            try:
                small, all_ = _read_reviewed_gpkg(gpkg_candidates[0])
            except Exception:  # noqa: BLE001
                if has_csv:
                    small, all_ = _read_review_csv_only(csv_path)
        elif has_csv:
            small, all_ = _read_review_csv_only(csv_path)

        buckets.append(
            BucketCounts(
                region="cape_town",
                model_run="legacy_flat_batch003",
                grid_id=grid_dir.name,
                results_path="results",
                has_gpkg=has_gpkg,
                has_csv=has_csv,
                small=small,
                all_=all_,
                deprecated=False,
                notes="pre-PR3 flat results layout",
            )
        )
    return buckets


def _scan_external_labels() -> dict:
    """Count auxiliary label sources that are NOT review decisions."""
    external: dict = {"gt_heater_audit": {}, "small_fp_taxonomy": {}}

    for phase_csv in sorted(
        (RESULTS_DIR / "analysis" / "gt_heater_audit").glob("*/audit_labels_phase1.csv")
    ):
        counts: Counter = Counter()
        small: Counter = Counter()
        with open(phase_csv) as f:
            for row in csv.DictReader(f):
                label = row.get("audit_label", "")
                counts[label] += 1
                try:
                    area = float(row.get("area_m2", ""))
                except ValueError:
                    area = None
                if area is not None and area < AREA_CUTOFF_M2:
                    small[label] += 1
        external["gt_heater_audit"][str(phase_csv.relative_to(PROJECT_ROOT))] = {
            "all": dict(counts),
            "small": dict(small),
        }

    for taxo_csv in sorted(
        (RESULTS_DIR / "analysis" / "small_fp").glob("*/small_fp_taxonomy_labeled.csv")
    ):
        counts: Counter = Counter()
        small: Counter = Counter()
        with open(taxo_csv) as f:
            for row in csv.DictReader(f):
                label = row.get("human_label", "")
                counts[label] += 1
                try:
                    area = float(row.get("area_m2", ""))
                except ValueError:
                    area = None
                if area is not None and area < AREA_CUTOFF_M2:
                    small[label] += 1
        external["small_fp_taxonomy"][str(taxo_csv.relative_to(PROJECT_ROOT))] = {
            "all": dict(counts),
            "small": dict(small),
        }

    return external


def _aggregate(buckets: list[BucketCounts]) -> dict:
    """Roll up per-bucket counts to per-model_run, per-region, total."""
    per_model_run: dict = defaultdict(lambda: {"small": Counter(), "all": Counter(), "grids": 0})
    per_region: dict = defaultdict(lambda: {"small": Counter(), "all": Counter(), "grids": 0})
    grand = {"small": Counter(), "all": Counter(), "grids": 0}

    for b in buckets:
        mr_key = f"{b.region}:{b.model_run}"
        per_model_run[mr_key]["small"] += b.small
        per_model_run[mr_key]["all"] += b.all_
        per_model_run[mr_key]["grids"] += 1
        per_region[b.region]["small"] += b.small
        per_region[b.region]["all"] += b.all_
        per_region[b.region]["grids"] += 1
        grand["small"] += b.small
        grand["all"] += b.all_
        grand["grids"] += 1

    def _finalize(d: dict) -> dict:
        return {
            k: {
                "grids": v["grids"],
                "small": dict(v["small"]),
                "all": dict(v["all"]),
            }
            for k, v in d.items()
        }

    return {
        "per_model_run": _finalize(per_model_run),
        "per_region": _finalize(per_region),
        "grand_total": {
            "grids": grand["grids"],
            "small": dict(grand["small"]),
            "all": dict(grand["all"]),
        },
    }


def _read_deprecated_flags() -> dict[tuple[str, str], bool]:
    """Parse `deprecated: true` flags directly from regions.yaml
    (ModelRunConfig does not currently surface the field).
    """
    import yaml

    flags: dict[tuple[str, str], bool] = {}
    with open(region_registry.REGIONS_YAML) as f:
        raw = yaml.safe_load(f)
    for region_key, region_data in raw.get("regions", {}).items():
        for run_id, run_data in (region_data.get("model_runs") or {}).items():
            flags[(region_key, run_id)] = bool(run_data.get("deprecated", False))
    return flags


def _format_md(
    run_id: str,
    area_cutoff: float,
    buckets: list[BucketCounts],
    agg: dict,
    external: dict,
) -> str:
    lines: list[str] = []
    lines.append(f"# Classifier Data Source Inventory — {run_id}")
    lines.append("")
    lines.append(f"Small-target area cutoff: `{area_cutoff} m²` (matches classifier scope).")
    lines.append("")

    lines.append("## Grand total (reviewed decisions)")
    gt = agg["grand_total"]
    lines.append(f"- Grids with review data: {gt['grids']}")
    lines.append(f"- Small-target decisions (area < {area_cutoff} m²): {dict(gt['small'])}")
    lines.append(f"- All decisions (any area): {dict(gt['all'])}")
    lines.append("")

    lines.append("## Per region")
    for region, counts in sorted(agg["per_region"].items()):
        lines.append(f"### {region}")
        lines.append(f"- Grids: {counts['grids']}")
        lines.append(f"- Small: {counts['small']}")
        lines.append(f"- All: {counts['all']}")
        lines.append("")

    lines.append("## Per (region, model_run)")
    for mr_key, counts in sorted(agg["per_model_run"].items()):
        lines.append(f"### {mr_key}")
        lines.append(f"- Grids: {counts['grids']}")
        lines.append(f"- Small: {counts['small']}")
        lines.append(f"- All: {counts['all']}")
        # Surface deprecation / notes
        flagged = [b for b in buckets if f"{b.region}:{b.model_run}" == mr_key and b.deprecated]
        if flagged:
            lines.append("- **DEPRECATED model_run** (from regions.yaml).")
        lines.append("")

    lines.append("## External label sources (auxiliary)")
    for source_key, items in external.items():
        lines.append(f"### {source_key}")
        for path, counts in items.items():
            lines.append(f"- `{path}`")
            lines.append(f"  - small: {counts['small']}")
            lines.append(f"  - all:   {counts['all']}")
        lines.append("")

    lines.append("## Grids with csv but no reviewed.gpkg (area unknown, excluded from small counts)")
    for b in buckets:
        if b.has_csv and not b.has_gpkg:
            lines.append(f"- {b.region} / {b.model_run} / {b.grid_id} — {b.notes}")
    lines.append("")

    return "\n".join(lines) + "\n"


def run_audit(run_id: str, output_root: Path, area_cutoff: float) -> Path:
    deprecated_flags = _read_deprecated_flags()

    buckets: list[BucketCounts] = []
    for region in region_registry.list_regions():
        for run in region_registry.list_model_runs(region):
            is_dep = deprecated_flags.get((region, run), False)
            buckets.extend(_scan_model_run(region, run, deprecated=is_dep))
    buckets.extend(_scan_legacy_flat())

    agg = _aggregate(buckets)
    external = _scan_external_labels()

    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": run_id,
        "area_cutoff_m2": area_cutoff,
        "aggregate": agg,
        "buckets": [b.to_dict() for b in buckets],
        "external_label_sources": external,
    }
    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    md_path = out_dir / "summary.md"
    md_path.write_text(_format_md(run_id, area_cutoff, buckets, agg, external))

    print(f"[audit] buckets={len(buckets)}  grids={agg['grand_total']['grids']}")
    print(f"[audit] small-area totals: {agg['grand_total']['small']}")
    print(f"[audit] wrote {json_path}")
    print(f"[audit] wrote {md_path}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-id",
        default=date.today().isoformat(),
        help="Run identifier (default: today's ISO date)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=INVENTORY_ROOT,
        help="Inventory root (default: results/analysis/classifier_data_inventory/)",
    )
    parser.add_argument(
        "--area-cutoff",
        type=float,
        default=AREA_CUTOFF_M2,
        help=f"Small-target area cutoff in m² (default: {AREA_CUTOFF_M2})",
    )
    args = parser.parse_args()

    run_audit(args.run_id, args.output_root, args.area_cutoff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
