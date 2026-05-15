"""Bootstrap data/negative_pool/manifest.csv from cls_pv_thermal_v2.

Reads cls_pv_thermal_v2/subtype_labels.csv (the JHB v3c_sam_mask_geid_2024_02
bucket — currently the only subtype-labeled source) and appends one negative
pool row per non-PV chip. Excludes ``actually_pv_mislabeled`` because those
are GT gaps, not lookalike negatives.

Idempotent: skips chip_ids already present in manifest.csv. Optionally writes
224×224 preview symlinks under ``data/negative_pool/previews/<archetype>/``.

Usage:
    python scripts/training/negative_pool/bootstrap_from_cls_v2.py
    python scripts/training/negative_pool/bootstrap_from_cls_v2.py --no-symlinks
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUBTYPE_CSV = PROJECT_ROOT / "data" / "cls_pv_thermal_v2" / "subtype_labels.csv"
POOL_ROOT = PROJECT_ROOT / "data" / "negative_pool"
MANIFEST_CSV = POOL_ROOT / "manifest.csv"
PREVIEW_ROOT = POOL_ROOT / "previews"
SOURCE_CHIP_ROOTS = [
    PROJECT_ROOT / "data" / "cls_pv_thermal_v2" / "train" / "non_pv",
    PROJECT_ROOT / "data" / "cls_pv_thermal_v2" / "val" / "non_pv",
]

# Maps cls_pv_thermal_v2 source_bucket → (region, imagery_layer)
BUCKET_TO_REGION_LAYER = {
    "johannesburg:v3c_sam_mask_geid_2024_02": ("johannesburg", "geid_2024_02"),
}

EXCLUDED_SUBTYPES = {"actually_pv_mislabeled"}

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
]

# Filename pattern: jhbcbd_v3c_G0772_p0000__ground_road_marking.png
FILENAME_RE = re.compile(
    r"^(?P<prefix>[a-z0-9]+)_(?P<detector>v3c|v4_2)_(?P<grid>G\d+)_p(?P<pred>\d+)__(?P<subtype>[a-z_]+)\.png$"
)


def existing_chip_ids() -> set[str]:
    if not MANIFEST_CSV.exists():
        return set()
    with MANIFEST_CSV.open() as f:
        reader = csv.DictReader(f)
        return {row["chip_id"] for row in reader if row.get("chip_id")}


def find_source_chip(filename: str) -> Path | None:
    for root in SOURCE_CHIP_ROOTS:
        p = root / filename
        if p.exists() or p.is_symlink():
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        help="Skip creating preview symlinks (manifest-only update)",
    )
    parser.add_argument(
        "--source-run-tag",
        default="cls_pv_thermal_v2",
        help="Value to write into the source_run column",
    )
    args = parser.parse_args()

    if not SUBTYPE_CSV.exists():
        print(f"ERROR: {SUBTYPE_CSV} not found", file=sys.stderr)
        return 1

    seen = existing_chip_ids()

    new_rows: list[dict[str, str]] = []
    excluded_count = 0
    unknown_bucket_count = 0
    unparseable_count = 0
    missing_source_count = 0
    today = date.today().isoformat()

    with SUBTYPE_CSV.open() as f:
        for row in csv.DictReader(f):
            subtype = row["subtype"]
            if subtype in EXCLUDED_SUBTYPES:
                excluded_count += 1
                continue

            bucket = row["source_bucket"]
            if bucket not in BUCKET_TO_REGION_LAYER:
                unknown_bucket_count += 1
                continue
            region, imagery_layer = BUCKET_TO_REGION_LAYER[bucket]

            fname = row["chip_filename"]
            m = FILENAME_RE.match(fname)
            if not m:
                unparseable_count += 1
                continue

            grid = m.group("grid")
            detector = m.group("detector")
            pred = m.group("pred")
            chip_id = f"{region}_{grid}_{detector}_p{pred}"

            if chip_id in seen:
                continue
            seen.add(chip_id)

            preview_path = ""
            if not args.no_symlinks:
                src = find_source_chip(fname)
                if src is None:
                    missing_source_count += 1
                else:
                    archetype_dir = PREVIEW_ROOT / subtype
                    archetype_dir.mkdir(parents=True, exist_ok=True)
                    link = archetype_dir / f"{chip_id}.png"
                    if not link.exists():
                        target = src.resolve()
                        try:
                            link.symlink_to(target)
                        except OSError as e:
                            print(f"WARN: symlink failed for {chip_id}: {e}", file=sys.stderr)
                    preview_path = str(link.relative_to(POOL_ROOT))

            new_rows.append(
                {
                    "chip_id": chip_id,
                    "archetype": subtype,
                    "archetype_confidence": "A2",
                    "region": region,
                    "imagery_layer": imagery_layer,
                    "grid_id": grid,
                    "detector": detector,
                    "source_run": args.source_run_tag,
                    "source_pred_id": pred,
                    "bbox_geo_wkt": "",
                    "preview_path": preview_path,
                    "added_date": today,
                    "notes": "",
                }
            )

    if not new_rows:
        print("No new rows to append (manifest already in sync with cls_pv_thermal_v2)")
    else:
        write_header = not MANIFEST_CSV.exists() or MANIFEST_CSV.stat().st_size == 0
        # CSV with header but empty body still counts as "has header"
        if MANIFEST_CSV.exists() and MANIFEST_CSV.stat().st_size > 0:
            with MANIFEST_CSV.open() as f:
                first = f.readline()
            write_header = not first.strip().startswith("chip_id")

        with MANIFEST_CSV.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            if write_header:
                writer.writeheader()
            for row in new_rows:
                writer.writerow(row)
        print(f"Appended {len(new_rows)} rows to {MANIFEST_CSV.relative_to(PROJECT_ROOT)}")

    print("---")
    print(f"  excluded (actually_pv_mislabeled etc.): {excluded_count}")
    print(f"  unknown source_bucket: {unknown_bucket_count}")
    print(f"  unparseable filename: {unparseable_count}")
    print(f"  missing source chip (no symlink): {missing_source_count}")

    # Per-archetype summary
    if new_rows:
        from collections import Counter

        c = Counter(r["archetype"] for r in new_rows)
        print("  per-archetype rows added:")
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<32s} {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
