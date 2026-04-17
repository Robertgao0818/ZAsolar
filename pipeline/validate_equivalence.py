"""Mechanical equivalence checker for dataset builds.

Compares a new dataset build against a baseline build and reports
pass/fail on key metrics.  This is NOT a manual eyeball check — it
produces a structured comparison report covering:

- Image / annotation counts (train + val)
- Positive / negative / HN chip breakdown
- Per-grid annotation composition
- File-name set equality (train + val)
- Provenance CSV row counts

Usage::

    python -m pipeline.validate_equivalence \\
        --new /tmp/test_v4_1 \\
        --baseline /mnt/d/ZAsolar/coco_v4_1_hn
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


def _load_coco_stats(build_dir: Path) -> dict:
    """Extract key stats from a COCO dataset directory."""
    stats: dict = {"dir": str(build_dir)}

    for split in ("train", "val"):
        json_path = build_dir / f"{split}.json"
        if not json_path.exists():
            stats[f"{split}_exists"] = False
            continue

        with open(json_path) as f:
            data = json.load(f)

        images = data.get("images", [])
        annotations = data.get("annotations", [])

        n_pos = sum(1 for img in images if img.get("positive", True))
        n_neg = len(images) - n_pos

        # Count HN chips by ID range
        n_hn_b3 = sum(1 for img in images if 900000 <= img.get("id", 0) < 950000)
        n_hn_b4 = sum(1 for img in images if img.get("id", 0) >= 950000)

        # File-name set for membership comparison
        file_names = sorted(img.get("file_name", "") for img in images)

        # Per-grid annotation counts (from annotations, not images)
        grid_annot_counts: Counter = Counter()
        img_id_to_file = {img["id"]: img.get("file_name", "") for img in images}
        for ann in annotations:
            fname = img_id_to_file.get(ann["image_id"], "")
            # file_name typically starts with grid_id, e.g. "G1238_tile_..."
            grid_id = fname.split("_")[0] if fname else "unknown"
            grid_annot_counts[grid_id] += 1

        stats[f"{split}_exists"] = True
        stats[f"{split}_images"] = len(images)
        stats[f"{split}_annotations"] = len(annotations)
        stats[f"{split}_positive"] = n_pos
        stats[f"{split}_negative"] = n_neg
        stats[f"{split}_hn_batch003"] = n_hn_b3
        stats[f"{split}_hn_batch004"] = n_hn_b4
        stats[f"{split}_file_names"] = file_names
        stats[f"{split}_grid_annot_counts"] = dict(grid_annot_counts)

    # Provenance CSV row counts
    for prov_name in ("train_provenance.csv", "val_provenance.csv", "hn_provenance.csv"):
        prov_path = build_dir / prov_name
        if prov_path.exists():
            with open(prov_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                n_rows = sum(1 for _ in reader) - 1  # subtract header
            stats[f"prov_{prov_name.replace('.csv', '')}_rows"] = max(0, n_rows)
        else:
            stats[f"prov_{prov_name.replace('.csv', '')}_rows"] = None

    return stats


def check_equivalence(
    new_dir: Path,
    baseline_dir: Path,
    *,
    tolerance: int = 0,
) -> dict:
    """Compare two COCO dataset builds.

    Args:
        new_dir: Path to the new build.
        baseline_dir: Path to the baseline build.
        tolerance: Allowed difference in counts (default: exact match).

    Returns:
        Dict with per-check pass/fail and details.
    """
    new_stats = _load_coco_stats(new_dir)
    base_stats = _load_coco_stats(baseline_dir)

    checks: list[dict] = []
    all_pass = True

    def _add_check(description: str, passed: bool, detail: str) -> None:
        nonlocal all_pass
        checks.append({"check": description, "passed": passed, "detail": detail})
        if not passed:
            all_pass = False

    # ── Count-based checks ───────────────────────────────────────────
    count_keys = [
        ("train_images", "Train image count"),
        ("val_images", "Val image count"),
        ("train_annotations", "Train annotation count"),
        ("val_annotations", "Val annotation count"),
        ("train_positive", "Train positive chips"),
        ("train_negative", "Train negative chips (easy-neg + HN)"),
        ("train_hn_batch003", "Train HN batch 003 chips"),
        ("train_hn_batch004", "Train HN batch 004 chips"),
    ]

    for key, description in count_keys:
        new_val = new_stats.get(key, "MISSING")
        base_val = base_stats.get(key, "MISSING")

        if new_val == "MISSING" or base_val == "MISSING":
            _add_check(description, False, f"new={new_val}, baseline={base_val}")
        else:
            diff = abs(new_val - base_val)
            _add_check(
                description,
                diff <= tolerance,
                f"new={new_val}, baseline={base_val}, diff={new_val - base_val}",
            )

    # ── File-name set membership ─────────────────────────────────────
    for split in ("train", "val"):
        new_names = set(new_stats.get(f"{split}_file_names", []))
        base_names = set(base_stats.get(f"{split}_file_names", []))

        if not new_names and not base_names:
            _add_check(f"{split.title()} file_name set", True, "both empty")
            continue

        only_new = new_names - base_names
        only_base = base_names - new_names
        passed = len(only_new) == 0 and len(only_base) == 0

        detail_parts = [f"|new|={len(new_names)}, |baseline|={len(base_names)}"]
        if only_new:
            sample = sorted(only_new)[:5]
            detail_parts.append(f"only_in_new({len(only_new)})={sample}")
        if only_base:
            sample = sorted(only_base)[:5]
            detail_parts.append(f"only_in_baseline({len(only_base)})={sample}")

        _add_check(f"{split.title()} file_name set", passed, "; ".join(detail_parts))

    # ── Per-grid annotation composition ──────────────────────────────
    for split in ("train",):  # val typically has fewer grids, train is the key check
        new_grid = new_stats.get(f"{split}_grid_annot_counts", {})
        base_grid = base_stats.get(f"{split}_grid_annot_counts", {})
        all_grids = sorted(set(new_grid.keys()) | set(base_grid.keys()))

        mismatches = []
        for gid in all_grids:
            nv = new_grid.get(gid, 0)
            bv = base_grid.get(gid, 0)
            if abs(nv - bv) > tolerance:
                mismatches.append(f"{gid}(new={nv},base={bv})")

        passed = len(mismatches) == 0
        if passed:
            detail = f"{len(all_grids)} grids, all match"
        else:
            detail = f"{len(mismatches)}/{len(all_grids)} grids differ: {mismatches[:10]}"

        _add_check(f"{split.title()} per-grid annotation counts", passed, detail)

    # ── Provenance row counts ────────────────────────────────────────
    for prov_key in ("prov_train_provenance_rows", "prov_val_provenance_rows",
                     "prov_hn_provenance_rows"):
        label = prov_key.replace("prov_", "").replace("_rows", "").replace("_", " ").title()
        new_val = new_stats.get(prov_key)
        base_val = base_stats.get(prov_key)

        if new_val is None and base_val is None:
            _add_check(f"{label} rows", True, "both absent")
        elif new_val is None or base_val is None:
            _add_check(f"{label} rows", False,
                       f"new={'absent' if new_val is None else new_val}, "
                       f"baseline={'absent' if base_val is None else base_val}")
        else:
            diff = abs(new_val - base_val)
            _add_check(f"{label} rows", diff <= tolerance,
                       f"new={new_val}, baseline={base_val}, diff={new_val - base_val}")

    return {
        "verdict": "PASS" if all_pass else "FAIL",
        "tolerance": tolerance,
        "new_dir": str(new_dir),
        "baseline_dir": str(baseline_dir),
        "checks": checks,
        "new_stats": {k: v for k, v in new_stats.items()
                      if not k.endswith("_file_names")},
        "baseline_stats": {k: v for k, v in base_stats.items()
                           if not k.endswith("_file_names")},
    }


def print_report(result: dict) -> None:
    """Pretty-print an equivalence report."""
    print(f"\n{'=' * 60}")
    print(f"Equivalence Check: {result['verdict']}")
    print(f"  New:      {result['new_dir']}")
    print(f"  Baseline: {result['baseline_dir']}")
    print(f"  Tolerance: {result['tolerance']}")
    print(f"{'─' * 60}")
    for c in result["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['check']}: {c['detail']}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two COCO dataset builds for equivalence"
    )
    parser.add_argument("--new", type=Path, required=True,
                        help="Path to new build directory")
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Path to baseline build directory")
    parser.add_argument("--tolerance", type=int, default=0,
                        help="Allowed count difference (default: 0 = exact)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of human-readable report")
    args = parser.parse_args()

    result = check_equivalence(args.new, args.baseline, tolerance=args.tolerance)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)

    sys.exit(0 if result["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
