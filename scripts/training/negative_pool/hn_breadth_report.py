"""HN-breadth acceptance report for the negative pool (F1-gap plan C-1).

The pool design principle is **breadth dominates count**
(``feedback_hn_breadth_dominates_size``): the variety of hard-negative lookalike
archetypes must be at least as wide as the variety of positive appearance
contexts, *per region*. Adding more chips of one archetype does not reduce
lookalike FP; adding a *new* archetype does.

This tool produces two things:

1. A per ``region × imagery_layer × archetype`` count table of the negative pool
   (the raw breadth breakdown), split by ``training_eligible`` so it is obvious
   which rows can actually enter a training bundle.

2. A per-region **acceptance verdict**: HN-archetype breadth vs positive
   appearance-context breadth.  The positive pool is single-class (PV
   installations), so its "breadth" is measured as the number of distinct
   imagery layers (appearance domains) it spans in that region; the HN breadth
   is the number of distinct *training-eligible* archetypes in that region.  The
   gate PASSES when ``eligible_hn_archetypes >= positive_imagery_layers`` for the
   region — i.e. the HN stream covers at least as many distinct looks as the
   positives present.

Usage::

    python scripts/training/negative_pool/hn_breadth_report.py
    python scripts/training/negative_pool/hn_breadth_report.py --csv out.csv
    python scripts/training/negative_pool/hn_breadth_report.py \\
        --positive-manifests data/training_pool/positive_trusted_manifest.csv \\
                             data/training_pool/positive_untrusted_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POOL_ROOT = PROJECT_ROOT / "data" / "negative_pool"
MANIFEST_CSV = POOL_ROOT / "manifest.csv"
DEFAULT_POSITIVE_MANIFESTS = [
    PROJECT_ROOT / "data" / "training_pool" / "positive_trusted_manifest.csv",
    PROJECT_ROOT / "data" / "training_pool" / "positive_untrusted_manifest.csv",
]


def _is_eligible(row: dict) -> bool:
    val = (row.get("training_eligible") or "").strip().lower()
    return val != "false"  # "true" or blank/legacy => eligible


def load_hn_breakdown(manifest_csv: Path):
    """Return (counts, eligible_archetypes_by_region).

    counts: list of dicts {region, imagery_layer, archetype, eligible,
            n_chips, n_grids}.
    eligible_archetypes_by_region: {region: set(archetype)} for eligible rows.
    """
    agg: dict[tuple, dict] = defaultdict(
        lambda: {"n_chips": 0, "grids": set()}
    )
    eligible_arch: dict[str, set] = defaultdict(set)
    all_arch: dict[str, set] = defaultdict(set)
    with manifest_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("archetype") == "actually_pv_mislabeled":
                continue
            region = row.get("region", "")
            layer = row.get("imagery_layer", "")
            archetype = row.get("archetype", "")
            eligible = _is_eligible(row)
            key = (region, layer, archetype, eligible)
            agg[key]["n_chips"] += 1
            agg[key]["grids"].add(row.get("grid_id", ""))
            all_arch[region].add(archetype)
            if eligible:
                eligible_arch[region].add(archetype)

    counts = []
    for (region, layer, archetype, eligible), v in sorted(agg.items()):
        counts.append({
            "region": region,
            "imagery_layer": layer,
            "archetype": archetype,
            "training_eligible": "true" if eligible else "false",
            "n_chips": v["n_chips"],
            "n_grids": len(v["grids"]),
        })
    return counts, eligible_arch, all_arch


def load_positive_breadth(positive_manifests: list[Path]):
    """Return {region: set(imagery_layer)} for non-archived positives."""
    layers_by_region: dict[str, set] = defaultdict(set)
    for p in positive_manifests:
        if not p.exists():
            print(f"WARN: positive manifest not found: {p}", file=sys.stderr)
            continue
        with p.open(newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("archived") or "").strip().lower() in ("true", "1"):
                    continue
                region = row.get("region", "")
                layer = row.get("imagery_layer", "")
                if region and layer:
                    layers_by_region[region].add(layer)
    return layers_by_region


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--positive-manifests", type=Path, nargs="+",
                        default=DEFAULT_POSITIVE_MANIFESTS)
    parser.add_argument("--csv", type=Path, default=None,
                        help="write the breakdown table to this CSV path")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: {args.manifest} not found", file=sys.stderr)
        return 1

    counts, eligible_arch, all_arch = load_hn_breakdown(args.manifest)
    pos_layers = load_positive_breadth(args.positive_manifests)

    # ── breakdown table ──
    print("=== HN breadth: region × imagery_layer × archetype ===")
    hdr = f"{'region':<14}{'imagery_layer':<26}{'archetype':<28}{'elig':<6}{'chips':>6}{'grids':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in counts:
        print(f"{r['region'][:13]:<14}{r['imagery_layer'][:25]:<26}"
              f"{r['archetype'][:27]:<28}"
              f"{r['training_eligible']:<6}{r['n_chips']:>6}{r['n_grids']:>6}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "region", "imagery_layer", "archetype",
                "training_eligible", "n_chips", "n_grids"])
            w.writeheader()
            w.writerows(counts)
        print(f"\nWrote breakdown CSV: {args.csv}")

    # ── per-region acceptance verdict ──
    print("\n=== acceptance: HN archetype breadth >= positive appearance breadth ===")
    vhdr = (f"{'region':<14}{'pos_imagery_layers':>20}{'all_hn_arch':>14}"
            f"{'eligible_hn_arch':>18}{'verdict':>10}")
    print(vhdr)
    print("-" * len(vhdr))
    regions = sorted(set(list(pos_layers.keys()) + list(all_arch.keys())))
    any_fail = False
    for region in regions:
        n_pos = len(pos_layers.get(region, set()))
        n_all = len(all_arch.get(region, set()))
        n_elig = len(eligible_arch.get(region, set()))
        passed = n_elig >= n_pos and n_pos > 0
        if n_pos > 0 and not passed:
            any_fail = True
        verdict = "PASS" if passed else ("FAIL" if n_pos > 0 else "n/a")
        print(f"{region[:13]:<14}{n_pos:>20}{n_all:>14}{n_elig:>18}{verdict:>10}")

    print("\nNote: a FAIL means the training-eligible HN stream covers fewer "
          "distinct\narchetypes than the region's positive appearance domains. "
          "Today the\npool is JHB-GEID-only + cross-domain Vexcel empty-probe, "
          "all\ntraining_eligible=false (imagery-layer balance gate), so every "
          "in-domain\nregion FAILS until in-domain (CT-aerial / JHB-Vexcel) HN "
          "are ingested\nand flipped eligible. This is the documented C-1 "
          "follow-up, not a bug.")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
