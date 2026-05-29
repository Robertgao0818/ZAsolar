"""
Export V4.1 combined hard-negative chips: batch 003 (reviewed FPs) + batch 004 (curated shortlist).

Merges two HN sources into a base COCO dataset to achieve ~15-20% targeted HN ratio.
ID segments: base COCO 1-899999, batch 003 HN 900000-949999, batch 004 HN 950000+.

Usage:
    python scripts/training/export_v4_1_hn.py \
        --base-coco /workspace/coco_v4_1_base \
        --output-dir /workspace/coco_v4_1_hn \
        --batch004-sample-rate 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Reuse from existing exporters
from scripts.training.export_targeted_hn import (
    DEFAULT_BATCH_003_GRIDS,
    load_fp_locations,
    extract_fp_chips,
)
from scripts.training.export_v4_hn import (
    load_shortlist,
    stratified_sample,
    load_fp_geometries,
    extract_hn_chips,
    EXCLUDE_CORRECTIONS,
)
from core import region_registry
from core.grid_utils import get_results_root
from pipeline.manifests import (
    write_build_manifest,
    generate_build_id,
    build_source_inventory,
    compute_string_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ID segment boundaries
BATCH003_ID_START = 900000
BATCH004_ID_START = 950000


def merge_with_base(
    base_dir: Path,
    batch003_images: list[dict],
    batch003_prov: list[dict],
    batch004_images: list[dict],
    batch004_prov: list[dict],
    output_dir: Path,
) -> None:
    """Merge batch 003 + batch 004 HN chips into base COCO dataset."""
    with open(base_dir / "train.json") as f:
        base_train = json.load(f)
    with open(base_dir / "val.json") as f:
        base_val = json.load(f)

    # Hard-link/copy base images to output
    for split in ("train", "val"):
        src_split = base_dir / split
        dst_split = output_dir / split
        dst_split.mkdir(parents=True, exist_ok=True)
        if src_split.exists():
            for img_file in src_split.iterdir():
                dst_file = dst_split / img_file.name
                if not dst_file.exists():
                    try:
                        dst_file.hardlink_to(img_file)
                    except OSError:
                        shutil.copy2(img_file, dst_file)

    # Merge all HN into train
    all_hn_images = batch003_images + batch004_images
    merged_images = base_train["images"] + all_hn_images
    merged_annots = base_train["annotations"]  # HN chips have no annotations

    merged = {
        "info": {
            **base_train["info"],
            "description": base_train["info"]["description"]
            + " + V4.1 combined HN (batch003 + batch004)",
        },
        "licenses": base_train.get("licenses", []),
        "categories": base_train["categories"],
        "images": merged_images,
        "annotations": merged_annots,
    }

    with open(output_dir / "train.json", "w") as f:
        json.dump(merged, f)
    with open(output_dir / "val.json", "w") as f:
        json.dump(base_val, f)

    # Write provenance
    all_prov = batch003_prov + batch004_prov
    if all_prov:
        prov_path = output_dir / "v4_1_hn_provenance.csv"
        with open(prov_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_prov[0].keys())
            writer.writeheader()
            writer.writerows(all_prov)

    # Summary
    n_base = len(base_train["images"])
    n_pos = sum(1 for img in base_train["images"] if img.get("positive", True))
    n_easy_neg = n_base - n_pos
    n_b3 = len(batch003_images)
    n_b4 = len(batch004_images)
    n_hn = n_b3 + n_b4
    total = len(merged_images)
    hn_pct = n_hn / total * 100 if total else 0

    print(f"\n{'=' * 60}")
    print(f"V4.1 Combined HN Dataset Summary")
    print(f"{'=' * 60}")
    print(f"  Positive chips:      {n_pos:>6}")
    print(f"  Easy negatives:      {n_easy_neg:>6}")
    print(f"  Batch 003 HN:        {n_b3:>6}  (ID {BATCH003_ID_START}+)")
    print(f"  Batch 004 HN:        {n_b4:>6}  (ID {BATCH004_ID_START}+)")
    print(f"  ─────────────────────────────")
    print(f"  Total train:         {total:>6}")
    print(f"  Targeted HN ratio:   {hn_pct:>5.1f}%  ({n_hn}/{total})")
    print(f"  Annotations:         {len(merged_annots):>6}  (unchanged)")
    print(f"  Val:                 {len(base_val['images']):>6}  (unchanged)")
    print(f"{'=' * 60}")

    if hn_pct > 25:
        print(f"  WARNING: HN ratio {hn_pct:.1f}% is very high, risk of recall regression")
    elif hn_pct < 10:
        print(f"  WARNING: HN ratio {hn_pct:.1f}% is below 10%, may not suppress FP effectively")


def _src_rel(path: Path) -> str:
    """Repo-relative path string when possible, else absolute."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _emit_build_manifest(
    args: argparse.Namespace,
    batch003_grids: list[str],
    b3_prov: list[dict],
    b4_prov: list[dict],
) -> None:
    """Write build_manifest.json for the V4.1 combined HN export.

    Additive provenance only — wrapped by the caller in try/except so a
    failure here never breaks the build. HN chips are negatives (no GT
    polygons); ``selected_annotations`` records the selected HN *chips*
    (the supervision items this builder emits), keyed by source tile/type.
    """
    out_dir = Path(args.output_dir)

    # ── Source inventory ────────────────────────────────────────────────
    # batch003: reviewed FP gpkgs (one per grid, region-resolved).
    annotation_paths: list[Path] = []
    for gid in batch003_grids:
        rkey = region_registry.lookup_region(gid)
        results_root = get_results_root(region=rkey)
        reviewed_path = results_root / gid / "review" / f"{gid}_reviewed.gpkg"
        annotation_paths.append(reviewed_path)
    # batch004: curated small-FP shortlist CSV.
    source_inventory = build_source_inventory(
        annotation_paths,
        hn_shortlist_csvs=[Path(args.batch004_shortlist)],
    )

    # ── Selected supervision = the HN chips this builder emits ──────────
    selected_annotations: list[dict] = []
    for p in b3_prov:
        selected_annotations.append({
            "region": "cape_town",
            "imagery_layer": "aerial_2025",
            "label_source": "reviewed_prediction_fp_negative",
            "split": p.get("split", "train"),
            "source_file": "batch003_reviewed_fp",
            "source_id": p.get("image_id"),
            "source_tile": p.get("source_tile"),
            "source_type": p.get("source_type"),
        })
    for p in b4_prov:
        selected_annotations.append({
            "region": "cape_town",
            "imagery_layer": "aerial_2025",
            "label_source": "small_fp_hn_negative",
            "split": p.get("split", "train"),
            "source_file": _src_rel(Path(args.batch004_shortlist)),
            "source_id": p.get("image_id"),
            "source_tile": p.get("source_tile"),
            "source_type": p.get("source_type"),
        })

    resolved_spec = {
        "training_set_id": "v4_1_hn",
        "regions": ["cape_town"],
        "imagery_layer": "aerial_2025",
        "base_coco": _src_rel(Path(args.base_coco)),
        "batch003_grids": sorted(batch003_grids),
        "batch004_shortlist": _src_rel(Path(args.batch004_shortlist)),
        "batch004_sample_rate": args.batch004_sample_rate,
        "batch003_id_start": BATCH003_ID_START,
        "batch004_id_start": BATCH004_ID_START,
        "chip_size": args.chip_size,
        "seed": args.seed,
    }

    # Fingerprint: effective config + source sha256s; changes iff the HN
    # dataset contents would change. Excludes timestamps.
    fingerprint = {
        "resolved_spec": resolved_spec,
        "source_sha256": sorted(
            (e["path"], e["sha256"]) for e in source_inventory
        ),
    }
    build_fingerprint_json = json.dumps(fingerprint, sort_keys=True)
    resolved_spec_hash = compute_string_sha256(
        json.dumps(resolved_spec, sort_keys=True)
    )
    build_id = generate_build_id("v4_1_hn", build_fingerprint_json)

    write_build_manifest(
        out_dir,
        build_id=build_id,
        spec_path="bespoke:scripts/training/export_v4_1_hn.py",
        resolved_spec=resolved_spec,
        resolved_spec_hash=resolved_spec_hash,
        regions=["cape_town"],
        evaluation_regime="installation",
        # This is an HN export merged onto a base COCO; it defines no
        # benchmark holdout of its own.
        exclude_grids=[],
        excluded_grids_reason="n/a (HN export; holdout owned by base COCO build)",
        source_inventory=source_inventory,
        split_strategy="hn_merge_into_base_train",
        split_seed=args.seed,
        # HN export has no easy-negative sampling of its own.
        easy_neg_ratio=0.0,
        hard_negatives_config=[
            {
                "source": "batch003_reviewed_fp",
                "id_start": BATCH003_ID_START,
                "n_chips": len(b3_prov),
            },
            {
                "source": "batch004_small_fp_shortlist",
                "id_start": BATCH004_ID_START,
                "n_chips": len(b4_prov),
                "sample_rate": args.batch004_sample_rate,
            },
        ],
        selected_annotations=selected_annotations,
        resolved_tile_roots={
            "tiles_root": str(args.tiles_root) if args.tiles_root else "default(resolve_tiles_dir)",
        },
        resolved_output_root=str(out_dir),
        entrypoint="scripts/training/export_v4_1_hn.py",
    )
    print(f"[BUILD_MANIFEST] {out_dir / 'build_manifest.json'} "
          f"(build_id={build_id}, "
          f"{len(source_inventory)} sources, "
          f"{len(selected_annotations)} selected_annotations)")


def main():
    parser = argparse.ArgumentParser(
        description="Export V4.1 combined HN: batch 003 reviewed FPs + batch 004 curated shortlist"
    )
    parser.add_argument("--base-coco", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--batch004-shortlist", type=Path,
        default=PROJECT_ROOT / "results/analysis/small_fp/taxonomy_run/hn_small_fp_shortlist.csv",
    )
    parser.add_argument("--batch004-sample-rate", type=float, default=1.0)
    parser.add_argument("--chip-size", type=int, default=400)
    parser.add_argument("--tiles-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Batch 003 HN ─────────────────────────────────────────────────
    print("[1/4] Loading batch 003 reviewed FP predictions...")
    batch003_grids = DEFAULT_BATCH_003_GRIDS
    fp_by_grid = load_fp_locations(batch003_grids)
    total_b3_fp = sum(len(gdf) for gdf in fp_by_grid.values())
    print(f"  Found {total_b3_fp} FPs across {len(fp_by_grid)} grids")

    if total_b3_fp > 0:
        print(f"\n  Extracting {args.chip_size}x{args.chip_size} chips...")
        # Override img_id start in extract_fp_chips by monkey-patching isn't clean,
        # so we call it and then remap IDs
        b3_images, b3_prov = extract_fp_chips(
            fp_by_grid, args.output_dir,
            chip_size=args.chip_size,
            tiles_root=args.tiles_root,
        )
        # Remap IDs to batch 003 segment
        id_offset = BATCH003_ID_START - 900000  # extract_fp_chips starts at 900000
        for img in b3_images:
            img["id"] = img["id"]  # already starts at 900000, keep as-is
        for p in b3_prov:
            p["image_id"] = p["image_id"]  # same
        print(f"  Batch 003: {len(b3_images)} HN chips (ID {BATCH003_ID_START}-{BATCH003_ID_START + len(b3_images) - 1})")
    else:
        b3_images, b3_prov = [], []
        print("  No batch 003 FPs found")

    # ── Batch 004 HN ─────────────────────────────────────────────────
    print(f"\n[2/4] Loading batch 004 HN shortlist...")
    shortlist = load_shortlist(args.batch004_shortlist)
    print(f"  {len(shortlist)} candidates from {shortlist['grid_id'].nunique()} grids")

    if args.batch004_sample_rate < 1.0:
        print(f"\n  Stratified sampling ({args.batch004_sample_rate*100:.0f}%)...")
        sampled = stratified_sample(shortlist, args.batch004_sample_rate, seed=args.seed)
    else:
        sampled = shortlist
        print(f"  Using all {len(sampled)} candidates (100%)")

    print(f"\n[3/4] Loading batch 004 FP geometries and extracting chips...")
    fp_by_grid_b4 = load_fp_geometries(sampled)
    total_b4_fp = sum(len(gdf) for gdf in fp_by_grid_b4.values())
    print(f"  {total_b4_fp} FP geometries loaded")

    b4_images, b4_prov = extract_hn_chips(
        fp_by_grid_b4, args.output_dir,
        chip_size=args.chip_size,
        tiles_root=args.tiles_root,
    )
    # Remap batch 004 IDs to 950000+ segment
    for img in b4_images:
        img["id"] = img["id"] - 900000 + BATCH004_ID_START
    for p in b4_prov:
        p["image_id"] = p["image_id"] - 900000 + BATCH004_ID_START
    print(f"  Batch 004: {len(b4_images)} HN chips (ID {BATCH004_ID_START}-{BATCH004_ID_START + len(b4_images) - 1})")

    # ── Merge ─────────────────────────────────────────────────────────
    print(f"\n[4/4] Merging with base dataset...")
    merge_with_base(
        args.base_coco,
        b3_images, b3_prov,
        b4_images, b4_prov,
        args.output_dir,
    )
    print(f"\nOutput: {args.output_dir}")

    # ── Build provenance manifest (additive; never breaks the build) ─────
    try:
        _emit_build_manifest(args, batch003_grids, b3_prov, b4_prov)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] build_manifest write failed (non-fatal): {exc}")


if __name__ == "__main__":
    main()
