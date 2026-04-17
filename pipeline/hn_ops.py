"""Hard-negative operations — stable public API for the dataset builder.

Extracts reusable HN functions from the CLI scripts in
``scripts/training/``.  The builder calls these functions directly;
the CLI scripts become thin wrappers.

Public API
----------
- ``extract_reviewed_fp_hn(grids, output_dir, ...) -> HNResult``
- ``extract_small_fp_hn(shortlist_csv, output_dir, ...) -> HNResult``
- ``merge_hn_into_coco(base_dir, hn_images_list, output_dir) -> MergeResult``
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HNResult:
    """Result of an HN extraction step."""
    images: list[dict] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    source_type: str = ""
    n_grids: int = 0
    n_chips: int = 0


@dataclass
class MergeResult:
    """Result of merging HN chips into a base COCO dataset."""
    total_train_images: int = 0
    total_val_images: int = 0
    total_annotations: int = 0
    n_base_positive: int = 0
    n_base_easy_neg: int = 0
    n_hn_chips: int = 0
    hn_ratio: float = 0.0


def extract_reviewed_fp_hn(
    grids: list[str],
    output_dir: Path,
    chip_size: int = 400,
    tiles_root: Path | None = None,
    img_id_start: int = 900000,
) -> HNResult:
    """Extract HN chips from reviewed FP predictions.

    Wraps the logic from ``scripts/training/export_targeted_hn.py``.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.training.export_targeted_hn import (
        load_fp_locations, extract_fp_chips,
    )

    fp_by_grid = load_fp_locations(grids)
    total_fp = sum(len(gdf) for gdf in fp_by_grid.values())

    if total_fp == 0:
        return HNResult(source_type="reviewed_fp_hn")

    images, provenance = extract_fp_chips(
        fp_by_grid, output_dir,
        chip_size=chip_size,
        tiles_root=tiles_root,
    )

    # Remap IDs if needed
    if img_id_start != 900000:
        offset = img_id_start - 900000
        for img in images:
            img["id"] += offset
        for p in provenance:
            p["image_id"] += offset

    return HNResult(
        images=images,
        provenance=provenance,
        source_type="reviewed_fp_hn",
        n_grids=len(fp_by_grid),
        n_chips=len(images),
    )


def extract_small_fp_hn(
    shortlist_csv: Path,
    output_dir: Path,
    chip_size: int = 400,
    sample_rate: float = 0.5,
    tiles_root: Path | None = None,
    seed: int = 42,
    img_id_start: int = 950000,
) -> HNResult:
    """Extract HN chips from curated small-FP shortlist.

    Wraps the logic from ``scripts/training/export_v4_hn.py``.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.training.export_v4_hn import (
        load_shortlist, stratified_sample,
        load_fp_geometries, extract_hn_chips,
    )

    shortlist = load_shortlist(shortlist_csv)
    if len(shortlist) == 0:
        return HNResult(source_type="small_fp_hn")

    if sample_rate < 1.0:
        sampled = stratified_sample(shortlist, sample_rate, seed=seed)
    else:
        sampled = shortlist

    fp_by_grid = load_fp_geometries(sampled)
    if not fp_by_grid:
        return HNResult(source_type="small_fp_hn")

    images, provenance = extract_hn_chips(
        fp_by_grid, output_dir,
        chip_size=chip_size,
        tiles_root=tiles_root,
    )

    # Remap IDs to target segment
    offset = img_id_start - 900000
    for img in images:
        img["id"] += offset
    for p in provenance:
        p["image_id"] += offset

    return HNResult(
        images=images,
        provenance=provenance,
        source_type="small_fp_hn",
        n_grids=len(fp_by_grid),
        n_chips=len(images),
    )


def merge_hn_into_coco(
    base_dir: Path,
    hn_results: list[HNResult],
    output_dir: Path,
) -> MergeResult:
    """Merge HN chips into a base COCO dataset.

    Hard-links or copies base images and appends HN images to train.json.
    Val split is unchanged.
    """
    with open(base_dir / "train.json") as f:
        base_train = json.load(f)
    with open(base_dir / "val.json") as f:
        base_val = json.load(f)

    # Hard-link base images to output
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

    # Collect all HN images
    all_hn_images: list[dict] = []
    hn_descriptions: list[str] = []
    for hn in hn_results:
        all_hn_images.extend(hn.images)
        if hn.n_chips > 0:
            hn_descriptions.append(f"{hn.source_type}({hn.n_chips})")

    # Merge into train
    merged_images = base_train["images"] + all_hn_images
    merged_annots = base_train["annotations"]  # HN chips have no annotations

    merged = {
        "info": {
            **base_train["info"],
            "description": (
                base_train["info"].get("description", "")
                + " + " + ", ".join(hn_descriptions)
            ),
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

    # Copy base provenance files to output
    for prov_name in ("train_provenance.csv", "val_provenance.csv"):
        src_prov = base_dir / prov_name
        if src_prov.exists():
            dst_prov = output_dir / prov_name
            if not dst_prov.exists():
                shutil.copy2(src_prov, dst_prov)

    # Write combined HN provenance
    import csv
    all_prov = []
    for hn in hn_results:
        all_prov.extend(hn.provenance)
    if all_prov:
        prov_path = output_dir / "hn_provenance.csv"
        with open(prov_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_prov[0].keys())
            writer.writeheader()
            writer.writerows(all_prov)

    n_base_pos = sum(1 for img in base_train["images"] if img.get("positive", True))
    n_base_neg = len(base_train["images"]) - n_base_pos
    n_hn = len(all_hn_images)
    total = len(merged_images)

    return MergeResult(
        total_train_images=total,
        total_val_images=len(base_val["images"]),
        total_annotations=len(merged_annots),
        n_base_positive=n_base_pos,
        n_base_easy_neg=n_base_neg,
        n_hn_chips=n_hn,
        hn_ratio=n_hn / total if total else 0.0,
    )
