"""
Download full-resolution tiles for grids selected in a preview-review batch.

Supports tile-level OSM building mask: only download tiles that contain
buildings (+ neighbor buffer), skipping empty wasteland tiles.

Examples:
  python scripts/imagery/download_reviewed_grids.py \
    --batch-dir results/grid_previews/batch_002

  # With OSM tile mask (selective download):
  python scripts/imagery/download_reviewed_grids.py \
    --batch-dir results/grid_previews/batch_002 --use-tile-mask

  python scripts/imagery/download_reviewed_grids.py \
    --batch-dir results/grid_previews/batch_001 \
    --decision keep review \
    --dry
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from scripts.imagery.download_tiles import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_TIFF_COMPRESS,
    download_grid,
)

DEFAULT_DECISIONS = ("keep",)
DEFAULT_TILE_MASK = Path(__file__).resolve().parent.parent.parent / "cache" / "tile_download_mask.csv"


def load_grid_ids(batch_dir: Path, decisions: tuple[str, ...]) -> list[str]:
    decisions_path = batch_dir / "grid_review_decisions.csv"
    if not decisions_path.exists():
        raise FileNotFoundError(f"review decisions not found: {decisions_path}")

    wanted = {decision.strip().lower() for decision in decisions if decision.strip()}
    if not wanted:
        raise ValueError("at least one --decision value is required")

    grid_ids: list[str] = []
    seen: set[str] = set()
    with decisions_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            decision = str(row.get("decision", "")).strip().lower()
            grid_id = str(row.get("grid_id", "")).strip().upper()
            if decision not in wanted or not grid_id or grid_id in seen:
                continue
            seen.add(grid_id)
            grid_ids.append(grid_id)
    return grid_ids


def load_tile_mask(mask_path: Path, grid_ids: list[str]) -> dict[str, set[tuple[int, int]]]:
    """Load tile download mask CSV and return per-grid tile sets."""
    if not mask_path.exists():
        raise FileNotFoundError(f"tile mask not found: {mask_path}")

    df = pd.read_csv(mask_path)
    grid_set = set(grid_ids)
    masks: dict[str, set[tuple[int, int]]] = {}
    for _, row in df[df["grid_id"].isin(grid_set)].iterrows():
        gid = row["grid_id"]
        masks.setdefault(gid, set()).add((int(row["col"]), int(row["row"])))
    return masks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download full-resolution tiles for grids chosen in browser review output"
    )
    parser.add_argument(
        "--batch-dir",
        required=True,
        type=Path,
        help="Preview batch directory, e.g. results/grid_previews/batch_001",
    )
    parser.add_argument(
        "--decision",
        nargs="+",
        default=list(DEFAULT_DECISIONS),
        help="Decision labels to include, e.g. keep review",
    )
    parser.add_argument(
        "--use-tile-mask",
        action="store_true",
        help="Use OSM building tile mask for selective download",
    )
    parser.add_argument(
        "--tile-mask-path",
        type=Path,
        default=DEFAULT_TILE_MASK,
        help="Path to tile_download_mask.csv",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Only print the selected grids without downloading tiles",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of grids to download in parallel",
    )
    parser.add_argument(
        "--tiff-compress",
        default=DEFAULT_TIFF_COMPRESS,
        choices=["JPEG", "DEFLATE", "ZSTD", "LZW", "NONE"],
        help="GeoTIFF compression for saved tiles (default: JPEG)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality when --tiff-compress=JPEG (default: 95)",
    )
    parser.add_argument(
        "--no-tiled",
        action="store_true",
        help="Disable tiled GeoTIFF output",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing tiles",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    grid_ids = load_grid_ids(args.batch_dir, tuple(args.decision))
    print(f"[INFO] batch_dir={args.batch_dir}")
    print(f"[INFO] selected_grids={len(grid_ids)}")
    if not grid_ids:
        print("[INFO] no matching grids found")
        return

    # Load tile mask if requested
    tile_masks: dict[str, set[tuple[int, int]]] = {}
    if args.use_tile_mask:
        tile_masks = load_tile_mask(args.tile_mask_path, grid_ids)
        grids_with_mask = [g for g in grid_ids if g in tile_masks]
        grids_no_mask = [g for g in grid_ids if g not in tile_masks]
        total_tiles = sum(len(v) for v in tile_masks.values())
        print(f"[INFO] tile mask: {len(grids_with_mask)} grids with buildings "
              f"({total_tiles} tiles)")
        if grids_no_mask:
            print(f"[INFO] tile mask: {len(grids_no_mask)} grids with NO buildings "
                  f"in OSM (skipped): {', '.join(grids_no_mask)}")
        grid_ids = grids_with_mask

    print("[INFO] grid_ids=" + ", ".join(grid_ids))
    if args.dry:
        return

    workers = max(1, min(int(args.workers), len(grid_ids)))
    print(f"[INFO] workers={workers}")

    def run_one(job: tuple[int, str]) -> str:
        idx, grid_id = job
        mask = tile_masks.get(grid_id)
        mask_info = f" (mask: {len(mask)} tiles)" if mask else " (all tiles)"
        print(f"\n=== [{idx}/{len(grid_ids)}] {grid_id}{mask_info} ===", flush=True)
        download_grid(
            grid_id,
            dry_run=False,
            tile_mask=mask,
            tiff_compress=args.tiff_compress,
            jpeg_quality=args.jpeg_quality,
            tiled=not args.no_tiled,
            force=args.force,
        )
        return grid_id

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, job) for job in enumerate(grid_ids, start=1)]
        for future in as_completed(futures):
            grid_id = future.result()
            print(f"[DONE] grid={grid_id}", flush=True)


if __name__ == "__main__":
    main()
