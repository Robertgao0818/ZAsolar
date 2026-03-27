"""
补丁式 mosaic 构建：在 tile mask 基础上，手动追加额外 tiles 后重建 VRT。

用途：OSM 无建筑数据的新建区，tile mask 会漏掉，用此脚本补上。

Usage:
  # 补上 G1628 左下角缺失的 tiles
  python scripts/imagery/patch_grid_mosaic.py \
    --grid-id G1628 --extra-tiles 0,3 0,4 0,5 1,5

  # 干跑，只看哪些 tiles 会进 mosaic
  python scripts/imagery/patch_grid_mosaic.py \
    --grid-id G1628 --extra-tiles 0,3 0,4 0,5 1,5 --dry
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.imagery.build_grid_vrts import build_vrt_for_grid, iter_tile_paths, verify_vrt

TILE_MASK_CSV = Path("cache/tile_download_mask.csv")


def parse_tile_spec(spec: str) -> tuple[int, int]:
    """Parse 'col,row' string."""
    parts = spec.split(",")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="Patch grid mosaic: mask tiles + extra manual tiles")
    parser.add_argument("--grid-id", required=True)
    parser.add_argument(
        "--extra-tiles", nargs="+", required=True,
        help="Extra tile specs as col,row (e.g. 0,4 0,5 1,5)")
    parser.add_argument("--tile-mask", type=Path, default=TILE_MASK_CSV)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    grid_id = args.grid_id.upper()
    grid_dir = Path("tiles") / grid_id

    # Load mask tiles
    mask_tiles: set[tuple[int, int]] = set()
    if args.tile_mask.exists():
        df = pd.read_csv(args.tile_mask)
        gdf = df[df["grid_id"] == grid_id]
        mask_tiles = set(zip(gdf["col"].astype(int), gdf["row"].astype(int)))
    print(f"[INFO] mask tiles: {len(mask_tiles)}")

    # Parse extra tiles
    extra = {parse_tile_spec(s) for s in args.extra_tiles}
    print(f"[INFO] extra tiles: {len(extra)} -> {sorted(extra)}")

    keep = mask_tiles | extra
    print(f"[INFO] total tiles for mosaic: {len(keep)}")

    # Map to actual files
    all_tiles = iter_tile_paths(grid_dir)
    keep_paths = []
    hide_paths = []
    for p in all_tiles:
        parts = p.stem.replace("_geo", "").split("_")
        col, row = int(parts[-2]), int(parts[-1])
        if (col, row) in keep:
            keep_paths.append(p)
        else:
            hide_paths.append(p)

    print(f"[INFO] tiles in mosaic: {len(keep_paths)}, hidden: {len(hide_paths)}")

    if args.dry:
        print("\n[DRY] tiles that would be included:")
        for p in keep_paths:
            print(f"  {p.name}")
        return

    # Temporarily hide non-selected tiles, build VRT, restore
    renamed = []
    for p in hide_paths:
        tmp = p.with_suffix(".tif.bak")
        p.rename(tmp)
        renamed.append((tmp, p))

    try:
        vrt_path = build_vrt_for_grid(grid_id, force=True)
    finally:
        for tmp, orig in renamed:
            tmp.rename(orig)

    w, h, c = verify_vrt(vrt_path)
    print(f"[OK] {vrt_path} ({w}x{h}, bands={c})")


if __name__ == "__main__":
    main()
