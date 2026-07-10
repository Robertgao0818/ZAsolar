"""Inventory snapshot — scan all results trees and classify by region/imagery/model.

Produces docs/plans/joburg_pre_migration_snapshot.csv with one row per
<tree>/<grid> containing: region, inferred imagery source, model version,
result count, gpkg row count (if readable), vintage hint, created_at.

Reads config.json.model_path and config.json.config.tiles_dir as ground
truth — does NOT guess from grid ID range.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from core import region_registry  # noqa: E402


def resolve_region_with_fallback(grid_id: str, tiles_dir: str | None) -> str:
    """Try registry first, then fall back to tiles_dir path hint, then grid ID pattern."""
    registered = region_registry.lookup_region(grid_id)
    if registered:
        return registered
    # fallback: tiles_dir path hint
    if tiles_dir:
        lower = tiles_dir.lower()
        if "joburg" in lower or "jhb" in lower:
            return "johannesburg"
        if "capetown" in lower or "cape_town" in lower:
            return "cape_town"
    # fallback: grid ID pattern
    if grid_id.startswith("JHB"):
        return "johannesburg"
    # JHB grid numeric ranges known so far: G0772..G0926, G1110..G1xxx (but
    # only CBD 25 registered). Without a registered task grid for JHB
    # periphery we can't tell from ID alone. Mark as AMBIGUOUS.
    return "AMBIGUOUS"


RESULT_TREES = [
    REPO / "results",
    REPO / "results_joburg",
    Path("/home/gaosh/zasolar_data/results"),
]
OUTPUT_CSV = REPO / "docs" / "plans" / "joburg_pre_migration_snapshot.csv"


def infer_imagery_source(tiles_dir: str | None) -> str:
    if not tiles_dir:
        return "unknown"
    lower = tiles_dir.lower()
    if "geid" in lower:
        return "geid"
    if "aerial" in lower:
        return "aerial"
    # default to aerial if path is tiles_joburg (historically aerial)
    if "tiles_joburg" in lower or "tiles/joburg" in lower:
        return "aerial"
    return "unknown"


def extract_model_version(model_path: str | None) -> str:
    if not model_path:
        return "unknown"
    # pattern: .../checkpoints/<version>/best_model.pth
    m = re.search(r"checkpoints/([^/]+)/", model_path)
    if m:
        return m.group(1)
    return Path(model_path).parent.name


def try_gpkg_row_count(grid_dir: Path) -> int | None:
    gpkg = grid_dir / "predictions_metric.gpkg"
    if not gpkg.exists():
        return None
    try:
        import geopandas as gpd  # local import to avoid hard dep
        gdf = gpd.read_file(gpkg)
        return len(gdf)
    except Exception as e:
        return -1  # sentinel: file present but unreadable


def scan_tree(tree_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not tree_root.exists():
        return rows

    for grid_dir in sorted(tree_root.iterdir()):
        if not grid_dir.is_dir():
            continue
        grid_id = grid_dir.name
        if not re.fullmatch(r"(G\d{4}|JHB\d{2})", grid_id):
            continue

        cfg_path = grid_dir / "config.json"
        row: dict[str, Any] = {
            "tree_root": str(tree_root),
            "grid_id": grid_id,
            "grid_dir": str(grid_dir),
            "region": None,  # filled after config read so tiles_dir can inform
            "has_config": cfg_path.exists(),
            "model_version": None,
            "imagery_source": None,
            "config_tiles_dir": None,
            "model_path": None,
            "result_count": None,
            "gpkg_row_count": try_gpkg_row_count(grid_dir),
            "created_at_utc": None,
        }

        tiles_dir_hint: str | None = None
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                inner = cfg.get("config", {})
                row["model_path"] = inner.get("model_path")
                row["config_tiles_dir"] = inner.get("tiles_dir")
                row["model_version"] = extract_model_version(inner.get("model_path"))
                row["imagery_source"] = infer_imagery_source(inner.get("tiles_dir"))
                row["result_count"] = cfg.get("result_count")
                row["created_at_utc"] = cfg.get("created_at_utc")
                tiles_dir_hint = inner.get("tiles_dir")
            except (json.JSONDecodeError, OSError) as e:
                row["model_version"] = f"CONFIG_READ_ERROR: {e}"

        row["region"] = resolve_region_with_fallback(grid_id, tiles_dir_hint)
        rows.append(row)
    return rows


def main() -> None:
    all_rows: list[dict[str, Any]] = []
    for tree in RESULT_TREES:
        print(f"[scan] {tree} ... ", end="", flush=True)
        rows = scan_tree(tree)
        print(f"{len(rows)} grids")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[write] {OUTPUT_CSV} ({len(df)} rows)")

    print("\n=== Summary ===")
    print("\nBy (region, model_version, imagery_source):")
    grouped = (
        df.groupby(["region", "model_version", "imagery_source"], dropna=False)
        .size()
        .reset_index(name="grid_count")
        .sort_values(["region", "grid_count"], ascending=[True, False])
    )
    print(grouped.to_string(index=False))

    print("\nBy tree_root:")
    print(df.groupby("tree_root").size().to_string())

    print("\nGrids without config.json:")
    no_cfg = df[~df["has_config"]]
    if len(no_cfg):
        print(no_cfg[["tree_root", "grid_id", "region"]].to_string(index=False))
    else:
        print("(none)")


if __name__ == "__main__":
    main()
