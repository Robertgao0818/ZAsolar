"""PR5: migrate results trees into results/<region>/<model_run>/<grid>/.

Reads the PR1 inventory CSV, classifies each <tree>/<grid> by reading
config.json.model_path and config.json.tiles_dir (NEVER by grid ID range),
then mv's matching grids into their canonical destination.

Grids without a config.json (NaN) or with unclear lineage stay in place;
user reviews them in PR8 cleanup.

Run with --dry-run first to preview classification.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from core import region_registry  # noqa: E402

INVENTORY_CSV = REPO / "docs" / "plans" / "joburg_pre_migration_snapshot.csv"

# Canonical destinations: (region, run_id) -> absolute path where grids land.
RUN_PATHS = {
    ("johannesburg", "v4_aerial_2023"):            REPO / "results" / "johannesburg" / "v4_aerial_2023",
    ("johannesburg", "v3c_geid_2024_02"):          REPO / "results" / "johannesburg" / "v3c_geid_2024_02",
    ("cape_town",    "v3c_targeted_hn_aerial_2025"): REPO / "results" / "cape_town" / "v3c_targeted_hn_aerial_2025",
    ("cape_town",    "v4_hn_aerial_2025"):         REPO / "results" / "cape_town" / "v4_hn_aerial_2025",
    ("cape_town",    "v3c_geid_experiment"):       REPO / "results" / "cape_town" / "v3c_geid_experiment",
}


@dataclass
class Routing:
    region: str | None
    run_id: str | None
    reason: str


def infer_region_from_tiles_dir(tiles_dir: str | None, grid_id: str) -> str | None:
    """Region inference that does NOT trust grid_id alone (overlap with CT/JHB)."""
    if not tiles_dir:
        return None
    lower = tiles_dir.lower()
    if "joburg" in lower or "jhb" in lower:
        return "johannesburg"
    if "capetown" in lower or "cape_town" in lower:
        return "cape_town"
    # Plain /workspace/tiles/ or /dev/shm/tiles/ without region hint ==
    # CT aerial by historical convention.
    if re.search(r"(/workspace|/dev/shm)/tiles/G\d{4}", tiles_dir):
        return "cape_town"
    return None


def classify(row: pd.Series) -> Routing:
    model = row.get("model_version")
    imagery = row.get("imagery_source")
    tiles_dir = row.get("config_tiles_dir")
    grid_id = row["grid_id"]

    if pd.isna(model) or not model or (isinstance(model, str) and model.startswith("CONFIG_READ_ERROR")):
        return Routing(None, None, "no_config_or_error")

    region = infer_region_from_tiles_dir(tiles_dir if isinstance(tiles_dir, str) else None, grid_id)

    if model == "exp004_v4_hn" and imagery == "aerial":
        if region == "johannesburg":
            return Routing("johannesburg", "v4_aerial_2023", "V4 on JHB aerial")
        if region == "cape_town":
            return Routing("cape_town", "v4_hn_aerial_2025", "V4 on CT aerial")
    if model == "exp003_C_targeted_hn" and imagery == "geid":
        # Special-case: G1189/G1190 on GEID are CT cross-region experiments
        # (user confirmed 2026-04-19: "卫星图迁移结果, 可以删"). Route to
        # a discardable CT run; physical aerial_2023 JHB coverage of the same
        # IDs is a real separate area and lands in v4_aerial_2023 via a
        # different path.
        if grid_id in ("G1189", "G1190"):
            return Routing("cape_town", "v3c_geid_experiment", "V3C cross-region experiment on CT grid (user: discardable)")
        if region == "johannesburg":
            return Routing("johannesburg", "v3c_geid_2024_02", "V3C on JHB GEID")
        if region == "cape_town":
            return Routing("cape_town", "v3c_geid_experiment", "V3C cross-region experiment (user: discardable)")
    if model == "exp003_C_targeted_hn" and imagery in ("unknown", "aerial"):
        if region == "cape_town":
            return Routing("cape_town", "v3c_targeted_hn_aerial_2025", "V3C on CT aerial")
        if region == "johannesburg":
            return Routing("johannesburg", "v3c_geid_2024_02", "V3C on JHB (imagery unknown, routed by region)")
    if model == "checkpoints_cleaned":
        return Routing(None, None, "legacy_checkpoints_cleaned_defer")

    return Routing(None, None, f"unrouted: model={model} imagery={imagery} region={region}")


def main(dry_run: bool) -> None:
    df = pd.read_csv(INVENTORY_CSV)
    print(f"Inventory: {len(df)} rows")

    # Classify
    routings = df.apply(classify, axis=1)
    df["target_region"] = [r.region for r in routings]
    df["target_run"] = [r.run_id for r in routings]
    df["route_reason"] = [r.reason for r in routings]

    routable = df[df["target_run"].notna()]
    unrouted = df[df["target_run"].isna()]

    print("\n=== Routing summary ===")
    print(routable.groupby(["target_region", "target_run"]).size().to_string())
    print(f"\nUnrouted: {len(unrouted)} grids")
    print(unrouted.groupby("route_reason").size().to_string())

    if dry_run:
        print("\n[dry-run] classification preview only; no mv executed.")
        out = INVENTORY_CSV.parent / "pr5_routing_preview.csv"
        df.to_csv(out, index=False)
        print(f"[dry-run] full routing written to {out}")
        return

    # Create target directories
    print("\n=== Creating target directories ===")
    for (region, run), path in RUN_PATHS.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"  mkdir -p {path}")

    # Execute mv
    print("\n=== Executing mv ===")
    moved, failed, skipped = 0, 0, 0
    for _, row in routable.iterrows():
        src = Path(row["grid_dir"])
        region = row["target_region"]
        run_id = row["target_run"]
        dst_root = RUN_PATHS.get((region, run_id))
        if dst_root is None:
            print(f"  [SKIP] ({region}, {run_id}) not in RUN_PATHS: {src}")
            skipped += 1
            continue
        dst = dst_root / row["grid_id"]
        if not src.exists():
            print(f"  [MISS] src gone: {src}")
            skipped += 1
            continue
        if dst.exists():
            print(f"  [COLL] dst exists, skipping: {dst}")
            skipped += 1
            continue
        try:
            shutil.move(str(src), str(dst))
            moved += 1
        except Exception as e:
            print(f"  [FAIL] {src} -> {dst}: {e}")
            failed += 1

    print(f"\nmv results: moved={moved} skipped={skipped} failed={failed}")

    # Write RUN_MANIFEST.json for each target
    now = datetime.now(timezone.utc).isoformat()
    print("\n=== Writing RUN_MANIFEST.json ===")
    for (region, run_id), path in RUN_PATHS.items():
        grid_dirs = sorted(d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith("."))
        if not grid_dirs:
            print(f"  [skip empty] {path}")
            continue
        # Try to enrich with registry metadata
        model_cfg = None
        try:
            model_cfg = region_registry.get_model_run(region, run_id)
        except KeyError:
            pass
        manifest = {
            "region": region,
            "model_run_id": run_id,
            "model_version": model_cfg.model_version if model_cfg else "unknown",
            "imagery_layer": model_cfg.imagery_layer if model_cfg else "unknown",
            "inference_date": model_cfg.inference_date if model_cfg else None,
            "grids": grid_dirs,
            "grid_count": len(grid_dirs),
            "created_at_utc": now,
            "schema_version": 1,
            "notes": f"Written by PR5 migration. Source inventory: {INVENTORY_CSV.relative_to(REPO)}",
        }
        manifest_path = path / "RUN_MANIFEST.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"  wrote {manifest_path}  ({len(grid_dirs)} grids)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
