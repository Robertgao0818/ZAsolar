#!/usr/bin/env python3
"""Fetch teacher vectors and aerial tiles for the frozen learning-curve grids.

Reads ``learning_curve_2026-09-28/selection.json``. Does not train, does
not score, and does not download reserve grids. A grid is complete when
both production GeoPackages are on disk and the aerial folder has 42
tiles. The sentinel ``fetch.done`` is written only after every primary
grid is complete.

Resume is safe: completed grids are skipped. Two imagery failures in a
row stop the run, because that pattern means the city WMS is down rather
than one grid being bad.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import region_registry
from scripts.gate_pilot.common import (
    DROPBOX_PERDET,
    REPO_ROOT,
    VECTOR_FILES,
    setup_logging,
)
from scripts.gate_pilot.learning_curve_select import SELECTION_DIR, SELECTION_PATH

TILES_ROOT = Path.home() / "zasolar_data" / "tiles" / "cape_town" / "aerial_2025"
VECTOR_DEST = SELECTION_DIR / "vectors"
STATUS_PATH = SELECTION_DIR / "fetch_status.json"
SENTINEL = SELECTION_DIR / "fetch.done"
FILES_FROM = SELECTION_DIR / "rclone_files_from.txt"
EXPECTED_TILES = 42
# Teacher labels are the two GeoPackages. config.json and predictions.geojson
# are copied when the backup has them. CPT2776's backup has no geojson;
# that duplicate is not a reason to drop the grid.
REQUIRED_VECTORS = (
    "predictions_metric.gpkg",
    "predictions_metric_cls_filtered.gpkg",
)


def _tile_paths(grid_id: str) -> list[Path]:
    source = region_registry.resolve_source_grid_id(grid_id, "cape_town")
    return sorted((TILES_ROOT / source).glob(f"{grid_id}_*_*_geo.tif"))


def _vectors_ready(grid_id: str) -> bool:
    for name in REQUIRED_VECTORS:
        path = VECTOR_DEST / grid_id / name
        if not path.exists() or path.stat().st_size == 0:
            return False
    return True


def _tiles_ready(grid_id: str) -> bool:
    return len(_tile_paths(grid_id)) == EXPECTED_TILES


def _write_status(payload: dict) -> None:
    STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _rclone(relative_paths: list[str], log) -> list[str]:
    missing = [
        rel for rel in relative_paths
        if not (VECTOR_DEST / rel).exists() or (VECTOR_DEST / rel).stat().st_size == 0
    ]
    if not missing:
        log.info("vectors already on disk: %d files", len(relative_paths))
        return []
    FILES_FROM.write_text("\n".join(missing) + "\n", encoding="utf-8")
    cmd = [
        "rclone", "copy", DROPBOX_PERDET, str(VECTOR_DEST),
        "--files-from", str(FILES_FROM),
        "--transfers", "8",
        "--checkers", "8",
        "--retries", "5",
        "--low-level-retries", "10",
        "--stats", "30s",
        "--stats-one-line",
    ]
    log.info("rclone copy %d missing vector files", len(missing))
    completed = subprocess.run(cmd, check=False)
    still = [
        rel for rel in missing
        if not (VECTOR_DEST / rel).exists() or (VECTOR_DEST / rel).stat().st_size == 0
    ]
    if completed.returncode != 0 or still:
        log.error(
            "rclone exited %s with %d files still missing, first %s",
            completed.returncode, len(still), still[:5],
        )
    return still


def _child_env() -> dict:
    # download_tiles.py reads SOLAR_TILES_ROOT at import and writes
    # <root>/<source_id>/. Step 1 pointed that root at the aerial_2025
    # layer. Without it, tiles land in the repo's tiles/ directory and
    # this script's 42-file check never sees them.
    env = os.environ.copy()
    env["SOLAR_TILES_ROOT"] = str(TILES_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _download_tiles(grid_id: str, log) -> bool:
    if _tiles_ready(grid_id):
        log.info("%s already has %d tiles", grid_id, EXPECTED_TILES)
        return True
    for attempt in (1, 2):
        log.info("%s imagery attempt %d", grid_id, attempt)
        subprocess.run(
            [
                sys.executable, "scripts/imagery/download_tiles.py",
                "--grid-id", grid_id, "--region", "ct",
            ],
            cwd=REPO_ROOT,
            env=_child_env(),
            check=False,
        )
        if _tiles_ready(grid_id):
            log.info("%s imagery complete", grid_id)
            return True
    log.error("%s has %d tiles after two attempts", grid_id, len(_tile_paths(grid_id)))
    return False


def main() -> int:
    log = setup_logging("learning_curve_fetch")
    if not SELECTION_PATH.exists():
        log.error("missing %s", SELECTION_PATH)
        return 1
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("schema") != "learning_curve_selection_v1":
        log.error("unexpected selection schema %s", selection.get("schema"))
        return 1
    primary = list(dict.fromkeys([*selection["train_30"], *selection["holdout"]]))
    if len(primary) != 39:
        log.error("expected 30 train + 9 holdout, found %d", len(primary))
        return 1
    if SENTINEL.exists():
        SENTINEL.unlink()
    VECTOR_DEST.mkdir(parents=True, exist_ok=True)
    relative = [f"{grid_id}/{name}" for grid_id in primary for name in VECTOR_FILES]
    missing = _rclone(relative, log)
    missing_required = [rel for rel in missing if Path(rel).name in REQUIRED_VECTORS]
    missing_optional = [rel for rel in missing if Path(rel).name not in REQUIRED_VECTORS]
    if missing_optional:
        log.warning("optional vector files missing, grids stay in the draw: %s", missing_optional)
    failed_vectors = sorted({Path(rel).parts[0] for rel in missing_required})
    status = {
        "primary": primary,
        "vector_failed": failed_vectors,
        "imagery_ok": [],
        "imagery_failed": [],
        "stopped_early": False,
    }
    _write_status(status)
    consecutive = 0
    for index, grid_id in enumerate(primary, start=1):
        if grid_id in failed_vectors:
            log.error("skip imagery for %s because vectors are missing", grid_id)
            status["imagery_failed"].append(grid_id)
            consecutive += 1
        elif _download_tiles(grid_id, log):
            status["imagery_ok"].append(grid_id)
            consecutive = 0
            log.info("complete %d/%d %s", index, len(primary), grid_id)
        else:
            status["imagery_failed"].append(grid_id)
            consecutive += 1
        _write_status(status)
        if consecutive >= 2:
            status["stopped_early"] = True
            _write_status(status)
            log.error("two grids in a row failed; stopping")
            return 1
    if status["imagery_failed"] or failed_vectors:
        log.error(
            "fetch incomplete: vectors %s imagery %s",
            failed_vectors, status["imagery_failed"],
        )
        return 1
    SENTINEL.write_text(f"ok {len(primary)}\n", encoding="utf-8")
    log.info("fetch.done %d grids", len(primary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
