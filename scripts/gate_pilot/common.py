"""Shared paths and small helpers for the presence-gate pilot."""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_ROOT = Path.home() / "zasolar_data" / "gate_pilot_2026-09"
VECTOR_DIR = OUT_ROOT / "vectors"
CANDIDATE_DIR = OUT_ROOT / "step0" / "empty_candidates"
GEOMETRY_CARRIER_DIR = OUT_ROOT / "step0" / "geometry_carriers"
STEP0_DIR = OUT_ROOT / "step0"
LOG_DIR = OUT_ROOT / "logs"

DROPBOX_PERDET = "dropbox:RA_Solar/Gao/ct_census/results/unifiedA_census_perdet"
CROSSWALK_CSV = REPO_ROOT / "data" / "ct_grid_crosswalk_g_to_cpt.csv"
INVENTORY_CSV = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "ct_census_output_table"
    / "ct_full_inventory_2026-06-21_merged.csv"
)
INVENTORY_GPKG = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "ct_census_output_table"
    / "ct_full_inventory_2026-06-21_merged.gpkg"
)
RESULT_MD = REPO_ROOT / "docs" / "experiments" / "2026-09-yolo-gate-pilot" / "RESULT.md"

TRAIN_PV_GRIDS = ("CPT2932", "CPT2373", "CPT2182")
HOLDOUT_PV_GRID = "CPT1238"
EMPTY_SEED = 20260923

CHIP_SIZE = 400
OVERLAP = 0.25
TILE_PIXELS = 2000
G2_RECALL = 0.99
G3_PASS_RATE = 0.20
SMALL_AREA_M2 = 30.0

VECTOR_FILES = (
    "config.json",
    "predictions.geojson",
    "predictions_metric.gpkg",
    "predictions_metric_cls_filtered.gpkg",
)


def setup_logging(name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
