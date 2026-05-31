#!/usr/bin/env python3
"""Build a stratified candidate manifest to calibrate Gemini review against RA.

The reviewed-prediction GeoPackages (``<grid>/review/<grid>_reviewed.gpkg``)
are the RA ground truth: each row carries the detector ``confidence`` and the
human ``review_status`` ({correct, edit} -> PV, {delete} -> non_PV) on the same
geometry.  This script scans those files, filters to a detector-confidence band
(default the hardest one, conf >= 0.95, where the detector is most confident yet
still produces lookalike FPs), and stratified-samples PV vs non_PV.

Output is a candidate manifest directly consumable by
``scripts/training/build_gemini_detection_review_chips.py``: each row sets
``predictions_path`` to the reviewed gpkg itself and ``pred_id`` to the row's
positional index, so the rendered chip and the RA label come from the same row
(no spatial join).  ``ra_label`` / ``confidence`` / ``review_status`` are carried
as extra columns for the downstream Gemini-vs-RA eval join (by candidate_id).
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core import region_registry  # noqa: E402
from core.grid_utils import resolve_tiles_dir  # noqa: E402

PV_STATUSES = {"correct", "edit"}
NONPV_STATUSES = {"delete"}

MANIFEST_FIELDS = [
    "candidate_id",
    "grid_id",
    "pred_id",
    "region_key",
    "predictions_path",
    "tile_path",
    "source_tile",
    "confidence",
    "area_m2",
    "review_status",
    "ra_label",
]


def _grid_id_from_gpkg(path: Path) -> str:
    # .../<grid>/review/<grid>_reviewed.gpkg
    stem = path.stem
    return stem[:-9].upper() if stem.lower().endswith("_reviewed") else stem.upper()


def _resolve_tile_path(grid_id: str, region: str, imagery_layer: str | None, source_tile: str) -> Path | None:
    try:
        tiles = resolve_tiles_dir(grid_id, region=region, imagery_layer=imagery_layer)
    except Exception:
        return None
    if tiles.is_file():  # mosaic layout
        return tiles
    if not source_tile:
        return None
    candidate = tiles / f"{source_tile}.tif"
    return candidate


def collect_rows(
    gpkgs: list[Path],
    *,
    region: str,
    imagery_layer: str | None,
    min_conf: float,
    max_conf: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    pv_rows: list[dict[str, Any]] = []
    nonpv_rows: list[dict[str, Any]] = []
    stats = {
        "gpkgs_scanned": 0,
        "rows_in_band": 0,
        "skipped_status": 0,
        "skipped_no_conf": 0,
        "skipped_tile_missing": 0,
    }
    for path in gpkgs:
        try:
            gdf = gpd.read_file(path)
        except Exception:
            continue
        if "confidence" not in gdf.columns or "review_status" not in gdf.columns:
            continue
        stats["gpkgs_scanned"] += 1
        grid_id = _grid_id_from_gpkg(path)
        gdf = gdf.reset_index(drop=True)  # positional index == pred_id for iloc
        for pred_id, row in gdf.iterrows():
            conf = row.get("confidence")
            if conf is None or (isinstance(conf, float) and math.isnan(conf)):
                stats["skipped_no_conf"] += 1
                continue
            conf = float(conf)
            if not (min_conf <= conf < max_conf):
                continue
            status = str(row.get("review_status") or "").strip().lower()
            if status in PV_STATUSES:
                ra_label = "pv"
            elif status in NONPV_STATUSES:
                ra_label = "non_pv"
            else:
                stats["skipped_status"] += 1
                continue
            source_tile = str(row.get("source_tile") or "").strip()
            tile_path = _resolve_tile_path(grid_id, region, imagery_layer, source_tile)
            if tile_path is None or not tile_path.exists():
                stats["skipped_tile_missing"] += 1
                continue
            stats["rows_in_band"] += 1
            area = row.get("area_m2")
            rec = {
                "candidate_id": f"{grid_id}_pred{int(pred_id):06d}",
                "grid_id": grid_id,
                "pred_id": int(pred_id),
                "region_key": region,
                "predictions_path": str(path.resolve()),
                "tile_path": str(tile_path.resolve()),
                "source_tile": source_tile,
                "confidence": f"{conf:.6f}",
                "area_m2": "" if area is None or (isinstance(area, float) and math.isnan(area)) else f"{float(area):.4f}",
                "review_status": status,
                "ra_label": ra_label,
            }
            (pv_rows if ra_label == "pv" else nonpv_rows).append(rec)
    return pv_rows, nonpv_rows, stats


def sample(rows: list[dict[str, Any]], cap: int | None, seed: int) -> list[dict[str, Any]]:
    if cap is None or cap >= len(rows):
        return rows
    # Deterministic sample without Random() global state issues: sort by a
    # seeded hash of candidate_id, take the first `cap`.
    import hashlib

    def key(r: dict[str, Any]) -> str:
        h = hashlib.sha256(f"{seed}:{r['candidate_id']}".encode()).hexdigest()
        return h

    return sorted(rows, key=key)[:cap]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reviewed-glob",
        default="results/G*/review/G*_reviewed.gpkg",
        help="Glob for reviewed gpkgs (default: CT legacy-flat results).",
    )
    ap.add_argument("--region", default="cape_town")
    ap.add_argument("--imagery-layer", default=None, help="Override imagery layer; else registry default.")
    ap.add_argument("--min-conf", type=float, default=0.95)
    ap.add_argument("--max-conf", type=float, default=1.0001)
    ap.add_argument("--nonpv-cap", type=int, default=None, help="Cap non_PV samples (default: take all).")
    ap.add_argument(
        "--pv-cap",
        type=int,
        default=None,
        help="Cap PV samples (default: match #non_PV kept, for a balanced discrimination set).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    region = region_registry.normalize_region_key(args.region) or args.region
    gpkgs = [Path(p) for p in sorted(glob.glob(args.reviewed_glob))]
    if not gpkgs:
        raise SystemExit(f"No reviewed gpkgs matched: {args.reviewed_glob}")

    pv_rows, nonpv_rows, stats = collect_rows(
        gpkgs,
        region=region,
        imagery_layer=args.imagery_layer,
        min_conf=args.min_conf,
        max_conf=args.max_conf,
    )

    nonpv_kept = sample(nonpv_rows, args.nonpv_cap, args.seed)
    pv_cap = args.pv_cap if args.pv_cap is not None else len(nonpv_kept)
    pv_kept = sample(pv_rows, pv_cap, args.seed)

    out_rows = nonpv_kept + pv_kept
    out_rows.sort(key=lambda r: (r["grid_id"], r["pred_id"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    print("=== calibration manifest summary ===")
    print(f"reviewed gpkgs scanned : {stats['gpkgs_scanned']} / {len(gpkgs)} matched")
    print(f"confidence band        : [{args.min_conf}, {args.max_conf})")
    print(f"available in band      : pv={len(pv_rows)}  non_pv={len(nonpv_rows)}")
    print(f"sampled (written)      : pv={len(pv_kept)}  non_pv={len(nonpv_kept)}  total={len(out_rows)}")
    print(f"skipped: status={stats['skipped_status']} no_conf={stats['skipped_no_conf']} tile_missing={stats['skipped_tile_missing']}")
    print(f"out_csv                : {args.out_csv}")


if __name__ == "__main__":
    main()
