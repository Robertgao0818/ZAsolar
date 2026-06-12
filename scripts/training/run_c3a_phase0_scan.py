#!/usr/bin/env python3
"""C-3(a) Phase 0 — low-confidence scan runner (deliverable B).

Runs the production detector (``exp_unified_reviewall_A``) over the grids that
the sampler drew chips from, at a low score threshold (~0.05), and extracts the
*background-region* proposals inside each sampled chip — the candidates the
audit must adjudicate (real PV vs lookalike vs ignore).

Two phases, both parameterized and runnable as-is:

  ``--phase detect``  : invoke ``detect_direct.py`` per (grid, imagery_layer)
                        with ``--detector-score-threshold``.  THIS IS THE ONLY
                        GPU STEP — run it on the pod (or a 5090).  It is a thin
                        subprocess wrapper; no inference logic lives here.
  ``--phase extract`` : CPU-only.  Reads the raw_detections.pkl artifacts +
                        the sampler's chip_manifest.csv + gt_refs gpkgs, and
                        writes ``proposals.csv`` (one row per background-region
                        proposal inside a sampled chip) + per-proposal mask
                        crops for the audit renderer.

The two phases are split so the GPU step and the CPU bookkeeping can run on
different machines.  ``--phase both`` runs detect then extract.

Detection<->chip matching is done in *source-tile pixel space*: both the raw
artifact's ``box_source_xyxy`` and the sampler's chip windows are pixel
coordinates on the same source TIF (joined by ``source_tile_id`` == tile_stem),
so no CRS round-trip is needed for the spatial test.  Metric proposal area is
computed from the tile's transform via a metric CRS lookup.

Usage
-----
    # On the pod (GPU):
    python scripts/training/run_c3a_phase0_scan.py --phase detect \
        --run-dir results/analysis/c3a_phase0/<run_id> \
        --model-path checkpoints/exp_unified_reviewall_A/best_model.pth \
        --model-run exp_unified_reviewall_A \
        --score-threshold 0.05

    # Anywhere (CPU):
    python scripts/training/run_c3a_phase0_scan.py --phase extract \
        --run-dir results/analysis/c3a_phase0/<run_id>
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.training.c3a_phase0 import proposal_overlaps_gt  # noqa: E402

DEFAULT_DETECT_OUTPUT = "raw_scans"  # sub-dir under run-dir for raw_detections.pkl


# ──────────────────────────────────────────────────────────────────────────
# Manifest IO
# ──────────────────────────────────────────────────────────────────────────


def _read_manifest(run_dir: Path) -> list[dict]:
    path = run_dir / "chip_manifest.csv"
    if not path.exists():
        sys.exit(f"[scan] missing {path} — run the sampler first")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _grid_jobs(rows: list[dict]) -> list[dict]:
    """Distinct (region, imagery_layer, grid_id) jobs to run the detector on."""
    seen: dict[tuple, dict] = {}
    for r in rows:
        key = (r["region"], r["imagery_layer"], r["grid_id"])
        seen.setdefault(key, {
            "region": r["region"],
            "imagery_layer": r["imagery_layer"],
            "grid_id": r["grid_id"],
        })
    return list(seen.values())


# ──────────────────────────────────────────────────────────────────────────
# Phase: detect (GPU; thin subprocess wrapper around detect_direct.py)
# ──────────────────────────────────────────────────────────────────────────


def phase_detect(args) -> int:
    rows = _read_manifest(args.run_dir)
    jobs = _grid_jobs(rows)
    out_root = args.run_dir / DEFAULT_DETECT_OUTPUT
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[scan][detect] {len(jobs)} grid jobs @ score>={args.score_threshold}")

    failures = []
    for job in jobs:
        grid = job["grid_id"]
        out_dir = out_root / job["region"] / job["imagery_layer"] / grid
        raw_path = out_dir / "raw_detections.pkl"
        if raw_path.exists() and not args.force:
            print(f"[scan][detect] skip {grid} (exists; --force to rerun)")
            continue
        cmd = [
            sys.executable, str(REPO_ROOT / "detect_direct.py"),
            "--grid-id", grid,
            "--region", job["region"],
            "--imagery-layer", job["imagery_layer"],
            "--model-run", args.model_run,
            "--model-path", str(args.model_path),
            "--detector-score-threshold", str(args.score_threshold),
            "--output-dir", str(out_dir),
        ]
        if args.detections_per_img is not None:
            cmd += ["--detections-per-img", str(args.detections_per_img)]
        if args.max_chips is not None:
            cmd += ["--max-chips", str(args.max_chips)]
        print(f"[scan][detect] {grid}: {' '.join(cmd)}")
        if args.dry_run:
            continue
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"[scan][detect][ERROR] {grid} exited {rc}")
            failures.append(grid)

    if failures:
        print(f"[scan][detect] FAILED grids: {failures}")
        return 1
    print("[scan][detect] done")
    return 0


# ──────────────────────────────────────────────────────────────────────────
# Phase: extract (CPU; background-region proposals per sampled chip)
# ──────────────────────────────────────────────────────────────────────────


def _load_gt_pixel_geoms(run_dir: Path, manifest_rows: list[dict]):
    """Load GT refs (in tile CRS) and convert to source-tile pixel space,
    keyed by chip_uid.  Returns {chip_uid: [pixel_polygon, ...]}."""
    import geopandas as gpd
    import rasterio
    from export_coco_dataset import polygon_to_pixel_coords

    # Map chip_uid -> tile_path for transform lookup.
    tile_for_chip = {r["chip_uid"]: r["tile_path"] for r in manifest_rows}

    # Cache transforms per tile.
    transform_cache: dict[str, object] = {}

    def _transform(tile_path: str):
        if tile_path not in transform_cache:
            with rasterio.open(tile_path) as src:
                transform_cache[tile_path] = src.transform
        return transform_cache[tile_path]

    gt_by_chip: dict[str, list] = {}
    for gpkg in sorted(run_dir.glob("gt_refs__*.gpkg")):
        gdf = gpd.read_file(gpkg)
        for _, row in gdf.iterrows():
            chip_uid = row["chip_uid"]
            tp = tile_for_chip.get(chip_uid)
            if tp is None:
                continue
            transform = _transform(tp)
            pgeom = polygon_to_pixel_coords(row.geometry, transform)
            if not pgeom.is_empty and pgeom.is_valid:
                gt_by_chip.setdefault(chip_uid, []).append(pgeom)
    return gt_by_chip


def phase_extract(args) -> int:
    from shapely.geometry import box as shapely_box

    from core.grid_utils import get_metric_crs
    from core.inference.raw_artifact import read_artifact

    run_dir = args.run_dir
    manifest_rows = _read_manifest(run_dir)
    out_root = run_dir / DEFAULT_DETECT_OUTPUT

    # GT in source-tile pixel space, per chip.
    gt_by_chip = _load_gt_pixel_geoms(run_dir, manifest_rows)

    # Group sampled chips by (region, imagery_layer, grid) and by source tile.
    chips_by_tile: dict[tuple, list[dict]] = {}
    for r in manifest_rows:
        key = (r["region"], r["imagery_layer"], r["grid_id"], r["tile_stem"])
        chips_by_tile.setdefault(key, []).append(r)

    # Metric CRS per (region, grid) for area_m2.
    def _metric_crs(region: str, grid_id: str) -> str:
        return get_metric_crs(grid_id, region=region)

    chips_dir = run_dir / "chips"
    (chips_dir / "rgb").mkdir(parents=True, exist_ok=True)

    proposals: list[dict] = []
    missing_artifacts: list[str] = []

    for (region, layer, grid, tile_stem), chip_rows in sorted(chips_by_tile.items()):
        raw_path = out_root / region / layer / grid / "raw_detections.pkl"
        if not raw_path.exists():
            missing_artifacts.append(f"{region}/{layer}/{grid}")
            continue
        artifact = read_artifact(raw_path)

        # Collect all detections on this tile_stem in source-pixel space.
        # Each detection: box_source_xyxy (pixel coords on the source TIF).
        tile_dets = []
        for ch in artifact.chips:
            if ch.source_tile_id != tile_stem:
                continue
            for d in ch.detections:
                tile_dets.append(d)

        # Metric area scale: project a unit pixel box to metric CRS once.
        tile_path = chip_rows[0]["tile_path"]
        try:
            m_crs = _metric_crs(region, grid)
            px_area_m2 = _pixel_area_m2(tile_path, m_crs)
        except Exception as e:  # noqa: BLE001
            print(f"[scan][extract][WARN] metric area for {grid}: {e}; using NaN")
            px_area_m2 = float("nan")

        for chip in chip_rows:
            chip_uid = chip["chip_uid"]
            x0, y0 = int(chip["x0"]), int(chip["y0"])
            w, h = int(chip["w"]), int(chip["h"])
            chip_box = shapely_box(x0, y0, x0 + w, y0 + h)
            gt_geoms = gt_by_chip.get(chip_uid, [])

            # Render chip RGB once (used by the audit overlay renderer).
            png_rel = _render_chip_rgb(
                tile_path, x0, y0, w, h, int(chip["chip_size"]),
                chips_dir / "rgb" / f"{_safe(chip_uid)}.png",
                run_dir,
            )

            pidx = 0
            for d in tile_dets:
                bx0, by0, bx1, by1 = d.box_source_xyxy
                det_box = shapely_box(bx0, by0, bx1, by1)
                if not det_box.intersects(chip_box):
                    continue
                # Keep the part of the detection inside the chip.
                det_in_chip = det_box.intersection(chip_box)
                if det_in_chip.is_empty or det_in_chip.area < 1:
                    continue
                # Background-region test: NOT overlapping any existing GT.
                if proposal_overlaps_gt(det_in_chip, gt_geoms,
                                        iof_threshold=args.gt_iof_threshold):
                    continue
                # Max IoF vs GT (diagnostic).
                max_iof = _max_iof(det_in_chip, gt_geoms)
                area_m2 = det_in_chip.area * px_area_m2
                proposals.append({
                    "chip_uid": chip_uid,
                    "region": region,
                    "imagery_layer": layer,
                    "grid_id": grid,
                    "tile_stem": tile_stem,
                    "x0": x0,
                    "y0": y0,
                    "chip_size": int(chip["chip_size"]),
                    "proposal_index": pidx,
                    "score": round(float(d.score), 4),
                    "proposal_area_m2": round(area_m2, 2) if area_m2 == area_m2 else "",
                    "max_iof_vs_gt": round(max_iof, 4),
                    "n_gt_in_chip": len(gt_geoms),
                    # proposal box in chip-local pixel coords (for the overlay):
                    "box_chip_x0": round(bx0 - x0, 1),
                    "box_chip_y0": round(by0 - y0, 1),
                    "box_chip_x1": round(bx1 - x0, 1),
                    "box_chip_y1": round(by1 - y0, 1),
                    "chip_png": png_rel,
                })
                pidx += 1

    # Write proposals.csv (renderer + audit builder input).
    prop_cols = [
        "chip_uid", "region", "imagery_layer", "grid_id", "tile_stem",
        "x0", "y0", "chip_size", "proposal_index", "score",
        "proposal_area_m2", "max_iof_vs_gt", "n_gt_in_chip",
        "box_chip_x0", "box_chip_y0", "box_chip_x1", "box_chip_y1", "chip_png",
    ]
    prop_path = run_dir / "proposals.csv"
    with open(prop_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=prop_cols)
        w.writeheader()
        for p in proposals:
            w.writerow(p)
    print(f"[scan][extract] wrote {prop_path} ({len(proposals)} background-region "
          f"proposals over {len(chips_by_tile)} tile groups)")

    n_chips_with_prop = len({p["chip_uid"] for p in proposals})
    print(f"[scan][extract] {n_chips_with_prop} chips have >= 1 background proposal")
    if missing_artifacts:
        print(f"[scan][extract][WARN] missing raw_detections.pkl for: "
              f"{sorted(set(missing_artifacts))}")
    return 0


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _safe(s: str) -> str:
    return s.replace(":", "_").replace("/", "_")


def _max_iof(geom, gt_geoms) -> float:
    if geom is None or geom.is_empty or geom.area <= 0 or not gt_geoms:
        return 0.0
    a = geom.area
    best = 0.0
    for g in gt_geoms:
        if g is None or g.is_empty or not geom.intersects(g):
            continue
        best = max(best, geom.intersection(g).area / a)
    return best


def _pixel_area_m2(tile_path: str, metric_crs: str) -> float:
    """Area (m^2) of one source pixel, via the tile transform projected to a
    metric CRS.  Robust to non-metric (4326) source CRS."""
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import box as shapely_box
    from shapely.ops import transform as shp_transform

    with rasterio.open(tile_path) as src:
        t = src.transform
        src_crs = src.crs
    # One-pixel box in source pixel space -> source CRS world coords.
    x0w, y0w = t * (0, 0)
    x1w, y1w = t * (1, 1)
    px_world = shapely_box(min(x0w, x1w), min(y0w, y1w),
                           max(x0w, x1w), max(y0w, y1w))
    if str(src_crs) == metric_crs:
        return px_world.area
    transformer = Transformer.from_crs(src_crs, metric_crs, always_xy=True)
    px_metric = shp_transform(
        lambda xs, ys: transformer.transform(xs, ys), px_world
    )
    return px_metric.area


def _render_chip_rgb(tile_path, x0, y0, w, h, chip_size, out_path: Path,
                     run_dir: Path) -> str:
    """Render the chip RGB to PNG; return path relative to run_dir."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return str(out_path.relative_to(run_dir))
    with rasterio.open(tile_path) as src:
        n_bands = min(3, src.count)
        data = src.read(list(range(1, n_bands + 1)),
                        window=Window(x0, y0, w, h))
    # (bands, h, w) -> (chip_size, chip_size, 3) uint8 padded.
    arr = np.zeros((chip_size, chip_size, 3), dtype=np.uint8)
    bands = data.shape[0]
    rgb = np.transpose(data[:bands], (1, 2, 0))
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    hh, ww = rgb.shape[:2]
    arr[:hh, :ww, :bands] = rgb[:, :, :bands]
    if bands == 1:
        arr[:hh, :ww, 1] = rgb[:, :, 0]
        arr[:hh, :ww, 2] = rgb[:, :, 0]
    try:
        from PIL import Image
        Image.fromarray(arr).save(out_path)
    except Exception:  # noqa: BLE001 — PIL optional; fall back to rasterio PNG
        import rasterio
        with rasterio.open(
            out_path, "w", driver="PNG", height=chip_size, width=chip_size,
            count=3, dtype="uint8",
        ) as dst:
            dst.write(np.transpose(arr, (2, 0, 1)))
    return str(out_path.relative_to(run_dir))


def main() -> int:
    ap = argparse.ArgumentParser(description="C-3(a) Phase 0 low-conf scan runner")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Sampler output dir (has chip_manifest.csv + gt_refs)")
    ap.add_argument("--phase", choices=["detect", "extract", "both"],
                    default="both")
    # detect phase
    ap.add_argument("--model-path", type=Path,
                    default=REPO_ROOT / "checkpoints/exp_unified_reviewall_A/best_model.pth")
    ap.add_argument("--model-run", default="exp_unified_reviewall_A")
    ap.add_argument("--score-threshold", type=float, default=0.05,
                    help="detect_direct --detector-score-threshold (low-conf scan)")
    ap.add_argument("--detections-per-img", type=int, default=None)
    ap.add_argument("--max-chips", type=int, default=None,
                    help="cap chips per grid (smoke only)")
    ap.add_argument("--force", action="store_true",
                    help="re-run detect even if raw_detections.pkl exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="detect phase: print commands, do not launch GPU")
    # extract phase
    ap.add_argument("--gt-iof-threshold", type=float, default=0.10,
                    help="a proposal with IoF >= this vs any GT is 'labeled' "
                         "(excluded from background-region candidates)")
    args = ap.parse_args()

    if args.phase in ("detect", "both"):
        rc = phase_detect(args)
        if rc != 0:
            return rc
    if args.phase in ("extract", "both"):
        return phase_extract(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
