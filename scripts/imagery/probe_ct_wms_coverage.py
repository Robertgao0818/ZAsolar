"""
Cape Town WMS 影像覆盖探测
Probe per-cell aerial-imagery coverage for the Cape Town task grid.

For every cell in data/task_grid.gpkg this issues a small WMS GetMap request
to the City of Cape Town ERDAS IWS server and computes a coverage_fraction =
fraction of non-blank pixels. The server renders pure white (RGB 255,255,255,
std 0) where no aerial imagery exists (ocean, out-of-coverage), so a pixel is
counted as "covered" when it is NOT pure white. Calibration evidence and the
blank signature are documented in
results/analysis/ct_wms_coverage_probe/README.md.

数据源: City of Cape Town — Aerial Imagery 2025Jan (same WMS as download_tiles.py)
WMS:    https://cityimg.capetown.gov.za/erdas-iws/ogc/wms/GeoSpatial Datasets
Layer:  Aerial Imagery_Aerial Imagery 2025Jan

Usage:
  python scripts/imagery/probe_ct_wms_coverage.py                 # full grid
  python scripts/imagery/probe_ct_wms_coverage.py --limit 100     # first 100 unprobed
  python scripts/imagery/probe_ct_wms_coverage.py --resume        # skip done cells
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── WMS 配置（与 download_tiles.py 一致） ────────────────────────────────
WMS_URL = "https://cityimg.capetown.gov.za/erdas-iws/ogc/wms/GeoSpatial Datasets"
WMS_LAYER = "Aerial Imagery_Aerial Imagery 2025Jan"
WMS_FORMAT = "image/jpeg"

DEFAULT_GRID = PROJECT_ROOT / "data" / "task_grid.gpkg"
DEFAULT_OUT = (
    PROJECT_ROOT / "results" / "analysis" / "ct_wms_coverage_probe" / "probe.csv"
)

# Blank signature: ERDAS IWS renders no-imagery areas as pure white. A pixel is
# "blank" when all three bands are >= this threshold (JPEG can shift 255 a hair).
WHITE_THRESHOLD = 250

DEFAULT_WIDTH = 64
DEFAULT_HEIGHT = 64
DEFAULT_WORKERS = 6
DEFAULT_TIMEOUT = 60
MAX_ATTEMPTS = 3
BACKOFF_BASE = 2.0  # seconds; attempt n waits BACKOFF_BASE ** n

CSV_FIELDS = ["gridcell_id", "coverage_fraction", "status", "attempts"]


def coverage_fraction_from_array(arr: np.ndarray) -> float:
    """Fraction of non-blank (non-pure-white) pixels in an RGB array."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    blank = (arr >= WHITE_THRESHOLD).all(axis=-1)
    return float(1.0 - blank.mean())


def fetch_cell_coverage(
    bounds: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    timeout: int,
    session: requests.Session,
) -> float:
    """Issue one GetMap and return coverage_fraction. Raises on failure."""
    xmin, ymin, xmax, ymax = bounds
    params = {
        "service": "WMS",
        "version": "1.1.1",
        "request": "GetMap",
        "layers": WMS_LAYER,
        "srs": "EPSG:4326",
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "width": width,
        "height": height,
        "format": WMS_FORMAT,
        "styles": "",
    }
    response = session.get(WMS_URL, params=params, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "xml" in content_type.lower() or "html" in content_type.lower():
        raise RuntimeError(f"WMS returned non-image response: {content_type}")

    img = Image.open(BytesIO(response.content)).convert("RGB")
    return coverage_fraction_from_array(np.array(img))


def probe_cell(
    grid_id: str,
    bounds: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    timeout: int,
    session: requests.Session,
) -> dict:
    """Probe a single cell with retries + exponential backoff."""
    last_err = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            frac = fetch_cell_coverage(
                bounds,
                width=width,
                height=height,
                timeout=timeout,
                session=session,
            )
            return {
                "gridcell_id": grid_id,
                "coverage_fraction": round(frac, 6),
                "status": "ok",
                "attempts": attempt,
            }
        except Exception as exc:  # noqa: BLE001 — record any failure mode
            last_err = str(exc)
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_BASE ** attempt)
    return {
        "gridcell_id": grid_id,
        "coverage_fraction": "",
        "status": "error",
        "attempts": MAX_ATTEMPTS,
        "_error": last_err,
    }


def load_done_ids(out_path: Path) -> set[str]:
    """Return gridcell_ids already present with status=ok in the output CSV."""
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                done.add(row["gridcell_id"])
    return done


def read_existing_rows(out_path: Path) -> dict[str, dict]:
    """Read existing CSV rows keyed by gridcell_id (last write wins)."""
    rows: dict[str, dict] = {}
    if out_path.exists():
        with out_path.open("r", newline="") as f:
            for row in csv.DictReader(f):
                rows[row["gridcell_id"]] = {k: row.get(k, "") for k in CSV_FIELDS}
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells already recorded with status=ok in --out.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Probe at most this many (unresolved) cells this run.",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(args.grid)
    id_col = "gridcell_id" if "gridcell_id" in gdf.columns else "Name"

    done = load_done_ids(args.out) if args.resume else set()
    existing = read_existing_rows(args.out)

    todo: list[tuple[str, tuple[float, float, float, float]]] = []
    for _, row in gdf.iterrows():
        gid = str(row[id_col])
        if args.resume and gid in done:
            continue
        todo.append((gid, tuple(float(v) for v in row.geometry.bounds)))

    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"Grid: {args.grid}  ({len(gdf)} cells)")
    print(f"Output: {args.out}")
    print(f"Already ok (resume): {len(done)}")
    print(f"Probing this run: {len(todo)}  (workers={args.workers})")
    if not todo:
        print("Nothing to do.")
        return

    results: dict[str, dict] = {}
    lock = threading.Lock()
    n_ok = 0
    n_err = 0

    def _worker(item):
        gid, bounds = item
        session = requests.Session()
        try:
            return probe_cell(
                gid,
                bounds,
                width=args.width,
                height=args.height,
                timeout=args.timeout,
                session=session,
            )
        finally:
            session.close()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, item): item[0] for item in todo}
        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            err = res.pop("_error", "")
            with lock:
                results[res["gridcell_id"]] = res
                completed += 1
                if res["status"] == "ok":
                    n_ok += 1
                else:
                    n_err += 1
            if completed % 100 == 0 or completed == len(todo):
                rate = completed / max(time.time() - t0, 1e-6)
                print(
                    f"  [{completed}/{len(todo)}] ok={n_ok} err={n_err} "
                    f"({rate:.1f} cells/s)"
                )
            if err and res["status"] == "error":
                print(f"  [ERROR] {res['gridcell_id']}: {err}")

    # Merge with existing rows and write out (resume-safe).
    existing.update(results)
    ordered_ids = [str(r[id_col]) for _, r in gdf.iterrows()]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for gid in ordered_ids:
            if gid in existing:
                writer.writerow({k: existing[gid].get(k, "") for k in CSV_FIELDS})

    print(
        f"\n[DONE] probed={len(todo)} ok={n_ok} err={n_err} "
        f"elapsed={time.time() - t0:.1f}s"
    )
    if n_err:
        print(f"  {n_err} cell(s) still in error — re-run with --resume to retry.")


if __name__ == "__main__":
    main()
