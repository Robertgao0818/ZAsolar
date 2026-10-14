"""Query Vexcel /ortho/dates API for each JNB grid cell centroid.

Writes: data/analysis/vexcel_jhb_per_grid_capture_dates_2026-06-04.csv

Usage:
    cd /home/gaosh/projects/ZAsolar
    source scripts/activate_env.sh
    python scripts/analysis/query_vexcel_capture_dates.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import MultiPolygon, Polygon
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Bootstrap: allow running from repo root without install
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.vexcel_auth import load_env, mint_token  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = "https://api.vexcelgroup.com/v2"
COLLECTION = "za-gp-johannesburg-2024"
LAYER = "urban"
CONCURRENCY = 8          # polite concurrency for metadata-only queries
RETRY_SLEEP = 0.5        # seconds between retries on transient errors
MAX_RETRIES = 3          # retries on 5xx / network errors

OUTPUT_CSV = REPO_ROOT / "data/analysis/vexcel_jhb_per_grid_capture_dates_2026-06-04.csv"
GRID_FILE = REPO_ROOT / "data/vexcel_task_grids/joburg_task_grid.gpkg"

# Quarter-point offsets (fraction of cell bounds) for fallback probing
# Each is (lon_frac, lat_frac) relative to cell centroid displaced to a
# quarter-point of the cell bounding box.
FALLBACK_OFFSET_FRACS = [
    (+0.25, +0.25),
    (+0.25, -0.25),
    (-0.25, +0.25),
    (-0.25, -0.25),
]
# We estimate the cell half-extent from coverage_fraction/area; simpler: just
# use a fixed small offset (~250m in degrees at JHB latitude).
FALLBACK_DELTA_DEG = 0.002  # ~220m at 26°S, well within 1km cell


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def query_dates(lon: float, lat: float, token: str, session: requests.Session) -> dict | None:
    """
    Query /ortho/dates at (lon, lat).
    Returns parsed JSON dict on success, None on 404 (outside coverage).
    Raises RuntimeError on unexpected errors after retries.
    """
    wkt = f"POINT({lon:.8f} {lat:.8f})"
    params = {
        "collection": COLLECTION,
        "layer": LAYER,
        "wkt": wkt,
        "srid": "4326",
        "token": token,
    }
    url = f"{BASE_URL}/ortho/dates"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=20)
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Network error querying {wkt}: {exc}") from exc
            time.sleep(RETRY_SLEEP * attempt)
            continue

        if resp.status_code == 404:
            return None
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 401:
            # Signal caller to re-mint token
            raise RuntimeError("AUTH_EXPIRED")
        if resp.status_code >= 500:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Server error {resp.status_code} querying {wkt}: {resp.text[:200]}"
                )
            time.sleep(RETRY_SLEEP * attempt)
            continue
        # Other 4xx
        raise RuntimeError(
            f"Unexpected HTTP {resp.status_code} querying {wkt}: {resp.text[:200]}"
        )
    raise RuntimeError(f"All retries exhausted for {wkt}")


def query_with_fallback(
    grid_id: str, clon: float, clat: float, token: str, session: requests.Session
) -> dict:
    """
    Try centroid first, then 4 quarter-point offsets.
    Returns a result dict with all output fields filled.
    """
    probes = [(clon, clat)] + [
        (clon + dx * FALLBACK_DELTA_DEG, clat + dy * FALLBACK_DELTA_DEG)
        for dx, dy in FALLBACK_OFFSET_FRACS
    ]

    for idx, (lon, lat) in enumerate(probes):
        result = query_dates(lon, lat, token, session)
        if result is not None:
            fcd = result.get("first-capture-date", "")
            ecd = result.get("estimate-date", "")
            lcd = result.get("last-capture-date", "")
            # Bucket = calendar date portion of last-capture-date
            bucket = lcd[:10] if lcd else ""
            coverage_status = "centroid_hit" if idx == 0 else f"offset_{idx}_hit"
            return {
                "grid_id": grid_id,
                "centroid_lon": clon,
                "centroid_lat": clat,
                "first_capture_date": fcd,
                "estimate_date": ecd,
                "last_capture_date": lcd,
                "flight_bucket": bucket,
                "coverage_status": coverage_status,
            }

    # All probes returned 404
    return {
        "grid_id": grid_id,
        "centroid_lon": clon,
        "centroid_lat": clat,
        "first_capture_date": "",
        "estimate_date": "",
        "last_capture_date": "",
        "flight_bucket": "",
        "coverage_status": "no_coverage",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load grid
    print(f"Loading grid: {GRID_FILE}")
    gdf = gpd.read_file(GRID_FILE)
    assert len(gdf) == 382, f"Expected 382 JNB grids, got {len(gdf)}"
    assert "gridcell_id" in gdf.columns, "Missing gridcell_id column"
    assert "lon" in gdf.columns and "lat" in gdf.columns, "Missing lon/lat columns"
    print(f"  Loaded {len(gdf)} grids. CRS={gdf.crs}")

    # Confirm all IDs are JNB####
    non_jnb = gdf[~gdf["gridcell_id"].str.startswith("JNB")]
    if len(non_jnb) > 0:
        print(f"  WARNING: {len(non_jnb)} non-JNB IDs found: {non_jnb['gridcell_id'].tolist()[:5]}")

    # Build list of (grid_id, lon, lat)
    grids = [
        (row["gridcell_id"], float(row["lon"]), float(row["lat"]))
        for _, row in gdf.iterrows()
    ]

    # 2. Mint token
    env = load_env(REPO_ROOT / ".env")
    print("Minting Vexcel token...")
    token = mint_token(BASE_URL, env["VEXCEL_USER"], env["VEXCEL_PASSWORD"])
    print("  Token minted (not printed for security)")

    # 3. Query all grids with thread pool
    results: list[dict] = [None] * len(grids)  # type: ignore[list-item]
    token_lock = __import__("threading").Lock()
    token_holder = {"token": token}

    def worker(idx_grid):
        idx, (grid_id, lon, lat) = idx_grid
        for _attempt in range(2):  # allow one token re-mint
            try:
                result = query_with_fallback(
                    grid_id, lon, lat, token_holder["token"], session
                )
                return idx, result
            except RuntimeError as exc:
                if "AUTH_EXPIRED" in str(exc):
                    with token_lock:
                        # Re-mint once
                        print("  Token expired — re-minting...")
                        token_holder["token"] = mint_token(
                            BASE_URL, env["VEXCEL_USER"], env["VEXCEL_PASSWORD"]
                        )
                    continue
                raise
        raise RuntimeError(f"Token re-mint did not help for {grid_id}")

    session = requests.Session()
    done = 0
    offset_hits = 0
    no_coverage = 0

    print(f"Querying {len(grids)} grids (concurrency={CONCURRENCY})...")
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(worker, (i, g)): i for i, g in enumerate(grids)}
        for fut in as_completed(futures):
            idx, res = fut.result()
            results[idx] = res
            done += 1
            status = res["coverage_status"]
            if status == "no_coverage":
                no_coverage += 1
            elif status != "centroid_hit":
                offset_hits += 1
            if done % 50 == 0 or done == len(grids):
                print(f"  {done}/{len(grids)} done | no_coverage={no_coverage} | offset_hits={offset_hits}")

    # 4. Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "grid_id", "centroid_lon", "centroid_lat",
        "first_capture_date", "estimate_date", "last_capture_date",
        "flight_bucket", "coverage_status",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nWrote {len(results)} rows → {OUTPUT_CSV}")

    # 5. Summary
    from collections import Counter
    buckets = Counter(r["flight_bucket"] for r in results if r["flight_bucket"])
    print("\n=== Flight bucket distribution ===")
    for bucket, count in sorted(buckets.items()):
        print(f"  {bucket}: {count} grids")
    print(f"  (no_coverage): {no_coverage} grids")

    covered = [r for r in results if r["last_capture_date"]]
    if covered:
        all_lcd = [r["last_capture_date"] for r in covered]
        print(f"\nMin last_capture_date: {min(all_lcd)[:10]}")
        print(f"Max last_capture_date: {max(all_lcd)[:10]}")

    print(f"\nOffset-fallback hits: {offset_hits}")
    print("Done.")


if __name__ == "__main__":
    main()
