#!/usr/bin/env python3
"""Bulk-download Vexcel 2024 ortho tiles for all 382 JNB JHB grids.

Writes directly to the canonical inference layout expected by detect_direct.py:

    <TILES_ROOT>/<grid_id>/<grid_id>_<col>_<row>_geo.tif

Resume:
    A tile is skipped if the file exists AND starts with a TIFF signature.

Per-tile failures (HTTP 4xx other than 429) are logged to the manifest but do
not abort the run — common for edge cells outside the collection footprint.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV = PROJECT_ROOT / ".env"
DEFAULT_GRID_GPKG = PROJECT_ROOT / "data" / "vexcel_task_grids" / "joburg_task_grid.gpkg"
DEFAULT_TILES_ROOT = Path("/home/gaosh/zasolar_data/tiles/johannesburg/vexcel_2024")
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "vexcel_eval_samples" / "jhb_jnb382_tile_manifest.csv"
)
DEFAULT_COLLECTION = "za-gp-johannesburg-2024"

sys.path.insert(0, str(PROJECT_ROOT))
from core.vexcel_auth import load_env, resolve_token  # noqa: E402

TIFF_SIGNATURES = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def tiff_like(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 8:
        return False
    with path.open("rb") as f:
        return f.read(4) in TIFF_SIGNATURES


def iter_chunks(geom, n_cols: int, n_rows: int):
    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    height = maxy - miny
    for col in range(n_cols):
        txmin = minx + width * col / n_cols
        txmax = minx + width * (col + 1) / n_cols
        for row in range(n_rows):
            tymax = maxy - height * row / n_rows
            tymin = maxy - height * (row + 1) / n_rows
            tile_box = box(txmin, tymin, txmax, tymax)
            clipped = tile_box.intersection(geom)
            if not clipped.is_empty and clipped.area > 1e-12:
                yield col, row, clipped


def fetch_tile(
    *,
    base_url: str,
    token: str,
    collection: str,
    wkt: str,
    out_path: Path,
    layer: str,
    retries: int,
) -> dict[str, Any]:
    if tiff_like(out_path):
        return {"status": "skipped", "bytes": out_path.stat().st_size, "error": ""}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_suffix(out_path.suffix + ".part")
    params = {
        "layer": layer,
        "collection": collection,
        "wkt": wkt,
        "srid": "4326",
        "image-format": "tiff",
        "token": token,
    }
    last = ""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(
                f"{base_url.rstrip('/')}/ortho/extract", params=params, timeout=300
            )
            if r.status_code != 200:
                last = f"HTTP {r.status_code}: {r.text[:240]}"
                # 4xx (except 429) is permanent — no point retrying
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    return {"status": "failed", "bytes": 0, "error": last}
                raise RuntimeError(last)
            if r.content[:4] not in TIFF_SIGNATURES:
                last = f"non-TIFF response: {r.text[:240]}"
                raise RuntimeError(last)
            temp.write_bytes(r.content)
            temp.replace(out_path)
            return {"status": "downloaded", "bytes": out_path.stat().st_size, "error": ""}
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            if temp.exists():
                try:
                    temp.unlink()
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(min(30, 3 * attempt))
    return {"status": "failed", "bytes": 0, "error": last}


def _task(task: dict[str, Any]) -> dict[str, Any]:
    result = fetch_tile(
        base_url=task["base_url"],
        token=task["token"],
        collection=task["collection"],
        wkt=task["wkt"],
        out_path=task["out_path"],
        layer=task["layer"],
        retries=task["retries"],
    )
    return {
        "grid_id": task["grid_id"],
        "col": task["col"],
        "row": task["row"],
        "collection": task["collection"],
        "tile_path": str(task["out_path"]),
        "status": result["status"],
        "bytes": result["bytes"],
        "error": result["error"],
        "wkt": task["wkt"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-gpkg", type=Path, default=DEFAULT_GRID_GPKG)
    p.add_argument("--tiles-root", type=Path, default=DEFAULT_TILES_ROOT)
    p.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    p.add_argument("--n-cols", type=int, default=2)
    p.add_argument("--n-rows", type=int, default=2)
    p.add_argument("--layer", default="urban")
    p.add_argument("--collection", default=DEFAULT_COLLECTION,
                   help="Fallback collection_id if gpkg row lacks one")
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--grid-ids", nargs="*", help="Only these JNB IDs (default: all)")
    p.add_argument("--limit", type=int, help="First N grids only (smoke test)")
    p.add_argument("--skip-edge", action="store_true",
                   help="Skip grids where is_edge=True (61 of 382)")
    p.add_argument("--progress-every", type=int, default=10)
    args = p.parse_args()

    env = load_env(args.env_file)
    base_url = env.get("VEXCEL_API_BASE", "https://api.vexcelgroup.com/v2")
    token = resolve_token(env, base_url)

    gdf = gpd.read_file(args.grid_gpkg)
    if "gridcell_id" not in gdf.columns:
        sys.exit(f"[fatal] {args.grid_gpkg} missing gridcell_id column")

    if args.skip_edge and "is_edge" in gdf.columns:
        gdf = gdf[~gdf["is_edge"].astype(bool)].copy()
    if args.grid_ids:
        gdf = gdf[gdf["gridcell_id"].isin(set(args.grid_ids))].copy()
    if args.limit:
        gdf = gdf.head(args.limit).copy()
    if gdf.empty:
        sys.exit("[fatal] no grids selected")

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.tiles_root.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    for _, row in gdf.iterrows():
        gid = row["gridcell_id"]
        coll = row.get("collection_id") if "collection_id" in gdf.columns else None
        if not coll or (isinstance(coll, float) and coll != coll):
            coll = args.collection
        for col, r, clipped in iter_chunks(row.geometry, args.n_cols, args.n_rows):
            out_path = args.tiles_root / gid / f"{gid}_{col}_{r}_geo.tif"
            tasks.append(
                {
                    "grid_id": gid,
                    "col": col,
                    "row": r,
                    "collection": coll,
                    "wkt": clipped.wkt,
                    "out_path": out_path,
                    "base_url": base_url,
                    "token": token,
                    "layer": args.layer,
                    "retries": args.retries,
                }
            )

    print(
        f"queued {len(tasks)} tiles across {len(gdf)} grids "
        f"({args.n_cols}x{args.n_rows} per grid); workers={args.workers}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    n_dl = n_skip = n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_task, t): t for t in tasks}
        for done_n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            if r["status"] == "downloaded":
                n_dl += 1
            elif r["status"] == "skipped":
                n_skip += 1
            else:
                n_fail += 1
            if done_n % args.progress_every == 0 or done_n == len(tasks):
                elapsed = time.time() - t0
                rate = done_n / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done_n) / rate if rate > 0 else 0
                print(
                    f"  [{done_n}/{len(tasks)}] dl={n_dl} skip={n_skip} fail={n_fail} "
                    f"rate={rate:.2f}/s eta={eta/60:.1f}min",
                    flush=True,
                )

    rows.sort(key=lambda r: (r["grid_id"], r["col"], r["row"]))
    with args.manifest_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(
        f"\ndone. downloaded={n_dl} skipped={n_skip} failed={n_fail} "
        f"in {(time.time()-t0)/60:.1f}min\n"
        f"manifest: {args.manifest_out}\n"
        f"tiles:    {args.tiles_root}",
        flush=True,
    )
    if n_fail:
        sys.exit(3)


if __name__ == "__main__":
    main()
