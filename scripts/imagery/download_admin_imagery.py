#!/usr/bin/env python3
"""Orchestrator: download per-grid imagery for full-admin inference.

Reads `data/imagery_plans/<region>_plan.csv`, fetches each grid via its
primary_source (ArcGIS or Vexcel), falls back on empty response, writes
chunks to `~/zasolar_data/tiles/<region>/<imagery_layer>/<grid_id>/`,
emits MANIFEST.json, and updates the plan CSV in place.

Usage:
    # Smoke: 5 grids per source
    python scripts/imagery/download_admin_imagery.py --region durban --smoke 5

    # Resume full run with 6 workers
    python scripts/imagery/download_admin_imagery.py --region joburg --workers 6 --resume

Source kinds supported:
  - arcgis_imageserver  → _arcgis_fetch.fetch_arcgis_chunk
  - vexcel              → defers to download_vexcel_eval_sample.py (TODO)

Vexcel sources are stubbed in this version (smoke focuses on ArcGIS).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import urllib3
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "imagery"))
from _arcgis_fetch import fetch_arcgis_chunk, grid_chunks_3857, FetchResult  # noqa: E402

DEFAULT_TILES_ROOT = Path.home() / "zasolar_data" / "tiles"
PLANS_DIR = PROJECT_ROOT / "data" / "imagery_plans"
SOURCES_CONFIG = PROJECT_ROOT / "configs" / "datasets" / "aerial_sources.yaml"

# region_key (vexcel_urban_coverage.yaml) → canonical tile-dir name
# (matches existing tiles_root in regions.yaml; new cities default to key)
REGION_TILE_DIR = {
    "joburg": "johannesburg",
    "durban": "durban",
    "pretoria": "pretoria",
    "bloemfontein": "bloemfontein",
    "east_london": "east_london",
    "gqeberha": "gqeberha",
    "pietermaritzburg": "pietermaritzburg",
}


def load_sources(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def grid_dir(tiles_root: Path, region: str, layer: str, grid_id: str) -> Path:
    region_dir = REGION_TILE_DIR.get(region, region)
    return tiles_root / region_dir / layer / grid_id


def _imagery_layer_id(source_key: str) -> str:
    """Map source registry key to imagery_layer id used in tiles path.

    e.g. vexcel_durban_2026 → vexcel_2026
         coj_aerial_2023    → aerial_2023
         ethekwini_2023     → ethekwini_2023
         ethekwini_2022     → ethekwini_2022
         vexcel_joburg_2024 → vexcel_2024
    """
    if source_key.startswith("vexcel_"):
        # extract trailing year
        return f"vexcel_{source_key.rsplit('_', 1)[-1]}"
    if source_key == "coj_aerial_2023":
        return "aerial_2023"
    return source_key  # ethekwini_*


def fetch_grid_arcgis(*, grid_row: pd.Series, source: dict[str, Any],
                      source_key: str, fallback_key: str | None,
                      sources_cfg: dict[str, Any], chunk_cfg: dict[str, Any],
                      tiles_root: Path, force: bool = False) -> dict[str, Any]:
    region = grid_row["region_key"]
    grid_id = grid_row["gridcell_id"]
    layer_id = _imagery_layer_id(source_key)
    out_dir = grid_dir(tiles_root, region, layer_id, grid_id)

    manifest_path = out_dir / "MANIFEST.json"
    if manifest_path.exists() and not force:
        try:
            existing = json.loads(manifest_path.read_text())
            if existing.get("status") == "ok":
                return {"grid_id": grid_id, "skipped": True,
                        "source_used": existing.get("source_used", source_key)}
        except Exception:
            pass

    cps = chunk_cfg["per_grid"][0]  # square
    sub_max = chunk_cfg["arcgis_max_request_px"]
    chunks = grid_chunks_3857(grid_row["lon"], grid_row["lat"],
                              grid_size_m=1000.0, chunks_per_side=cps)
    # Per-source chunk_px: never request finer than native GSD (server rejects upsample)
    chunk_m = chunks[0]["width_m"]
    primary_gsd = float(source["gsd_m"])
    chunk_px = max(100, round(chunk_m / primary_gsd / 100) * 100)

    # Try primary; on empty_response or any chunk fail, retry that chunk on fallback (if any).
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_results: list[dict[str, Any]] = []
    layer_used = source_key
    sources_used: set[str] = set()
    primary_empty_chunks = 0

    def attempt(src_key: str, chunk: dict, out_path: Path) -> FetchResult:
        s = sources_cfg["sources"][src_key]
        return fetch_arcgis_chunk(
            base_url=s["url"], chunk=chunk, chunk_px=chunk_px,
            sub_max_px=sub_max, output_sr=int(s.get("output_sr", 3857)),
            out_path=out_path, verify_ssl=bool(s.get("verify_ssl", True)),
        )

    t_start = time.perf_counter()
    for chunk in chunks:
        chunk_name = f"{grid_id}_{chunk['col']}_{chunk['row']}_geo.tif"

        # Try primary then fallback per chunk (matches user's policy choice (b))
        result = attempt(source_key, chunk, out_dir / chunk_name)
        used = source_key
        if not result.ok and result.reason == "empty_response" and fallback_key:
            primary_empty_chunks += 1
            result = attempt(fallback_key, chunk, out_dir / chunk_name)
            used = fallback_key if result.ok else source_key
        if not result.ok:
            chunk_results.append({"chunk": chunk_name, "ok": False,
                                  "reason": result.reason, "source": used})
            continue
        sources_used.add(used)
        chunk_results.append({
            "chunk": chunk_name, "ok": True,
            "bytes": result.bytes_out, "elapsed_s": round(result.elapsed_s, 2),
            "sub_requests": result.sub_requests, "source": used,
        })

    n_ok = sum(1 for c in chunk_results if c["ok"])
    n_total = len(chunks)
    status = "ok" if n_ok == n_total else ("partial" if n_ok > 0 else "failed")
    is_mixed = False
    if len(sources_used) == 1:
        layer_used = next(iter(sources_used))
    elif len(sources_used) > 1:
        layer_used = f"mixed:{','.join(sorted(sources_used))}"
        is_mixed = True
    else:
        layer_used = source_key

    # Provenance: if all chunks served by one source != primary, move files to that source's dir
    final_layer_dir_id = _imagery_layer_id(source_key)
    if not is_mixed and layer_used != source_key and status == "ok":
        new_layer_id = _imagery_layer_id(layer_used)
        new_dir = grid_dir(tiles_root, region, new_layer_id, grid_id)
        new_dir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.iterdir():
            if f.suffix == ".tif":
                f.rename(new_dir / f.name)
        # cleanup primary dir if empty
        try: out_dir.rmdir()
        except OSError: pass
        out_dir = new_dir
        final_layer_dir_id = new_layer_id

    manifest_path = out_dir / "MANIFEST.json"
    manifest = {
        "grid_id": grid_id, "region": region,
        "imagery_layer": final_layer_dir_id,
        "source_key": source_key,
        "fallback_chain": [source_key] + ([fallback_key] if fallback_key else []),
        "source_used": layer_used,
        "mixed_sources": is_mixed,
        "primary_empty_chunks": primary_empty_chunks,
        "status": status,
        "n_chunks_ok": n_ok, "n_chunks_total": n_total,
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "centroid_lon": float(grid_row["lon"]),
        "centroid_lat": float(grid_row["lat"]),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "chunk_px": chunk_px, "chunks_per_side": cps,
        "chunks": chunk_results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {"grid_id": grid_id, "status": status, "source_used": layer_used,
            "n_ok": n_ok, "n_total": n_total,
            "elapsed_s": manifest["elapsed_s"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True)
    parser.add_argument("--plan", type=Path,
                        help="Override plan CSV path (default: data/imagery_plans/<region>_plan.csv)")
    parser.add_argument("--sources-config", type=Path, default=SOURCES_CONFIG)
    parser.add_argument("--tiles-root", type=Path, default=DEFAULT_TILES_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", type=int, default=0,
                        help="Smoke mode: cap to N grids per primary_source")
    parser.add_argument("--source-only", default="",
                        help="Filter to grids with this primary_source")
    parser.add_argument("--resume", action="store_true",
                        help="Skip grids whose plan status='ok'")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if manifest is ok")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan_path = args.plan or (PLANS_DIR / f"{args.region}_plan.csv")
    if not plan_path.exists():
        print(f"[ERR] plan not found: {plan_path}")
        sys.exit(2)
    sources_cfg = load_sources(args.sources_config)
    chunk_cfg = sources_cfg["chunk"]

    plan = pd.read_csv(plan_path)
    plan = plan[plan.region_key == args.region]
    if args.source_only:
        plan = plan[plan.primary_source == args.source_only]
    if args.resume:
        plan = plan[plan.status != "ok"]
    if args.smoke > 0:
        plan = plan.groupby("primary_source", group_keys=False).head(args.smoke)

    print(f"[plan] region={args.region}  grids to fetch: {len(plan)}")
    if args.dry_run:
        print(plan[["gridcell_id","primary_source","fallback_source","lon","lat"]].head(20).to_string(index=False))
        print(f"\n(dry run; would write under {args.tiles_root})")
        return

    if not args.tiles_root.exists():
        args.tiles_root.mkdir(parents=True, exist_ok=True)

    # Filter to ArcGIS-only for smoke (vexcel sources stubbed below)
    arcgis_keys = {k for k, v in sources_cfg["sources"].items()
                   if v["type"] == "arcgis_imageserver"}
    arcgis_plan = plan[plan.primary_source.isin(arcgis_keys)].copy()
    vexcel_plan = plan[~plan.primary_source.isin(arcgis_keys)].copy()
    if len(vexcel_plan):
        print(f"[note] {len(vexcel_plan)} Vexcel grids deferred (use download_vexcel_full_admin.py)")

    if not len(arcgis_plan):
        print("[done] no ArcGIS grids to fetch in this slice")
        return

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    def task(row: pd.Series) -> dict[str, Any]:
        return fetch_grid_arcgis(
            grid_row=row,
            source=sources_cfg["sources"][row["primary_source"]],
            source_key=row["primary_source"],
            fallback_key=(row["fallback_source"] or None) if isinstance(row["fallback_source"], str) and row["fallback_source"] else None,
            sources_cfg=sources_cfg, chunk_cfg=chunk_cfg,
            tiles_root=args.tiles_root, force=args.force,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(task, row): row["gridcell_id"]
                   for _, row in arcgis_plan.iterrows()}
        for fut in as_completed(futures):
            grid = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                tag = "✓" if r.get("status") == "ok" else ("◑" if r.get("status") == "partial" else "✗")
                if r.get("skipped"):
                    print(f"  ↷ {grid:<10} skipped (manifest ok)")
                else:
                    print(f"  {tag} {grid:<10} {r.get('source_used','?'):<24} "
                          f"{r.get('n_ok','?')}/{r.get('n_total','?')} chunks  "
                          f"{r.get('elapsed_s','?')}s")
            except Exception as e:
                print(f"  ✗ {grid} EXC: {e}")
                results.append({"grid_id": grid, "status": "error", "error": str(e)})

    # Write back plan status
    out_idx = {r["grid_id"]: r for r in results}
    plan_full = pd.read_csv(plan_path)
    for col in ("status", "source_used", "downloaded_at", "error"):
        if col in plan_full.columns:
            plan_full[col] = plan_full[col].astype(object)
    for i, row in plan_full.iterrows():
        r = out_idx.get(row["gridcell_id"])
        if not r: continue
        plan_full.at[i, "status"] = r.get("status", row["status"])
        plan_full.at[i, "source_used"] = r.get("source_used", "")
        plan_full.at[i, "downloaded_at"] = datetime.now(timezone.utc).isoformat() if r.get("status")=="ok" else ""
        plan_full.at[i, "error"] = r.get("error", "") if r.get("status") not in ("ok","partial") else ""
    plan_full.to_csv(plan_path, index=False)

    # Summary
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_partial = sum(1 for r in results if r.get("status") == "partial")
    n_failed = sum(1 for r in results if r.get("status") == "failed")
    n_err = sum(1 for r in results if r.get("status") == "error")
    print(f"\n[done] {len(results)} grids  ok={n_ok} partial={n_partial} failed={n_failed} err={n_err}  "
          f"total {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    main()
