#!/usr/bin/env python3
"""Batch tile downloader replicating Allmapsoft GEID via the legacy Google
Earth Pro API on kh.google.com — runs headless on any platform.

This is the operational follow-up to ``geid_python_prototype.py``: instead
of one tile, it walks a Joburg grid (or arbitrary bbox) and downloads every
tile in the GE Pro quadtree at the requested zoom, decrypting on the fly
with the recovered XOR cipher key.

Output layout (mirrors GEID's nesting so existing stitch tools work):
    <output-dir>/<GridID>/<GridID>/<zoom>/<x>/ges_<x>_<y>_<zoom>.jpg

Resume: tiles already on disk are skipped, so reruns are cheap.

A captured ``SessionId`` is required (the geauth handshake is not yet
replicated; capture one with the Frida hook in ``windows/hook_geid_ssl.js``
and pass it via --session-id or store in --session-id-file).

Known limitations:
- Per-tile XOR cipher key is fixed at 19759 B (data/geid_protocol/cipher_key.bin).
  ~0.1% of Joburg z=21 tiles in dense urban blocks are 20-22 KB and exceed
  this; their tails can't be decrypted. Extend the key in a Windows+Frida
  capture session and re-run failed grids (per-grid manifest.failed_xy).
- imgVer varies per geographic area. Script auto-probes the center tile of
  each grid against IMGVER_CANDIDATES; pass --img-ver N to force one.

Usage:
    # single grid (test)
    python scripts/imagery/geid_python_batch.py \\
        --grid-id G0772 \\
        --output-dir /mnt/d/ZAsolar/joburg_geid_python \\
        --session-id 'egPTzQADEAEAAQB...=='

    # full Joburg batch 1
    python scripts/imagery/geid_python_batch.py \\
        --all-batch1 \\
        --output-dir /mnt/d/ZAsolar/joburg_geid_python \\
        --session-id-file ~/.geid_session
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import requests
import geopandas as gpd

# Layout: <repo>/geid_reverse_engineering/python/this_file.py
#         <repo>/geid_reverse_engineering/artifacts/cipher_key.bin   (RE artifact)
#         <repo>/data/jhb_task_grid.gpkg                              (project data)
RE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RE_ROOT.parent
CIPHER_KEY_PATH = RE_ROOT / "artifacts" / "cipher_key.bin"
JHB_GRID_PATH = PROJECT_ROOT / "data" / "jhb_task_grid.gpkg"

USER_AGENT = (
    "GoogleEarth/7.3.6.9345(Windows;Microsoft Windows (6.2.9200.0);"
    "en;kml:2.2;client:Pro;type:default)"
)
GE_DIGIT_MAP = (0, 3, 1, 2)

# Per-area imagery versions. CBD uses 1010; Sandton/Midrand and most other
# Joburg areas use 1020/1022/1024/1029/1033 etc. Empirically derived by
# brute-forcing imgVer 1010-1099 against tiles in failed grids. Sorted by
# preference: highest = most recent capture, with 1010 last as the original
# CBD baseline. The downloader probes the center tile of each grid to pick
# its primary version, then falls back through this list per-tile if needed.
IMGVER_CANDIDATES = (1033, 1029, 1024, 1022, 1020, 1018, 1015, 1014, 1013, 1012, 1011, 1010)


# ---------------------------------------------------------------------------
# coordinate <-> tile  (GE Pro legacy quadtree, linear equirectangular)
# ---------------------------------------------------------------------------

def quadkey(x: int, y: int, z: int) -> str:
    qk = []
    for i in range(z - 1, -1, -1):
        x_bit = (x >> i) & 1
        y_bit = (y >> i) & 1
        qk.append(str(GE_DIGIT_MAP[(x_bit << 1) | y_bit]))
    return "".join(qk)


def bbox_to_tile_range(lon_min: float, lon_max: float, lat_min: float, lat_max: float, z: int):
    """Compute (x_min, x_max, y_min, y_max) tile range covering a bbox at GE zoom z.

    Empirically derived from GEID 6.48's own list1.txt: at z=N the world is a
    2^(N-1) × 2^(N-1) quadtree on a linear (NOT Mercator) projection. y grows
    NORTH (TMS-style)."""
    n = 2 ** (z - 1)
    factor = n / 360.0
    x_min = int((lon_min + 180.0) * factor)
    x_max = int((lon_max + 180.0) * factor)
    y_min = int((lat_min + 180.0) * factor)
    y_max = int((lat_max + 180.0) * factor)
    return x_min, x_max, y_min, y_max


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

@dataclass
class GridResult:
    grid_id: str
    total: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    failed_tiles: list[tuple[int, int]] = field(default_factory=list)
    elapsed: float = 0.0


class GeidClient:
    def __init__(self, session_id: str, cipher_key: bytes, timeout: float = 30.0):
        self.session_id = session_id
        self.cipher_key = cipher_key
        self.timeout = timeout
        s = requests.Session()
        s.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "identity",
            "Content-Type": "application/octet-stream",
            "Cookie": f'$Version="0"; SessionId="{session_id}"; State="1"',
        })
        self.session = s

    def _fetch_raw(self, qk: str, img_ver: int) -> tuple[int, bytes]:
        url = f"https://kh.google.com/flatfile?f1-{qk}-i.{img_ver}"
        last_exc = None
        for attempt in range(3):
            try:
                r = self.session.get(url, timeout=self.timeout)
                return r.status_code, r.content
            except (requests.Timeout, requests.ConnectionError) as e:
                last_exc = e
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"network error after retries: {last_exc}")

    def fetch_and_decrypt(self, x: int, y: int, z: int, img_ver: int,
                          fallback_vers: tuple[int, ...] = ()) -> bytes:
        """Fetch a tile and XOR-decrypt it. Tries primary img_ver first,
        then walks `fallback_vers` if 404."""
        qk = quadkey(x, y, z)
        tried = []
        for v in (img_ver,) + tuple(fallback_vers):
            if v in tried:
                continue
            tried.append(v)
            status, wire = self._fetch_raw(qk, v)
            if status == 200:
                if len(wire) > len(self.cipher_key):
                    raise RuntimeError(
                        f"tile {len(wire)}B > key {len(self.cipher_key)}B; extend cipher key"
                    )
                return bytes(a ^ b for a, b in zip(wire, self.cipher_key))
            if status in (401, 403):
                raise RuntimeError(f"auth failed ({status}) — SessionId likely expired")
            if status == 404:
                continue
            raise RuntimeError(f"HTTP {status}: {wire[:200]!r}")
        raise FileNotFoundError(f"no tile at zoom {z} for ({x},{y}) — tried imgVers {tried}")

    def probe_imgver(self, x: int, y: int, z: int,
                     candidates: tuple[int, ...] = IMGVER_CANDIDATES) -> int | None:
        """Find the highest imgVer that returns a valid tile at (x,y,z).
        Returns None if no candidate works."""
        qk = quadkey(x, y, z)
        for v in candidates:
            status, _ = self._fetch_raw(qk, v)
            if status == 200:
                return v
            if status in (401, 403):
                raise RuntimeError(f"auth failed ({status}) — SessionId likely expired")
        return None


def download_grid(
    client: GeidClient,
    grid_id: str,
    geom_bounds: tuple[float, float, float, float],
    output_dir: Path,
    z: int,
    workers: int,
    primary_imgver: int,
    quiet: bool = False,
) -> GridResult:
    lon_min, lat_min, lon_max, lat_max = geom_bounds
    x_min, x_max, y_min, y_max = bbox_to_tile_range(lon_min, lon_max, lat_min, lat_max, z)
    coords = [(x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]

    grid_dir = output_dir / grid_id
    grid_root = grid_dir / grid_id / str(z)
    grid_dir.mkdir(parents=True, exist_ok=True)
    result = GridResult(grid_id=grid_id, total=len(coords))
    started = time.monotonic()

    # Per-tile fallback list = all candidates minus the primary (already first)
    fallbacks = tuple(v for v in IMGVER_CANDIDATES if v != primary_imgver)

    def out_path(x: int, y: int) -> Path:
        return grid_root / str(x) / f"ges_{x}_{y}_{z}.jpg"

    def work(xy: tuple[int, int]):
        x, y = xy
        p = out_path(x, y)
        if p.exists() and p.stat().st_size > 0:
            return ("skip", x, y)
        try:
            data = client.fetch_and_decrypt(x, y, z, primary_imgver, fallbacks)
            if not data.startswith(b"\xff\xd8\xff"):
                return ("bad", x, y, "not a JPEG after decrypt")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            return ("ok", x, y)
        except Exception as e:
            return ("fail", x, y, str(e))

    last_log = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, c) for c in coords]
        for fut in as_completed(futures):
            r = fut.result()
            if r[0] == "ok":
                result.downloaded += 1
            elif r[0] == "skip":
                result.skipped += 1
            else:
                result.failed += 1
                result.failed_tiles.append((r[1], r[2]))
                if "auth failed" in (r[3] if len(r) > 3 else ""):
                    # Hard-stop the whole grid on auth failure
                    for f in futures:
                        f.cancel()
                    raise RuntimeError(r[3])
            now = time.monotonic()
            if not quiet and now - last_log > 2.0:
                done = result.downloaded + result.skipped + result.failed
                rate = result.downloaded / max(now - started, 0.01)
                print(f"  [{grid_id}] {done}/{result.total}  d={result.downloaded} s={result.skipped} f={result.failed}  {rate:.1f} dl/s",
                      file=sys.stderr, flush=True)
                last_log = now
    result.elapsed = time.monotonic() - started

    # Manifest
    manifest = {
        "grid_id": grid_id,
        "zoom": z,
        "primary_imgver": primary_imgver,
        "bbox_lon_lat": [lon_min, lat_min, lon_max, lat_max],
        "tile_range": {"x": [x_min, x_max], "y": [y_min, y_max]},
        "tile_count": result.total,
        "downloaded": result.downloaded,
        "skipped": result.skipped,
        "failed": result.failed,
        "failed_xy": result.failed_tiles,
        "elapsed_sec": round(result.elapsed, 2),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (grid_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_grid_geometries(grid_ids: list[str] | None, all_batch1: bool):
    g = gpd.read_file(JHB_GRID_PATH).set_index("gridcell_id")
    if all_batch1:
        # Batch 1 = the 100 G####-prefixed grids (excluding 6 legacy JHB##)
        g = g[g.index.str.startswith("G")]
        return list(g.index), g
    if not grid_ids:
        raise SystemExit("must pass --grid-id or --all-batch1")
    missing = [gid for gid in grid_ids if gid not in g.index]
    if missing:
        raise SystemExit(f"unknown grid IDs: {missing}")
    return grid_ids, g.loc[grid_ids]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-id", nargs="*", help="One or more grid IDs (e.g. G0772 G0773)")
    ap.add_argument("--all-batch1", action="store_true", help="All 100 batch1 grids")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--session-id", help="SessionId base64 (or use --session-id-file)")
    ap.add_argument("--session-id-file", type=Path, help="File with SessionId on first line")
    ap.add_argument("--cipher-key", type=Path, default=CIPHER_KEY_PATH)
    ap.add_argument("--zoom", type=int, default=21)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--img-ver", type=int, default=None,
                    help="Force specific imgVer for all grids; default = auto-probe per grid")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip grids that already have a manifest.json (resume mode)")
    ap.add_argument("--dry-run", action="store_true", help="List tiles per grid, don't download")
    args = ap.parse_args()

    if args.session_id:
        session_id = args.session_id
    elif args.session_id_file:
        session_id = args.session_id_file.read_text().strip().splitlines()[0]
    else:
        raise SystemExit("--session-id or --session-id-file required")

    if not args.cipher_key.exists():
        raise SystemExit(f"cipher key not found: {args.cipher_key}")
    cipher_key = args.cipher_key.read_bytes()

    grid_ids, geoms = load_grid_geometries(args.grid_id, args.all_batch1)
    print(f"target: {len(grid_ids)} grid(s) at zoom {args.zoom}, output → {args.output_dir}", file=sys.stderr)

    if args.dry_run:
        total = 0
        for gid in grid_ids:
            b = geoms.loc[gid].geometry.bounds
            x1, x2, y1, y2 = bbox_to_tile_range(b[0], b[2], b[1], b[3], args.zoom)
            n = (x2 - x1 + 1) * (y2 - y1 + 1)
            total += n
            print(f"  {gid}: {n} tiles  x=[{x1},{x2}] y=[{y1},{y2}]")
        print(f"TOTAL: {total} tiles")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = GeidClient(session_id, cipher_key)

    overall_started = time.monotonic()
    summary = []
    skipped_grids = []
    try:
        for i, gid in enumerate(grid_ids, 1):
            if args.skip_existing and (args.output_dir / gid / "manifest.json").exists():
                skipped_grids.append(gid)
                print(f"\n[{i}/{len(grid_ids)}] {gid}  SKIP (manifest exists)", file=sys.stderr)
                continue
            b = geoms.loc[gid].geometry.bounds
            print(f"\n[{i}/{len(grid_ids)}] {gid}  bbox={b}", file=sys.stderr)

            # Pick primary imgVer: forced, or probe the grid's center tile
            if args.img_ver is not None:
                primary = args.img_ver
                print(f"  imgVer (forced): {primary}", file=sys.stderr)
            else:
                x1, x2, y1, y2 = bbox_to_tile_range(b[0], b[2], b[1], b[3], args.zoom)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                primary = client.probe_imgver(cx, cy, args.zoom)
                if primary is None:
                    print(f"  ✗ no working imgVer found at center ({cx},{cy}) — skipping",
                          file=sys.stderr)
                    skipped_grids.append(gid)
                    continue
                print(f"  imgVer (probed): {primary}", file=sys.stderr)

            r = download_grid(client, gid, b, args.output_dir, args.zoom, args.workers, primary)
            summary.append(r)
            print(f"  → {gid}: {r.downloaded} dl, {r.skipped} skip, {r.failed} fail in {r.elapsed:.1f}s",
                  file=sys.stderr)
    except KeyboardInterrupt:
        print("\n[geid] interrupted by user — partial state saved per-grid manifest", file=sys.stderr)

    elapsed = time.monotonic() - overall_started
    print(f"\n=== batch summary ===", file=sys.stderr)
    print(f"grids done : {len(summary)}/{len(grid_ids)}", file=sys.stderr)
    print(f"tiles dl   : {sum(r.downloaded for r in summary)}", file=sys.stderr)
    print(f"tiles skip : {sum(r.skipped for r in summary)}", file=sys.stderr)
    print(f"tiles fail : {sum(r.failed for r in summary)}", file=sys.stderr)
    print(f"elapsed    : {elapsed:.1f}s", file=sys.stderr)
    return 0 if all(r.failed == 0 for r in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
