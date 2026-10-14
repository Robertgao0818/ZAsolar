#!/usr/bin/env python3
"""Ingest Li RA cross-domain annotation gpkgs (6 new Vexcel cities) into the repo.

Source: Dropbox ``RA_Solar/Li/<CITY>/<CITY><id>(<count>).gpkg`` (GeoSAM review-GUI
exports, EPSG:3857, one or more ``sam_*`` layers per file).

Output: ``data/annotations/Vexcel/<region>/<GRID_ID>.gpkg`` — single ``annotations``
layer, reprojected to EPSG:4326 (project vector convention), grid-id-canonical
filename so ``core.annotation_loader`` / ``region_registry`` resolve them as GT.

Handled anomalies (resolved from each file's internal ``sam_*`` layer name, which
embeds the true grid id):
  * ``PTA03220(60).gpkg`` -> ``PTA0322``  (trailing typo digit)
  * ``ELS165(90).gpkg``   -> ``ELS0165``  (missing leading zero)
  * ``ELS0180(0).gpkg``   dropped (empty duplicate of ``ELS0180(4).gpkg``)

Empty (0-feature) files are kept as zero-GT grids (valid cross-domain
false-positive-rate checks) by raw-copying the source gpkg.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd

BASE = Path("/mnt/c/Users/gaosh/Dropbox/RA_Solar/Li")
ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "annotations" / "Vexcel"

CITY_TO_REGION = {
    "PTA": "pretoria",
    "BFN": "bloemfontein",
    "DBN": "durban",
    "ELS": "east_london",
    "GQB": "gqeberha",
    "PMB": "pietermaritzburg",
}

OVERRIDE_GID = {            # malformed filename -> canonical grid id (from layer name)
    "PTA03220(60).gpkg": "PTA0322",
    "ELS165(90).gpkg": "ELS0165",
}
SKIP_FILES = {"ELS0180(0).gpkg"}   # empty duplicate of ELS0180(4)

GRID_RE = re.compile(r"^([A-Za-z]+\d{4})")


def grid_id_for(name: str) -> str | None:
    if name in OVERRIDE_GID:
        return OVERRIDE_GID[name]
    m = GRID_RE.match(name.replace(".gpkg", ""))
    return m.group(1) if m else None


def read_all_features(path: Path) -> gpd.GeoDataFrame:
    frames = []
    for lyr in fiona.listlayers(path):
        g = gpd.read_file(path, layer=lyr)
        if len(g):
            frames.append(g)
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def main() -> int:
    rows = []
    seen: dict[str, int] = {}   # grid_id -> n (for intra-run dedup safety)
    for city, region in CITY_TO_REGION.items():
        dst_dir = DEST / region
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted((BASE / city).glob("*.gpkg")):
            if src.name in SKIP_FILES:
                rows.append(dict(region=region, grid_id="", src=src.name,
                                 n=0, action="skip(empty-dup)", dst=""))
                continue
            gid = grid_id_for(src.name)
            if gid is None:
                rows.append(dict(region=region, grid_id="", src=src.name,
                                 n=-1, action="skip(no-grid-id)", dst=""))
                continue
            gdf = read_all_features(src)
            n = len(gdf)
            dst = dst_dir / f"{gid}.gpkg"

            if gid in seen:   # defensive: prefer the richer file
                if n <= seen[gid]:
                    rows.append(dict(region=region, grid_id=gid, src=src.name,
                                     n=n, action="skip(dup<=kept)", dst=""))
                    continue
            seen[gid] = n

            if n == 0:
                shutil.copyfile(src, dst)         # keep as zero-GT grid
                action = "copy-empty"
                cx = cy = None
            else:
                g4326 = gdf.to_crs(4326) if (gdf.crs and gdf.crs.to_epsg() != 4326) else gdf
                out = g4326[["geometry"]].copy()
                out["src_file"] = src.name
                out.to_file(dst, driver="GPKG", layer="annotations")
                action = "reproject-write"
                c = g4326.union_all().centroid
                cx, cy = round(c.x, 5), round(c.y, 5)
            rows.append(dict(region=region, grid_id=gid, src=src.name, n=n,
                             action=action, dst=str(dst.relative_to(ROOT)),
                             lon=cx, lat=cy))

    df = pd.DataFrame(rows)
    manifest = DEST / "li_xdomain_manifest.csv"
    df.to_csv(manifest, index=False)

    kept = df[df.action.isin(["reproject-write", "copy-empty"])]
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"manifest: {manifest.relative_to(ROOT)}")
    print(f"grids written: {len(kept)}  (empties: {(kept.n==0).sum()}, "
          f"with-polys: {(kept.n>0).sum()})  total polygons: {int(kept.n[kept.n>0].sum())}")
    print("\nPer-region grids written:")
    print(kept.groupby('region').agg(grids=('grid_id', 'nunique'),
                                     polys=('n', lambda s: int(s[s > 0].sum()))).to_string())
    # per-region grid list for pod inference driving
    for region, sub in kept.groupby('region'):
        (DEST / region / "_grids.txt").write_text(
            "\n".join(sorted(sub.grid_id.tolist())) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
