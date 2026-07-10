"""Stage Li's Cape Town GT into data/annotations/Capetown_Li/ as L-prefix gpkgs.

One-shot ingest script (Phase 0.5 of ct-census-model-comparison-plan). Reads the
Dropbox source gpkgs (filenames carry spaces / full-width parens; the paren
number is the polygon count), copies each to a clean ``L<NNNN>.gpkg``, merges the
3-part G1896 zip into a single layer, validates each file (polygon count vs the
paren count, CRS, layer name) and writes the manifest CSV.

Li uses an INDEPENDENT grid scheme: source Gao-named IDs (G1842..G1954) are
mapped to L-prefix IDs (L1842..L1954) to avoid collision with Gao's grid
namespace (Li's G1895 != Gao's G1895). Run from repo root with the project venv.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

SRC_DIR = Path("/mnt/c/Users/gaosh/Dropbox/RA_Solar/Li/capetown")
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "annotations" / "Capetown_Li"
G1896_PARTS_DIR = OUT_DIR / "_g1896_parts"

# Source Gao-named gpkg filename -> (L-prefix grid id, paren count from filename).
# Filenames use a mix of ASCII () and full-width （） parens.
SOURCE_FILES: dict[str, str] = {
    "G1842  (31).gpkg": "L1842",
    "G1843（21）.gpkg": "L1843",
    "G1844 (46).gpkg": "L1844",
    "G1846 (5).gpkg": "L1846",
    "G1895 (260).gpkg": "L1895",
    # G1896 handled separately (zip with 3 parts)
    "G1897（112）.gpkg": "L1897",
    "G1898(26）.gpkg": "L1898",
    "G1899（94）.gpkg": "L1899",
    "G1900(41).gpkg": "L1900",
    "G1901（12）.gpkg": "L1901",
    "G1902(8).gpkg": "L1902",
    "G1950（82）.gpkg": "L1950",
    "G1951（135）.gpkg": "L1951",
    "G1952（210）.gpkg": "L1952",
    "G1953(55).gpkg": "L1953",
    "G1954(153).gpkg": "L1954",
}

OUT_LAYER = "li_ct_gt"  # canonical layer name in the staged L<NNNN>.gpkg


def _paren_count(filename: str) -> int | None:
    """Extract the integer inside () or （） in a source filename."""
    m = re.search(r"[(（](\d+)[)）]", filename)
    return int(m.group(1)) if m else None


def _first_layer(path: Path) -> str:
    return pyogrio.list_layers(str(path))[0][0]


def stage_simple(src_name: str, lid: str) -> dict:
    src = SRC_DIR / src_name
    layer = _first_layer(src)
    gdf = gpd.read_file(src, layer=layer)
    paren = _paren_count(src_name)
    crs = str(gdf.crs)
    out = OUT_DIR / f"{lid}.gpkg"
    gdf.to_file(out, layer=OUT_LAYER, driver="GPKG")
    n = len(gdf)
    return {
        "grid_id": lid,
        "src_filename": src_name,
        "n_polygons": n,
        "paren_count": paren,
        "count_match": "yes" if paren == n else "MISMATCH",
        "crs": crs,
        "src_layer_name": layer,
        "out_layer_name": OUT_LAYER,
        "label_source": "human_manual_sam_assisted",
        "semantic_confidence": "A2",
        "quality_tier": "T2",
    }


def stage_g1896() -> dict:
    """Merge the 3-part G1896 zip (already extracted to _g1896_parts/) into L1896."""
    parts = sorted(G1896_PARTS_DIR.glob("*.gpkg"))
    frames = []
    src_layers = []
    for p in parts:
        lyr = _first_layer(p)
        src_layers.append(lyr)
        frames.append(gpd.read_file(p, layer=lyr))
    merged = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), crs=frames[0].crs
    )
    out = OUT_DIR / "L1896.gpkg"
    merged.to_file(out, layer=OUT_LAYER, driver="GPKG")
    n = len(merged)
    return {
        "grid_id": "L1896",
        "src_filename": "G1896(202).zip (3 parts merged)",
        "n_polygons": n,
        "paren_count": 202,
        "count_match": "yes" if n == 202 else "MISMATCH",
        "crs": str(merged.crs),
        "src_layer_name": "|".join(src_layers),
        "out_layer_name": OUT_LAYER,
        "label_source": "human_manual_sam_assisted",
        "semantic_confidence": "A2",
        "quality_tier": "T2",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for src_name, lid in SOURCE_FILES.items():
        print(f"[stage] {src_name} -> {lid}.gpkg")
        rows.append(stage_simple(src_name, lid))
    print("[stage] G1896 zip (3 parts) -> L1896.gpkg")
    rows.append(stage_g1896())

    rows.sort(key=lambda r: r["grid_id"])
    manifest = OUT_DIR / "annotation_manifest_li.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== Wrote {len(rows)} gpkgs + manifest -> {manifest}")
    print(f"{'grid':<7}{'n':>5}{'paren':>7}  {'match':<9}{'crs':<12}layer")
    for r in rows:
        print(
            f"{r['grid_id']:<7}{r['n_polygons']:>5}"
            f"{str(r['paren_count']):>7}  {r['count_match']:<9}"
            f"{r['crs']:<12}{r['src_layer_name']}"
        )


if __name__ == "__main__":
    main()
