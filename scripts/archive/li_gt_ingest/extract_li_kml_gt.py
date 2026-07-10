"""Extract Li's Cape Town panel GT from the Google-Earth KML into Capetown_Li/.

Companion to ``stage_li_ct_gt.py`` (which ingested the L1842..L1954 gpkg batch).
This script ingests the *bulk* KML export
``cape_town_grid_Li_G0029_G1841.kml`` (EPSG:4326, exported from Google Earth
Pro) covering the L0029..L1841 range.

KML structure (reconnoitred 2026-06-10):
  - 8 top-level ``<Folder>`` named by grid-ID segments.
  - Panel annotations are ``<Placemark>`` named ``Gxxxx_NNN`` (grid prefix +
    panel index), geometry = ``<Polygon>``.  A handful (5) use a hyphen typo
    ``Gxxxx-NNN`` -- those are genuine panel-sized polygons and ARE kept (regex
    accepts ``[_-]``).
  - Two folders (G0436_G1114, G1115_G1506) also carry 2215 *task-grid* cells
    each, named bare ``Gxxxx`` (no separator).  These are NOT panels and are
    excluded.
  - One stray placemark named ``02/2024`` (a 1.9 m^2 date-label scribble) is
    excluded by the panel regex.

Namespace rule (see memory project_li_grid_namespace + repo CLAUDE rules):
  Li uses G-numbers on an INDEPENDENT physical grid scheme (east/west split
  re-uses the same G-number for a different physical cell).  Ingest renames
  G -> L verbatim, zero-padded to 4 digits (G0269 -> L0269, G1520 -> L1520).
  The resulting ``^L[0-9]{4}$`` namespace does not collide with the report
  suite L1842..L1954, so all grids are leakage-free; nothing is dropped on a
  G-id basis.

Output mirrors stage_li_ct_gt.py:
  - ``data/annotations/Capetown_Li/L<NNNN>.gpkg`` (layer ``li_ct_gt``, GPKG,
    CRS preserved EPSG:4326).
  - Appends rows to ``annotation_manifest_li.csv`` (this batch is A2/T2,
    label_source human_manual_sam_assisted, same provenance class as L18xx).

Run from repo root with the project venv.  ``--dry-run`` parses + reports
without writing any file.
"""

from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

SRC_KML = Path(
    "/mnt/c/Users/gaosh/Dropbox/RA_Solar/Li/capetown/"
    "cape_town_grid_Li_G0029_G1841.kml"
)
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "annotations" / "Capetown_Li"
MANIFEST = OUT_DIR / "annotation_manifest_li.csv"

KML_NS = {"k": "http://www.opengis.net/kml/2.2"}
OUT_LAYER = "li_ct_gt"

# Panel placemark: grid prefix + separator + index.  Accepts the '-' typo
# variant; rejects bare 'Gxxxx' task-grid cells and non-grid labels.
PANEL_RE = re.compile(r"^G(\d{3,4})[_-]\d+$")

# Manifest columns (must match stage_li_ct_gt.py output exactly).
FIELDNAMES = [
    "grid_id",
    "src_filename",
    "n_polygons",
    "paren_count",
    "count_match",
    "crs",
    "src_layer_name",
    "out_layer_name",
    "label_source",
    "semantic_confidence",
    "quality_tier",
]


def _placemark_polygon(pm: ET.Element) -> Polygon | None:
    """Return the outer-ring Polygon of a Placemark, or None if not a polygon."""
    coords = pm.find(".//k:Polygon//k:outerBoundaryIs//k:coordinates", KML_NS)
    if coords is None:
        coords = pm.find(".//k:Polygon//k:coordinates", KML_NS)
    if coords is None or not coords.text:
        return None
    pts = []
    for tok in coords.text.split():
        parts = tok.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 3:
        return None
    return Polygon(pts)


def parse_kml(src: Path) -> tuple[dict[str, list[Polygon]], dict[str, int]]:
    """Parse the KML.

    Returns ({L<NNNN>: [Polygon, ...]}, {L<NNNN>: n_empty_geom}). Panel-named
    placemarks whose geometry is empty/degenerate (no parseable ring) are
    counted in the second dict so the QA can account for the matched-vs-valid
    delta (Li's KML has 1 such empty Polygon under G0269_007).
    """
    root = ET.parse(src).getroot()
    doc = root.find("k:Document", KML_NS)
    if doc is None:
        raise RuntimeError("KML has no <Document> element")
    grid_polys: dict[str, list[Polygon]] = defaultdict(list)
    empty_geom: dict[str, int] = defaultdict(int)
    for folder in doc.findall("k:Folder", KML_NS):
        for pm in folder.findall(".//k:Placemark", KML_NS):
            name_el = pm.find("k:name", KML_NS)
            name = name_el.text.strip() if name_el is not None and name_el.text else ""
            m = PANEL_RE.match(name)
            if not m:
                continue
            lid = f"L{int(m.group(1)):04d}"
            poly = _placemark_polygon(pm)
            if poly is None:
                empty_geom[lid] += 1
                continue
            grid_polys[lid].append(poly)
    return grid_polys, empty_geom


def clean_geoms(polys: list[Polygon]) -> tuple[list[Polygon], int, int]:
    """buffer(0) invalid geoms, drop empties. Return (clean, fixed, dropped)."""
    out: list[Polygon] = []
    fixed = 0
    dropped = 0
    for p in polys:
        if p is None or p.is_empty:
            dropped += 1
            continue
        if not p.is_valid:
            p = p.buffer(0)
            fixed += 1
            if p.is_empty:
                dropped += 1
                continue
        out.append(p)
    return out, fixed, dropped


def stage_grid(lid: str, polys: list[Polygon], dry_run: bool) -> dict:
    clean, fixed, dropped = clean_geoms(polys)
    gdf = gpd.GeoDataFrame(geometry=clean, crs="EPSG:4326")
    n = len(gdf)
    out = OUT_DIR / f"{lid}.gpkg"
    if not dry_run:
        if out.exists():
            raise RuntimeError(
                f"REFUSING to overwrite existing {out} -- L-id collision; stop and report."
            )
        gdf.to_file(out, layer=OUT_LAYER, driver="GPKG")
    return {
        "grid_id": lid,
        "src_filename": f"{SRC_KML.name} (KML folder placemarks)",
        "n_polygons": n,
        "paren_count": "",  # KML carries no filename count
        "count_match": "n/a",
        "crs": "EPSG:4326",
        "src_layer_name": "kml_placemark",
        "out_layer_name": OUT_LAYER,
        "label_source": "human_manual_sam_assisted",
        "semantic_confidence": "A2",
        "quality_tier": "T2",
        "_geom_invalid_fixed": fixed,
        "_geom_dropped": dropped,
    }


def append_manifest(rows: list[dict]) -> None:
    write_header = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDNAMES})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-kml", type=Path, default=SRC_KML)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report counts without writing gpkgs or manifest.",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_polys, empty_geom = parse_kml(args.src_kml)
    total_empty = sum(empty_geom.values())

    # Collision guard against the already-staged L1842..L1954 batch.
    existing = {
        p.stem
        for p in OUT_DIR.glob("L*.gpkg")
    }
    collisions = sorted(set(grid_polys) & existing)
    if collisions:
        raise RuntimeError(
            f"L-id collision with already-staged batch: {collisions}. STOP and report."
        )

    rows: list[dict] = []
    for lid in sorted(grid_polys):
        r = stage_grid(lid, grid_polys[lid], args.dry_run)
        rows.append(r)
        print(
            f"[{'dry' if args.dry_run else 'stage'}] {lid}: "
            f"{r['n_polygons']:>4} panels "
            f"(fixed={r['_geom_invalid_fixed']}, dropped={r['_geom_dropped']})"
        )

    if not args.dry_run:
        append_manifest(rows)

    total = sum(r["n_polygons"] for r in rows)
    total_fixed = sum(r["_geom_invalid_fixed"] for r in rows)
    total_dropped = sum(r["_geom_dropped"] for r in rows)
    print(
        f"\n=== {'DRY-RUN ' if args.dry_run else ''}"
        f"{len(rows)} grids, {total} panels "
        f"(geom fixed={total_fixed}, dropped={total_dropped}, "
        f"empty-geom skipped at parse={total_empty})"
    )
    if not args.dry_run:
        print(f"manifest appended -> {MANIFEST}")


if __name__ == "__main__":
    main()
