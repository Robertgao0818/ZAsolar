#!/usr/bin/env python3
"""Probe eThekwini (Durban) GIS ArcGIS ImageServer for aerial-imagery
viability across 5 vintages × 3 representative grids.

Each probe fetches a 200 m × 200 m centered window at native GSD
(stays under maxImage 15000×4100 even for the 5 cm 2025 service)
and writes TIFF + records timing / status / file-size / pixel size.

Output: probes/durban_aerial/<grid_id>__<service>.tif + summary.csv
"""

from __future__ import annotations

import csv
import time
import urllib.parse
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADMIN_GRID = PROJECT_ROOT / "data" / "admin_grids" / "durban_admin_grid.gpkg"
OUTPUT_DIR = PROJECT_ROOT / "probes" / "durban_aerial"

SERVICES = [
    ("AerialPhoto2020",          "10cm", "WG31"),
    ("AerialPhotography20212022", "10cm", "WG31"),
    ("AerialPhotography2022",     "10cm", "wg31_wgs84"),
    ("AerialPhotography2023",     "10cm", "wg31_wgs84"),
    ("LowRangeAerialPhotography2025", "5cm", "EPSG:3857"),
]

PROBE_GRIDS = ["ETK1554", "ETK0891", "ETK1383"]
PROBE_LABELS = {"ETK1554": "CBD_in_vexcel",
                "ETK0891": "north_township_aerial",
                "ETK1383": "west_periphery_aerial"}

WINDOW_M = 200  # 200 m × 200 m centered probe window


def webmercator_window(lon: float, lat: float, half_m: float) -> tuple[float, float, float, float]:
    """Return EPSG:3857 bbox of `2*half_m` square centered on (lon, lat)."""
    import math
    R = 6378137.0
    x = lon * math.pi / 180.0 * R
    y = math.log(math.tan(math.pi / 4 + lat * math.pi / 360.0)) * R
    return (x - half_m, y - half_m, x + half_m, y + half_m)


def probe_service(service: str, gsd_label: str, native_crs: str,
                  bbox_3857: tuple[float, float, float, float],
                  out_path: Path) -> dict:
    xmin, ymin, xmax, ymax = bbox_3857
    width_m = xmax - xmin

    # Pick output size targeting native GSD where possible
    if gsd_label == "5cm":
        size_px = 4000  # 4000 px over 200 m → 5 cm/px
    else:  # 10 cm
        size_px = 2000  # 2000 px over 200 m → 10 cm/px

    base = f"https://gis.durban.gov.za/server/rest/services/AerialPhotography/{service}/ImageServer/exportImage"
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": "3857",
        "imageSR": "3857",
        "size": f"{size_px},{size_px}",
        "format": "tiff",
        "pixelType": "U8",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    url = f"{base}?{urllib.parse.urlencode(params)}"

    t0 = time.perf_counter()
    try:
        r = requests.get(url, verify=False, timeout=120, stream=True)
        elapsed = time.perf_counter() - t0
        status = r.status_code
        if status == 200 and "image/tiff" in r.headers.get("Content-Type", ""):
            out_path.write_bytes(r.content)
            size_kb = out_path.stat().st_size / 1024
            return {"ok": True, "status": status, "size_kb": round(size_kb, 1),
                    "elapsed_s": round(elapsed, 2),
                    "px": size_px, "gsd_m": round(width_m / size_px, 4)}
        else:
            preview = r.text[:200] if r.text else ""
            return {"ok": False, "status": status, "size_kb": 0,
                    "elapsed_s": round(elapsed, 2), "preview": preview}
    except Exception as e:
        return {"ok": False, "status": "EXC",
                "elapsed_s": round(time.perf_counter() - t0, 2),
                "preview": str(e)[:200]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grids = gpd.read_file(ADMIN_GRID).set_index("gridcell_id").loc[PROBE_GRIDS]

    rows: list[dict] = []
    for grid_id in PROBE_GRIDS:
        row = grids.loc[grid_id]
        bbox = webmercator_window(row["lon"], row["lat"], WINDOW_M / 2)
        label = PROBE_LABELS[grid_id]
        print(f"\n=== {grid_id} ({label})  lon={row['lon']:.4f} lat={row['lat']:.4f}  "
              f"buildings={row['n_buildings']:,}  vexcel={row['vexcel_fraction']:.2f} ===")

        for service, gsd, crs in SERVICES:
            out = OUTPUT_DIR / f"{grid_id}__{service}.tif"
            r = probe_service(service, gsd, crs, bbox, out)
            row_out = {"grid_id": grid_id, "label": label, "service": service,
                       "gsd_label": gsd, "native_crs": crs, **r}
            rows.append(row_out)
            tag = "✓" if r["ok"] else "✗"
            extra = f"{r.get('size_kb',0):.0f} KB  px={r.get('px','?')}  gsd={r.get('gsd_m','?')}m" if r["ok"] else f"status={r['status']}  {r.get('preview','')[:80]}"
            print(f"  {tag} {service:<32} {r['elapsed_s']:>5.1f}s  {extra}")

    csv_path = OUTPUT_DIR / "summary.csv"
    fieldnames = ["grid_id", "label", "service", "gsd_label", "native_crs",
                  "ok", "status", "size_kb", "elapsed_s", "px", "gsd_m", "preview"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary CSV: {csv_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    main()
