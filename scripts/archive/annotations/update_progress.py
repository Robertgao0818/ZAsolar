#!/usr/bin/env python3
"""
标注进度自动更新脚本
Scan cleaned annotation directory and generate PROGRESS.md

用法：
    python scripts/annotations/update_progress.py
"""

import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import fiona
except ImportError:
    fiona = None

BASE_DIR = Path(__file__).resolve().parents[2]
CLEANED_DIR = BASE_DIR / "data" / "annotations" / "cleaned"
PROGRESS_FILE = BASE_DIR / "data" / "annotations" / "PROGRESS.md"

# ── Batch definitions by grid ID range ──────────────────────────────
BATCHES = [
    ("Legacy",       lambda g: g.startswith("G") and _gnum(g) <= 1134),
    ("Early SAM2",   lambda g: g.startswith("G") and _gnum(g) in (1189, 1190, 1238)),
    ("Batch 001",    lambda g: g.startswith("G") and 1240 <= _gnum(g) <= 1411),
    ("Batch 002",    lambda g: g.startswith("G") and 1466 <= _gnum(g) <= 1572),
    ("Batch 002b",   lambda g: g.startswith("G") and 1573 <= _gnum(g) <= 1635),
    ("Batch 003",    lambda g: g.startswith("G") and 1636 <= _gnum(g) <= 1847),
    ("Batch 004+",   lambda g: g.startswith("G") and _gnum(g) > 1847),
    ("JHB",          lambda g: g.startswith("JHB")),
]


def _gnum(grid_id: str) -> int:
    """Extract numeric part from grid ID like G1238 → 1238."""
    m = re.search(r"\d+", grid_id)
    return int(m.group()) if m else -1


def count_features(gpkg_path: Path) -> int:
    """Count features in a GPKG file without loading geometries."""
    if fiona is not None:
        try:
            with fiona.open(gpkg_path) as src:
                return len(src)
        except Exception:
            pass
    # Fallback: ogrinfo
    import subprocess
    try:
        result = subprocess.run(
            ["ogrinfo", "-sql", "SELECT COUNT(*) FROM \"\"", "-q", str(gpkg_path)],
            capture_output=True, text=True, timeout=10,
        )
        m = re.search(r"COUNT\(\*\).*?=\s*(\d+)", result.stdout)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    # Last resort: geopandas
    try:
        import geopandas as gpd
        return len(gpd.read_file(gpkg_path))
    except Exception:
        return -1


def scan_cleaned_dir() -> dict:
    """Scan cleaned dir, return {grid_id: {path, count, date, filename}}."""
    if not CLEANED_DIR.exists():
        return {}

    grids = {}
    for f in sorted(CLEANED_DIR.glob("*.gpkg")):
        if f.name == "all_annotations_cleaned.gpkg":
            continue

        # Extract grid_id and date from filename
        # Patterns: G1238.gpkg, G1240_SAM2_260322.gpkg
        m = re.match(r"((?:G|JHB)\d+)(?:_SAM2_(\d{6}))?\.gpkg", f.name)
        if not m:
            continue

        grid_id = m.group(1)
        date_str = m.group(2)  # e.g. "260322" → 2026-03-22

        # If grid already seen, prefer SAM2 version (has date)
        if grid_id in grids:
            if date_str and not grids[grid_id]["date"]:
                pass  # overwrite legacy with SAM2
            else:
                continue  # keep existing

        date_fmt = None
        if date_str:
            try:
                date_fmt = datetime.strptime("20" + date_str, "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError:
                date_fmt = date_str

        grids[grid_id] = {
            "path": f,
            "filename": f.name,
            "date": date_fmt,
            "count": None,  # lazy-count below
        }

    # Count features
    for grid_id, info in grids.items():
        info["count"] = count_features(info["path"])

    return grids


def assign_batch(grid_id: str) -> str:
    """Assign a grid to its batch."""
    for name, pred in BATCHES:
        if pred(grid_id):
            return name
    return "Other"


def generate_markdown(grids: dict) -> str:
    """Generate PROGRESS.md content."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Group by batch
    batches = defaultdict(list)
    for grid_id, info in sorted(grids.items(), key=lambda x: x[0]):
        batch = assign_batch(grid_id)
        batches[batch].append((grid_id, info))

    total_grids = len(grids)
    total_polys = sum(v["count"] for v in grids.values() if v["count"] and v["count"] > 0)

    lines = [
        "# Annotation Progress",
        "",
        f"Last updated: {now}",
        "",
        f"**Total: {total_grids} grids, {total_polys} installations**",
        "",
    ]

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Batch | Grids | Installations | Date |")
    lines.append("|-------|-------|---------------|------|")

    for batch_name, _ in BATCHES:
        if batch_name not in batches:
            continue
        items = batches[batch_name]
        n_grids = len(items)
        n_polys = sum(info["count"] for _, info in items if info["count"] and info["count"] > 0)
        dates = sorted(set(info["date"] for _, info in items if info["date"]))
        date_range = dates[0] if len(dates) == 1 else f"{dates[0]} ~ {dates[-1]}" if dates else "-"
        lines.append(f"| {batch_name} | {n_grids} | {n_polys} | {date_range} |")

    lines.append("")

    # Detailed per-grid listing
    lines.append("## Detail")
    lines.append("")

    for batch_name, _ in BATCHES:
        if batch_name not in batches:
            continue
        items = batches[batch_name]
        lines.append(f"### {batch_name}")
        lines.append("")
        for grid_id, info in items:
            count = info["count"] if info["count"] and info["count"] > 0 else "?"
            date = info["date"] or "-"
            lines.append(f"- {grid_id}: {count} installations ({date})")
        lines.append("")

    return "\n".join(lines)


def main():
    grids = scan_cleaned_dir()
    if not grids:
        print("[WARN] No cleaned annotations found.")
        return

    md = generate_markdown(grids)
    PROGRESS_FILE.write_text(md, encoding="utf-8")
    print(f"[OK] Updated {PROGRESS_FILE} — {len(grids)} grids")


if __name__ == "__main__":
    main()
