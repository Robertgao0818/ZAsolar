# download-grids Skill

Preview, filter, and download new grid tiles for annotation or inference.

## When to use

When expanding coverage to new geographic areas — preview grids, select ones with imagery, download tiles.

## Prerequisites

- Environment: `source scripts/activate_env.sh`
- OSM building data cached in `cache/osm_buildings_centroids.gpkg`
- Tiles output to D drive: `SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles`

## Steps

### 1. Generate grid previews

```bash
# Start from where last batch ended (check previous batch's last grid)
python scripts/imagery/grid_preview_batch.py \
  --start-grid-id <START_GRID> --batch-index 1 --batch-size 100 \
  --output-dir results/grid_previews/batch_<NNN>
```

Batch history:
- batch_001: G1247 → G1410
- batch_002: G1424 → G1635
- batch_003: G1636 → G1847
- batch_004: starts from G1848

### 2. Auto-exclude blank grids

```python
import csv, pandas as pd
from datetime import datetime, timezone
metrics = pd.read_csv("results/grid_previews/batch_<NNN>/grid_preview_metrics.csv")
path = "results/grid_previews/batch_<NNN>/grid_review_decisions.csv"
now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
blank = metrics[metrics["valid_imagery_ratio"] == 0.0]
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["grid_id", "decision", "notes", "updated_at"])
    w.writeheader()
    for _, r in blank.iterrows():
        w.writerow({"grid_id": r["grid_id"], "decision": "exclude",
                    "notes": "auto: 0% valid imagery", "updated_at": now})
print(f"Auto-excluded {len(blank)} blank, {len(metrics)-len(blank)} remaining")
```

### 3. Review GUI

```bash
python scripts/imagery/review_grid_previews.py \
  --batch-dir results/grid_previews/batch_<NNN>
```

Open **http://localhost:8765**. Mark grids as keep/exclude/review.

### 4. Generate OSM building mask

```bash
python scripts/imagery/filter_grids_osm.py --buffer-rings 2
```

### 5. Download tiles

```bash
# Dry run first
SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles python scripts/imagery/download_reviewed_grids.py \
  --batch-dir results/grid_previews/batch_<NNN> --dry

# Actual download (full, no OSM mask)
SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles python scripts/imagery/download_reviewed_grids.py \
  --batch-dir results/grid_previews/batch_<NNN> --workers 4
```

## Constraints

- Always set `SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles` — C drive space is limited
- Default is full download (all tiles). Only add `--use-tile-mask` in dense urban areas where OSM data is reliable
- OSM tile mask causes issues in suburban areas (incomplete building data → missed tiles, boundary cuts)
- Grid preview GUI uses port 8765 (detection review uses 8766)
- Grids with no OSM buildings are auto-skipped by tile mask
- `download_tiles.py` uses `TILES_ROOT` from `core/grid_utils.py` (reads env var)
