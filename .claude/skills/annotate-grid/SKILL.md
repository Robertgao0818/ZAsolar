# annotate-grid Skill

Semi-auto annotation workflow: model detection → review GUI → SAM auto-fill → export GT.

## When to use

When annotating new grids for training data expansion.

## Prerequisites

- Tiles downloaded to `/mnt/d/ZAsolar/tiles/<GridID>/`
- Best model at `checkpoints/best_model.pth`
- Environment: `source scripts/activate_env.sh`
- Full workflow doc: `docs/semi_auto_annotation_workflow.md`

## Steps

### 1. Run detection (no GT needed)

```bash
SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles python detect_and_evaluate.py \
  --grid-id <GRID_ID> --model-path checkpoints/best_model.pth --force
```

Output: `results/<GRID_ID>/predictions_metric.gpkg`

### 2. Launch review GUI

```bash
SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles python scripts/annotations/review_detections.py \
  --grid-id <GRID_ID>
```

Open **http://localhost:8766**

Key shortcuts:
- Click polygon → select, then **A** correct / **D** delete / **E** edit
- **Q** = quick-pass entire tile as correct
- **M** = FN marker mode (click to mark missed panels, right-click to remove)
- Scroll wheel zoom, drag to pan, double-click reset
- **← →** = prev/next tile

### 3. Export and SAM auto-fill

Click **Export GPKG** button in GUI, then run SAM for FN markers:

```python
# Change GRID_ID, then run the SAM script from docs/semi_auto_annotation_workflow.md Step 4
```

SAM mask selection: always use index 1 (medium granularity, best for solar panels).

### 4. Verify SAM results

Render verification crops:
```python
# Quick verification — outputs to D:\ZAsolar\sam_fixed_<tile>.png
```

Or reload GUI with reviewed GPKG:
```bash
SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles python scripts/annotations/review_detections.py \
  --grid-id <GRID_ID> \
  --predictions results/<GRID_ID>/review/<GRID_ID>_reviewed.gpkg
```

### 5. Export as training GT

```bash
python3 -c "
import geopandas as gpd
gdf = gpd.read_file('results/<GRID_ID>/review/<GRID_ID>_reviewed.gpkg')
gt = gdf[gdf['review_status'] == 'correct'].copy()
gt.to_file('data/annotations/cleaned/<GRID_ID>_SAM2_$(date +%y%m%d).gpkg', driver='GPKG')
print(f'Exported {len(gt)} GT polygons')
"
```

### 6. Rebuild COCO dataset

```bash
python export_coco_dataset.py --output-dir data/coco
```

## Batch mode

For multiple grids, run detection in a loop:
```bash
for gid in G1682 G1683 G1685 G1686; do
  SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles python detect_and_evaluate.py \
    --grid-id $gid --model-path checkpoints/best_model.pth --force
done
```

Then review each grid one at a time via the GUI.

## Constraints

- SAM model (~1.2GB) downloads to `~/.cache/huggingface/`. Delete after use to free C drive.
- Review GUI runs on port 8766 (grid preview GUI uses 8765).
- When loading reviewed GPKG in GUI, decisions are read from the `review_status` column.
- `download_tiles.py` respects `SOLAR_TILES_ROOT` env var.
