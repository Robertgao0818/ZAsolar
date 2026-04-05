# Johannesburg Batch 1 — Inference & Validation Plan

**Created**: 2026-04-05
**Source**: `Dropbox/RA_Solar/grid_data_joburg/plan_batch1.xlsx`
**Total grids**: 100 (4 categories x 25 grids)

---

## 1. Data Overview

### Imagery Source

| Field | Value |
|-------|-------|
| Provider | City of Johannesburg |
| Service | ArcGIS ImageServer (`AerialPhotography/2023/ImageServer`) |
| GSD | 0.15 m/px (15 cm) |
| CRS | EPSG:3857 (stored as EPSG:4326) |
| Year | 2023 |
| Tile split | 6x6 equal-split per grid (~1248x1254 px/tile) |
| Tiles per grid | 36 |
| Total tiles | 3,600 |

### Grid Categories

| Category | Area | Grids | Description |
|----------|------|-------|-------------|
| **CBD** | Joburg CBD | 25 | High-rise, commercial, mixed-use |
| **Residential (Rich)** | Sandton | 25 | Affluent suburb, large rooftops |
| **Residential (Poor)** | Alexandra | 25 | Dense informal/low-income housing |
| **Industrial** | Midrand | 25 | Warehouses, factories, business parks |

### Grid ID List

<details>
<summary>CBD — Joburg (25 grids)</summary>

| # | Grid ID |
|---|---------|
| 1 | G0772 |
| 2 | G0773 |
| 3 | G0774 |
| 4 | G0775 |
| 5 | G0776 |
| 6 | G0814 |
| 7 | G0815 |
| 8 | G0816 |
| 9 | G0817 |
| 10 | G0818 |
| 11 | G0853 |
| 12 | G0854 |
| 13 | G0855 |
| 14 | G0856 |
| 15 | G0857 |
| 16 | G0888 |
| 17 | G0889 |
| 18 | G0890 |
| 19 | G0891 |
| 20 | G0892 |
| 21 | G0922 |
| 22 | G0923 |
| 23 | G0924 |
| 24 | G0925 |
| 25 | G0926 |

</details>

<details>
<summary>Residential Rich — Sandton (25 grids)</summary>

| # | Grid ID |
|---|---------|
| 1 | G1110 |
| 2 | G1111 |
| 3 | G1112 |
| 4 | G1113 |
| 5 | G1114 |
| 6 | G1144 |
| 7 | G1145 |
| 8 | G1146 |
| 9 | G1147 |
| 10 | G1148 |
| 11 | G1179 |
| 12 | G1180 |
| 13 | G1181 |
| 14 | G1182 |
| 15 | G1183 |
| 16 | G1214 |
| 17 | G1215 |
| 18 | G1216 |
| 19 | G1217 |
| 20 | G1218 |
| 21 | G1250 |
| 22 | G1251 |
| 23 | G1252 |
| 24 | G1253 |
| 25 | G1254 |

</details>

<details>
<summary>Residential Poor — Alexandra (25 grids)</summary>

| # | Grid ID |
|---|---------|
| 1 | G1151 |
| 2 | G1152 |
| 3 | G1153 |
| 4 | G1154 |
| 5 | G1155 |
| 6 | G1186 |
| 7 | G1187 |
| 8 | G1188 |
| 9 | G1189 |
| 10 | G1190 |
| 11 | G1221 |
| 12 | G1222 |
| 13 | G1223 |
| 14 | G1224 |
| 15 | G1225 |
| 16 | G1257 |
| 17 | G1258 |
| 18 | G1259 |
| 19 | G1260 |
| 20 | G1261 |
| 21 | G1293 |
| 22 | G1294 |
| 23 | G1295 |
| 24 | G1296 |
| 25 | G1297 |

</details>

<details>
<summary>Industrial — Midrand (25 grids)</summary>

| # | Grid ID |
|---|---------|
| 1 | G1512 |
| 2 | G1513 |
| 3 | G1514 |
| 4 | G1515 |
| 5 | G1516 |
| 6 | G1541 |
| 7 | G1542 |
| 8 | G1543 |
| 9 | G1544 |
| 10 | G1545 |
| 11 | G1570 |
| 12 | G1571 |
| 13 | G1572 |
| 14 | G1573 |
| 15 | G1574 |
| 16 | G1600 |
| 17 | G1601 |
| 18 | G1602 |
| 19 | G1603 |
| 20 | G1604 |
| 21 | G1630 |
| 22 | G1631 |
| 23 | G1632 |
| 24 | G1633 |
| 25 | G1634 |

</details>

---

## 2. Pipeline Stages

### Stage 1: Tile Download — IN PROGRESS

- **Script**: `scripts/imagery/download_jhb_tiles.py --batch batch1 --workers 6`
- **Storage**: RunPod `/workspace/tiles_joburg/{GridID}/`
- **Env**: `SOLAR_TILES_ROOT=/workspace/tiles_joburg`
- **Note**: Joburg tiles MUST be separate from Cape Town tiles (46 overlapping grid IDs)

| Milestone | Status | Date |
|-----------|--------|------|
| Download script (ImageServer) | Done | 2026-04-05 |
| Grid ID overlap bug fix (46 IDs) | Done | 2026-04-05 |
| Equal-split tiling (6x6, no thin strips) | Done | 2026-04-05 |
| 100 grid directories created | Done | 2026-04-05 |
| Tile download complete (3600 tiles) | In progress | — |

### Stage 2: Batch Inference

Run V4 model (or best available) on all 100 grids.

| Item | Plan |
|------|------|
| Model | V4 HN fine-tuned (pending V4 training), fallback: V3-C (`exp003_C`) |
| Script | `detect_and_evaluate.py --model-path <ckpt> --evaluation-profile installation` |
| Env | `SOLAR_TILES_ROOT=/workspace/tiles_joburg` |
| Output | `/workspace/ZAsolar/results/{GridID}/` |
| Parallelism | Sequential per grid (GPU-bound), batch script with grid list |

| Milestone | Status | Date |
|-----------|--------|------|
| V4 model ready | Pending (blocked on V4 HN training) | — |
| Batch inference script for Joburg | Not started | — |
| Run inference on 100 grids | Not started | — |
| Results collected | Not started | — |

### Stage 3: Cross-City Validation

Compare model performance across Joburg area types. No Joburg GT annotations exist yet, so validation is visual + statistical.

| Metric | Method |
|--------|--------|
| Detection count per grid | Automated (pipeline output) |
| Detection density (per km2) | Automated (count / grid area) |
| Confidence distribution | Automated (per-category histogram) |
| FP rate estimate | Manual spot-check (sample 5 grids/category, 20 total) |
| Size distribution | Automated (area histogram by category) |

| Milestone | Status | Date |
|-----------|--------|------|
| Aggregate stats per category | Not started | — |
| Visual spot-check (20 grids) | Not started | — |
| Cross-category comparison report | Not started | — |

### Stage 4: Transfer Quality Assessment

Assess Cape Town model's generalization to Johannesburg.

| Question | How to Answer |
|----------|---------------|
| Does the model detect at all? | Check detection count > 0 on residential grids |
| Precision sanity | Manual review of top-50 confident detections per category |
| Recall proxy | Compare detection density to expected solar adoption rates |
| CBD vs Residential vs Industrial | Per-category density + size distribution |
| Failure modes | Taxonomy of FP types (water heater? shadow? roofing?) |

| Milestone | Status | Date |
|-----------|--------|------|
| Per-category detection summary | Not started | — |
| FP taxonomy (Joburg-specific) | Not started | — |
| Transfer assessment report | Not started | — |
| Decision: Joburg-specific fine-tune needed? | Not started | — |

---

## 3. Key Differences: Cape Town vs Johannesburg

| Aspect | Cape Town | Johannesburg |
|--------|-----------|-------------|
| Imagery GSD | 0.05 m (5 cm) | 0.15 m (15 cm) |
| Imagery year | Jan 2025 | 2023 |
| Imagery source | WMS (open) | ImageServer (open) |
| Grid system | Cape Town 1km grid | Joburg 1km grid |
| Grid ID overlap | G0001–G2214 | G0001–G1830 (46 IDs shared) |
| Tiles root | `tiles/` | `tiles_joburg/` |
| GT annotations | Yes (456 polygons, SAM2) | None yet |
| Model trained on | Cape Town only | Cape Town only (transfer test) |

### Resolution Impact

At 15 cm GSD (Joburg) vs 5 cm GSD (Cape Town):

| Object | Cape Town (px) | Joburg (px) |
|--------|---------------|-------------|
| Residential panel (1.7x1.0m) | 34x20 | 11x7 |
| Commercial array (10x5m) | 200x100 | 67x33 |
| Water heater (2x1m) | 40x20 | 13x7 |

Smaller pixel footprint means lower detection sensitivity, especially for residential panels. Model may need confidence threshold adjustment for Joburg.

---

## 4. Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| 15cm too coarse for residential panels | Missed detections in Alexandra/Sandton | Evaluate recall by category; consider upscaling if needed |
| Water heater FP in Sandton | Inflated detection count | Apply V4 HN filter (trained on Cape Town water heaters) |
| Model overfits to Cape Town roofing | Systematic FP/FN on Joburg architecture | Visual spot-check; if bad, collect Joburg GT for fine-tune |
| CoJ ImageServer rate limiting | Slow/failed downloads | Retry logic + concurrent workers (already implemented) |
| Grid ID collision (CT/JHB) | Wrong coordinates in pipeline | Fixed: Joburg script reads `jhb_task_grid.gpkg` directly |

---

## 5. File Registry

| File | Purpose |
|------|---------|
| `data/jhb_task_grid.gpkg` | Joburg grid geometries (106 grids: 6 legacy JHB + 100 batch1) |
| `scripts/imagery/download_jhb_tiles.py` | Tile download from CoJ 2023 Aerial ImageServer |
| `configs/datasets/regions.yaml` | Dataset registry (Joburg section to be updated) |
| `docs/joburg_batch1_plan.md` | This document |
