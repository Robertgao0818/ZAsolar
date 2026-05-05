# GEID Temporal Anchor-Presence Detection Architecture

> **DEPRECATED 2026-05-05.** Migrated to solar_backdating subrepo (V1.4 sub-line pivot).
> Authoritative copy: `/home/gaosh/projects/solar_backdating/docs/geid_temporal_anchor_presence_architecture.md`.
> This file is frozen; scheduled for removal after 2026-05-31. Edits go to subrepo first.

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task if this evolves beyond the initial skeleton.

**Goal:** Infer rooftop PV installation timing by anchoring on known installation locations, downloading GEID historical chips, scoring PV presence/absence per capture date, and finding the first stable absent→present breakpoint.

**Architecture:** Treat existing aerial/Vexcel GT polygons as spatial anchors, not pixel-perfect masks for historical satellite imagery. The historical task is an anchor-conditioned binary presence problem over buffered chips, with explicit alignment uncertainty and monotonic time-series inference. The first implementation should produce manifests, GEID task CSVs, presence time-series schemas, and breakpoint outputs before training a new model.

**Tech Stack:** Python, GeoPandas/Shapely/PyProj for geometry, GEID `downloader.exe` bridge for historical chips, CSV manifests for pipeline boundaries, optional PyTorch classifier/detector plugged in later.

---

## 1. Core decision: anchor-presence, not historical mask IoU

Existing GT masks are valuable, but they should not be treated as exact historical satellite masks.

Reasons:

1. Aerial/Vexcel imagery and GEID historical satellite imagery can have different orthorectification, viewing angle, parallax, GSD, and roof-edge registration.
2. Existing annotations are installation-footprint polygons under the project spec, but some Joburg rows are reviewed prediction / SAM-refined review semantics, not individually verified gold masks.
3. For install-date inference, the needed signal is whether the known installation is visibly present near that roof location at a given historical capture date.

Therefore the geometry contract is:

- Use GT polygon centroid / representative point / bounding envelope as an anchor.
- Export a buffered chip bbox around the anchor, large enough to tolerate mask/source offset.
- Do not require historical detection masks to overlap the original GT polygon exactly.
- Score PV presence inside a search window around the anchor.

## 2. Data products and schemas

### 2.1 Anchor manifest

File: `data/geid_temporal/anchors.csv`

One row per known installation anchor.

Required fields:

- `anchor_id`: stable unique ID, e.g. `johannesburg_G0922_a000001`
- `region_key`
- `grid_id`
- `source_annotation_path`
- `source_feature_id`
- `quality_tier` if known
- `anchor_policy`: e.g. `gt_centroid_buffered_bbox`
- `centroid_lon`, `centroid_lat`
- `source_area_m2`
- `source_width_m`, `source_height_m`
- `chip_half_m`
- `search_radius_m`
- `chip_lon_min`, `chip_lat_min`, `chip_lon_max`, `chip_lat_max`
- `alignment_note`: e.g. `mask_not_used_as_exact_history_gt`

### 2.2 GEID historical task CSV

File: `data/geid_temporal/geid_tasks.csv`

Compatible with the `geid_reverse_engineering/python/geid_historical_cli_batch.py` bridge.

Required fields:

- `grid_id`: use anchor ID or source grid ID depending on download grouping
- `task_name`: e.g. `johannesburg_G0922_a000001_20190615`
- `save_to`: Windows root path for GEID CLI
- `date`: requested date, e.g. `2019-06-15`
- `zoom_from`, `zoom_to`
- `left_longitude`, `right_longitude`, `top_latitude`, `bottom_latitude`

Important: downstream must parse embedded JPEG capture date (`*AD*YYYY:MM:DD*`), because GEID can return nearest available imagery rather than the requested date.

### 2.3 Presence time series

File: `data/geid_temporal/presence_timeseries.csv`

One row per `(anchor_id, capture_date)` after chip scoring.

Required fields:

- `anchor_id`
- `requested_date`
- `capture_date`: true GEID embedded capture date when available
- `pv_score`: continuous classifier/detector score
- `pv_present`: binary decision `0/1`, nullable if chip invalid
- `decision_source`: `manual`, `classifier`, `detector`, `ensemble`, `missing_chip`
- `chip_path` or `chip_dir`
- `alignment_score` optional
- `quality_flag`: `ok`, `missing_chip`, `no_date_metadata`, `cloud_shadow`, `ambiguous`, etc.

### 2.4 Installation interval output

File: `data/geid_temporal/install_intervals.csv`

One row per anchor.

Required fields:

- `anchor_id`
- `status`: `appears`, `already_present`, `not_seen`, `no_valid_observations`, `ambiguous_nonmonotonic`, `ambiguous_sporadic_positive`
- `latest_absent_date`
- `earliest_present_date`
- `install_interval_start`: latest known absent date, open lower bound
- `install_interval_end`: earliest known present date, closed upper bound
- `n_observations`, `n_absent`, `n_present`
- `confidence`: `high`, `medium`, `low`
- `notes`

## 3. Model strategy

### Phase 0: manual/heuristic smoke

- Generate anchors from known GT.
- Download historical chips for a small sample.
- Manually inspect a few anchor time stacks.
- Fill `pv_present` manually for smoke tests.
- Validate breakpoint inference before training anything.

### Phase 1: anchor-conditioned presence classifier

Input: buffered RGB chip centered near an existing GT installation.

Output: three-class frame decision:

- `present`: PV is visibly present near the anchor.
- `absent`: imagery is readable and PV is not visible near the anchor.
- `unusable`: imagery is too blurred, low-resolution, shadowed, mixed-date, missing, or otherwise not reliable enough to score as present/absent.

Recommended robustness:

- Use chip context larger than the GT mask.
- Evaluate shifted crops / patch tokens around the anchor within `search_radius_m`.
- Freeze the first search-window geometry for reproducibility: start with a 3×3 patch-token grid around the anchor for DINOv2 ViT-S/14 at GEID z=21; revise only through an explicit experiment note if GSD/source changes demand it.
- Aggregate by top-k mean score (`k=2` or `k=3`) rather than raw max, to reduce single-patch artifact sensitivity.
- Keep an `alignment_score`/`best_offset_m` so large shifts can be flagged.

This is preferred over immediately running Mask R-CNN segmentation on historical GEID because the temporal objective is binary presence, not exact footprint segmentation.

Backbone order for the first real scoring experiment:

1. LightGBM on handcrafted visual + metadata features, as the no-GPU floor.
2. DINOv2 ViT-S/14 frozen encoder with patch-token MIL / top-k pooling and a small three-class MLP head.
3. ConvNeXt-Tiny frozen encoder as the CNN local-texture baseline.

Do not use a Siamese / change-detection architecture in Phase 1. The project has spatial anchors, not a same-sensor current GEID reference image for every anchor. Current aerial/Vexcel imagery is useful for anchor geometry, but cross-sensor difference features are expected to be dominated by sensor/domain gap.

Calibration policy:

- Calibrate `present` probabilities before temporal inference; do not feed raw softmax scores directly to breakpoint logic.
- Use per-vintage-source temperature scaling at minimum, separating CBD GEID aerial-mosaic style from non-CBD true satellite style when labels are available.
- Keep calibration separate from the monotonic time-series smoother so classifier calibration errors remain diagnosable.

### Phase 2: detector fallback

If classifier false positives are too high, run a lightweight detector/localizer inside the anchor chip:

- Accept detection if any PV-like object intersects the anchor search window.
- Do not require exact overlap with the original aerial mask.
- Record detector score and offset.

## 4. Offset and viewpoint risk policy

The pipeline should explicitly assume offset can happen.

Mitigations:

1. Use chip bbox = source polygon bbox expanded by margin, with a minimum chip half-size.
2. Use a smaller search window around the anchor for scoring, not the exact source mask.
3. Use shifted-crop or detector max-pooling for binary presence.
4. Store `best_offset_m` where possible; if best offset exceeds a threshold, mark `ambiguous` rather than forcing a date.
5. Use monotonic time-series logic: real PV installation should not repeatedly disappear/reappear unless imagery/model quality is bad.
6. Prefer strict monotonic / isotonic temporal smoothing over unconstrained changepoint fitting. If non-monotonic segments concentrate in poor-quality vintages, output `ambiguous_nonmonotonic` instead of forcing a breakpoint.

## 5. Minimal implementation tasks

### Task 1: Anchor manifest builder

Create `scripts/temporal/build_gt_anchor_manifest.py`.

- Load annotations through `core.annotation_loader`.
- Reproject to region metric CRS.
- Compute centroid, source bbox size, buffered chip bbox.
- Export `data/geid_temporal/anchors.csv`.
- Never mark these masks as exact historical GT.

### Task 2: GEID task exporter

Create `scripts/temporal/export_geid_temporal_tasks.py`.

- Read anchor manifest.
- Expand requested years/dates.
- Export a CSV compatible with `geid_historical_cli_batch.py`.

### Task 3: Presence template / Phase-0 manual scoring bridge

Create `scripts/temporal/score_anchor_presence.py`.

- Read `anchors.csv` and `geid_tasks.csv`.
- Resolve each GEID task output directory (`save_to` + `task_name`), including Windows `D:\...` roots translated to WSL `/mnt/d/...` paths.
- Scan downloaded JPGs and parse embedded GEID capture dates (`*AD*YYYY:MM:DD*`).
- Emit `presence_timeseries.csv` rows with `decision_source=manual_template` and blank `pv_present` / `pv_score` for human review.
- The QA gallery must expose `present`, `absent`, `unusable`, and `unsure` labels. `unusable` and `unsure` rows keep `pv_present` blank so they do not become false absences.
- Emit explicit `missing_chip` rows instead of guessing absence when downloads are absent.
- Optionally merge a manually reviewed decisions CSV into the same schema.
- Optionally write a local HTML QA gallery for small smoke batches.

### Task 4: Presence schema and breakpoint inference

Create `scripts/temporal/infer_install_dates.py` plus shared pure functions.

- Read `presence_timeseries.csv`.
- Sort by true `capture_date` when available.
- Infer latest absent / earliest present interval.
- Flag non-monotonic or sparse ambiguous sequences.

### Task 5: Verification smoke

- Unit-test breakpoint inference, task-row generation, and presence-template generation.
- Dry-run anchor generation on one Joburg grid.
- Dry-run GEID task CSV for 2 anchors × 2 dates.
- Build a tiny manual presence template/QA page from synthetic or downloaded chips.
- Do not run large downloads by default.

## 6. Acceptance criteria for the skeleton

- A user can generate anchors from existing GT without downloading imagery.
- A user can generate historical GEID task CSVs from anchors.
- A user can infer install intervals from a hand-filled or model-filled presence CSV.
- The docs state clearly that original aerial masks are anchors, not exact historical masks.
- The architecture preserves capture-date provenance and flags ambiguous temporal sequences.
