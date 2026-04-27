# Experiment: Classifier Dataset Protocol

**Date**: 2026-04-22
**Status**: Active
**Plan**: `/home/gaosh/.claude/plans/codex-efficientnetb0-convnexttiny-found-swirling-muffin.md`

Authoritative data protocol for the PV vs non-PV binary classifier. This
document freezes the rules used by `scripts/classifier/build_cls_dataset.py`
and is the reference both the backbone ablation and downstream detector
integration depend on.

## Source of truth

Discovery is **registry-driven** via `core.region_registry`. For every
registered `(region, model_run)` that is not marked `deprecated: true` in
`configs/datasets/regions.yaml`, the builder scans
`<results_path>/G*/review/` for `{grid}_reviewed.gpkg`. Legacy flat
`results/G*/review/` (pre-PR3 CT batch 003) is included as a
`cape_town:legacy_flat_batch003` pseudo-bucket.

`{grid}_reviewed.gpkg` is the **authoritative source** for both `area_m2`
and `review_status`. `predictions_metric.gpkg` is only consulted when
joining auxiliary labels (taxonomy, GT audit) that reference pred_id.

## Source buckets

| Bucket | Registered model_run | results_path | Grids |
|---|---|---|---|
| `cape_town:legacy_flat_batch003` | pseudo (pre-PR3 flat) | `results/` | 21 |
| `cape_town:v3c_targeted_hn_aerial_2025` | registered | `results/cape_town/v3c_targeted_hn_aerial_2025/` | 36 (1 csv-only) |
| `johannesburg:v4_aerial_2023` | registered | `results/johannesburg/v4_aerial_2023/` | 50 |

Exploratory / held-out (not used for training, not in promotion rule):

| Bucket | Reason |
|---|---|
| `johannesburg:v3c_geid_2024_02` | `deprecated: true` in regions.yaml (GEID bounds bug); 1 csv-only grid |
| `cape_town:v3c_targeted_hn_aerial_2025/G1918` | csv-only (no `_reviewed.gpkg`, area unknown) |

## Label map

**Reviewed gpkg (primary)**

| `review_status` | Label |
|---|---|
| `correct` | `pv` |
| `edit` | `pv` (real panel, polygon needs fix) |
| `delete` | `non_pv` |
| `unreviewed` / other | dropped |

**Taxonomy CSV (auxiliary, `--include-taxonomy`)**

| `human_label` | Label |
|---|---|
| `correct_detection` | `pv` |
| `solar_thermal_water_heater` / `skylight_roof_window` / `roof_shadow_dark_fixture` / `pergola_carport_shadow` | `non_pv` |

**GT heater audit CSV (auxiliary, `--include-gt-audit`)**

| `audit_label` | Label |
|---|---|
| `pv` | `pv` |
| `heater_or_non_pv` | `non_pv` |
| `uncertain` | dropped |

GT audit centroids come from annotation gpkgs
(`data/annotations/<region>/<source_file>`, row by `row_index`), not from
predictions — they annotate GT polygons, not model detections.

## Area cutoff

`area_cutoff_m2 = 30` (classifier scope). Decisions with `area_m2 >= 30` are
not used for training (they already bypass the classifier at inference time
per `classify_predictions.py`).

## Split

**Region-stratified whole-grid** holdout. Each source bucket gets its own
`GroupShuffleSplit(test_size=0.2, random_state=42)`; results are
concatenated. This prevents one bucket from dominating either train or val.

Auxiliary labels (taxonomy, GT audit) go **train-only**, restricted to
grids that are already in the reviewed train set, to prevent test-set
leakage.

## Current reproducible counts (2026-04-22)

Run on the current working tree with
`scripts/classifier/audit_cls_sources.py --run-id 2026-04-22`:

- Reviewed pool (area < 30 m²): **7,446** chips (5,502 PV / 1,944 non-PV)
  across 105 grids / 3 buckets
- Taxonomy add (train-only): 100 chips (3 PV / 97 non-PV)
- GT audit add (train-only, restricted to train buckets): 570 chips

Typical split outcome (seed=42):

| Bucket | Train | Val |
|---|---|---|
| cape_town:legacy_flat_batch003 | 2,011 | 260 |
| cape_town:v3c_targeted_hn_aerial_2025 | 1,600 | 811 |
| johannesburg:v4_aerial_2023 | 2,169 | 595 |
| **Total (reviewed)** | **5,780** | **1,666** |
| + taxonomy + GT audit (train) | +670 | 0 |
| **Total (selected)** | **~6,450** | **~1,666** |

Final chip counts after extraction (`extraction_stats.*_saved`) can be
slightly lower if a detection centroid falls outside available tiles or
the tile is blank/overexposed.

## Parameters (locked)

| Parameter | Value | Notes |
|---|---|---|
| `area_cutoff_m2` | 30 | Matches classifier inference gate |
| Extraction chip size | 400 × 400 px | Centered on detection centroid |
| Output image size | 224 × 224 px | Resized via `cv2.INTER_AREA` |
| Channels | RGB | First three bands of GeoTIFF |
| Val fraction | 0.2 (per bucket) | `GroupShuffleSplit` |
| Seed | 42 | |

Augmentation profile (`--aug-profile {current, flip_only}`) is recorded in
the manifest but does not affect the build step. It is consumed at training
time; see `exp_cls_augmentation_ablation.md`.

## Manifest

`dataset_manifest.json` is written alongside the chip tree. Includes per-
bucket / per-source / per-region / per-class counts for both the full
reviewed pool and the selected train/val splits, as well as the
extraction success/skip stats. This is the single file every downstream
consumer (training, evaluation, reproducibility checks) reads.

## Known limitations

- **CT batch 003 has no `predictions_metric.gpkg`** — the pre-PR3 layout
  only preserves the `review/` subtree. Taxonomy joining still works
  because taxonomy references batch 004 grids whose `predictions_metric.gpkg`
  lives under the registered model_run path. GT audit works because it
  references annotation gpkgs under `data/annotations/`, not predictions.
- **GEID G1110 (1 grid)** is deprecated by the registry and has
  csv-only review without area info. It is excluded from the mainline
  dataset; exploratory domain-shift evaluation in Task 5b reads this
  grid separately via `--classifier-filtered-gpkg` once its own
  predictions + review are re-generated against corrected GEID mosaics.
- **Grid ID overlap across regions** (e.g., G1189 in both CT and JHB)
  is handled because the split key is `(source_bucket, grid_id)`, not
  `grid_id` alone.
