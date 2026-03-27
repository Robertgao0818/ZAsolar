# Project Status — Cape Town Solar Panel Detection

**Last Updated**: 2026-03-25

## Current Phase

V3 — Scaled annotation (43 grids) & retrain (IN PROGRESS)

---

## Progress Tracker
<!-- progress:status:start -->
### Recent Updates
- 2026-03-25: V3 retrain on 43-grid cleaned annotations — Stage 2 epoch 4/20, val_AP50=0.788 (+0.099 vs V2)
- 2026-03-25: export_coco_dataset.py updated to auto-discover cleaned/ annotations; 2982 polygons, 5636 train chips
- 2026-03-25: Project cleanup: removed old COCO datasets & intermediate checkpoints (~6G freed)
- 2026-03-24: Annotation expansion to 43 grids (data/annotations/cleaned/), cumulative 2982 installations
- 2026-03-22: OSM building filter for tile-level selective download (82% savings)
- 2026-03-20: V2 SAM2 annotation + retrain complete (val_AP50=0.6889, +0.268 vs V1)

### Current Ops Focus
- V3 training in progress (checkpoints_cleaned/best_model.pth, AP50=0.788 at epoch 4/20)
- Cloud migration planning: RunPod Pod + Network Volume for scalable training
- Continue annotation expansion beyond 43 grids
<!-- progress:status:end -->

## Evaluation Results Summary (V2, SAM2 GT, F1@IoU0.5)
<!-- progress:results:start -->
| Grid | P@IoU0.5 | R@IoU0.5 | F1@IoU0.5 | mean IoU | Status |
|------|----------|----------|-----------|----------|--------|
| G1238 | 0.509 | 0.569 | 0.537 | 0.691 | evaluated |
| G1190 | 0.559 | 0.717 | 0.628 | 0.742 | evaluated |
| G1189 | 0.555 | 0.560 | 0.557 | 0.729 | evaluated |
| JHB01–06 | — | — | — | — | not evaluated |
<!-- progress:results:end -->

## V0: Baseline Detection Pipeline — COMPLETE

| Grid | Precision | Recall | F1 | Notes |
|------|-----------|--------|----|-------|
| G1238 | 0.62 | 0.66 | 0.64 | Best grid, 124 annotations |
| G1189 | — | 0.33 | — | Low recall |
| G1190 | — | 0.39 | — | Low recall |
| JHB01–06 | — | 0.28 (macro) | — | Cross-city transfer |

## V1: Cape Town Fine-Tune — SUPERSEDED BY V2

Fine-tuned Mask R-CNN on 257 Cape Town annotations (legacy weak-supervision).

**Best checkpoint**: `checkpoints/v1_ft_cs400_tileval_20260317_r4/best_model.pth` (val_AP50=0.4205)

Full details → see `ROADMAP.md` V1 section.

---

## V1.2: Evaluation Profile & Annotation Alignment — COMPLETE

Task frozen as **installation-level footprint segmentation** (not panel-level). No model change this round.

### Work Package Progress

| WP | Description | Status |
|----|-------------|--------|
| Pre | STATUS.md progress tracker | Done |
| WP0 | Annotation spec (ANNOTATION_SPEC.md) | Done |
| WP1 | Annotation manifest + bootstrap script | Done |
| WP2 | Evaluation profile (presence/footprint/area) | Done |
| WP3 | config.json extension (evaluation_config) | Done |
| WP4 | COCO export manifest-aware filtering | Done |
| WP5 | train.py minor update | Done |
| WP6 | Documentation updates | Done |
| -- | GPU integration test (3 grids) | Done |

### V1.2 Acceptance Criteria

- full-grid F1@IoU0.3: G1189 >= 0.595, G1190 >= 0.649, G1238 no regression
- presence recall@IoU0.1: no regression from fine-tuned baseline
- 5-20m² bucket recall: must be reported; G1189/G1190 no regression
- area error: establish baseline (no hard gate yet)

Acceptance status: all current V1.2 gates satisfied on GPU-validated full-grid runs.

GPU integration test details → [`docs/experiment-archive/gpu-integration-test-v1.2.md`](docs/experiment-archive/gpu-integration-test-v1.2.md)

---

## V2: SAM2 Annotation & Retrain — COMPLETE

Re-annotated all 3 grids with SAM2.1 (GeoSAM/QGIS), 456 polygons total (T1 quality).

**Best checkpoint**: `checkpoints/v2_sam2_260320/best_model.pth` (val_AP50=0.6889)

| Metric | V1 | V2 | Change |
|--------|-----|-----|--------|
| val_AP50 | 0.4205 | 0.6889 | **+0.268** |
| Annotations | 257 (T2) | 456 (T1 SAM2) | +77% |
| Primary metric | F1@IoU0.1 | **F1@IoU0.5** | upgraded |

Key: precision (~0.63) is now the bottleneck, not recall. Semi-auto annotation workflow ready for scaling.

---

## V3: Scaled Annotation & Retrain — IN PROGRESS

Expanded annotations from 3 grids (456 polygons) to 43 grids (2982 polygons) using SAM2 cleaned workflow.

**Training (in progress)**: `checkpoints_cleaned/best_model.pth`

| Metric | V2 (3 grids) | V3 (43 grids, epoch 4/20) | Change |
|--------|-------------|---------------------------|--------|
| val_AP50 | 0.6889 | 0.7880 | **+0.099** |
| Annotations | 456 (T1 SAM2) | 2982 (T1 SAM2 cleaned) | **+553%** |
| Train chips | ~900 | 5636 | +526% |

COCO export: `data/coco_cleaned/` (28G, 74k files). Next step: WebDataset sharding for cloud training.

### Infrastructure Plan
- Local: annotation (QGIS), scripting, reports
- RunPod (planned): export, train, evaluate — Network Volume for tiles + COCO + checkpoints
- Old COCO datasets & intermediate epoch weights cleaned up (2026-03-25)

### Next Steps
- Complete V3 training (remaining 16 epochs)
- Full-grid evaluation with V3 checkpoint
- RunPod migration for faster iteration
- Continue annotation expansion
- FP reduction via hard negative mining
