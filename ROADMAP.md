# Roadmap — Cape Town Solar Panel Detection

## Execution Track
<!-- progress:roadmap:start -->
### Recently Completed
- 2026-04-03: Code updates: benchmark runner, batch inference, hard-negative training, review GUI improvements

### Next Up
- Repository structure cleanup: reduce root-level script clutter and group workflows by purpose.
- Export reviewed keep/exclude decisions into a reusable grid manifest for later tile downloads.
<!-- progress:roadmap:end -->

## V0: Baseline Detection Pipeline — DONE

Stock geoai `SolarPanelDetector` (Mask R-CNN ResNet50-FPN) + post-processing.

### Completed
- [x] Detection pipeline (`detect_and_evaluate.py`): geoai → mask → vectorize → filter → evaluate
- [x] Building footprint filter (`building_filter.py`): OSM + Microsoft buildings
- [x] Tile pipeline (`tiles/build_vrt.py`): WMS download → GeoTIFF → VRT mosaic
- [x] Multi-threshold IoU evaluation (0.1–0.7), merge-matching and strict modes
- [x] Size-stratified recall analysis (<10m², 10–50m², 50–100m², >100m²)
- [x] Per-tile evaluation, error analysis (FP/FN classification), confidence histograms
- [x] Parameter tuning: chip_size=400, overlap=0.25, conf=0.3, post_conf=0.70

### Baseline Numbers (stock weights)
| Grid | Precision | Recall | F1 | Notes |
|------|-----------|--------|----|-------|
| G1238 | 0.62 | 0.66 | 0.64 | Best grid, 124 annotations |
| G1189 | — | 0.33 | — | Low recall |
| G1190 | — | 0.39 | — | Low recall |
| JHB01–06 | — | 0.28 (macro) | — | Cross-city transfer |

### Known Issues
- Low recall on G1189/G1190
- Large arrays (>100m²) match at low IoU — mask quality insufficient
- JHB generalization poor with stock weights

---

## V1: Cape Town Fine-Tune — MINIMUM V1 COMPLETE

Fine-tune Mask R-CNN on 257 Cape Town annotations across 3 grids.

### Completed
- [x] `export_coco_dataset.py`: annotation → COCO exporter with 400×400 chips, tile-level 80/20 split, 1:1 pos:neg balancing
- [x] `train.py`: 2-stage training (heads-only 3 epochs → full fine-tune 20 epochs), cosine LR, augmentations (flip/rotate/color jitter/scale)
- [x] `detect_and_evaluate.py --model-path`: CLI integration for custom weights
- [x] Smoke test: annotation loading, tile splitting, chip extraction logic verified
- [x] Minimum V1 run completed on CUDA machine (`RTX 4070 Laptop GPU`)
- [x] Exact val-split baseline snapshot recorded on the same 20 validation tiles
- [x] Acceptance cleared: Cape Town val F1 improved by +0.2587 over baseline
- [x] Best checkpoint selected: `checkpoints/v1_ft_cs400_tileval_20260317_r4/best_model.pth` (`val_AP50=0.4205`)

### TODO
- [x] Post-training calibration sweep (min_object_area 2→5, max_elongation 4→8, post_conf unchanged at 0.70)
- [ ] Freeze best parameter set as v1 inference bundle
- [ ] Merge additional high-value annotations and rerun the same val-split protocol
- [ ] Leave-one-grid-out cross-validation
- [ ] JHB transfer evaluation (reporting only, baseline macro recall = 0.281)
- [ ] Size-stratified recall diagnostics focused on G1189 small panels (<20m²)

### Training Data
- G1238: 248 valid polygons (layer `SAM_Residential_merged`, SAM2.1 detailed segmentation)
- G1189: 58 polygons (from combined GPKG, Name prefix filter)
- G1190: 76 polygons (from combined GPKG, Name prefix filter)
- Total: 257 polygons across 126 source tiles (42 per grid)
- Split: ~80% train / ~20% val per grid, tile-level, no tile overlap
- Export used for the minimum V1 run: `data/coco_v1_ft_cs400_tileval_20260317_r3`
- Chips after balancing: train = 804 (402 positive, 402 negative), val = 218 (109 positive, 109 negative)
- Validation scope for decision-making: 20 val tiles, 64 GT polygons total

### Training Config
- Architecture: `maskrcnn_resnet50_fpn`, num_classes=2, init from geoai weights
- Stage 1: heads-only, 3 epochs, LR=1e-3
- Stage 2: full, ≤20 epochs, LR=1e-4, cosine decay
- SGD momentum=0.9, weight_decay=1e-4, batch_size=4
- Augmentations: flip H/V, 90°/180°/270° rotation, color jitter, 0.8–1.2× scale
- Checkpoint: best val `segm_AP50`

### Minimum V1 Val-Split Result
Same tile-level validation split for both stock baseline and fine-tuned model. Inference kept the current best small-chip setup and current post-processing bundle.

| Scope | Precision | Recall | F1 | ΔP | ΔR | ΔF1 |
|------|-----------|--------|----|----|----|-----|
| Overall val | 0.7031 | 0.7031 | 0.7031 | +0.0746 | +0.3594 | +0.2587 |
| G1238 | 0.7576 | 0.9259 | 0.8333 | +0.1465 | +0.5185 | +0.3444 |
| G1189 | 0.5625 | 0.4500 | 0.5000 | +0.0170 | +0.1500 | +0.1129 |
| G1190 | 0.7333 | 0.6471 | 0.6875 | -0.1000 | +0.3529 | +0.2527 |

### Size Subsets
- Small-panel recall (`<20m²`): `0.3939 -> 0.4848` (`+0.0909`)
- Large-array recall (`>100m²`): `0.2500 -> 1.0000` (`+0.7500`)
- Main residual weakness: G1189 small-panel recall still regressed (`0.3077 -> 0.2308`) despite the overall gain

### Decision
- Current evidence supports staying on the "add labels + fine-tune" path rather than changing model family
- Next iteration should target G1189-style small panels first, while preserving the strong gains already seen on G1238 and large arrays

---

## V1.2: Evaluation Profile & Annotation Alignment — COMPLETE

Task frozen as **installation-level footprint segmentation**. No model change. Focus: evaluation caliber, annotation consistency, experiment traceability.

### Completed
- [x] Annotation specification (`data/annotations/ANNOTATION_SPEC.md`): installation footprint definition, quality tiers T1/T2, correction policy
- [x] Annotation manifest (`data/annotations/annotation_manifest.csv`): 257 annotations with per-annotation quality tier tracking
- [x] Bootstrap script (`scripts/bootstrap_manifest.py`): generates manifest from existing GPKGs
- [x] Evaluation profile (`detect_and_evaluate.py --evaluation-profile`): installation vs legacy_instance modes
- [x] Three-layer installation metrics: presence (P/R/F1 @IoU0.1), footprint (IoU/Dice distribution), area error (per size class)
- [x] config.json extension: `evaluation_config` section (profile, label_definition, data_scope, tier_mix)
- [x] COCO export manifest-aware filtering (`--manifest`, `--tier-filter`, `--category-name`)
- [x] Documentation updates (CLAUDE.md, ROADMAP.md, annotations/README.md)
- [x] GPU integration validation completed on all 3 Cape Town grids (`G1189`, `G1190`, `G1238`) using installation profile

### Acceptance Criteria
- full-grid F1@IoU0.3: G1189 >= 0.595, G1190 >= 0.649, G1238 no regression
- presence recall@IoU0.1: no regression from current fine-tuned baseline
- 5-20m² bucket recall reported; G1189/G1190 no regression
- area error: establish V1.2 baseline (no hard gate yet)

### GPU-Validated Full-Grid Results

All runs below used:
`./.venv/bin/python detect_and_evaluate.py --model-path checkpoints/v1_ft_cs400_tileval_20260317_r4/best_model.pth --evaluation-profile installation --force`

| Grid | Presence P/R/F1 @IoU0.1 | Merge F1@IoU0.3 | Notes |
|------|--------------------------|-----------------|-------|
| G1189 | 0.6667 / 0.7241 / 0.6942 | 0.5950 | Meets threshold exactly |
| G1190 | 0.7600 / 0.7500 / 0.7550 | 0.6490 | Meets threshold exactly |
| G1238 | 0.7037 / 0.9268 / 0.8000 | 0.7789 | Clear no-regression pass |

### Outcome

- V1.2 acceptance gate cleared on GPU-validated full-grid runs
- Installation-level evaluation profile is now the project-default reporting frame
- Area-error baselines are established for all 3 Cape Town grids

### TODO
- [ ] Manual T1 review of G1189/G1190 val annotations (upgrade from T2)
- [x] Post-training calibration sweep (completed: min_area=5, max_elongation=8, macro mean F1 +0.021)
- [ ] G1189 small-panel (5-20m²) targeted annotation augmentation

---

## V2: SAM2 Annotation & Retrain — COMPLETE

Re-annotated all 3 Cape Town grids with SAM2.1 (GeoSAM plugin in QGIS), retrained Mask R-CNN, switched primary evaluation metric to F1@IoU0.5.

### Completed
- [x] SAM2.1 re-annotation: G1238 (248), G1189 (109), G1190 (99) — total 456 polygons, all T1 quality
- [x] Annotation file naming convention: `{GridID}_SAM2_{YYMMDD}.gpkg`
- [x] COCO export with SAM2 annotations (`data/coco_sam2_260320`)
- [x] Retrain: val_AP50 = 0.6889 (V1: 0.4205, **+0.268**)
- [x] GT resolution: `core/grid_utils.py` auto-selects SAM2 annotations for evaluation
- [x] Full-grid evaluation on all 3 grids with SAM2 GT
- [x] Primary metric switched from F1@IoU0.1 to **F1@IoU0.5**
- [x] Model-assisted annotation script (`scripts/annotations/export_hints.py`)

### Training Data
- G1238: 248 polygons (SAM2.1, layer `SAM_Residential_merged`)
- G1189: 109 polygons (SAM2.1, layer `sam_residential_g1189_mosa_109_rgb255105180`)
- G1190: 99 polygons (SAM2.1, layer `SAM_Residential_20260320_221905`)
- Total: 456 polygons across 126 source tiles
- Chips after balancing: train = 820 (410 pos, 410 neg), val = 204 (102 pos, 102 neg)

### Best Checkpoint
`checkpoints/v2_sam2_260320/best_model.pth` (val_AP50=0.6889, epoch 18)

### Full-Grid Results (SAM2 GT, installation profile)

| Grid | P@IoU0.5 | R@IoU0.5 | **F1@IoU0.5** | mean IoU | IoU>=0.5 rate |
|------|----------|----------|---------------|----------|---------------|
| G1238 | 0.509 | 0.569 | 0.537 | 0.691 | 81.0% |
| G1190 | 0.559 | 0.717 | **0.628** | 0.742 | 87.7% |
| G1189 | 0.555 | 0.560 | 0.557 | 0.729 | 87.1% |

### Key Findings
- Recall greatly improved vs V1; precision is now the bottleneck (~0.63 across all grids)
- SAM2 GT is stricter than legacy annotations — F1 numbers are lower but more honest
- Footprint quality high: mean IoU 0.69-0.74, mean Dice 0.80-0.84
- TP confidence (0.977) well-separated from FP confidence (0.865) — enables tiered hint workflow

---

## V3: Future Directions (not started)

- [ ] Semi-automatic annotation: expand to additional Cape Town grids via export_hints.py + SAM2 workflow
- [ ] FP reduction: hard negative mining, confidence threshold tuning
- [ ] Detector + SAM2 inference pipeline (bbox detection → SAM2 mask refinement)
- [ ] JHB annotations + fine-tuning for cross-city generalization
- [ ] Stronger backbone (Swin Transformer, ConvNeXt) if F1@IoU0.5 plateaus
- [ ] Active learning: prioritize annotation on high-uncertainty tiles
