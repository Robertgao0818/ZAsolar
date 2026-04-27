# Classifier → Detector Evaluation Integration Contract

**Status**: Active
**Plan reference**: Task 5a in
`/home/gaosh/.claude/plans/codex-efficientnetb0-convnexttiny-found-swirling-muffin.md`
**Related**: `exp_cls_backbone_ablation.md`, `exp_cls_dataset_protocol.md`

Defines how the PV vs non-PV classifier plugs into the installation-level
evaluation pipeline. The two components stay **decoupled** — the classifier
is not injected into `detect_and_evaluate.py`'s main pipeline.

## Why decoupled

Injecting the classifier into the detector pipeline would:

1. Conflate classifier config changes with detection cache keys, causing
   unnecessary re-runs every time a classifier threshold shifts.
2. Break existing `configs/postproc/*.json` semantics (`load_postproc_config`
   is for geometric post-processing only).
3. Force every caller of `detect_and_evaluate.py` to carry classifier
   knobs even when classifier filtering is not wanted.

Keeping the classifier as an **external pre-evaluation step** means each
piece can be iterated independently, and cache invalidation stays tight.

## Three-step data flow

```
┌─────────────────────┐    ┌──────────────────────────┐    ┌─────────────────────┐
│ detect_and_eval.py  │ →  │ classify_predictions.py  │ →  │ detect_and_eval.py  │
│ (detection run)     │    │ (post-hoc classifier)    │    │ --classifier-       │
│                     │    │                          │    │ filtered-gpkg       │
│ writes              │    │ reads                    │    │                     │
│ predictions_metric  │    │ predictions_metric.gpkg  │    │ eval-only path      │
│ .gpkg               │    │                          │    │                     │
│                     │    │ writes                   │    │ reads the filtered  │
│                     │    │ predictions_metric_     │    │ gpkg and reports    │
│                     │    │ filtered.gpkg            │    │ installation-level  │
│                     │    │ (+ _cls.gpkg provenance) │    │ P/R/F1              │
└─────────────────────┘    └──────────────────────────┘    └─────────────────────┘
```

## Artifacts

All artifacts live under `results/<region-or-flat>/<model-run-or-flat>/<GRID>/`:

| File | Writer | Contents |
|---|---|---|
| `predictions_metric.gpkg` | `detect_and_evaluate.py` | Raw detections (source of truth for classifier input). |
| `predictions_metric_cls.gpkg` | `classify_predictions.py` | All detections + `cls_score`, `cls_label`, `cls_applied` columns. Full provenance record. |
| `predictions_cls.geojson` | `classify_predictions.py` | EPSG:4326 export of above, for QGIS inspection. |
| `predictions_metric_filtered.gpkg` | `classify_predictions.py` | **Canonical filtered GPKG** — only rows with `cls_label == 'pv'`. This is the file consumed by `detect_and_evaluate.py --classifier-filtered-gpkg`. |
| `predictions_metric_cls_filtered.gpkg` | `classify_predictions.py` | Identical copy for backwards compat with earlier review scripts. |
| `cls_summary.json` | `classify_predictions.py` | Counts, threshold, area_cutoff, model_path, timestamp. |

## Command sequence

**Detection (produces raw GPKG)**

```bash
python detect_and_evaluate.py --grid-id G1687 \
  --model-path checkpoints/exp003_C_targeted_hn/best_model.pth \
  --postproc-config configs/postproc/v4_canonical.json
```

**Classifier filter (produces filtered GPKG)**

```bash
python scripts/classifier/classify_predictions.py \
  --grid-id G1687 \
  --model-path checkpoints/cls_pv_thermal/best_cls.pth \
  --pv-threshold 0.5
```

**Evaluation with classifier filter applied**

```bash
python detect_and_evaluate.py --grid-id G1687 \
  --evaluation-profile installation \
  --classifier-filtered-gpkg results/G1687/predictions_metric_filtered.gpkg \
  --classifier-model-path checkpoints/cls_pv_thermal/best_cls.pth \
  --classifier-threshold 0.5
```

When `--classifier-filtered-gpkg` is provided, `detect_and_evaluate.py`:

1. **Skips detection entirely** — goes straight to `load_predictions(override_path=...)`.
2. Writes `classifier_filtered_gpkg`, `classifier_model_path`,
   `classifier_threshold` into the grid's `config.json` so the
   provenance-hashed cache key differs from the unfiltered run.
3. Uses the overridden GPKG for both presence matching and installation-
   level evaluation. `--evaluation-profile` semantics are unchanged.

## CLI contract (`detect_and_evaluate.py`)

New flags (all optional, all no-op if absent):

| Flag | Purpose |
|---|---|
| `--classifier-filtered-gpkg <path>` | Swap in the classifier-filtered GPKG. Triggers eval-only mode. |
| `--classifier-model-path <path>` | Provenance only — recorded in `config.json`. |
| `--classifier-threshold <float>` | Provenance only — recorded in `config.json`. |

`--force`, `--postproc-config`, `--evaluation-profile`, `--data-scope` all
continue to work as before. The classifier filter flags are **additive** —
they never silently change evaluation profile or postproc params.

## Cache behavior

`should_reuse_predictions` hashes `detection_config` to decide whether the
detection cache is reusable. When `--classifier-filtered-gpkg` is given,
the cache check is bypassed entirely (we are in eval-only mode). The
additional provenance fields (`classifier_*`) are recorded in
`detection_config` so that a subsequent detection-mode run against the same
grid will not reuse results that were generated under a different
classifier.

## Non-goals for this task

- **Not** fusing the classifier into `detect_solar_panels`.
- **Not** adding classifier_model as a `configs/postproc/*.json` key
  (existing `_meta` note in `detect_and_evaluate.py:226-227` about a future
  `classifier_model` postproc extension is now superseded by this
  decoupled design).
- **Not** changing evaluation profile semantics — `installation` stays
  installation-level evaluation of reviewed-prediction footprints vs
  installation-level GT.

## Verification steps

For any grid with reviewed GT available:

1. Run classifier filter to produce `predictions_metric_filtered.gpkg`.
2. Run `detect_and_evaluate.py` without the flag → record pre-filter
   installation-level P/R/F1.
3. Run `detect_and_evaluate.py --classifier-filtered-gpkg ...` → record
   post-filter P/R/F1.
4. Delta must match what `compare_results.py` reports on reviewed-grid
   filtered precision; otherwise the decoupling is leaking state.
5. `config.json` in the post-filter run must contain the three classifier
   provenance fields.

## Detector × SAM mode benchmark (2026-04-26)

Why this matters for classifier integration: the classifier is the second
stage of a (detector → SAM mask refine → classifier) cascade. Picking the
detector + SAM mode that *enters* the classifier determines (a) how many FPs
the classifier has to kill, (b) the ceiling area F1 the cascade can reach,
and (c) the recall budget. The 9-cell ablation below was run on the V1.4
benchmark blueprint (25 JHB CBD grids × Li hand-labeled GT × GEID 2024-02
chunked imagery, post-proc `configs/postproc/v4_canonical.json`,
cluster-level eval via `scripts/analysis/cluster_level_eval.py`).

### Full 3 detector × 3 SAM mode matrix

| Detector / SAM mode | matched | FP | FN | cluster R | cluster F1 | **area F1** | **balanced** |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3-C / no SAM            |  977 |  578 |  364 | 0.729 | **0.675** | 0.833 | 0.685 |
| **V3-C / box-only**      |  948 |  591 |  495 | 0.657 | 0.636 | **0.918** ⭐ | **0.798** ⭐ |
| V3-C / mask+box          |  981 |  578 |  389 | 0.716 | 0.670 | 0.895 | 0.743 |
| V4.1 / no SAM            |  851 |  508 |  332 | 0.719 | 0.670 | 0.791 | 0.620 |
| V4.1 / box-only          |  831 |  519 |  467 | 0.640 | 0.628 | 0.891 | 0.750 |
| V4.1 / mask+box          |  857 |  506 |  339 | 0.717 | 0.670 | 0.855 | 0.676 |
| V4.2 / no SAM            |  826 | 1211 |  159 | **0.839** | 0.547 | 0.730 | 0.558 |
| V4.2 / box-only          |  821 | 1211 |  244 | 0.771 | 0.530 | 0.860 | 0.696 |
| V4.2 / mask+box          |  841 | 1198 | **155** | **0.844** | 0.554 | 0.796 | 0.613 |

Source artifacts: `results/analysis/{v3c_rerun,v4_1,v4_2}_vs_li_20260426/`
(detector-only) and `results/analysis/{v3c,v4_1,v4_2}_sam_{box,mask}_vs_li_20260426/`
(+ SAM); raw masks in `results/johannesburg/{v3c,v4_1,v4_2}_sam_{box,mask}_geid_2024_02/`.

### Three findings that shape classifier-stage planning

1. **"SAM is the area F1 ceiling" — falsified.** V4.2+SAM tops at area F1
   0.860; V3-C+SAM tops at 0.918. **Detector box quality propagates
   directly into SAM output** — clean boxes let SAM pick the right
   sub-component. Detector choice is *not* recall-only.
2. **Box-only FN flip is detector-invariant.** V3-C +131 FN, V4.1 +135 FN,
   V4.2 +85 FN — every detector loses recall when SAM gets bbox-only
   prompts. The "SAM `argmax(score)` specificity bias picks
   sub-components" failure mode is general, not over-segmentation-
   specific. Classifier integration cannot fix this — it must be solved
   at the SAM-prompt layer.
3. **mask+box gives a +6pp area F1 gain across all detectors.**
   V3-C +6.2 / V4.1 +6.4 / V4.2 +6.6. The role of mask-prompt is
   sub-component-flip suppression, not over-seg repair.

### Implications for classifier integration

- **Primary cascade input = V3-C + SAM mask+box.** Detector matched=981,
  FP=578, FN=389, area F1 0.895. Classifier needs to kill ~70% of 578 FPs
  to push cluster P from 0.629 → ~0.85; estimated three-stage cluster F1
  ≈ 0.78, area F1 0.895, balanced ≈ 0.80+.
- **Aggressive variant = V3-C + SAM box-only.** Starts at area F1 0.918 /
  balanced 0.798 (already best in the no-classifier matrix). Same FP count
  (591), but recall is capped at 0.657 — choose this only if downstream KPI
  is weighted toward area precision and recall is allowed to drop.
- **Control = V4.2 + SAM mask+box.** recall 0.844 ceiling but FP 1198
  forces classifier recall ≥ 90% to break even on cluster F1; reserved for
  the case where someone trains a near-perfect water-heater discriminator.
- **V4.1 is dominated** at every SAM mode by V3-C at the same SAM mode;
  drop from candidate cascades, keep only as ablation control.

### Classifier dataset implication

The 981 matched + 578 FP from V3-C + SAM mask+box on the 25 CBD grids form
a natural labeled pool for the binary PV vs non-PV classifier:

- 981 matched (vs Li GT, IoU ≥ 0.1) → positive examples
- 578 FPs → negative examples (subtype audit feeds
  `scripts/analysis/cls_nonpv_subtype_audit.py` to ensure water-heater /
  fixture / shadow / road-marking coverage)

This dataset is built and consumed via the protocol in
`exp_cls_dataset_protocol.md`; see also `exp_cls_backbone_ablation.md` for
the backbone evaluation that will be run on it.
