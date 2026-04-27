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
