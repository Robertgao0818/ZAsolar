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

## Frozen deployment baseline (CT, 2026-04-28): v1 ConvNeXt-tiny

Cape Town in-domain cascade is **frozen as a deployment baseline** at
v1 + ConvNeXt-tiny + threshold 0.5 + V3-C + SAM mask+box. All future
classifier work (v2 / v3 / ensembles / per-imagery thresholds) is
evaluated *against this baseline* on the CT side, not against raw V3-C.

### Frozen numbers

CT 6-grid holdout (no training-set leakage), 798 GT polygons,
`results/analysis/cls_cascade_holdout/backbone_compare_thr050/summary_metrics.csv`:

| Stage | P | R | F1 | FP | TP-lost vs raw |
|---|---:|---:|---:|---:|---:|
| Raw V3-C (no classifier) | 0.685 | 0.931 | 0.789 | 342 | — |
| **+ v1 ConvNeXt @ 0.5** | **0.835** | **0.890** | **0.862** | **140** | **33** |
| + v1 EffB0 @ 0.5 | 0.824 | 0.880 | 0.851 | 150 | 41 |
| + v1 DINOv2 @ 0.5 | 0.843 | 0.888 | 0.865 | 132 | 34 |

**Headline**: F1 0.789 → 0.862 (**+7.3 pp**); FP kill 202 / TP lost 33
≈ **6.1 : 1** net-positive trade. Recall drop only −4.1 pp on a CT
in-domain holdout that contains no training grids.

DINOv2 edges F1 by +0.003 but with one extra TP loss; ConvNeXt is
chosen as the default for **operational reasons**: faster inference,
fewer dependency surprises (timm vs DINOv2 weights), and the
recall-budget headroom (3 pp slack vs ConvNeXt's 4.1 pp) is too small
to outweigh the operational simplicity.

### Why this is defensible despite the OOD failure

The same v1 ConvNeXt scores **bal_acc 0.653** on the JHB CBD GEID
audit set (462 chips, see `results/analysis/cls_audit_eval_20260427/`)
and only kills **15.5%** of non-PVs at PV-recall ≥ 0.95 — far below the
40% promotion bar for v2. The two regimes are not in conflict:

- **CT in-domain non-PV ≈ {water heater, skylight, shadow}**, dominated
  77% by water heaters in the original taxonomy. v1's CT training pool
  covers this distribution well; the model learned a tight one-class PV
  boundary against this specific non-PV mix.
- **JHB CBD GEID non-PV ≈ {skylight 22%, corrugated metal 15%, road
  marking 10%, HVAC, water heater 7%}**. v1 has near-zero training
  signal for corrugated metal / HVAC / road marking and does not see
  GEID's color/sharpness statistics — the same one-class boundary
  generalises poorly.

→ v1 is a *CT-shaped* PV-vs-non-PV filter. Within CT it functions as
the intended water-heater suppressor (the dominant FP class). Outside
CT — especially on GEID CBD — it should not be deployed.

### Deployment scope

v1 ConvNeXt is approved for inference-time filtering on results runs
whose `config.json.imagery_layer_id == aerial_2025` (Cape Town
municipal WMS). Other imagery layers (`aerial_2023` JHB, `geid_2024_02`
JHB CBD, future Vexcel layers) **must not** route through v1 by
default; they wait for v2 with per-imagery thresholds (see
`exp_cls_dataset_protocol.md` v2 protocol section).

### Replacement criteria

v1 ConvNeXt is replaced as the CT default when *both* hold:

1. v2 (or later) achieves CT 6-grid holdout F1 ≥ 0.862 + 1 pp
   (i.e. ≥ 0.872) at the same recall budget (≤ 4.5 pp drop vs raw),
   *and*
2. v2 satisfies its own JHB GEID promotion rule (PV-recall ≥ 0.95,
   non-PV kill ≥ 40%) so the replacement is one model, not a per-domain
   patchwork.

If only (1) holds and (2) does not, v1 ConvNeXt **stays as the CT
deployment** while v2 ships only on the imagery layers it qualifies
for. Per-imagery model selection is acceptable; per-imagery thresholds
on a single model is preferred but not required.

### What this freeze unblocks

- Vexcel pull / new-city domain-shift work can proceed without waiting
  on classifier completion: CT is already stable with v1, and new
  cities use raw V3-C + SAM mask+box until v2 has a per-imagery
  threshold for the relevant layer.
- V1.4 Channel 1 RA precision audit on CT can use v1-filtered output as
  its input, since v1 is now the official CT inventory pipeline rather
  than a candidate.
- v2 work can be evaluated honestly against +7.3 pp baseline, not
  against raw V3-C.
