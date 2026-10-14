# scripts/archive/ — retired one-off analysis & campaign scripts

Scripts moved here on 2026-07-10 after a repo-wide audit (4-area classify +
per-candidate reference verification). Everything in this tree belongs to a
**concluded** experiment/campaign and has **no living reference** from active
code, tests, configs, `.claude/`, or canonical docs at move time. They are
kept for provenance and potential re-runs, not maintained.

DO NOT import from this tree in new code. If a script here turns out to be
needed again, move it back out (git mv) and re-wire references — do not import
it in place.

Caveats:

- Scripts may still reference their **old paths** in docstrings/usage lines
  (e.g. `scripts/analysis/...`), and sibling imports written as
  `from scripts.analysis.X import ...` (e.g. `tta_probe_2026-06/tta_probe_audit.py`
  → `raw_hint_audit`) need a one-line path fix if re-run. Siblings were always
  archived together into the same subdir.
- Historical docs (`docs/experiment-archive/`, `docs/handoffs/`, ADRs) still
  cite old paths; those are historical records and were intentionally not
  rewritten.

Subdir → campaign map:

| Subdir | Campaign / why concluded |
|---|---|
| `annotations/` | Early manual-annotation phase tools (superseded by review_detections.py chain) |
| `backdating_reports/` | JHB install-date delivery reports (delivered 2026-06-02; temporal work now in solar_backdating) |
| `calibration_sweeps/` | Pre-run_benchmark multi-grid GPU driver (retired G-ids) |
| `ch1_precision_campaign/` | V1.4 Ch1 precision campaign (concluded per validation_strategy.md 2026-07-03) |
| `cls_v1_holdout_audit/` | cls v1 holdout audit one-offs |
| `gemini_fp_review_2026-05/` | Two-stage Gemini FP-review calibration (production version = gemini_fp_review_two_stage.py, kept) |
| `imagery/` | Dormant nationwide admin-grid imagery subsystem + legacy VRT builder |
| `jhb_cbd25_3model/` | 2026-05-14 3-model comparison (report: docs/experiments/2026-05-14-jhb-cbd25-3model-sam.md) |
| `li_gt_ingest/` | Li KML / xdomain GT ingestion batches (L-namespace, ingested) |
| `merge_experiments/` | One-off 382-grid gpkg concatenation (superseded by finalize.py) |
| `postprocess_legacy/` | Pre-finalize postproc ablation / NMS-based posthoc eval |
| `pre_migration_2026-04/` | 2026-04 infra migration snapshot tooling |
| `runpod/` | Session-specific pod watchers (hardcoded grid lists) |
| `sam_halo_audit/` | SAM halo per-polygon audit (25-grid JHB CBD, concluded) |
| `small_fp_taxonomy/` | Early small-FP taxonomy tools (superseded by heater-audit + Gemini review chains) |
| `training/` | Sandton batch1 ingest one-off; `jhb_phaseA/` = failed PhaseA retrain tooling |
| `tta_probe_2026-06/` | f1-gap TierB TTA probe cluster (concluded) |
| `validation/` | Ch2 / clean_gt / Vexcel generalization sample builders (GT locked) |
| `vexcel_capture_dates/` | One-off Vexcel /ortho/dates API query |
| `vexcel_sync/` | Dropbox sync hardcoding superseded model_run v3c_vexcel_2024 |
