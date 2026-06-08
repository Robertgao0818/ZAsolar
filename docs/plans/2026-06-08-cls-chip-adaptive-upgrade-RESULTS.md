# solar_cls adaptive-chip upgrade — RESULTS (2026-06-08)

**Status:** ✅ LANDED & VALIDATED. Adaptive bbox+margin chip (design B,
`chip_spec_version=adaptive_v1`) **beats the locked fixed-400 baseline on every
metric** on the 16 held-out Li grids. Plan:
[`2026-06-08-cls-chip-adaptive-upgrade-plan.md`](2026-06-08-cls-chip-adaptive-upgrade-plan.md).

## What changed
Replaced solar_cls's fixed **400px centroid crop → resize 224** with a unified
**adaptive bbox+margin → 224** chip (bbox-center, MARGIN=0.6, clamp [96,512]px,
edge-reflect pad, INTER_CUBIC up / INTER_AREA down). The crop is now defined in
ONE shared module `scripts/classifier/chip_extraction.py` imported by all
extractors (train + infer mirror, guaranteed byte-identical + golden-hash test).
Retrained dinov2_vits14 on the rebuilt v1+cascade chips, recalibrated per-layer
thresholds, A/B'd CLS-only vs the locked baseline.

## Mirror guarantee
- `tests/classifier/test_chip_extraction.py`: 7/7. Golden pixel hash
  `546f348f…b9` verified **byte-identical on local WSL and the RunPod pod**
  (cv2 4.13.0, rasterio 1.5.0) — so locally-rebuilt training chips match
  pod-side inference chips exactly.
- Rebuilt v1 (6005 train / 1661 val) and v2 (train pv4668/npv1995, val
  pv1306/npv600) reproduce the **original dataset composition exactly** (same
  grids, same split seed 0, same counts) — only the chip pixels differ. Clean A/B.

## Tier-0 (per-layer val, same val detections as baseline)
nonpv_kill at 0.95 PV recall, aerial_2025 dinov2:

| chip spec | nonpv_kill@0.95R | Δ |
|---|---|---|
| fixed-400 (baseline) | 0.871 | — |
| **adaptive_v1** | **0.959** | **+8.8pp** |

(geid_2024_02 promotable=True; aerial_2023 kill 0.962.)

## Tier-1 (CLS-only inventory, 16 Li grids, GT=Capetown_Li, n_gt=1233)
σ_Bw + RMSE primary referees; bulk ∈ [0.5,2.0] gate; cov50 census tiebreaker.

| run | filter | F1 | bulk | **σ_Bw** | RMSE | R² | **cov50** | kept |
|---|---|---|---|---|---|---|---|---|
| `unifiedA_li_clsonly` | fixed-400, prod (area≥30 ∨ calib) | 0.723 | 1.427 | 0.297 | 1505 | 0.942 | 0.815 | 2211 |
| `unifiedA_li_clsonly_v2_bypass` | adaptive, area≥30 bypass | 0.731 | 1.414 | 0.286 | 1443 | 0.948 | 0.830 | 2182 |
| **`unifiedA_li_clsonly_v2_all`** | **adaptive, classify-all** | **0.742** | **1.376** | **0.260** | **1332** | **0.955** | **0.829** | 2144 |

Baseline row reproduced the locked numbers exactly (0.815 / 0.297 / 1.427) →
harness is apples-to-apples.

## Verdict: ship **config (c) classify-all** (adaptive_v1)
- σ_Bw **0.260** < 0.277 (calib-only baseline) and < 0.297 (prod baseline) —
  the #1 dispersion judge. **WIN.**
- nonpv_kill@0.95R **0.959 ≥ 0.871**. **WIN.**
- bulk **1.376** ∈ [0.5,2.0]. Gate **PASS** (and closer to 1.0 than baseline).
- cov50 **0.829 > 0.815** (+1.4pp) — keeps MORE true installations while
  dropping MORE FPs (2144 kept vs 2211). Escapes the recall/dispersion tradeoff.
- Threshold-robust: v2 dominates baseline cov50 at **every** sweep threshold
  (0.85→0.97), not just the calibrated operating point.
- classify-all loses **no** large-band recall vs bypass (cov50 0.829 ≈ 0.830),
  so the plan's gate "config c must hold large-band pvR" passes — drop the
  area_cutoff bypass for CT census (a clipped-detection safety floor remains:
  ext_px>MAX_SIDE → kept PV; ~0.1% of CT dets).

## Why it works (mechanism)
The fixed 400px→224 squeeze downsampled 80% of detections (<30 m²) to an
effective 14.8 cm/px, aliasing small dark PV with water heaters. The adaptive
crop UPSAMPLES the small band (side floors to 96px → INTER_CUBIC to 224),
preserving the texture that separates PV cells from a smooth heater tank, and
only downsamples genuinely large detections. Result: a sharper small-band
decision boundary (kill ↑) plus a newly-adjudicated large band (classify-all)
that removes big lookalike FPs without clipping true large PV.

## Artifacts
- Checkpoint (not committed): `cls_pv_thermal_v2_dinov2_vits14_adaptive/best_cls.pth`
  (best bal-acc 0.9607 vs baseline 0.9044); mirrored to
  `~/zasolar_data/cls/checkpoints/`.
- Thresholds: `solar_cls/configs/classifier/thresholds_v2_adaptive.json`
  (`chip_spec_version=adaptive_v1`; aerial_2025 thr 0.7168).
- Datasets (not committed): `data/cls_pv_thermal_v1_adaptive`,
  `data/cls_pv_nonpv_v3c_v42_cascade_adaptive`, `data/cls_pv_thermal_v2_adaptive`.
- Eval: `results/analysis/chip_upgrade_ab/{per_run_summary.csv,per_grid.csv,cov50/}`.
- Runs registered: `cape_town/model_runs/unifiedA_li_clsonly_v2_{bypass,all}`.

## Domain boundary (unchanged)
CT aerial_2025 in-domain only. JHB Vexcel / GEID keep the Gemini cross-domain FP
reviewer. The large-clipping case (pixel-or merge, industrial roofs) barely
triggers on CT per-det; re-validate the classify-all large band on JHB CBD
25-grid before extending config (c) cross-domain.

## Follow-ups
- ✅ **DONE 2026-06-08: promoted `unifiedA_li_clsonly_v2_all` to the locked
  CT-census post-proc baseline** (supersedes fixed-400 `unifiedA_li_clsonly`).
  Recorded in `configs/datasets/regions.yaml` (both model_run notes updated),
  `results/analysis/gemini_fpcut_li/PATH1_vs_PATH2_decision.md`, and memory.
- (optional) Retrain convnext_tiny + efficientnet_b0 on adaptive chips for the
  backbone ablation (dinov2 already decisive; not needed for the decision).
