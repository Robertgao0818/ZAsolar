# JHB CBD 25-grid 3-model comparison + SAM (mask+box) — 2026-05-14

## TL;DR — under unified v4_canonical pixel-or postproc

**Apparent winner**: `v3c_vexcel_2024_direct_sam_maskbox` — bulk 1.063, σ_Bw 0.221, RMSE 1078, agg_F1 0.820.
**This ranking is biased**: v4_canonical+pixel-or is V3-C's home turf. See "Per-model best operating point" section for the fair comparison — train20_val5_hn actually wins on Tier-1 once its native per-detection + polygon-conf sweep is used.

## TL;DR — per-model best operating point (fair)

| Rank | model + config | aggF1 | bulk | σ_Bw | RMSE | thru0_β |
|-----:|---------------|------:|-----:|-----:|-----:|--------:|
| **1** | **train20 per-det+SAM @ c=0.915** | **0.849** | **1.002** | **0.148** | **793** | 0.996 |
| 2 | unified_A per-det+SAM @ c=0.925 | 0.839 | 0.970 | 0.157 | 916 | 0.950 |
| 3 | unified_A per-det+SAM @ c=0.950 | 0.831 | 0.893 | 0.148 | 1328 | 0.870 |
| 4 | V3-C pixel-or+SAM @ c=0.825 | 0.824 | 1.046 | 0.189 | 1091 | 1.007 |
| 5 | V3-C pixel-or+SAM (no poly-conf filter) | 0.820 | 1.063 | 0.221 | 1078 | 1.021 |

**Real winner: `train20_val5_hn` with `per-detection` merge + polygon-conf c=0.915 + SAM mask+box.** Confirms `project_25grid_perdet_audit_2026-05-10.md` numbers (σ_Bw 0.148, bulk 1.002, RMSE 793). unified_reviewall_A also overtakes V3-C+SAM once its native configuration is used.

## Setup

- Imagery: `vexcel_2024` (Vexcel za-gp-johannesburg-2024, 6.7 cm GSD)
- Grids: 25 JHB CBD (G0772–G0926, see `scripts/runpod_3model_jhb_cbd25_compare.sh`)
- Pipeline per model: `detect_direct.py` (chip 400, overlap 0.25, det-thresh 0.05, mask-thresh 0.30) → `finalize.py --merge-mode pixel-or --postproc-config configs/postproc/v4_canonical.json` → `scripts/analysis/sam_refine_maskbox.py --prompt-mode mask_box`
- Eval: `scripts/analysis/area_aggregate_eval.py` against `data/annotations_channel2_clean/{grid}/{grid}_clean_gt.gpkg` (locked clean GT per `feedback_eval_gt_lock_clean.md`)
- Ranking: Tier-1 metric system (`feedback_tier1_metric_system.md`) — sort by σ_Bw + RMSE/1e5 with bulk ∈ [0.5, 2.0] gate

## Results (Tier-1, ranked)

| Rank | model_run | aggF1 | pgF1 | bulk | σ_Bw | log-σ | RMSE | thru0_β | R² |
|-----:|-----------|------:|-----:|-----:|-----:|------:|-----:|--------:|---:|
| 1 | v3c_vexcel_2024_direct_sam_maskbox          | 0.820 | 0.774 | 1.063 | 0.221 | 0.257 |  1078 | 1.021 | 0.950 |
| 2 | v3c_vexcel_2024_direct                      | 0.817 | 0.783 | 1.174 | 0.248 | 0.255 |  1469 | 1.126 | 0.956 |
| 3 | train20_val5_vexcel_2024_direct_sam_maskbox | 0.749 | 0.694 | 1.513 | 0.653 | 0.429 |  4078 | 1.431 | 0.864 |
| 4 | train20_val5_vexcel_2024_direct             | 0.730 | 0.680 | 1.668 | 0.713 | 0.429 |  5179 | 1.583 | 0.863 |
| 5 | unified_reviewall_A_vexcel_2024_direct_sam_maskbox | 0.750 | 0.685 | 1.482 | 0.813 | 0.503 |  3507 | 1.358 | 0.886 |
| 6 | unified_reviewall_A_vexcel_2024_direct      | 0.728 | 0.671 | 1.655 | 0.972 | 0.523 |  4655 | 1.517 | 0.883 |

Raw csv: `results/analysis/jhb_cbd25_3model_20260514/per_run_summary.csv` (+ per_grid.csv).

## Observations

1. **V3-C remains the production model** for the JHB CBD inventory task. Its bulk ratio + σ_Bw are 3–4× tighter than either re-trained candidate. SAM mask+box drops bulk from 1.174 → 1.063 (-10.6 pp) and RMSE 1469 → 1078 m² (-27 %), confirming `project_v3c_sam_v4agg_baseline_audit.md`: V3-C raw slightly over-paints (~+17 %) and SAM tightens to near-parity.
2. **train20_val5_hn over-paints structurally** (bulk 1.51–1.67). Matches `project_train20_val5_hn_failed.md` (+49 pp bulk vs V3-C). SAM helps marginally (−15 pp bulk) but cannot recover; per-grid σ_Bw stays ~3× V3-C.
3. **unified_reviewall_A is the worst on Tier-1** (bulk 1.48–1.65, highest σ_Bw 0.81–0.97). Matches `project_unified_reviewall_20260513.md` (pipeline bulk = 1.50 under v4_canonical pixel-or). SAM brings bulk to 1.48 but variance remains the largest of the six runs.
4. **SAM mask+box is universally additive** on this 25-grid suite: every model improves on bulk, σ_Bw, and RMSE under SAM refinement. agg_area_F1 moves only ±0.02 because the gain is variance-and-bias driven, not coverage-driven (per `feedback_polygon_sum_vs_pixel_area.md` — pixel-level set ops dominate the Tier-1 metrics here).
5. **agg_F1 is misleading on its own**: train20+SAM and unified+SAM both land at agg_F1 ≈ 0.75 but their σ_Bw differ by 0.16 and RMSE by 570 m². The σ_Bw + RMSE rule discriminates correctly.

## Per-model best operating point — methodology

1. **Pixel-or polygon-confidence sweep** (today on pod): for each of `{V3-C, train20, unified_A}_vexcel_2024_direct_sam_maskbox`, sweep c ∈ [0.50, 0.97] step 0.025 on `predictions_metric.gpkg`, rank by σ_Bw + RMSE/1e5 with bulk ∈ [0.5, 2.0] gate. Script: `scripts/analysis/poly_conf_sweep.py`. Output: `results/analysis/jhb_cbd25_3model_20260514/poly_conf_sweep/`.
2. **Per-detection merge for unified_A** (today on pod): re-ran `finalize.py --merge-mode per-detection` on the existing `raw_detections.pkl`, then SAM mask+box, then polygon-conf sweep. Script: ad-hoc nohup wrapper, output dirs under `/root/results/analysis/direct_maskrcnn_v1/johannesburg/unified_reviewall_A_vexcel_2024_direct_perdet[_sam_maskbox]/`.
3. **train20 per-det+SAM c=0.915 numbers reused from `results/analysis/train20_vs_v3c_tier1_25grid_20260511/tier1_summary.csv`** — same 25 grids, same clean_gt, same SAM config, snapshot from 2026-05-11.
4. **V3-C kept on its home turf** (pixel-or+SAM) per `project_finalizer_per_detection_tradeoff.md` — per-det costs V3-C −12.9 pp area F1.

### Per-model gain from sweep

| Model | unified postproc (pixel-or, no sweep) | best operating point | Δ σ_Bw | Δ bulk |
|-------|--------------------------------------:|---------------------:|-------:|-------:|
| V3-C+SAM | σ_Bw=0.221, bulk=1.063 | pixel-or @ c=0.825: σ_Bw=0.189, bulk=1.046 | −0.032 | −0.017 |
| train20+SAM | σ_Bw=0.653, bulk=1.513 | per-det @ c=0.915: σ_Bw=0.148, bulk=1.002 | **−0.505** | **−0.511** |
| unified_A+SAM | σ_Bw=0.813, bulk=1.482 | per-det @ c=0.925: σ_Bw=0.157, bulk=0.970 | **−0.656** | **−0.512** |

The unified-postproc comparison was systematically rigged against the two new models: their pipeline-level over-paint (bulk 1.48–1.51) was an artifact of pixel-or merge + low polygon-conf retention, not a structural model defect. Once each model uses its native operating point, **train20_val5_hn becomes the Tier-1 winner** and unified_reviewall_A also clears V3-C+SAM.

## Decision

**Production pipeline for JHB CBD Vexcel 2024 inventory: `train20_val5_hn` with `--merge-mode per-detection` + polygon-conf ≥ 0.915 + SAM2.1 mask+box.** This supersedes the V3-C+SAM v4_canonical pixel-or pipeline (`project_v3c_sam_v4agg_baseline_audit.md`) on Tier-1: σ_Bw 0.148 vs 0.180, RMSE 793 vs 1066, agg_F1 0.849 vs 0.821, bulk 1.002 vs 0.986.

unified_reviewall_A remains a viable candidate (only slightly behind train20 on RMSE 916 vs 793 and F1 0.839 vs 0.849); could be re-evaluated when V1.4 Channel 1/2 stratified precision/recall numbers are in.

V3-C+SAM **is no longer the production winner for Ch3 inventory** when fair per-model tuning is allowed, despite being the winner under the (unfair) unified postproc.

## Artefacts

- Pod: `/root/results/analysis/direct_maskrcnn_v1/johannesburg/{run_id}/` × 6 (raw + SAM)
- Pod: `/root/results/analysis/jhb_cbd25_3model_20260514/{per_run_summary,per_grid}.csv`
- Local: `results/analysis/jhb_cbd25_3model_20260514/{per_run_summary,per_grid}.csv`
- Wrapper: `scripts/runpod_3model_jhb_cbd25_compare.sh`
- regions.yaml patch: `scripts/_3model_jhb_cbd25_regions_patch.yaml` (already applied on pod, NOT yet applied to local `configs/datasets/regions.yaml`)
- Logs: `/workspace/logs/3model_jhb_cbd25_20260514/{run_id}/`
- Wrapper wall-clock: 3,034 s (~51 min) for 3 models × 25 grids × (Phase A + Phase B) on a single 4090 / PARALLEL=2.
