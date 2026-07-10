# Cross-Domain Generalization — `unified_reviewall_A` on 6 new Vexcel cities

**Date**: 2026-05-31
**Model**: `exp_unified_reviewall_A` (V3-C warm-start, `best_model.pth`, trained on JHB/CT only)
**Pipeline**: `detect_direct.py` → `finalize.py --merge-mode per-detection` (v4_canonical) → `sam_refine_maskbox.py --prompt-mode mask_box` (SAM2.1) → polygon-conf **c = 0.925** — identical to the JHB 382-grid production inventory (`unified_reviewall_A_perdet_sam_maskbox_vexcel_2024_full382`). **No PV/non-PV classifier attached** (pure detector cross-domain baseline).
**GT**: Li RA cross-domain annotations, `data/annotations/Vexcel/<city>/<GRID>.gpkg` — GeoSAM review-GUI exports, **sub-array / panel level (A2 / T2)**, EPSG:4326.
**Scope**: 60 grids (10 per city), 2 564 GT polygons, 10 confirmed-empty grids. Cities are all Vexcel 2025–2026 ortho (~6.9–7.4 cm GSD), unseen during training.

---

## TL;DR

Deployed as-is on 6 unseen South-African cities, the production detector **holds first-order aggregate inventory calibration** but **loses ~8× per-grid reliability** versus in-domain JHB.

- **Aggregate (50 non-empty grids)**: total predicted area 180 571 m² vs GT 173 384 m² → **bulk 1.04**, through-origin slope **1.09**, **R² 0.97**, **agg area-F1 0.763** (in-domain JHB: 0.839). The cross-grid pred-vs-GT regression is still strong → usable for **first-order census totals**.
- **Per-grid dispersion explodes**: **σ_Bw 0.157 → 1.228** (B-weighted ratio std), RMSE 916 → 1628 m². Reliability at the individual-grid level (what economic analysis needs) degrades sharply and unevenly.
- **Two distinct failure modes** (below): lookalike-FP **over-paint** on sparse low-density grids (durban, bloemfontein) and **recall collapse** on dense / fine-carved grids (pretoria).
- **2 of 6 cities generalize cleanly**: **pietermaritzburg** (agg-F1 0.872, σ_Bw 0.214) and **east_london** (σ_Bw 0.245) are near in-domain quality.
- **Empty-grid FP probe**: on the 10 confirmed-zero-PV grids the model is conservative — 2/10 perfectly clean, mean **3.3 small FP polygons / grid** (944 m² total), i.e. it does **not** hallucinate wholesale; FPs cluster where real PV is nearby.

**Feasibility verdict**: nationwide *aggregate* census is plausible with the current model; *grid-level economic-grade* reliability requires (a) attaching the existing `solar_cls` PV-vs-lookalike classifier to kill the over-paint, and (b) a recall fix / domain adaptation for dense roofs.

---

## Method

1. **Imagery**: 240 Vexcel ortho tiles (2×2 per grid) downloaded on-pod via `download_vexcel_jhb382.py` (layer `urban`, collection per `regions.yaml`) to `/workspace/tiles/vexcel/<city>/ortho`. 0 download failures.
2. **Inference** (RTX 5090): per-city via the JHB-382 overnight runner with env overrides (`--region <city> --imagery-layer <layer>`), tiles staged to `/dev/shm`, **PARALLEL=3** (PARALLEL=6 OOM'd on the larger tiles), detect batch 8, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. 60/60 grids, **0 inference failures**.
3. **Consumption filter**: `filter_sam_inventory.py --config configs/postproc/xdomain_c0925.json` (detector_confidence_min = 0.925) → `results/vexcel/<city>/unified_reviewall_A_perdet_sam_maskbox_xdomain_c0925/`.
4. **Evaluation**: `scripts/analysis/eval_xdomain60.py`, which reuses `area_aggregate_eval.summarize()` (canonical Tier-1 suite) verbatim. Metric CRS looked up per grid via `get_metric_crs` (PTA/BFN/ELS/GQB = 32735, DBN/PMB = 32736). `installation` semantics (no `legacy_instance` switch). Empty-GT grids are auto-excluded from area-aggregate (zero GT area) and scored separately as an FP probe.

> **GT-semantics caveat.** The in-domain JHB baseline below was computed against **installation-merged clean_gt**; this cross-domain eval uses **Li sub-array / panel-level GT (A2, possibly non-exhaustive)**. Pixel-union *area* is largely invariant to sub-array carving, so bulk / slope / σ_Bw remain comparable — this is exactly why area-aggregate is the cross-domain signal. **Polygon F1 is depressed by carving and is diagnostic only.** High per-grid over-paint ratios (below) may be partly Li under-annotation, not pure model FP — flagged for RA spot-check.

---

## Overall Tier-1 vs JHB in-domain

| metric | **xdomain (50 grids)** | JHB in-domain¹ (25 grids) | Δ |
|---|---|---|---|
| agg area-F1 | **0.763** | 0.839 | −0.076 |
| agg area-R / -P | 0.778 / 0.747 | 0.93 / 0.63² | — |
| mean per-grid F1 | 0.630 [0.556, 0.707] | — | — |
| bulk (pred/GT) | **1.042** | 0.970 | +0.072 |
| σ_Bw (B-weighted ratio std) | **1.228** | 0.157 | **+1.07 (≈8×)** |
| std log-ratio | 0.868 | — | — |
| RMSE (m²/grid) | 1628 | 916 | +712 |
| through-origin slope | 1.093 | 0.950 | +0.14 |
| OLS R² | 0.971 | — | — |
| grids within ±20 % | 0.48 | — | — |

¹ `unified_reviewall_A` per-det + SAM mask+box @ c=0.925, JHB CBD 25-grid clean_gt (`docs/experiments/2026-05-14-jhb-cbd25-3model-sam.md`).
² in-domain R/P from the unfiltered `_direct_sam_maskbox` row; the @c=0.925 operating point reports agg-F1 0.839.

**Read**: level (bulk, slope, R²) is preserved; **dispersion (σ_Bw, RMSE) is the casualty.** Per `feedback_tier1_metric_system`, σ_Bw + RMSE are the primary judges and bulk ∈ [0.5, 2.0] is a sanity gate — bulk passes, σ_Bw fails.

---

## Per-city Tier-1

Sorted by σ_Bw + RMSE (primary Tier-1 ranking):

| rank | city | n | agg-F1 | mean-pg-F1 | bulk | σ_Bw | RMSE | thru0-slope | R² | ±20 % |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **pietermaritzburg** | 8 | 0.872 | 0.844 | 1.068 | **0.214** | 466 | 1.049 | 0.982 | 0.75 |
| 2 | **east_london** | 9 | 0.725 | 0.799 | 0.970 | **0.245** | 481 | 0.952 | 1.000 | 0.78 |
| 3 | pretoria | 9 | 0.543 | 0.507 | 0.920 | 0.272 | 223 | 0.992 | 0.991 | 0.44 |
| 4 | gqeberha | 6 | 0.765 | 0.659 | 1.203 | 0.533 | 4330 | 1.251 | 0.998 | 0.17 |
| 5 | bloemfontein | 10 | 0.829 | 0.552 | 1.115 | 2.556 | 444 | 1.083 | 0.979 | 0.30 |
| 6 | durban | 8 | 0.757 | 0.442 | 0.876 | 3.837 | 1311 | 0.844 | 1.000 | 0.375 |

Per-city mean area-R / -P (recall vs precision split):

| city | mean area-R | mean area-P | dominant issue |
|---|---|---|---|
| pietermaritzburg | 0.916 | 0.801 | — (clean) |
| east_london | 0.860 | 0.765 | — (clean) |
| bloemfontein | 0.844 | 0.474 | over-paint (precision) |
| gqeberha | 0.829 | 0.603 | over-paint (precision) |
| durban | 0.619 | 0.434 | over-paint + recall |
| pretoria | 0.498 | 0.592 | recall collapse |

---

## Failure modes

### Mode 1 — lookalike-FP over-paint on sparse, low-density grids (precision collapse)

Grids where Li marked 1–4 panels but the detector fired 9–29 polygons. These dominate the σ_Bw blow-up in **durban** and **bloemfontein**:

| grid | city | n_pred | n_gt | pred m² | GT m² | ratio | area-P |
|---|---|---|---|---|---|---|---|
| BFN0126 | bloemfontein | 29 | 4 | 871 | 51 | **17.0×** | 0.06 |
| DBN0044 | durban | 9 | 2 | 234 | 18 | 12.7× | 0.07 |
| DBN0100 | durban | 13 | 2 | 322 | 33 | 9.8× | 0.02 |
| DBN0666 | durban | 13 | 2 | 261 | 40 | 6.5× | 0.04 |

These are precisely the residential lookalikes (skylights, bright roof facets, solar water heaters, roof textures) that the post-hoc **`solar_cls` PV-vs-lookalike classifier targets — and it was deliberately not attached this round** (pure-detector baseline). This is the single most addressable degradation. **Caveat**: on sparse grids Li may have under-annotated; some of these "FP" may be real PV, so the true precision floor is bracketed between this and the empty-grid probe.

### Mode 2 — recall collapse on dense / fine-carved grids

Grids where Li carved many small modules and the detector recovered a small fraction of the area. Drives **pretoria** down:

| grid | city | n_pred | n_gt | pred m² | GT m² | area-R |
|---|---|---|---|---|---|---|
| PTA0292 | pretoria | 11 | 287 | 68 | 631 | **0.06** |
| PTA0360 | pretoria | 16 | 147 | 122 | 443 | 0.13 |
| BFN0233 | bloemfontein | 13 | 2 | 117 | 171 | 0.04 (spatial miss) |
| PTA0322 | pretoria | 14 | 60 | 114 | 191 | 0.35 |

PTA0292 (287 GT modules ≈ 2.2 m²/poly → module-level carving) is the extreme: the installation-trained detector recovered 6 % of GT area. Needs visual confirmation that these are genuine misses vs. Li over-carving of a few installations.

---

## Empty-grid FP probe (10 confirmed-zero-PV grids)

Every prediction here is a false positive (cleanest precision signal — GT is confirmed zero):

| grid | city | FP polys | FP area m² |
|---|---|---|---|
| ELS0064 | east_london | **0** | 0 |
| PMB0226 | pietermaritzburg | **0** | 0 |
| PMB0042 | pietermaritzburg | 3 | 20 |
| GQB0204 | gqeberha | 4 | 21 |
| DBN0402 | durban | 4 | 37 |
| GQB0203 | gqeberha | 4 | 58 |
| GQB0327 | gqeberha | 5 | 90 |
| GQB0202 | gqeberha | 6 | 116 |
| DBN0403 | durban | 4 | 265 |
| PTA0738 | pretoria | 3 | 337 |

**Totals**: 33 FP polygons, 944 m², mean **3.3 polys / grid** (~94 m²/grid). 2/10 grids perfectly clean. The model does **not** hallucinate at scale on empty land — consistent with the over-paint in Mode 1 being localized to grids that already contain some PV, where lookalike structures cluster.

---

## Polygon F1 @ IoU 0.5 (diagnostic only)

Plain greedy 1-1 IoU matching (NOT installation-merge). **Expected-low** because GT is carved sub-array and predictions are installation-level.

- **Overall**: P 0.291, R 0.474, **F1 0.360** (TP 1216 / FP 2969 / FN 1348).
- Per city: pmb 0.45, els 0.45, bfn 0.58, gqb 0.24, durban 0.17, pretoria 0.20.

Do not over-read these — the carving mismatch alone caps polygon F1 well below the area metrics. They are reported for continuity, not for model selection.

---

## Recommendations

1. **Attach `solar_cls`** (PV vs non-PV) and re-score — Mode-1 over-paint (durban/bloemfontein, mean area-P 0.43–0.47) is the classifier's exact target; expect σ_Bw and bulk to tighten most there.
2. **RA spot-check the high-ratio sparse grids** (BFN0126, DBN0044/0100/0666) to partition Mode-1 area into genuine lookalike FP vs. Li under-annotation, calibrating the true cross-domain precision floor.
3. **Visual review of dense recall failures** (PTA0292, PTA0360) — confirm genuine miss vs. Li module-level over-carving; if genuine, this is a recall/domain-adaptation gap (consistent with `feedback_recall_recovery_constraints`: training-time fix only).
4. **First-order census is usable now** (overall bulk 1.04, slope 1.09, R² 0.97); two cities (pmb, els) are already grid-level usable. Prioritize the other four for classifier + recall work before economic-grade per-grid use.

---

## Reproducibility / artifacts

- Driver (pod): `/workspace/infer_xdomain.sh` (city loop over `scripts/runpod_vexcel_jhb382_overnight.sh`).
- Consumption filter config: `configs/postproc/xdomain_c0925.json`.
- Eval: `python scripts/analysis/eval_xdomain60.py --output-dir results/analysis/xdomain60`.
- Predictions (per city): `results/vexcel/<city>/unified_reviewall_A_perdet_sam_maskbox_xdomain_{sam_maskbox,c0925}/<grid>/predictions_metric.gpkg`.
- Metrics: `results/analysis/xdomain60/{per_grid,per_city_tier1,overall_tier1,empty_grid_fp}.csv`.
- GT manifest: `data/annotations/Vexcel/li_xdomain_manifest.csv`.
