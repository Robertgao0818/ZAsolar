# Finalizer ablation: pixel-or vs per-detection (V1.4 calibration)

**Date**: 2026-05-08
**Status**: Decided
**Owner**: gaosh
**Inputs**: V3-C Vexcel 2024 inference, JHB CBD val grids G0817 / G0816 / G0925, evaluated against `data/annotations_channel2_clean/`.
**Code**: `scripts/analysis/{finalizer_envelope_group_sweep,finalizer_stage3b_sam,finalizer_stage3b_postnms,raw_hint_audit}.py` (commit `b5764b87`).
**Per-grid summaries already written**:
- `results/analysis/finalizer_mask_shaping_ablation/cross_grid_envelope_group_summary.md`
- `results/analysis/finalizer_mask_shaping_ablation/cross_grid_stage3b_summary.md`
- `results/analysis/finalizer_mask_shaping_ablation/raw_hint_audit/summary.md`
- `results/analysis/finalizer_mask_shaping_ablation/G0817/stage3_envelope_group_fill_summary.md`

This doc is the **decision synthesis** that routes those findings into the V1.4 finalizer config (`configs/postproc/v4_canonical.json`, `v4_agg.json`, `v4_high.json`) and into the Channel 2 polygon-diagnostic path.

---

## Why we ran this

V1.4 reframes the success metric from per-polygon F1 to **aggregate area at grid level** (see [`docs/validation_strategy.md`](../validation_strategy.md)). The legacy `finalize.py --merge-mode pixel-or` works well for area inventory but loses polygon-level matching on multi-array roofs where one big envelope swallows several distinct GT installations. Switching to `--merge-mode per-detection` (one polygon per raw detection) recovers polygon F1 — but at what area cost? G0817 alone showed the trade is real:

| variant on G0817 (clean GT) | area_F1 | poly_F1@0.3 |
| --- | ---: | ---: |
| `old_pixel_or` (production) | **0.717** | 0.646 |
| `instance_only` (per-detection) | 0.582 | **0.777** |
| `ref_sam_maskbox` (V4_agg policy) | 0.729 | 0.659 |
| `pixel_or_adapt_hyst` (mask hysteresis) | 0.629 | 0.524 |

Per-detection gives **+13.1pp polygon F1** but **-13.5pp area F1** on a single grid. V1.4's primary metric is area_F1, so the headline question for production is: *does the per-detection trade transport cross-grid, and is there a hybrid that gets polygon recall without giving up area?*

The ablation suite (`b5764b87`) ran four independent probes:
1. **Per-detection vs pixel-or** baseline on three val grids.
2. **Envelope-group fill** — replace mutual-IoU cluster groups with their pixel-or envelope when (env_area, group_density, n_clusters) trigger fires.
3. **Stage 3B (cluster → SAM)** — rebuild each cluster polygon via SAM2 mask+box prompts, with optional NMS.
4. **Raw hint audit** — for every clean-GT polygon, did the raw detector even propose a box, and does per-detection finalize actually recover GTs that pixel-or merges away?

---

## Cross-grid headline (area_F1)

| variant | G0817 | G0816 | G0925 | mean | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `old_pixel_or` | 0.717 | **0.751** | **0.539** | 0.669 | best on 2/3 grids |
| `ref_sam_maskbox` (current `v4_agg`) | **0.729** | 0.692 | 0.482 | 0.634 | best on G0817 |
| `mutual_iou_0.3` (cluster baseline) | 0.639 | 0.584 | 0.458 | 0.560 | |
| `instance_only` (per-detection) | 0.582 | 0.492 | 0.356 | 0.477 | -19pp mean vs old |
| `pixel_or_adapt_hyst` | 0.629 | (n.a.) | (n.a.) | — | hurts even on G0817 |
| `group_a100_d0.45_c3` (envelope-group fill, loose) | 0.742 | 0.629 | 0.506 | 0.626 | wins G0817 only |
| `group_a150_d0.50_c3` (conservative) | 0.728 | 0.626 | 0.504 | 0.619 | safer trigger, still loses |
| `stage3b_hybrid_all_nms0.5` | 0.679 | 0.656 | 0.494 | 0.610 | new poly-quality leader |
| `stage3b_hybrid_all` (no NMS) | 0.403 | 0.094 | 0.302 | — | over-segmentation collapse |

Take-aways:

1. **`old_pixel_or` is the cross-grid area_F1 leader** (2/3 grids), but partly by being geometrically over-permissive — high `area_recall` on SAM_supp comes with very low `median_iou` (envelopes much larger than GT polygons; G0816 median_iou = 0.082).
2. **`ref_sam_maskbox` is the most stable second**: within 1.4pp of leader on G0817 / G0925, 5.9pp below on G0816. As a no-trigger no-tuning baseline that doesn't over-fit any one grid, it is the right production area-inventory default.
3. **Per-detection is *not* a viable area-F1 path.** -19pp mean is too steep, even on grids where it gains polygon recall.
4. **Envelope-group fill does not transport.** The G0817 +2.5pp lift from `group_a100_d0.45_c3` reverses to **-12pp** on G0816 and **-3pp** on G0925. The trigger structure assumes V3-C-style fragmentation into ≥3 mutual-IoU clusters per envelope (G0817 has many; G0816/G0925 have few). Single-grid optimum did not survive cross-grid validation.
5. **Stage 3B + NMS** trails on area_F1 (5–9pp behind `old_pixel_or` cross-grid), so it does not displace `ref_sam_maskbox` for V1.4 inventory. But it has a separate use — see polygon-quality below.

---

## Polygon-quality channel (Stage 3B hybrid + NMS)

`SAM_supp+V3C_TP` `recall_iou05` (sub-array detection — Channel 2 GT category):

| variant | G0817 | G0816 | G0925 |
| --- | ---: | ---: | ---: |
| `old_pixel_or` | 0.622 | 0.255 | 0.110 |
| `ref_sam_maskbox` | 0.635 | 0.245 | 0.068 |
| **`stage3b_hybrid_all_nms0.5`** | **0.811** | **0.277** | **0.123** |

`SAM_supp+V3C_TP` `median_iou`:

| variant | G0817 | G0816 | G0925 |
| --- | ---: | ---: | ---: |
| `old_pixel_or` | 0.786 | 0.082 | 0.213 |
| `ref_sam_maskbox` | 0.925 | 0.045 | 0.086 |
| **`stage3b_hybrid_all_nms0.5`** | **0.943** | 0.064 | **0.265** |

Polygon `F1@0.3` against full clean GT:

| variant | G0817 | G0816 | G0925 |
| --- | ---: | ---: | ---: |
| `old_pixel_or` | 0.648 | 0.382 | 0.244 |
| `ref_sam_maskbox` | **0.661** | 0.382 | 0.193 |
| **`stage3b_hybrid_all_nms0.5`** | 0.626 | **0.390** | **0.314** |

Hybrid + NMS wins SAM_supp `recall_iou05` on **all three grids** and wins polygon F1 on G0816 / G0925. The cost is a 7–13pp drop in V3C_TP `recall_iou05` (NMS occasionally suppresses an envelope-spanning SAM polygon in favour of a high-score smaller-cluster polygon). For Channel 2 (sub-array recall against clean GT), this is the right trade.

Without NMS the hybrid path collapses (`area_F1` 0.094 on G0816): clusters that share an envelope each generate near-identical envelope-spanning SAM polygons, so `mean_n_pred_hits=22` and `pred_gt = 14.9×`. NMS at IoU 0.3 dedups those duplicates and lifts area_F1 by 27–58pp.

---

## Raw hint audit — does post-processing have signal?

For every clean-GT polygon, was there at least one raw detector proposal touching it (before any post-processing)?

| grid | n_GT (SAM_supp+V3C_TP) | `raw_box_hint_rate` | `raw_mask_hint_rate` | strict no-proposal |
| --- | ---: | ---: | ---: | ---: |
| G0816 | 94 | 1.000 | 1.000 | 0 |
| G0817 | 74 | 1.000 | 1.000 | 0 |
| G0925 | 73 | 0.973 | 0.945 | 2 (2.7%) |

Per-detection's `iou05` recall lift over `old_pixel_or` on SAM_supp:

| grid | old | per-detection | mutual-IoU 0.3 | gain (perdet over old) |
| --- | ---: | ---: | ---: | ---: |
| G0816 | 0.255 | 0.298 | 0.309 | +4.3pp |
| G0817 | 0.622 | **0.784** | 0.784 | +16.2pp |
| G0925 | 0.110 | 0.068 | 0.082 | -4.2pp |

Bucket: `mask_hint_old_fail_perdet_pass` (proposals existed; old pixel-or merged them away; per-detection recovers):

| grid | n | share of SAM_supp |
| --- | ---: | ---: |
| G0816 | 15 | 16.0% |
| G0817 | 20 | 27.0% |
| G0925 | 3 | 4.1% |

The audit answers a question that the cross-grid metrics blur: **post-processing has signal to extract** on G0816/G0817 (15–27% of SAM_supp polygons fail the old envelope but had per-detection-recoverable proposals). On G0925 the dominant failure is "no useful proposal in raw" (mask-undersizing of large installations + 2 polygons with no box hint at all), which post-processing alone cannot fix. G0925's recall is **detector-bounded**, consistent with `feedback_recall_recovery_constraints.md` (TTA is the only path).

This frames the production split: post-processing variants matter on G0816/G0817 but cannot lift G0925 ceiling.

---

## Production routing

Channel 3 (area-aggregate inventory, V1.4 main metric) and Channel 2 (polygon-level diagnostic) are **decoupled** in the finalizer. One pipeline run can emit both.

| Config | merge-mode | refinement | NMS | use case |
| --- | --- | --- | --- | --- |
| `v4_canonical.json` | pixel-or | none | finalize default | Internal default; legacy parity for `detect_direct + finalize` smoke |
| `v4_agg.json` | pixel-or → SAM mask+box | filter via `filter_sam_inventory` | spatial NMS in finalize | **V1.4 area inventory (Channel 3)** — `ref_sam_maskbox` on Vexcel 25-grid |
| `v4_high.json` | pixel-or → SAM mask+box, post_conf=0.85 | aggressive filter | NMS | High-precision seed for solar_backdating (Channel 1 gates this) |
| **(new)** `v4_poly_diag.json` | per-detection → mutual-IoU 0.3 → SAM hybrid | cluster mask + envelope bbox SAM | spatial NMS at IoU 0.5 | **Channel 2 polygon diagnostic** — sub-array detection / TP-lost gallery |

Recommendations:

1. **Keep `v4_agg.json` (`ref_sam_maskbox`) as the V1.4 inventory production config.** No change. Cross-grid mean area_F1 leader 0.634 (vs 0.669 for raw `old_pixel_or`), with the right `median_iou` on V3C_TP that pixel-or doesn't have.
2. **Do not promote any envelope-group-fill trigger.** It over-fits G0817; cross-grid mean area_F1 0.619 < 0.634 for `ref_sam_maskbox`.
3. **Add `v4_poly_diag.json`** that wires `stage3b_hybrid_all_nms0.5` for the Channel 2 polygon-diagnostic output. Use it for TP-lost gallery and SAM_supp recall reporting; do **not** route it to inventory CSVs.
4. **Do not adopt `pixel_or_adapt_hyst`.** It hurts area_F1 even on G0817 (-8.8pp). Adaptive mask threshold helps in narrower contexts (large connected installations); not safe as a global switch.
5. **G0925's residual failure** is detector-bounded (mask-undersizing on the long contiguous PV strip + 2.7% no-box-hint polygons). Tag that grid in `regions.yaml` strata as `recall_detector_bounded`; finalizer changes will not move the needle until TTA or a retrained model fixes the upstream proposal stage.

---

## What this means for the next training run

Currently training: `train20_val5_hn_20260508_v3c` (V3-C continued on JHB CBD 20-grid Vexcel + CT 20-grid). Validation harness `scripts/analysis/validate_checkpoint.py` runs the new model through the same `v4_agg.json` policy, so:

- If the new model fixes G0925 mask-undersizing, **`old_pixel_or` may regain area_F1 leadership on G0925** (currently bounded at 0.539) — recheck `ref_sam_maskbox` vs `old_pixel_or` after the new run.
- If the new model fragments fewer G0816/G0817 multi-array roofs, the per-detection trade narrows — possibly making `instance_only` viable as a Channel 2 default (low priority; hybrid+NMS still wins on G0816/G0925).
- **The ablation conclusions are V3-C–specific.** Re-run the same suite on the new checkpoint when validation lands, especially `cross_grid_envelope_group_summary.md` and the raw-hint audit. Provenance lives in `comparison_metrics.csv` per grid.

---

## Followups (not blocking)

- Wire `v4_poly_diag.json` into the validation harness (today's harness emits Ch2 + Ch3 from `v4_agg.json` only; polygon-diagnostic remains a manual rerun).
- Promote the `cross_grid_*_summary.md` tables into `docs/experiments/exp_validation_v1_4.md` Channel 2 / Channel 3 sections so the running log captures the finalizer state alongside the channel results.
- Cluster-level provenance: hybrid+NMS clusters carry `cluster_id` + parent-envelope id. Surface those columns in the GeoPackage so QGIS review can colour by cluster.
