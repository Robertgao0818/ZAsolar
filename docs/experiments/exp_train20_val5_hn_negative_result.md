# Experiment: train20_val5_hn (V3-C continued on Ch2 clean GT) — negative result

**Date**: 2026-05-08
**Status**: Decided — do not promote
**Owner**: gaosh
**Model**: `checkpoints/train20_val5_hn_20260508_v3c/best_model.pth` (V3-C init, batch 16, stage1 3 ep + stage2 9 ep on RunPod RTX 4090)
**Training spec**: [`configs/datasets/train20_val5.yaml`](../../configs/datasets/train20_val5.yaml)
**Validation harness**: [`scripts/analysis/validate_checkpoint.py`](../../scripts/analysis/validate_checkpoint.py)
**Eval artifacts**: `results/validation/train20_val5_hn_20260508_v3c_eval/` + `results/validation/v3c_5grid_pixelor_2026-05-08_eval/`
**Related**: [`exp_finalizer_pixel_or_vs_per_detection.md`](exp_finalizer_pixel_or_vs_per_detection.md), [`docs/validation_strategy.md`](../validation_strategy.md), memory `project_channel2_clean_gt_25grid.md`

This is the first end-to-end run of the V1.4 four-channel validation harness on a freshly trained checkpoint. Result: **net loss across all four channels apples-to-apples vs V3-C** (the same model used as the pretrained init).

---

## TL;DR

| metric (val 5 grids, v4_canonical pixel-or, post_conf=0.85) | V3-C @ 5grid | new model @ 5grid | Δ |
|---|---:|---:|---:|
| Ch2 raw recall@0.3 | **0.443** | 0.321 | **−12.2pp** |
| Ch2 SAM+v4_agg recall@0.3 | **0.439** | 0.318 | **−12.1pp** |
| Ch3 raw bulk_pred/gt ratio | **1.019** (+1.9%) | 1.512 (+51.2%) | **+49.3pp over** |
| Ch3 SAM+v4_agg bulk ratio | **0.929** (−7.1%) | 1.408 (+40.8%) | **+47.9pp over** |
| CT G2030 polygon F1 | **0.496** (P 0.354 / R 0.829) | 0.404 (P 0.333 / R 0.514) | **−9.2pp** |
| CT G1971 polygon F1 | **0.866** (P 0.792 / R 0.955) | 0.832 (P 0.798 / R 0.870) | **−3.4pp** |
| Plausibility flags | 0 / 0 | 0 / 0 | tie |

V3-C remains the production model. The `train20_val5_hn` checkpoint must not be promoted.

---

## Why we ran this

The V1.4 pivot reframed the inventory metric to grid-level area aggregation (`docs/validation_strategy.md`). Two observations motivated extending V3-C with the JHB CBD Vexcel 2024 25-grid Ch2 clean GT:

1. V3-C area_F1 on JHB CBD Vexcel 2024 trailed pixel-level set-theoretic IoU/F1 expectations on grids with multi-array commercial roofs (memory `project_g0925_failure_modes.md`).
2. The Ch2 clean GT (`data/annotations_channel2_clean/`, 25 grids, 2083 polygons) was newly built at sub-array granularity (SAM_supp补标 fills V3-C missed sub-arrays). This was hypothesised to give the detector enough recall signal on multi-array roofs to beat V3-C's per-roof envelope output.

Training set: 20 JHB CBD train grids (Vexcel 2024) + 18 CT large-array grids + 2 CT residential diversity grids (`configs/datasets/train20_val5.yaml`). Holdout: 3 JHB val (G0816 / G0817 / G0925, the cross-grid failure-mode anchors from memory `project_g0925_failure_modes.md`) + 2 CT val (G2030 + G1971, picked from V3-C-already-inferred set so the new run could be compared directly).

---

## Validation protocol

`scripts/analysis/validate_checkpoint.py` was run **twice** for the new checkpoint, both on the same val grids:

- **P2 (per-detection)**: `v4_canonical.json` without `merge_mode` pinned → finalize.py default = per-detection (changed in commit `76c6b342`).
- **P3 (pixel-or)**: `v4_canonical.json` with `"merge_mode": "pixel-or"` pinned (commit `38a45729`) → matches V1.4 baseline policy.

V3-C was then re-run **on the same 5 val grids with the same `v4_canonical.json` pixel-or pin** (`results/validation/v3c_5grid_pixelor_2026-05-08_eval/`) so the comparison is apples-to-apples — finalize merge_mode, post_conf, NMS IoU, GT files, evaluation profile all identical. Historical V3-C 25-grid baseline numbers (`results/analysis/area_aggregate_ch3_jhb_cbd25_v3c_sam_fixed/`) served only as a sanity check that the 5-grid V3-C re-run was in the right neighbourhood.

---

## Mode-bias trap (P2 → P3 reversal)

P2 looked like a clear win for the new model: Ch2 recall +13.2pp, G2030 F1 +2.7pp, plausibility flags down 56→0. This was wrong, and the trap is worth documenting.

P2's gap was driven by `merge_mode=per-detection`. The new model's raw output is **finer-grained** than V3-C's (more, smaller masks per multi-array roof). Per-detection finalize keeps each small mask as a distinct polygon. Clean GT is also sub-array-sized. So per-detection + new-model output happens to match the GT granularity, inflating Ch2 recall and CT G2030 polygon F1.

But the V3-C baseline numbers we compared against were from `v4_canonical.json` runs that historically defaulted to pixel-or. The 13.2pp recall gain was therefore **mode bias**, not a model improvement. The matching V3-C run on per-detection mode was never done; with that fair comparison, the new-model lift would shrink or vanish.

P3 corrected the mode mismatch by pinning pixel-or on both runs. Result reversed: new model under-performs V3-C by 9-12pp on Ch2 / G2030 F1.

---

## Apples-to-apples (P3) discussion

### Channel 2 — exhaustive recall

V3-C raw 0.443 vs new 0.321 = **−12.2pp**. Even after SAM mask+box refinement + v4_agg filter, the gap holds (0.439 vs 0.318). The new model's per-detection masks are too small to merge into installation-sized polygons under pixel-or, so they get filtered out by the area threshold (`min_object_area=5` is fine but post_conf=0.85 still drops detections whose individual mask scores are diffused across many small fragments).

### Channel 3 — area aggregate

V3-C raw bulk_ratio 1.019 (the model nearly perfectly matches GT total area on these 3 val grids) vs new 1.512 (+51% over-counting). After SAM+v4_agg filter: V3-C 0.929 (−7%) vs new 1.408 (+41%). The new model's pixel-or output overlap pattern produces **inflated total area** because dense multiple-detection clusters generate envelopes much larger than the underlying ground truth. The same effect was seen in G0817 envelope-group sweeps in the prior `exp_finalizer_pixel_or_vs_per_detection.md` ablation, but at a smaller scale.

### CT polygon F1

G2030: V3-C 0.496 → new 0.404 (−9.2pp). Recall drops from 0.829 → 0.514 — losing nearly half the GT installations. This is the most damaging single number, since CT residential test grids are the closest proxy for "does this still work as a general detector". The new model also has lower precision (0.354 → 0.333), so it isn't simply a precision/recall tradeoff.

G1971: V3-C 0.866 → new 0.832 (−3.4pp). Smaller drop because G1971 has 354 GT polygons (vs G2030's 35), so the absolute precision impact is averaged across more matches.

Note: the V3-C 5-grid CT G2030 F1 = 0.496 here is **lower** than the historical `presence_metrics.csv` value of 0.700 from the 2026-03-29 V3-C run. The difference comes from finalize pipeline changes (direct vs legacy detect_and_evaluate, parity mode, post_conf consistency); the 0.496 is the apples-to-apples reference, while 0.700 reflects different evaluation harnesses.

### Plausibility

Both V3-C @ 5 grid and the new model emit zero flags from `grid_plausibility.py` — the model's grid-level density / area-coverage / mean polygon area all stay within the per-stratum bounds. Plausibility is a guardrail, not a comparator, and on these 5 grids both models comfortably stay inside it.

---

## Root cause: SAM_supp granularity contamination

Source breakdown of the training GT (25-grid Ch2 clean_gt, before whole-grid splits):

| source | n polygons | total m² | mean m² | median m² |
| --- | ---: | ---: | ---: | ---: |
| SAM_supp+V3C_TP | 1673 | 108460 | **64.8** | **26.8** |
| V3C_TP | 405 | 44470 | 109.8 | 49.2 |
| Li_marked | 5 | 36 | 7.3 | 6.2 |

**80.3% of the training GT is `SAM_supp+V3C_TP` at sub-array granularity** (median 26.8 m² ≈ a 4×8 panel sub-array, not a 30+ panel residential installation). The labels were built by SAM-refining model proposals where V3-C had failed to detect, so they sit at the cluster / sub-array level, not the installation level (memory `feedback_sam_tool_ceiling.md` and `feedback_fragmented_label_semantics.md` flagged this).

Training a detector on 80% sub-array targets pushes the output distribution toward sub-array-sized masks — exactly what we observed. The model "learns" to fragment, then under pixel-or finalize the fragments either don't survive the per-component vectorize threshold or they merge into envelopes that exceed the GT's actual physical footprint.

The clean_gt was the right artifact for **evaluation** (Ch2 exhaustive recall) but the wrong artifact for **direct training supervision** without granularity correction.

---

## Decision

1. **Do not promote** `train20_val5_hn_20260508_v3c`. V3-C remains production for V1.4 inventory.
2. **Quarantine the checkpoint** under `checkpoints/train20_val5_hn_20260508_v3c/` (no symlink to `production`, no `regions.yaml` model_run entry).
3. **Pin `merge_mode: pixel-or` in `v4_canonical.json` going forward** (commit `38a45729`) so future P2-style mode-bias traps are avoided. Per-detection mode remains available via explicit `--merge-mode per-detection` for diagnostic Channel 2 sub-array recall, but not as the default.
4. **Re-run the finalizer ablation suite** (`docs/experiments/exp_finalizer_pixel_or_vs_per_detection.md`) on the next viable model — its conclusions are V3-C-specific and may shift if the next model has different mask-shape bias.

---

## Next experiment design hypotheses

The path forward is changing **what the model is trained against**, not the post-processing. Three candidates, ordered by complexity:

### Option A — exclude SAM_supp source from training target

Use only `V3C_TP + Li_marked` (410 / 2083 = 20% of clean GT). Trains the model on installation-level GT only.

Pros: minimal infrastructure work — `export_train20_val5.py` filters by source column on load.
Cons: 76% reduction in training signal; effectively trains the model on what V3-C already detects, so the lift over V3-C will be marginal at best. Likely a no-op fine-tune.

### ~~Option B — spatial-merge clean GT to installation-level before training~~ (REJECTED 2026-05-10)

Originally proposed: run a `merge_sub_arrays_to_installations.py` step that dissolves overlapping / adjacent SAM_supp + V3C_TP polygons within ~3-5 m buffer per cluster, train against the merged GT.

**Rejected.** Reasons:

1. SAM tool ceiling on large connected PV (`feedback_sam_tool_ceiling.md`): merged installation polygons inherit the halo / roof-swallow of the worst SAM-supp component, then pixel-BCE学进 mask head — same failure mode as `train20_val5_hn` but compounded by union envelope.
2. Two-Axis Model contract (`project_two_axis_model.md` + `feedback_volume_loss_avoid.md`): only A1 = T1 qualifies as gold mask supervision. A buffer-3m connected-component union of A2 SAM-supp + A2 reviewed_prediction is **not** A1 — buffer dissolve does not upgrade semantic conformance, it just hides the boundary noise inside a larger blob.
3. Phase A (`project_jhb_phaseA_failed.md`) already showed that fixing halo at training time via loss/postproc can't beat V3-C raw when GT边界 noisy. Pre-training spatial-merge concentrates the noise instead of removing it.
4. Literature record (`docs/literature/2026-05-09-training-supervision-literature-record.md` §3.7) explicitly rejects pre-training sub-array → installation-blob synthesis as a wrong direction.

The right path is supervision layering — see `docs/plans/2026-05-09-training-supervision-layering.md` (freeze-mask-head, source-aware loss weighting, accumulation principle, area-adaptive boundary ignore band, selective relabelling). `merge_sub_arrays_to_installations.py` was deleted on 2026-05-10 to prevent future agents from landing this option.

**Allowed exception**: GT-side spatial-merge of sibling polygons that are clearly the same installation split into adjacent fragments (`feedback_fragmented_label_semantics.md` `split_within_gt`). That is a cleanup of a single GT cluster, not a pre-training upgrade of all sub-arrays to installation blobs.

### Option C — per-source loss weighting at training time

Keep clean_gt as-is but weight the loss inversely to area within source bucket: `SAM_supp` contributions get smaller weight (or area-aware weighting). Modify `train.py` loss assembly.

Pros: zero data changes; keeps all 2083 polygons.
Cons: complex implementation and tuning; train.py loss path is not currently source-aware; risk of unintended changes affecting V3-C re-trains.

---

## Followups

- ~~(B) implementation: write `scripts/training/build_train20_val5_v2_installation_level.py` ...~~ — withdrawn 2026-05-10 alongside Option B rejection. The replacement plan is the supervision-layering action list in `docs/plans/2026-05-09-training-supervision-layering.md` (freeze-mask-head + source-aware loss weighting + accumulation + ignore band + selective relabelling).
- Add a `quality_tier` column to clean_gt build pipeline (`scripts/validation/build_clean_gt_jhb_cbd25.py`) so source-filter / merge logic can target T1/T2 distinctly per memory `project_two_axis_model.md`.
- This experiment's negative-result lesson — "evaluation GT and training GT are not the same artifact" — should land in `docs/validation_strategy.md` as a **note on training-target design** to avoid the next contributor re-running this same trap.
- Re-run `exp_finalizer_pixel_or_vs_per_detection.md` ablation on the next viable checkpoint; current conclusions are bound to V3-C output statistics.
