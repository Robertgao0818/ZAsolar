# JHB Phase A retrain — boundary-aware mask supervision

**Status**: design + tooling staged 2026-05-08. train.py changes pending.
**Origin**: train20_val5_hn (2026-05-08) failed because clean_gt SAM-supp
boundary was treated as strict mask BCE target, reinforcing V3-C halo.
Codex review identified the root cause as a *biased pseudo-mask feedback
loop*: review accepted predictions provide existence labels, not boundary
labels, but training code consumed the dense pixel boundary as supervision.

## Goal

Break the feedback loop on the JHB CBD 25 Vexcel grid pool. CT data fully
withheld from training in this phase.

## Data pool — clean_gt with `source`-aware mask supervision

**Correction (after RA review of overlap semantics)**: V3C_correct often
overlaps sam_added because RA used SAM as a re-annotation tool to fill in
panels V3C had cut too small — they are the same install, V3C providing the
core and SAM the missing portion. Treating them as separate instances would
duplicate boxes; dedup-by-drop would lose the V3C portion. The correct
representation is a **union of overlapping parts = one install = one polygon**
— which is exactly what `build_clean_gt_jhb_cbd25.py` produced via
`dissolve_cluster(min_overlap_m2=0.01)`.

So the train target is `data/annotations_channel2_clean/<G>/<G>_clean_gt.gpkg`
(same file as evaluation GT — eliminates train/eval skew). Mask supervision
is differentiated by the dissolved polygon's `source` provenance string:

| `source` (in clean_gt)      | mask_weight | boundary_ignore_px |
|-----------------------------|------------:|-------------------:|
| `SAM_supp+V3C_TP`           | 1.0         | 2 (<600 m²) / 3 (≥600 m²) |
| `SAM_supp` only             | 1.0         | 2 / 3              |
| `V3C_TP` only               | 0.0         | — (no mask BCE; halo not learned) |
| any with `Li_marked`        | drop        | — (5 polygons in micro 3 grids) |

V3C_edit + V3C_delete were already excluded by `build_clean_gt`'s composition
rule; they don't enter clean_gt.

Train grids: `G0772/G0773/G0774/G0775/G0814/G0815/G0818/G0853-G0857/G0888-G0892/
G0922-G0926/G0926` (drops `G0776/G0891` for skylight-FP).
Val grids: `G0816/G0817/G0925`.

### Actual chip + per-instance counts after build

`JHBPhaseADataset` builds these (chip 400×400, overlap 0.25):

| split | n_chips | pos | neg | sam-inst | v3c-only-inst |
|---|---:|---:|---:|---:|---:|
| train | 3059 | 2660 | 399 | 4124 | 1174 |
| val   | 6336 | 351  | 5985| 831  | 137  |

Train per-instance: 78% sam-supervised / 22% v3c-only-fallback. Each polygon
appears in ≈ 2.9 chips due to overlap. Negative chips balanced to 15% in
train; val keeps deployment-time distribution (lots of negatives).

The 5 `Li_marked` polygons (G0774×2, G0816×2, G0922×1) are dropped at
load time by `_classify_source`; eval still treats them as part of GT
because we read the same file but downstream eval scripts don't filter.
**TODO**: separately, the build_clean_gt script should be patched to not
include Li_marked in clean_gt at all (orthogonal cleanup, see prior
discussion).

## Mask supervision rationale

`results/analysis/jhb_phaseA_prep/wobble_bucket_summary.csv`:

- Edge wobble (perimeter / DP-simplified-perimeter @ tol=1m) is ~1.21 for
  both SAM and V3C across all area buckets. Wobble is a generic
  raster→polygon baseline, not SAM-specific.
- Therefore boundary_ignore band of 2 px (≈ 13.4 cm at Vexcel 6.7 cm GSD)
  fully absorbs SAM's edge ragging.
- ≥600 m² polygons have wobble 1.37 (SAM) and step-structure (verified by
  user) — band widened to 3 px to also absorb step transitions.
- V3C halo is systemic outward bias (bulk_ratio 1.019→1.408 in
  train20_val5_hn). Wobble metric does NOT detect halo; halo can only be
  ignored by mask_weight=0 because a wide enough band (5 px) to cover halo
  would erase small panels entirely (a 10 m² panel is ~47 px on a side).
- Where V3C overlaps SAM in the union polygon, SAM-corrected boundary is
  the geometry that goes into mask BCE — V3C halo doesn't enter even when
  V3C contributed the box detection signal.

## Tooling staged

`scripts/training/jhb_phaseA/`:

- `count_raw_parts.py` — inventory CSV (run; output committed).
- `boundary_ignore.py` — `rasterize_polygon_with_ignore(pts, h, w, band_px)`
  returns (fg_mask, ignore_mask). Includes CLI for visual dry-run on real
  Vexcel chips.
- `wobble_audit.py` — 25-grid SAM vs V3C edge-wobble comparison (run;
  output committed).
- `test_boundary_loss.py` — 3 smoke tests for the patched mask loss
  (vanilla equivalence / full-ignore / zero-weight). All pass.
- `test_e2e_forward.py` — full forward+backward through Mask R-CNN with
  patch installed + dataset class. Confirms `mask_weight=0` instances
  contribute 0 to `loss_mask` (V3-C halo insulation works).

`core/training/`:

- `boundary_aware_mask.py` — monkey-patch for
  `torchvision.models.detection.roi_heads.maskrcnn_loss` adding per-pixel
  ignore band + per-instance mask weight. Reads supervision tensors from
  module-level batch state stashed by training-loop pre-hook.
- `jhb_phaseA_dataset.py` — `JHBRawPartsDataset` chip-scanner over raw
  reviewed.gpkg + sam_added.gpkg. Per-chip dedup: V3C_correct is dropped
  when it overlaps any sam_added by IoU > 0.3 (SAM provides better
  geometry). V3C_edit / V3C_delete dropped at this layer.
- `jhb_phaseA_transforms.py` — `BoundaryAwareTrainTransforms` mirrors
  train.py's transforms but synchronises `ignore_masks` and filters
  `mask_weights` alongside `masks` for every spatial op.

`configs/datasets/jhb_phaseA.yaml` — declarative spec (grids, supervision
schedule, chip params). Loaded by the dataset class.

`train.py` — added `--jhb-phaseA-spec PATH` flag; when set, swaps in the
new dataset + transforms, calls `install_patch()`, and registers a
`forward_pre_hook` on the model to stash supervision per batch.

## Actual dataset stats (full 23 grid build)

| split | n_chips | pos | neg | sam_added inst | v3c_correct inst |
|---|---:|---:|---:|---:|---:|
| train | 3059 | 2660 | 399 | 4138 | 1255 |
| val   | 6912 |  351 | 6561 | 845 | 152 |

Each polygon appears in ≈ 2.9 chips (chip overlap=0.25). The val set has
6561 negative chips because val grids were not balanced (only train is —
val should reflect deployment-time distribution).

**Unexpected finding**: 90.5% of V3C_correct polygons on G0922 overlap a
sam_added polygon by IoU > 0.3. This contradicts the spec narrative that
sam_added contains only FN polygons. In practice RA used SAM as a
re-annotation tool over many V3C predictions, not just FNs. The dedup
rule (drop V3C_correct when overlapping sam_added) handles this cleanly:
- Where SAM exists → SAM geometry drives box+cls+mask supervision
- Where SAM doesn't exist → V3C_correct drives box+cls (mask_weight=0)

Net effect: V3-C halo never enters mask BCE *and* SAM-refined geometry
takes over for the bulk of supervision. This is more robust than the
original "SAM = FN only" assumption would have allowed.

## train.py changes (DONE)

1. **Dataset class**: replace `CocoSolarDataset` consumption of polygon
   geometry with our raw-parts source. Inject `mask_weight` and
   `boundary_ignore_px` from a per-instance lookup table. Output target
   gains:
   - `target["masks"]` — uint8 (N, H, W) — fg as before
   - `target["ignore_masks"]` — uint8 (N, H, W) — boundary band, 1=ignore
   - `target["mask_weights"]` — float (N,)
2. **Augmentation**: `TrainTransforms` must transform `ignore_masks` in
   lockstep with `masks` (flip, rot, scale).
3. **Mask BCE patch**: monkey-patch
   `torchvision.models.detection.roi_heads.maskrcnn_loss` to:
   - RoIAlign `ignore_masks` alongside `gt_masks` using the same
     `mask_matched_idxs`.
   - `loss = (BCE(pred, fg) * (1 - ignore_resized) * mask_weight).mean()`
   - Skip mask loss entirely for instances with `mask_weight == 0`.
4. **Stage flags** (Phase B prep, not used in A):
   `--freeze-mask-head` / `--freeze-det-head`.
5. **Warm-start**: `--init-from checkpoints/exp003_C_targeted_hn/best_model.pth`.
6. **Data spec**: `configs/datasets/jhb_phaseA.yaml` — declares train/val
   grids, per-source paths, mask supervision schedule. NOT a COCO export
   — train.py reads gpkgs directly via the new dataset class.

## Evaluation

Headline metrics on val grids (G0816/G0817/G0925):
- `compute_ch2_recall.py` against `<G>_clean_gt.gpkg`
- `area_aggregate_eval.py` (Ch3) — bulk_ratio + area_F1
- v4_canonical pixel-or post-proc → 5-grid head-to-head vs `exp003_C_targeted_hn`

Chip-level F1@0.85 only as secondary, with the same boundary_ignore_px
applied to the eval GT to match training semantics.

Pass criteria for Phase A vs V3-C raw baseline:
- bulk_ratio: 1.019 ± 0.05 (must NOT exceed 1.10 — overshoot would imply
  the boundary-ignore band failed to suppress halo)
- Ch2 recall@0.3: ≥ 0.443 (V3-C raw); target ≥ 0.50
- area_F1 (Ch3): ≥ V3-C+SAM(mask+box) baseline

## Phase B (later, requires RA work)

Clean-boundary anchor set ~240 polygons across 6 archetypes (small
residential / trapezoidal multi-step / dense rooftop / shadowed /
V3C-halo-heavy / SAM-wobble-heavy). RA hand-edits in QGIS; result feeds
`mask_weight=2.0, boundary_ignore_px=0` as the strong-supervision anchor
for the mask head. Not blocking Phase A.
