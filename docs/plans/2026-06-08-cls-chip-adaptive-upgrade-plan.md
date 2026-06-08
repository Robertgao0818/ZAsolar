# solar_cls chip upgrade — unified adaptive bbox+margin (design B)

**Date:** 2026-06-08  **Status:** ✅ LANDED & VALIDATED 2026-06-08 —
`chip_spec_version=adaptive_v1` beats the locked fixed-400 baseline on every
metric (σ_Bw 0.260<0.297, nonpv_kill@0.95R 0.959>0.871, cov50 0.829>0.815).
Winner = config (c) classify-all. Full results:
[`2026-06-08-cls-chip-adaptive-upgrade-RESULTS.md`](2026-06-08-cls-chip-adaptive-upgrade-RESULTS.md).
**Owner repo:** `/home/gaosh/projects/solar_cls/` (mirrors into ZAsolar via shared venv).

## Decision
Replace solar_cls's fixed **400px centroid crop → resize 224** with a single
**unified adaptive bbox+margin → 224** chip (Option B). Two-path size routing
(Option A) was rejected: the small band is already ≤160px so an adaptive rule
*upsamples* it (achieving native-resolution preservation for free) and only
downsamples genuinely large detections (which have pixels to spare); a hard size
router adds an effective-GSD discontinuity + a second code path to mirror, for no
measurable gain. This is a **next-iteration** upgrade — the current CLS-only
(dinov2, 400px) is already the selected CT census baseline and must not be blocked.

## Why (data-grounded, aerial_2025 CT, n=3277 over 16 Li grids)
- **GSD ≈ 8.3 cm/px** (anisotropic 7.6×9.0). 400px window = 30×36 m ground; after
  400→224 resize, **effective GSD = 14.8 cm/px** (1.79× downsample).
- **Small-target resolution loss (the active problem):** a 16 m² panel ≈48px native
  → 27px after the squeeze; a 6 m² panel → ~20px (a smudge). 80.3% of detections
  are <30 m² and are all currently downsampled — exactly the hard
  "small dark PV ≡ water heater" aliasing band.
- **Large-band auto-bypass:** ≥30 m² (19.7%) currently BYPASS the classifier
  (`area_cutoff=30`, auto-kept PV) — that band is where large lookalike FPs live
  (bright industrial roofs, big skylights), never adjudicated today.
- **Large-PV clipping is latent on this run:** only ~0.1% (4/3277) exceed the 400px
  window (the per-detection finalizer caps single-detection extent ~400px). The
  clipping problem bites harder on **pixel-or merge mode** and **JHB Vexcel /
  industrial** scenes — validate the large-band win there too, not only CT.

## Chip spec (train + infer MUST be byte-identical)
```
1. bbox = detection geometry bounds in PIXEL space (src.index on 4 corners; NOT centroid)
2. side = max(w_px, h_px) * (1 + 2*MARGIN)        # MARGIN = 0.6 (roof context)
3. side = clamp(side, MIN_SIDE=96, MAX_SIDE=512)
4. center on bbox CENTER (not centroid — L-shaped array centroid can sit off-panel)
5. crop square `side`, edge-reflect/replicate pad at tile edges (NOT zero-pad)
6. resize side→224: INTER_CUBIC if upsampling (side<224), INTER_AREA if down (side>224)
   # keep 224 (multiple of 14 for dinov2 patch size)
```
- **MARGIN=0.6** keeps disambiguating roof context (pure tight bbox would strip the
  context that separates a dark panel from a water heater). Sweep {0.6,1.0,1.5} in
  the ablation if small-band kill regresses.
- **centroid → bbox-center** is also a free correctness fix (off-panel centroids).

## Adjacent decisions
- **Mask channel: NO (defer).** A 4th rasterized-shape channel forces dinov2
  patch-embed surgery and discards pretrained transfer (the source of the current
  kill=0.871). Adaptive framing already centers+scales on the detection, covering
  most of what a mask would add. Revisit only if error analysis shows
  multi-object-in-window confusion post-rebuild.
- **area_cutoff bypass: REPLACE with classify-all.** The bypass existed because the
  400px window clipped large panels; adaptive removes that reason. Classify all
  sizes; keep a tiny safety floor only for ext_px>MAX_SIDE (~0 here). **Validate
  large-band PV recall first** (ablation config c) before dropping the bypass in
  production — fall back to bypass-kept (config b) if large-band pvR<0.95.

## Change list (mirror-critical)
Three chip extractors must stay geometry-identical — **factor the crop into one
shared module imported by all** to prevent drift, + a golden-chip regression test
(fixed lon/lat/bbox → assert pixel-hash):
1. `solar_cls/scripts/classifier/classify_predictions.py` — rewrite
   `extract_detection_chips` (centroid+fixed window → adaptive bbox); relax the
   `area>=area_cutoff` bypass + `cls_score=1.0` large-default; direction-aware resize
   in `ChipDataset`.
2. `solar_cls/scripts/classifier/build_cls_dataset.py` — mirror in `extract_chip`;
   it currently reads only centroid_lon/lat — must carry bbox/geometry. Update
   manifest provenance (`chip_extraction_size`).
3. `solar_cls/scripts/analysis/label_cls_nonpv_subtype.py::extract_chip_with_bbox`
   (cascade chips feeding v2) — re-run with the new spec; v2 symlinks those PNGs.
4. `build_cls_dataset_v2.py` — re-link after #2/#3 re-extracted.
5. `configs/classifier/thresholds_v2.json` — **fully invalidated**; regenerate via
   `calibrate_v2_thresholds.py` after retrain. Add a `chip_spec_version` field;
   `classify_predictions.py` refuses to run on a mismatch.
6. `docs/experiments/cls_training_registry.json` — update build_parameters + new gen.

## Sequencing (~2-3 days, GPU for 3-5)
1. Shared adaptive-chip helper + wire into all 3 extractors + golden test. (~0.5d)
2. Rebuild training chips (v1 + cascade) → re-run build_cls_dataset_v2 relink. (~0.5d)
3. Retrain dinov2_vits14 head (+ convnext/efficientnet for ablation). (~hrs/backbone)
4. Re-calibrate per-layer thresholds → new thresholds_v2.json. (~1h)
5. Re-validate CLS-only on the 16 Li grids. (~1h)

## Validation plan (A/B, same 16 held-out Li grids)
- **Tier-0 gate:** nonpv_kill @ 0.95 PV recall per layer. Current aerial_2025 dinov2
  = **0.871**; new chip must hold pvR≥0.95 and not regress kill (thesis: kill↑ in
  small band + new measurable kill in the ≥30 m² band).
- **Tier-1 frontier:** CLS-only `area_aggregate_eval` vs current **cov50 0.805 /
  σ_Bw 0.277 / bulk 1.344** (calib-only) and the prod baseline **0.815 / 0.297 /
  1.427**. **σ_Bw + RMSE are the primary referees**; bulk ∈ [0.5,2.0] sanity gate.
  Win = σ_Bw↓ AND nonpv_kill@0.95R ≥ current AND bulk in-gate.
- **Ablation (isolate cause):** (a) fixed-400 baseline → (b) adaptive + bypass kept
  [resolution win] → (c) adaptive + classify-all [large-band win].
- If σ_Bw improvement is small-sample-noisy, add JHB CBD 25-grid CLS-only as a
  second surface (also exercises the large-clipping case).

## Risks
- Train/infer mirror drift (mitigate: shared fn + golden test + chip_spec_version).
- Calibration invalidation (mitigate: version field, refuse-on-mismatch).
- Context loss from tight crops (mitigate: MARGIN sweep).
- Bypass-removal could kill true large PV (gate: config c must pass large-band
  pvR≥0.95, else keep config b).
- dinov2 patch constraint: IMG_SIZE must stay a multiple of 14 (keep 224).

## Supersedes / relates
- Current CT census post-proc baseline = CLS-only @ prod (area≥30 ∨ dinov2 calib),
  cov50 0.815 / σ_Bw 0.297 / bulk 1.427 — see
  `results/analysis/gemini_fpcut_li/PATH1_vs_PATH2_decision.md` (final section).
- Domain boundary unchanged: this is CT aerial_2025 in-domain; JHB Vexcel / GEID
  keep the Gemini cross-domain FP reviewer.
