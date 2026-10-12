# Durban census: stitching v1, SAM3 box refinement, missing-piece recovery

Date: 2026-10-09 (NZDT). Work dir: `~/zasolar_data/durban_stitch_sam3_20261009/` (= `~/data/ZAsolar/...`).
Nothing in `gate_pilot_2026-09/durban_census_delivery_20261008/`, Dropbox, or any Cape Town output was changed. No RunPod used.
Coding agent: local grok build (`grok` 1.0.46, headless, `--always-approve`); specs in `prompts/A.md B.md B2.md C.md`.
The only repo change is one new untracked file, `scripts/postproc/stitch_census_v1.py`, plus this doc. Nothing was committed.

## Claude's diagnosis: what I checked

- Confirmed: 400 px chips, stride 300 (overlap 0.25), per-detection merge, `largest`, and v4_canonical tiers (0.85 below 100 m2, 0.65 at 100 m2, 0.70 at 200 m2). The gate skipped 88.24% of 2,212,555 chips. There are 65,040 raw polygons, the largest is 661.6 m2, and 14.08% have `n_components_dropped>0`. `reap_intermediates` deletes `raw_detections.pkl`.
- Confirmed exactly: the legacy IoU>0.1 rule gives 36,654 blocks, 1,927,374 m2 and 41,562 m2 of residual overlap. The raw union is 1,886,314 m2, so the delivered layer double-counts about 2.2%.
- Confirmed: on the delivered merged layer, 2,378 polygons are >50% inside another (29,937 m2). The v1 script prints 2,176 for its own legacy rebuild. That gap is not resolved yet.
- Correction: tile pixels are 0.0746 m in EPSG:3857, which is about 0.065 m on the ground at lat -29.8. So a chip is about 26 m on the ground, not 30 m (a 661 m2 maximum is consistent with this), and a 1008 px SAM3 window is about 65 m, not 75 m.
- Confirmed visually: tiles inside a cell do not overlap, and straight cuts at tile seams show in the overlays.
- Revised: the gate is not the main cause of pieces missing inside arrays (see C). The main causes are the 0.85 small-fragment bar and window overlap.
- Revised: multi_component=largest does drop components, but they recover only about 200 m2 on 5 cells.

## A: stitching v1, citywide (CPU, minutes)

Rule: union a pair if it overlaps or the gap is <= g. Then apply a fixed +0.15/-0.15 m mitre closing. Outputs are in `A_stitch/`: `stats.md`, `stats.json`, `durban_census_stitched_gap{0.25,1.0}_v1.gpkg`, `piece_to_block_gap*.parquet`.

| rule | blocks | area m2 | >=100 m2 | max m2 | residual overlap m2 | side>75 m (n / m2) | multi-cell |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy IoU>0.1 | 36,654 | 1,927,374 | 3,732 | 5,022 | 41,562 | 70 / 135,839 | 709 |
| gap 0 | 31,610 | 1,887,820 | 3,162 | 16,365 | 0 | 112 / 273,654 | 687 |
| gap 0.25 | 30,604 | 1,888,277 | 3,130 | 16,365 | 0 | 118 / 286,053 | 674 |
| gap 0.5 | 29,070 | 1,888,334 | 3,097 | 16,365 | 0 | 127 / 303,208 | 654 |
| gap 1.0 | 26,658 | 1,888,346 | 3,057 | 16,365 | 0 | 139 / 337,240 | 622 |

The gap changes the block count, not the area. Area sits at about the raw union for every gap.

## B: SAM3 box prompts (24 stitched blocks in the 5 C cells, local 4070)

Script `B_sam3/sam3_box_refine_v1.py`: `Sam3TrackerModel` with local weights, fp16, offline. Windows are 1008 px with stride 880; each polygon part gets one box. The bbox-fill pass uses one box per window. Runtime is about 2-5 s per block.
Results are in `B_sam3/summary.md`, `metrics.csv`, `refined_v1.gpkg` and `overlays/`.
- Area: 33,216 m2 became 30,448 m2 (-8.3%) under policy_sam, or 31,078 m2 (-6.4%) under policy_strict. Median kept change was -7.5% for L blocks, -12.8% for M and -13.6% for S.
- Agreement with the original: median IoU 0.92 (L), 0.81 (M), 0.84 (S). Boundary F at 0.3 m was 0.24 (L), 0.17 (M), 0.90 (S); boundaries move a lot.
- policy_sam rejected 3 blocks and flagged 4 for big change. One rejected block I looked at (000106) is a non-panel roof or shadow; the other two were not inspected.
- What works: SAM cuts car-park and shadow sprawl the census drew. On 000095 it removed 41%, which was correct, but the strict 0.6 rule reverted it, so that rule should be dropped. It also trims edges and walkways between sub-arrays.
- Per-part fill (>=8 m2, box iou_score>=0.8) found 983 m2. It catches real missed sections: 000716, about 130 m2 of missing array; 000729, two missing interior rows, although that fill also covers bare roof between the rows.
- The bbox-fill pass found 2,386 m2 and is not usable. On 028019 it paints the concrete margin strip, though it also picks up part of a neighbouring array.
- Open question: SAM masks leave out the gaps between modules and rows. Panel area versus installation footprint is the same unit question as A.

## C: recovering missing pieces (DBN0716, 0733, 0734, 0030, 0016)

Local tiles: 20 Vexcel extracts, 0.6 min, in `C_recall/tiles`. Detection was re-run locally with raw detections kept, in `C_recall/runs/{d25_gate,d25_nogate,d50_nogate}`.
GPU time per cell: about 80 s with the gate, about 133 s without the gate at overlap 0.25, and about 290 s at overlap 0.5 without the gate. All 15 runs took about 40 minutes.
F0 (the production recipe) matches the delivery on these cells: same polygon counts, union IoU 0.9997.
Script: `C_recall/recall_arms_v1.py`. Tables: `factor_table.md`, `metrics_by_cell.csv`, `replication.md`. Piece-level view: `new_pieces_v1.py`, `new_pieces_pooled.csv`, `new_pieces_near_arrays.gpkg`. Contact sheets: `sheets/`.

| factor | contrast | added m2 | in arrays (<=2 m) | elsewhere | lost m2 | new pieces near arrays | new pieces elsewhere | eyeballed real |
|---|---|---:|---:|---:|---:|---:|---:|---|
| gate off | F1 vs F0 | 5,347 | 84 | 5,263 | 9 | 1 | 24 | elsewhere ~5/20; rest are roads, roofs, cars |
| explode | F2 vs F0 | 200 | 67 | 133 | 4 | 1 | 4 | negligible |
| small-fragment bar 0.5, kept only if touching a >=100 m2 block | F3 vs F0 | 1,801 | 1,181 | 621 | 290 | 24 | 0 | ~13/20 real interior rows; rest are edge strips, roof, watermark |
| overlap 0.5 + centre-crop keep | F4 vs F1 | 12,378 | 5,057 | 7,321 | 8,045 | 9 | 38 | near ~3/9; elsewhere ~5/20 |
| centre-crop only | F4 vs F4b | 142 | 105 | 38 | 4,493 | - | - | crop removes seam double-count |
| all combined | F5 vs F0 | 22,737 | 6,298 | 16,439 | 5,197 | 43 | 68 | mostly rearrangement plus false positives |

Notes:
- The convex-hull "holes filled" column in factor_table.md overstates recovery, because the hulls include bare roof.
- Added area minus lost area is the honest net figure.
- Grok's `C_recall/overlays/*.png` read only one tile and are partly black. Use `sheets/` instead.
- The downloaded imagery carries a "Vexcel Imaging US, Inc." watermark at some tile edges, and the detector fires on it.

## Recommended order
1. Adopt stitching v1 citywide; it is done and the gpkg is written. Choose the gap with 葉. Re-stitch Cape Town the same way before comparing block counts between the two cities.
2. Add the F3 rescue as a post-processing step: on these 5 cells it adds about 1,100 m2 of in-array pieces, mostly real, with no new isolated detections. It needs raw detections, so the next citywide detect has to stop `reap_intermediates`, or the re-detect has to be done locally. Local re-detect with the gate costs about 80 s per cell, roughly 19 h for 846 cells on the 4070, or a pod.
3. Add a SAM3 boundary pass on stitched blocks: per-part boxes, policy_sam, a review flag when the change is over 25%, and per-part fill kept only as a candidate list. SAM3 must never delete on its own. Drop bbox-fill. Pilot on about 200 blocks before going citywide.
4. Keep the gate and overlap 0.25 for now. Overlap 0.5 recovers the most in-array area but adds a lot of false positives. Revisit it only together with a false-positive filter such as solar_cls.

## Decision needed from 葉
The census unit: "panel patch" (gap 0-0.25 m, 30,604 blocks) or "installation" (gap 1.0 m, 26,658 blocks). Area does not change with this choice.

## Decision 2026-10-09 (decided by 葉): census unit = one contiguous panel patch

Rule: union pieces that overlap or are within 0.25 m, then a +0.15/-0.15 m mitre closing (`stitch_census_v1.py` rule at gap 0.25).
Writer: `scripts/postproc/finalize_census_stitch_v1.py` (new). Every input column is carried from the highest-scoring member, which is the convention the IoU>0.1 merge used. `area_m2` is recomputed. New fields: `n_merged` (= `n_pieces`), `block_id`, `n_pieces`, `score_max`, `score_mean_area_weighted`, `cell_ids`, `n_cells`, `raw_area_sum_m2`, `max_side_m`, `n_parts`, `stitch_rule`.
Logs: `~/zasolar_data/durban_stitch_sam3_20261009/final_logs/`. Each output has a `.stats.json` beside it.

### Durban (from the 65,040 per-detection polygons)
Output: `~/zasolar_data/gate_pilot_2026-09/durban_census_delivery_20261008/durban_census_inventory_stitch025_v1.gpkg` (EPSG:32736).
This is a new file beside the delivery. The delivered files, `SHA256SUMS.txt`, README and Dropbox were not changed.
| layer | blocks | area m2 | pairwise overlap m2 | blocks >50% inside another |
|---|---:|---:|---:|---:|
| delivered merged_iou010 | 36,654 | 1,927,374 | 41,562 | 2,378 |
| stitch025_v1 | 30,604 | 1,888,277 | 0 | 0 |
These match experiment A exactly. The sum of `n_pieces` is 65,040. 118 blocks have a side longer than 75 m.

Equivalence check: stitching the delivered IoU-merged layer instead of the raw polygons gives the same 30,604 blocks, with area 1,887,986 m2 (-292 m2, -0.015%).
The components are provably identical: IoU gluing only joins overlapping pieces, and those are linked under the new rule too.
The small area difference comes from the closing step. It only runs on blocks with more than one input piece, so a group the IoU merge already fused is not closed again.
Check file: `~/zasolar_data/durban_stitch_sam3_20261009/A_stitch/check_stitch025_from_merged_iou010.gpkg`.

### Cape Town (from the delivered IoU-merged installations)
The per-detection layer (`ct_census_inventory_cpt.gpkg`, 170,605 rows) and the per-grid CPT outputs are not on this machine. They exist only on Dropbox: `RA_Solar/Gao/ct_census/` and `ZAsolar_archive/results.tar`. Dropbox was not touched.
The input was the local post-solar_cls merged table: `results/analysis/ct_census_output_table/ct_full_inventory_2026-06-21_merged.gpkg` (111,801 rows, EPSG:32734).
Output: `results/analysis/ct_census_output_table/ct_full_inventory_2026-06-21_stitch025_v1.gpkg`, plus `.stats.json`.
| layer | blocks | area m2 | pairwise overlap m2 | blocks >50% inside another |
|---|---:|---:|---:|---:|
| delivered merged (IoU>0.1) | 111,801 | 5,001,484 | 61,010 | 2,864 |
| stitch025_v1 | 96,489 | 4,945,796 | 0 | 0 |
The change is -13.7% in blocks and -1.1% in area. 290 blocks have a side longer than 75 m.
Caveats:
1. The input is the merged layer, so closing is under-applied by about 0.015% of area, the same as the Durban check.
2. In this file `n_pieces` counts old IoU-merged installations, not raw detections.
3. Cape Town polygons are after solar_cls; Durban polygons are before the classifier. Per-block counts and areas still compare different filters.
4. The JHB-schema columns, including `source_feature_id`, are carried from the best-confidence member, and the date block stays empty.
A re-detection is not needed. If one were ever wanted: 2,083 grids with no gate, at roughly 130-200 s per grid on the RTX 4070 (from the Durban no-gate rate), is about 75-115 GPU hours plus solar_cls. That also needs about 130-160 GB of tiles against 51 GB free on /home.

## Pilot 200: SAM3 boundary tightening (2026-10-09, approved by 葉, option 2 only)
Scope: no re-detection and no full-city run. Input: durban_census_inventory_stitch025_v1.gpkg (30,604 blocks). Local RTX 4070.

Method (B2_pilot200/sam3_box_refine_v2.py):
- One SAM3 box prompt per piece (part), using Sam3Tracker in fp16 with local weights.
- One Vexcel extract per block covering the block bounds + 8 m. This fixes the black-overlay bug: all 200 blocks have imagery cover 1.0.
- The candidate is the SAM mask, clipped to the prompt boxes and to the original polygon buffered by 0.5 m.
- Keep the original and flag the block when: mean SAM iou < 0.75 (low_sam_conf), the mask is empty, the candidate is < 0.25 × the original (collapse), or imagery cover < 0.9.
- A change of more than 25% is flagged as big_change; the refined polygon is still kept. There is no 60% fallback.
- Gap-fill goes only to a separate candidate layer.
- Sample: stratified, seed 20261009. XL (>75 m side) 30 of 118, L (>1000 m²) 50 of 136, M (100–1000 m²) 60 of 2,876, S (<100 m²) 60 of 27,474.

| stratum | n | sum orig → final m² | Δ sum | median Δ | median IoU | accepted | flagged | gap-fill m² |
|---|---|---|---|---|---|---|---|---|
| XL | 30 | 68,974 → 64,238 | −6.9% | −6.3% | 0.91 | 30 | 2 | 3,062 |
| L | 50 | 69,105 → 64,736 | −6.3% | −5.7% | 0.91 | 46 | 6 | 3,196 |
| M | 60 | 18,007 → 15,556 | −13.6% | −13.7% | 0.81 | 53 | 18 | 36.5 |
| S | 60 | 1,430 → 1,222 | −14.5% | −13.9% | 0.83 | 56 | 8 | 0 |

Flags: 34 blocks (17%). Reasons: big_change 31, low_sam_conf 15, collapse 1, empty 0, no_imagery 0 (a block can have several reasons). Failures: 0.

Runtime: SAM mean 0.7 s/block (median 0.4; XL 1.7, L 1.0, M 0.4, S 0.3). The Vexcel fetch takes 0.6–1.6 s/block. The 200 blocks took 409 s wall.
Full city (30,604 blocks): about 2.6 h of GPU time for SAM alone; about 8.2 h with serial fetch + SAM; about 3–4 h with a prefetch thread.

Visual QA (B2_pilot200/visual_qa_v2.md, 59 blocks): 34% better, 56% neutral, 10% worse.
- Unflagged: 15 better, 11 neutral, 3 worse out of 29. The worse cases are fragmentation along row gaps; no array is lost.
- Flagged: 5 better, 22 neutral, 3 worse out of 30. About 17 flagged blocks are census false positives (dark or corrugated roofs, an overpass, rail, plain roofs). SAM keeps them because it never deletes.
- low_sam_conf identifies false positives with a hit rate of about 12/15.

Outputs (~/zasolar_data/durban_stitch_sam3_20261009/B2_pilot200/):
- durban_sam3_refined_pilot200_v2.gpkg, with layers original / final / candidate / gapfill_candidates
- blocks_v2.csv, summary.md, summary.json, sample.csv, sample_config.json, sheets/, visual_qa_v2.md

Recommendation: a full-city run is technically ready, but it needs 葉's approval. Before running it:
1. Add a prefetch thread for the Vexcel fetch.
2. Drop specks < 0.5 m² and holes in the final layer.
3. Treat the low_sam_conf and big_change lists as an FP review queue rather than refinement QA.
4. Expect the city total to drop by roughly 7–14% per size class, and keep the original area alongside the refined area.

## Pilot 200 v3: text+box prompt arms and FP audit (2026-10-09, asked by 葉; no full-city run, no re-detection)

Same 200 blocks (seed 20261009), cached Vexcel extracts, local 4070. Outputs in `B2_pilot200/v3/` (arms) and `B2_pilot200/v3_fp/` (FP audit); v2 files untouched.

**How text and boxes are combined (TBX).** `Sam3Model` + `Sam3Processor` (PCS) on the v2 1008-px windows: `text="solar panel"` (TBX_pv: "photovoltaic panel") plus every piece box in the window as a positive geometric exemplar (`input_boxes`, `input_boxes_labels=1`). Instances with score > 0.3 are kept if >= 50% of their pixels lie inside a prompt box padded 0.5 m (or box IoU >= 0.3); the union is clipped to (prompt boxes) ∩ (original + 0.5 m), as in v2. TFB = text only, then the same box filter. T_solar = text only, everything in the block bbox. Text-arm low_conf = pixel-weighted instance score < 0.5. Same reject/flag rules as v2; SAM never deletes.

| arm (all 200) | med Δarea final | med IoU vs BOX | edge_mean / frac≤0.2 m (TP blocks) | TP blocks empty / <50% | low_conf | GPU s/block |
|---|---|---|---|---|---|---|
| ORIGINAL | 0 | 0.87 | 0.41 / 0.45 (all) | – | – | – |
| BOX (v2) | -10.4% | 1 | 0.879 / 0.905 | 0 / 1 | 15 | 0.70 |
| TBX_solar | -9.7% | 0.94 | 0.897 / 0.906 | 0 / 1 | 3 | 0.61 |
| TBX_pv | -9.8% | 0.94 | 0.908 / 0.911 | 0 / 1 | 4 | 0.59 |
| TFB_solar | -4.1% | 0.92 | 0.934 / 0.942 | 5 / 7 | 87 | 0.59 |
| T_solar | 0 | 0.78 | 0.922 / 0.924 | 5 / 6 | 90 | (reuses TFB) |

Visual QA TBX vs BOX, 27 blocks: 22% better, 74% neutral, 4% worse (`v3/visual_qa_v3.md`). TBX is marginally sharper (paired median +0.02 normalised gradient on TP blocks; 75 better vs 49 worse) and cleaner on large arrays, with no recall cost; every TBX <50% case is an FP or an FP extension. Text without boxes confirms the earlier failure: 5/150 true blocks return nothing. TBX presence is ~1.0 everywhere (the exemplar forces it), so TBX cannot flag FPs.

**FP audit** (`v3_fp/fp_analysis_v3.md`, `labels_v3.csv`): FP 30, PART 7, unsure 13, TP 150. FP rate XL 3%, L 8%, M 33%, S 8% (S 22% incl. unsure); city-weighted 10.7% of blocks (23% incl. unsure), ~17.7% of area (~0.33 km² of 1.89 km²). Mostly dark/corrugated/tile roofs; also white/skylight roofs, glass, an overpass, railway, drums.
**solar_cls not run:** production weights are missing on rog (broken symlinks to /home/gaosh; see solar_cls RUN-gate-limiters-cls-recheck-2026-10-02). Detector `score_max` is the strongest available signal (AUC 0.976): `score_max < 0.95` on blocks ≥100 m² removes 20/30 FP for 1/150 TP; < 0.97 removes 23 FP for 2 TP. Text-only presence (AUC 0.968) and v2 box low-conf add 1-4 S-block FPs at a 1-2 TP cost. Thresholds are tuned on the same 200 labels.

## FP stopgap flag v1 + solar_cls retrain estimate (2026-10-09, 葉: "do both")

Scope: flags only; no SAM full-city, no re-detection, no training, local 4070, $0.

### Part 1 — score-rule flag
- New file (source untouched): `~/zasolar_data/gate_pilot_2026-09/durban_census_delivery_20261008/durban_census_inventory_stitch025_fpflag_v1.gpkg` — adds `fp_flag_score` (area>=100 & score_max<0.95), `fp_flag_score_soft` (<0.97), `fp_flag_rule`. Not in the delivery's SHA256SUMS/manifest (new side file).
- City-wide: <0.95 flags 828 blocks / 211,809 m2 (2.7% / 11.2% of area): M 821 / 200,914; L 5 / 6,192; XL 2 / 4,703. <0.97: 926 / 238,410 m2.
- Blind validation, new sample (seed 20261011, disjoint from pilot 200, n=150: M 35f+35u, L 4f(all)+26u, XL 2f(all)+18u, S 30); labels TP 94 / PART 4 / FP 49 / UNS 3. Design-weighted, bootstrap 95% CI:
  - precision (<0.95): count 0.92 [0.83-1.00], area 0.95 [0.88-1.00]; (<0.97): 0.86 / 0.89.
  - recall of FPs >=100 m2 (<0.95): count 0.80 [0.65-0.99], area 0.71 [0.53-0.95]; of all FP area 0.58.
  - true panels wrongly flagged (<0.95): ~68 blocks / ~10,400 m2 [0-23,800]; (<0.97): ~126 / ~25,600 m2.
  - City FP area ~333,700 m2 [246k-442k] = 17.7% (pilot: 17.7%); after <0.95 flag ~140,300 m2 (~7%) remains, mostly small blocks and dark/corrugated roofs with score_max>0.95.
- Work folder: `~/zasolar_data/durban_stitch_sam3_20261009/fpflag_v1/` (`summary_fpflag_v1.md`, `fpflag_v1.py`, `validate_fpflag_v1.py`, `val_labels_v1.csv`, `validate_fpflag_v1.json`, sheets). Box sheet: `/workspace/durban_census/V1_fpflag_score_validation_flagged_true_and_missed_fp.jpg`.

### Part 2 — solar_cls status and retrain estimate (no training done)
- Surviving on rog: code `~/projects/solar_cls/scripts/classifier/` (train_cls.py, chip_extraction.py, build_cls_dataset*.py, rebuild_cascade_chips.py, calibrate_v2_thresholds.py, classify_predictions.py); configs `configs/classifier/dinov2_vits14.json` (3 frozen + 20 FT epochs, bs 32, AMP) and `thresholds_v2_adaptive.json`; label manifests in `data/` (v2_adaptive: 6,663 train + 1,906 val = 8,569 chips, CT v1 buckets + JHB GEID audit; JHB cascade manifest 3,913 rows; subtype labels 903 rows); registry `docs/experiments/cls_training_registry.json` (run completed 2026-06-08 on a RunPod 4090, best val bal-acc 0.961).
- Missing: all chip PNGs (0 in data dirs) and all weights (checkpoints/ symlinks -> nonexistent `/home/gaosh/zasolar_data/cls/checkpoints/`). Per 2026-10-02 RUN: not on HF `botao0818` (3 detector repos only), Dropbox `ZAsolar_archive/models/` detector only, `code_backup_2026-07-11` tar has 0-byte symlinks, Dropbox search was rate-limited (not exhaustive). Not re-scanned now (standing no-Dropbox rule). Unchecked: offline machines koko-82xm / MacBook, old RunPod volume.
- 0.7168 history: `thresholds_v2_adaptive.json`, aerial_2025 bucket calibrated on val n_pv 748 / n_nonpv 318 at target PV recall 0.95 (achieved 0.9505, non-PV kill 0.959). Valid only for the lost weights.
- Surviving old-model scores: `cls_score`/`cls_applied` in `ZAsolar/results/cape_town/unifiedA_li_clsonly_*` predictions (usable as a sanity reference for a retrained model).
- Key blocker beyond weights: the adaptive chip spec bypasses objects whose context crop exceeds 512 px (bbox x 2.2). At Durban GSD that is 93% of M, 100% of L/XL and 87% of the flagged blocks — the restored classifier as-is would not touch the medium/large FPs. A retrain must change the large-object chip rule (downscale full context to 224) for train and inference.
- Measured (synthetic random tensors, no data, nothing saved): dinov2_vits14 on the 4070 Laptop, AMP, bs 32: 413 img/s train, 1,217 img/s inference, 2 GB VRAM.
- Retrain plan (estimate; all local, $0 compute; Vexcel API quota/contract cost unknown):
  1. Chip spec v2 for large objects (downscale context >512 px to 224; keep adaptive for small) + Durban chip builder from per-block Vexcel extracts — 2-3 h code.
  2. Labels: reuse 350 Durban hand labels (pilot 200 + val 150) as the held-out TEST set only. New Durban train/calib labels ~2,000 blocks (~800 M/L/XL oversampling score_max<0.99, ~1,200 S oversampling score_max<0.95), labelled from contact sheets like this round (~150 blocks per 25 min) -> ~6 h, 葉 spot-check ~1 h. Optional: rebuild the 8,569 CT/JHB chips from the surviving manifests + surviving predictions (results/) — needs re-download of CT aerial_2025 for ~100 grids (~3-4 GB, ~1-3 h) and JHB GEID tiles (37 GB gehi cache on rog, coverage unverified) — +0.5-1 day; only 32% of JHB rows are >=100 m2 and those were bypassed in production, so value for Durban medium FPs is modest.
  3. Fetch Durban imagery: M/L/XL only 3,130 blocks ~50 min, ~1.3 GB; all 30,604 ~8 h at measured 0.93 s/block, ~12 GB (disk free now 50 GB).
  4. Train dinov2_vits14 3+20 epochs: ~10k chips -> ~10 min GPU at 413 img/s, ~20-30 min wall per run; 3 runs (seeds/variants) ~1.5 h.
  5. Calibrate new threshold per size class on a Durban calibration split (not the 350 test labels), target PV recall 0.95; 0.7168 is void.
  6. Evaluate on the 350 test labels vs the score rule (FP removed / TP removed, area-weighted); sanity check vs old cls_score on CT Li grids.
  7. Inference on 30,604 blocks: <1 min GPU at 1,217 img/s; write a new `_clsflag_v1.gpkg` flags file.
  Total: ~2 working days (labelling dominates), plus overnight fetch if S blocks are included. Cheaper alternative for >=100 m2: direct visual review of all 3,130 M/L/XL blocks (~8-9 h) covers 69% of census area without a model.

## Label audit v2 (2026-10-09 evening; 葉 corrections + Gemini/SAM3-text method)

- 葉 anchors: V1 sheet #1 g0.25_000715 = FP (translucent skylight sheets + shadow; I had PART), #2 g0.25_030281 = FP (see-through greenhouse/shade-house-type sheeting; I had TP), #3 g0.25_017294 = TP.
- Method per 葉: Gemini 3.5 flash-lite (local proxy, 12 concurrent, strict prompt p2) + SAM3 text-only 'solar panel' (cov<0.25 = FP) on all 350 audit blocks; agreement = label, 葉 anchors override, disagreements (41) keep old label and go to 葉. Folder `~/zasolar_data/durban_stitch_sam3_20261009/fpflag_v1/judge_v2/` (`summary_judge_v2.md`, `metrics_v2_table.csv`). New label files `fpflag_v1/val_labels_v2.csv`, `B2_pilot200/v3_fp/labels_v2.csv` (v1 files untouched).
- 26 labels changed: 6 toward FP, 20 toward TP (12 of my FPs that both models call PV: mostly dark/corrugated roofs = contested; conservative label set keeps them FP). Neither model catches 030281; Gemini p1 missed both anchors.
- Pooled 350, design-weighted (old -> v2 -> v2-conservative): city FP area 324.8k -> 283.3k -> 337.0k m2 (17.2% -> 15.0% -> 17.8%). Score<0.95 precision (area) 0.94 -> 0.93 -> 0.96; recall of >=100 m2 FP area 0.74 -> 0.76 -> 0.73; true panels wrongly flagged 12.1k -> 15.2k -> 7.6k m2.
- SAM3 text-only vs old labels (independent): precision 0.70 count / 0.88 area, recall 0.61 of >=100 m2 FP blocks; misses 7/255 old-TP blocks (block-level text-only miss ~3%). Combined (score<0.95 OR SAM cov<0.25, >=100 m2) vs old labels: recall of >=100 m2 FP area 0.86 (vs 0.74), true panels flagged 29k m2 (vs 12k).
- City-wide cost (local, $0 compute): imagery fetch 0.93 s/block (all 7.9 h / >=100 m2 0.8 h); SAM3 text 3.4 h / 0.5 h; Gemini 0.28 s/block at 12 concurrent (2.4 h / 15 min), ~2,640 input + 58 output tokens per block (81M / 8.3M input tokens) on the proxy quota. Not started.
- Box sheets: `/workspace/durban_census/R2_relabel_v1_to_v2_gemini_sam3text.jpg`, `D2_gemini_vs_sam3text_disagreements_for_review.jpg`, `F2_flagged_but_not_fp_v2_for_review.jpg`.
