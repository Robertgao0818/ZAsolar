# RUN: Durban city FP-rate correction + frozen estimation sample + blind labelling page (2026-10-12)

Local on rog, $0 (Vexcel academic extracts only). Built by grok build (grok-4.7-build-fast, reasoning high); orchestrator re-derived the FP numbers independently (identical).

## 1. FP-rate correction
`fpflag_v1/summary_fpflag_v1.md` (line 38) reported city FP area 333,700 m2 [246k-442k] = 17.7% of census area. That FP area is the val-150 frame only; the frame excludes the 200 pilot blocks (frame area 1,730,761 m2 of 1,888,277 m2). Corrected: FP_city = design-weighted FP_frame (same estimator + stratified bootstrap, B=2000, seed 7; UNS excluded, PART = true panel) + labelled FP area of the pilot-200 census (14,926 m2), over the full v1 census area.

| variant | FP city m2 (95% CI) | city rate (95% CI) |
|---|---|---|
| main (labels as-is) | 348,619 [261,205-456,519] | **18.5% [13.8-24.2]** |
| + 葉's anchor corrections | 355,617 [268,951-467,149] | 18.8% [14.2-24.7] |
| UNS as FP | 435,451 [298,103-564,368] | 23.1% [15.8-29.9] |
| pilot PART extension as FP | 352,528 [265,114-460,428] | 18.7% [14.0-24.4] |

Frame-only rate 19.3% [14.2-25.5]. Note: `~/zasolar_data/durban_stitch_sam3_20261009/fpflag_v1/CORRECTION_fp_rate_20261012.md` (summary not rewritten); script/json `estset_v1/fp_rate_correction_v2.{py,json}`.

## 2. Frozen held-out estimation sample (estset_v1)
Population holefill_v3 (30,604 blocks); strata per the sealed pilot rules (XL side>75 m; L >=1000 m2; M 100-1000; S 5-100). Frame excludes the 350 labelled blocks (pilot 200 + val 150; the 4 anchor blocks are among them). Seed 20261012. score_max used only to split M (fp_flag_score), never shown.

| cell | frame N | frame area m2 | n | pi |
|---|---|---|---|---|
| XL (all) | 68 | 162,204 | 68 | 1 |
| L (all) | 56 | 80,162 | 56 | 1 |
| M unflagged | 1,985 | 607,572 | 100 | 0.050 |
| M flagged (score<0.95) | 762 | 186,239 | 60 | 0.079 |
| S [5,25) | 19,842 | 264,096 | 20 | 0.0010 |
| S [25,50) | 5,771 | 197,648 | 20 | 0.0035 |
| S [50,100) | 1,770 | 119,879 | 20 | 0.0113 |
| total | 30,254 | | **344** | |

One block changes stratum v1->v3 (g0.25_002420 S->M, not drawn). Manifest `estset_v1/estset_manifest_v1.csv` (pi, weights), `estset_frame_cells_v1.csv`, `estset_config_v1.json` (sha256 seal af55de23...). Rule: freeze any FP rule before looking at these labels; do not train/tune on them.

## 3. Imagery + blind page
Per block one Vexcel extract, block bbox + 40 m margin: 309 downloaded (355 MB, no errors), 35 from cache/local tiles; 344 tifs, 0.8 GB. Page `~/zasolar_data/durban_estset_review_20261012/` served by `server.py` on port 8766 (tmux `estset-review`), Tailscale http://100.111.51.77:8766/ (same mechanism as the hole-review page on :8765). Same label set as labels v1/v3 (TP / FP / UNS / PART, optional fp_type from earlier files, notes); keys 1/2/3/4, arrows, O raw, Z zoom. Blind: items.json holds only anonymous est_id + image paths; no score/frac50/SAM3/Gemini/prior labels; randomized order. Labels autosave to `estset_labels_20261012.{json,csv}` (+ audit jsonl), resume-safe. Self-test: 2 dummy labels posted, survived a server restart, then cleared.
