# Durban FP-clean identity v2 (350 labelled blocks)

Date: 2026-10-11. Work dir: `~/zasolar_data/durban_stitch_sam3_20261009/fpclean_v2/`.
Nothing in `fpclean_v1/`, the census delivery, Dropbox, or any Cape Town output was changed. No RunPod. No paid API. The Gemini calls went to the local proxy on `127.0.0.1:8317` (`gemini-3.5-flash-lite`, 94 calls, 0 errors, about 412k prompt tokens and 14k completion tokens). No git commit.

This step only assigns an action in {keep, fp_candidate, review}. It does not edit polygons. `fp_candidate` is a flag.

## Rule
Nomination is the union of three signals already on disk. 94 of the 350 blocks were nominated.

- `nom_score` (66): inventory column `fp_flag_score`. Defined in `fpflag_v1/fpflag_v1.py` as area ≥ 100 m2 and `score_max < 0.95`. `fpc_common` reads the column. On these 350 it matches that formula with 0 mismatches.
- `nom_frac50` (75): `frac50 < 0.25` from `pregate_reverse_v1/blocks_scored_v1.csv`. Seven blocks are missing there, all `label_v1 = UNS` (they exist in `pregate_scores_v1.csv`; `analyse_v1.py` dropped non TP/FP before writing the scored file). Missing frac50 is not a nomination, and it cannot fire the safety rule.
- `nom_sam` (56): SAM3 text-only coverage `cov < 0.25`, from the v1 `test350` stage-1 file.

Not nominated stays `keep`. Nominated blocks go to Gemini with the v1 dual-scale crops and the v1 wide-context images. The prompt is v1 `fpclean_v1_p2` plus two sentences: stored or uninstalled modules are not installed PV, and rows of translucent skylight sheet on a corrugated / IBR roof are not PV. Version string `fpclean_v2_p3`. The v1 `decide()` rule is unchanged (not_pv, confidence ≥ 0.8, pv_fraction ≤ 0.1).

Safety, after that decision: if `cov ≤ 0.02` and `frac50 < 0.30` and Gemini said keep, the action becomes `review`. Six blocks moved. The flag set does not change.

Wide context: 85 blocks reused the v1 extracts. Nine new extracts were fetched with the same margin (max of 40 m and 1.5 × the block extent) into `fpclean_v2/context_extracts/`. All nine came from the Vexcel extract endpoint. Together they are 21.1 MB, under the cap of 80 files and 1 GB.

## Headline (L_v1anchor)
`L_v1anchor` is the v1 sheet (PART counted as TP) plus 葉's three anchors. UNS is excluded: 81 FP, 253 TP. This set does not depend on Gemini or SAM. `L_A` / `L_B` do, in part, so those rows are optimistic for any Gemini pipeline.

Proposed pipeline on all decided blocks: 65/81 FP blocks caught (recall 0.802; area recall 0.731), precision 1.000, 0 true-panel blocks flagged, 8 blocks in review (3,316 m2), 12 FPs left as keep (8,003 m2).
On blocks ≥ 100 m2: 58/70 FP blocks (recall 0.829; area recall 0.731), precision 1.000, 0 true panels flagged, 6 in review (3,303 m2), 8 FPs left.

The free rule (score OR frac50 < 0.25) catches more: 74/81 blocks and area recall 0.855, and flags 7 true panels (1,814 m2). On ≥ 100 m2 it flags 3 true panels (1,758 m2).

City-weighted with the v1 cells (stratum × `fp_flag_score`): the proposed pipeline removes about 263,000 of 329,000 m2 of FP area (recall 0.80, interval 0.68–0.91) and flags 0 m2 of true panels. Count recall is 0.68 with a wide interval (0.44–0.90); use the area number for the city total. The free rule removes about 288,000 m2 and flags about 27,000 m2 of true panels (8,000–53,000). Safety does not change these city numbers, because it only creates reviews.

Dark/corrugated FPs, same definition as `pregate_reverse_v1/analyse_v1.py` (the type text contains "dark" or "corrugat"): the proposed pipeline catches 37/48 (36/46 of those ≥ 100 m2). The free rule catches 43/48 (41/46).

Where `L_A` and `L_v1anchor` both say FP (69 blocks, 29,574 m2), 62 are flagged (26,750 m2), 4 are in review (1,916 m2) and 3 stay keep (907 m2): 021098 was not nominated (586 m2, corrugated, score 0.979, frac50 0.313, cov 0.91), 020731 was nominated and Gemini called it pv (316 m2, dark roof, frac50 0.003), and 022063 is a 5 m2 skylight Gemini called pv. The rest of the headline "missed" area is blocks `L_A` calls true panels.

## Anchors
000715: all three signals, both prompts say not_pv, flagged.
030281: score only. p2 said pv. p3 says not_pv (pool-heating mats, already in the v1 list), flagged.
021786: SAM only, cov 0, frac50 0.259. Both prompts say pv at confidence 1.0. The new skylight sentence did not change the answer. Safety puts it in review.
025757: SAM only. p2 said pv while describing palletised modules in a yard. p3 says not_pv. Flagged. This is the case the stored-module sentence was written for.

`L_A` / `L_B` for the same proposed pipeline: `L_B` flags one block both of those sets call a true panel, `g0.25_026777` (116 m2, v1 label UNS, so it is outside the headline). `L_A` recall of FP blocks is 0.86 (area 0.86) with 4 "true panels" flagged (317 m2), three of which are FP on `L_B`.

## Caveats
The thresholds were picked on these same 350 labels. A city forecast that treats 0.80 area recall and zero true-panel loss as confirmed will be too sure. `L_A` and `L_B` are partly circular with the older Gemini and SAM labels. `L_v1anchor` still calls several large dark roofs FP after `L_A` called them TP, which pulls the headline recall down. Neither issue is fixed by the bootstrap.

## Outputs
`fpclean_v2/summary_v2.md`, `README.md`, `common_v2.py`, `s0_fetch_context_v2.py`, `s2_gemini_verify_v2.py`, `s3_eval_v2.py`.
`fpclean_v2/runs/test350_v2/`: `stage2_decisions_v2.csv`, `eval_metrics_v2.csv`, `eval_city_weighted_v2.json`, `anchors.csv`, `missed_fp.csv`, `tp_flagged.csv`, `review.csv`, `sheet_errors.jpg`, `run.log`, `stage2_gemini.log`, `gemini/` (94 json).
`fpclean_v2/context_extracts/` (9 tif) and `context_extracts_log.csv`.
