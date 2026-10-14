# RUN: Durban estset_v1 evaluation of frozen FP rules A-E (2026-10-12)

Local rog RTX 4070, $0. Labels: 葉, blind page, 344/344 (TP 250 / FP 82 / PART 12 / UNS 0), 17:55-18:04 NZDT.
Rules frozen in `estset_v1/frozen_rules_v1.json` (sha256 1fb7de04..., commit e77d7c0 at 18:05 NZDT, before any signal was computed; the `frozen_at` field inside says 18:20 by mistake - the commit time is authoritative). Signals computed 18:15-18:40 with labels unread (`eval_v1/ORDER.log`), then joined. Built by grok build; orchestrator re-derived point estimates independently (identical).

Signals: frac50 + SAM3 via `fpcut_v4/podscore/score_city.py` (344 ok, 355 s) on extracts re-cut to the reference 8 m extent (score grid is anchored to image origin; no resampling). DINOv3-SAT native + CT-only logreg head refit reproduced dev 60/65 / 10/191; frozen thr 0.5612. Gemini 3.5 flash lite fpclean_v2_p3 (cfg 6dbc5221c760df93): 87 calls, 0 errors -> keep 273 / fp_candidate 62 / review 9 (5 safety). Wrapper bug (pandas `.cov`) fixed in the eval wrapper only; common_v2 unchanged.

Estimates on the frame (city minus the 350 labelled blocks: 30,254 blocks, 1,617,799 m2); stratified ratio estimator, 95% CI = stratified bootstrap (B=2000, XL/L certainty). Main: true = TP+PART.

| rule | flagged area m2 | area precision | FP-area capture (all) | capture >=100 m2 | true-panel area cut m2 (share) |
|---|---|---|---|---|---|
| A score<0.95 >=100 | 186,239 | 0.944 [0.861-1.00] | 0.50 [0.41-0.62] | 0.88 [0.81-0.93] | 10,380 [0-25,800] (0.8%) |
| B union >=100 | 234,812 | 0.838 [0.730-0.940] | 0.56 [0.46-0.70] | 0.98 [0.94-1.00] | 38,068 [13,094-69,400] (3.0%) |
| C union all sizes | 349,401 | 0.821 [0.714-0.917] | 0.82 [0.70-0.94] | 0.98 [0.94-1.00] | 62,630 [28,025-106,541] (4.9%) |
| D C->Gemini+safety | 250,243 | 0.951 [0.877-1.00] | 0.68 [0.55-0.82] | 0.76 [0.64-0.88] | 12,334 [0-30,781] (1.0%); review 143 blk / 38k m2 |
| E DINOv3-SAT CT head | 197,620 | 0.900 [0.791-0.982] | 0.51 [0.40-0.64] | 0.89 [0.81-0.96] | 19,860 [3,460-46,677] (1.6%) |

Bars. Codex (true-loss UB <=1%, precision LB >=0.98, capture LB >=0.75): all rules fail. Claude (precision LB >=0.90 and cut-true <=38k m2): none pass both; A and D pass the area limit (UB 25.8k / 30.8k) but precision LB is 0.861 / 0.877. With PART-as-FP sensitivity D and E pass Claude's bars.

FP rate. (i) estset frame 21.6% [17.5-26.3]. (ii) old corrected 18.5% [13.8-24.2] (different labeller/sample; not pooled). (iii) headline city = frame FP + labelled FP area of the 350 (L_v1anchor, 36,872 m2) over holefill_v3 city 1,889,920 m2: **20.5% [16.9-24.4]**. By stratum (frame): XL 1.3%, L 10.0% (census), M 24.0% [21.4-26.6], S 25.7% [14.5-38.7].

Files: `~/zasolar_data/durban_stitch_sam3_20261009/estset_v1/eval_v1/` (signals_v1.csv, eval_v1.json, eval_table_v1.csv, REPORT_eval_v1.md).
