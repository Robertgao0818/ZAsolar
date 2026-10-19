# Independent recheck of Joburg CBD25 zaeval-2 rescore (项目主管, 2026-10-19, CPU only, tmux jhb_z2_recheck)
- indep_score_v1.py (own code, no zaeval import; zaeval-2@8044d14 objects.py rules) per grid -> parts/; aggregate_v1.py -> indep_results_v1.json; group_iou_diag_v1.py -> group_iou_diag_v1.json (diagnostic, not zaeval-2).
- md5 infer_save_v1.py 349eeca6, score_z2_v1.py abf9245f, RESULTS_v1.csv b3cc743d: match.
- Result: tp/fp/fn at IoU 0.3/0.5/0.7, size bins and all 25 per-grid counts identical to scores/*__h0.9_0.5.json for all 4 models.
- Handling used by theirs (and replicated): probs carry the official 3x3 cell pregate (scores.json, >=0.45); zaeval-2 built-in cross-tile 8-adjacent merge; NO tile-level tau=0.70 pregate and no separate G7 xtile post-step. exclude-pixels with 6px GT band.
- uint8: binarize on u8>=230/>=128 == float /255 at 0.9/0.5 (0 px differ). Same 2500 tiles, same gate (0 px outside kept cells), 529 gated tiles for all.
- FN at IoU0.5 are ~80% "glued" (one prediction spans >=2 adjacent GT polygons). Group-IoU diagnostic F1: convnext .838, b2 .865, v2c .881, avg3 .877.
