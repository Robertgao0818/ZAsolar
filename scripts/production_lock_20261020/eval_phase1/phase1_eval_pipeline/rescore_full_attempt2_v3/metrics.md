# Phase-1 v3: validation selection and holdout

Main validation result: **val_holdout**. val_sel300 is the training checkpoint-selection diagnostic.
Intervals: 95% grid-cluster percentile bootstrap, 20,000 draws, seed 20261018;
ratios are recomputed from summed TP/FP/FN, not averaged grid rates. Undefined draws are counted in JSON.

| Class / metric | val_sel300 | val_holdout (main) |
|---|---:|---:|
| PV tiles | 300 | 1403 |
| PV pixel_iou | 0.88071135 [0.81022151, 0.92763993] | 0.89759735 [0.81695920, 0.94423184] |
| PV object_f1_0.5 | 0.83532220 [0.70618557, 0.92178771] | 0.82554162 [0.75645836, 0.87306526] |
| PV 0.3 | 0.83255814 / 0.87745098 / 0.85441527 | 0.82671082 / 0.88325472 / 0.85404789 |
| PV 0.5 | 0.81395349 / 0.85784314 / 0.83532220 | 0.79911700 / 0.85377358 / 0.82554162 |
| PV 0.7 | 0.76279070 / 0.80392157 / 0.78281623 | 0.75717439 / 0.80896226 / 0.78221209 |
| SWH tiles | 159 | 826 |
| SWH pixel_iou | 0.02206816 [0.00000000, 0.06850960] | 0.10056390 [0.03264804, 0.19903810] |
| SWH object_f1_0.5 | 0.00000000 [0.00000000, 0.00000000] | 0.06134969 [0.00952381, 0.13725490] |
| SWH 0.3 | 0.04545455 / 0.20000000 / 0.07407407 | 0.05673759 / 0.36363636 / 0.09815951 |
| SWH 0.5 | 0.00000000 / 0.00000000 / 0.00000000 | 0.03546099 / 0.22727273 / 0.06134969 |
| SWH 0.7 | 0.00000000 / 0.00000000 / 0.00000000 | 0.02127660 / 0.13636364 / 0.03680982 |

| Class / metric | eval |
|---|---:|
| PV tiles | 632 |
| PV pixel_iou | 0.91124778 [0.86507717, 0.93228731] |
| PV object_f1_0.5 | 0.85887709 [0.78030303, 0.91170431] |
| PV 0.3 | 0.87537994 / 0.87272727 / 0.87405159 |
| PV 0.5 | 0.86018237 / 0.85757576 / 0.85887709 |
| PV 0.7 | 0.77203647 / 0.76969697 / 0.77086495 |
| SWH tiles | 505 |
| SWH pixel_iou | 0.20271872 [0.00011005, 0.31920848] |
| SWH object_f1_0.5 | 0.23529412 [0.00000000, 0.33962264] |
| SWH 0.3 | 0.22000000 / 0.31428571 / 0.25882353 |
| SWH 0.5 | 0.20000000 / 0.28571429 / 0.23529412 |
| SWH 0.7 | 0.10000000 / 0.14285714 / 0.11764706 |

Object rows 0.3/0.5/0.7 show P / R / F1 under pinned zaeval one-to-one IoU matching.
SWH uses labelled grids only; unlabelled grids are excluded, labelled empty grids retained.
Objects are re-merged/matched within each scope; subtracting full-pool object counts would be wrong.
The split is by grid/image row; shared imagery across val grids may remain in both scopes.
CIs condition on this checkpoint; with five eval grids they are approximate and do not measure training-seed variability.
Future architecture comparisons should use paired grid draws; CI overlap alone is not a significance test.

Training log reconciliation:
```json
{
  "log_path": "/home/gao/capetown_eval/ds3m_ft/run_v2c/log.jsonl",
  "absolute_iou_tolerance": 0.001,
  "tolerance_note": "Operational diagnostic tolerance, not a theorem about bf16 error.",
  "status": "not_comparable",
  "best_step": 7500,
  "log_sha256": "fa0d1edca3d128dcfb1b367caba92dfcf03938a888654e9a11faae375dffa78d",
  "legacy_checkpoint": true,
  "selection_key": "iou",
  "selection_score": 0.7998345880435397,
  "logged_validation": {
    "iou": 0.7998345880435397,
    "prec": 0.9752515462044489,
    "rec": 0.8164051540057641,
    "area_bias": -0.1628773548905348,
    "biou": 0.44019910715801264,
    "rec_lt4": 0.4,
    "rec_4_10": 0.9565217391304348,
    "rec_10_25": 0.9833333333333333,
    "rec_gt25": 0.8852459016393442,
    "n_lt4": 5,
    "n_4_10": 23,
    "n_10_25": 60,
    "n_gt25": 61,
    "swh_iou": 0.18098102130965912,
    "swh_as_pv_frac": 0.0,
    "neg_fp_m2_per_ha": 0.5955034342120052,
    "step": 7500
  },
  "PV": {
    "saved_mask_iou": 0.8807113546087177,
    "logged_iou": 0.7998345880435397,
    "absolute_difference": 0.08087676656517806,
    "within_diagnostic_tolerance": false
  },
  "SWH": {
    "saved_mask_iou": 0.022068155857342304,
    "logged_iou": 0.18098102130965912,
    "absolute_difference": 0.1589128654523168,
    "within_diagnostic_tolerance": false
  },
  "comparability_issues": [
    "Legacy best.pt was selected by pixel IoU on v2c labels/pool, not v2f best_objf1.",
    "No matching v2f data_md5 in checkpoint-adjacent config.",
    "No matching training script_md5 in checkpoint-adjacent config.",
    "Cannot establish a full 300-tile training selection from config."
  ],
  "selection_verified_from_log_only": true,
  "checkpoint_step_binding": "inferred from best filename + logged save rule, not embedded checkpoint metadata"
}
```
