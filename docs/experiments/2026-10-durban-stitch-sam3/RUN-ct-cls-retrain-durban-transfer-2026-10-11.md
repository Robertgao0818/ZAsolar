# Cape Town solar_cls retrain, tested on the Durban 343

Date: 2026-10-11. Work dir: `~/zasolar_data/durban_stitch_sam3_20261009/ct_cls_transfer_v1/`.
Local CUDA. `summary_v1.md` names the machine as the rog RTX 4070. Cost $0. No RunPod. No Dropbox writes.
This note only records that run. Training was not repeated. The only new file is this note.
Decision: not adopted.

## Data

Cape Town labels are the human review of `cape_town:v3c_targeted_hn_aerial_2025`, plus legacy grid G1686.
`correct` and `edit` are PV (true panel). `delete` is non-PV (false panel). `s0_ct_labels_fetch.py` keeps only those three statuses.

`fetch.log` and `ct_labels.gpkg`: 3,299 objects, 2,506 true / 793 false, 37 grids.
36 grids are the v3c bucket (3,165 objects). G1686 is the legacy bucket (134 objects).
Review counts: correct 2,441, edit 65, delete 793.

Area is the polygon area in EPSG:32734. It matches the rounded table in `s2_eval.log`.

| | n | median m2 | p75 m2 | max m2 |
|---|---:|---:|---:|---:|
| false (delete) | 793 | 15.1 | 22.2 | 379.1 |
| true (correct+edit) | 2,506 | 19.6 | 33.2 | 2,231.2 |

| area | true | false | objects |
|---|---:|---:|---:|
| <30 m2 | 1,794 | 708 | 2,502 |
| 30–100 m2 | 585 | 80 | 665 |
| 100–1,000 m2 | 123 | 5 | 128 |
| >=1,000 m2 | 4 | 0 | 4 |

Cape Town has 5 false objects at or above 100 m2. None are at or above 1,000 m2.

Imagery: per-object crops, bbox plus 8 m, from the public City of Cape Town WMS (`cityimg.capetown.gov.za`, layer `Aerial Imagery_Aerial Imagery 2025Jan`, 8e-7 deg/px, the same resolution the script uses for tiles). `fetch.log`: `FETCH_DONE 3299 1754s`. Crops on disk: 140M (`du -sh ct_crops`). `summary_v1.md` says 123 MB; the size above is the current `du` figure.

This run does not rebuild the rest of the original solar_cls mix. The v1 adaptive manifest (`solar_cls/data/cls_pv_thermal_v1_adaptive/dataset_manifest.json`, area cutoff 30 m2) still records those chips: legacy_flat_batch003 2,581 train + 260 val, Johannesburg v4_aerial_2023 2,169 train + 595 val, GT heater audit 570 train chips. `summary_v1.md` also lists the Johannesburg GEID cascade as absent here; the solar_cls protocol note counts that bucket as 903 raw chips. Only G1686 from the legacy bucket was fetched.

Durban images are the cached Vexcel extracts already used for the 350-block audit (`fpflag_v1/extracts` for the val 150, `B2_pilot200/extracts` for the pilot 200). They were not fetched again.

The scored set is label_v2 with UNS dropped. Pilot `labels_v2.csv` is 200 rows with 7 UNS. Val `val_labels_v2.csv` is 150 rows and has no UNS. `blocks_features_oof_v1.csv` has the remaining 343: 270 true (`y=0`) and 73 false (`y=1`). These match label_v2.
At or above 100 m2: 191 true and 65 false. Below 100 m2: 79 true and 8 false.
Of the 73 false blocks, 8 are under 30 m2 (83.6 m2 together), none fall between 30 and 100 m2, 55 are 100–1,000 m2 (14,342 m2), and 10 are at or above 1,000 m2 (16,549 m2, 53.6% of the large-false area). The 65 large false blocks sum to 30,891 m2. The largest is `g0.25_026846` at 3,508 m2.

`darkcorr` is the flag from `pregate_reverse_v1/analyse_v1.py` (the false-type string contains "dark" or "corrugat"). 40 of the 65 large false blocks have it.

## Recipe

Two variants. Both use the unmodified `solar_cls` trainer and the unmodified adaptive chip code.

Code: `solar_cls/scripts/classifier/train_cls.py` and `chip_extraction.py` (`CHIP_SPEC_VERSION` `adaptive_v1`).
Config: `solar_cls/configs/classifier/dinov2_vits14.json`.
Backbone: `dinov2_vits14`. Image 224 px. Batch 32. AMP on. Class-balanced `WeightedRandomSampler` (inverse class frequency, with replacement).
Stage 1: 3 epochs, backbone frozen, Adam, lr 1e-3.
Stage 2: 20 epochs, all weights unfrozen, Adam, lr 1e-5, cosine decay to 1e-6.
The config text mentions layer-wise learning-rate decay. `train_cls.py` does not implement it. Stage 2 is one Adam learning rate on every parameter. `summary_v1.md` already notes this.

Chip geometry (`chip_extraction.py`): crop side = long side of the bbox times (1 + 2×0.6) = 2.2 times the long side, clamped to 96–512 px, then resized to 224. A chip is `clipped` when the unclamped side is over 512 px.

Split: whole grids. A grid is val if it is in `dataset_manifest.json` `counts.val_selected.grids`. Seven of those grids are in this label set: G1971, G1973, G1977, G1981, G2027, G2029, G2032. The other 30 grids are train. G1686 is train.

| variant | who is trained | chip rule | train images | val images |
|---|---|---|---:|---:|
| orig | area < 30 m2 only | max side 512 | 1,697 (1,225 PV, 472 non-PV) | 805 (569 PV, 236 non-PV) |
| nocap | all areas | max side removed; the whole bbox+margin is scaled to 224 | 2,213 (1,681 PV, 532 non-PV) | 1,086 (825 PV, 261 non-PV) |

The orig chip set is exactly the 2,502 objects under 30 m2. Largest orig chip: 29.98 m2. None of the 2,502 Cape Town orig chips are clipped. None of the 3,299 nocap chips are clipped.
The June v1 manifest lists 575 PV val chips for the v3c bucket. This rebuild has 569 PV val chips under the same <30 m2 rule and the same val grids. The non-PV val count matches (236). The 6-chip gap was not chased.

On the Durban chips the clip is the whole problem. Of the 343 scored blocks, orig-spec chips are clipped for 246 of 256 blocks at or above 100 m2 (59 of 65 false, 187 of 191 true). Six large false blocks and four large true blocks are not clipped. Nocap chips are never clipped. Their raw side runs from 96 px to 27,826 px before the resize to 224.

Thresholds, fixed before the Durban count is used:

- solar_cls score is P(PV). The cut is the 5th percentile of that score on Cape Town val true panels, so about 95% of those true panels stay PV. A block is flagged when P(PV) is below the cut.
- Frozen-head score is P(false). The cut is the 95th percentile of out-of-fold scores on Cape Town true panels, so about 95% of those true panels pass. A block is flagged when P(false) is at or above the cut and the area is at least 100 m2.
- The free rule is not fit on Cape Town. It is `score_max < 0.95` OR `frac50 < 0.25`, the rule already written in `pregate_reverse_v1/summary_v1.md`.

Deployment skip, from `classify_predictions.py`: a clipped chip is kept as PV and is not flagged. `s5_eval.py` calls this "orig deployment (clipped bypass)". The production default also skips every object at or above 30 m2. That stronger skip was not the row in the CSV. It would flag none of the 65 large false blocks, because every one of them is over 30 m2.

## Commands

Scripts, in run order, all in the work dir:

- `s0_ct_labels_fetch.py` writes `ct_labels.gpkg`, `ct_crops/`, and `fetch.log`.
- `s1_ct_embed.py` writes `ct_emb_dinov2s_gate.npz`, `ct_emb_dinov3l_sat.npz`, and `ct_pregate.csv`. Cape Town crops are resampled to 0.065 m pixels so the frozen features match the Durban extracts. Backbones stay frozen. Hugging Face cache is offline.
- `s2_train_eval.py` writes `ct_cv_and_durban_auc.csv`, `durban_heldout_metrics.csv`, `durban_blocks_ct_scores.csv`, and `s2_eval.log`. Heads are fit in memory. No head file is saved.
- `s3_build_chips.py` writes `chips_orig/`, `chips_nocap/`, `durban_chips/`, `ct_chip_manifest.csv`, and `durban_chip_manifest.csv`.
- `s4_train.sh` trains both variants. The command inside the script is:

```bash
export HF_HUB_CACHE=/home/gao/zasolar_data/models/dinov2_floor/hf_cache HF_HUB_OFFLINE=1
~/venvs/yolo_gate/bin/python -u /home/gao/projects/solar_cls/scripts/classifier/train_cls.py \
  --data-dir $PWD/$spec --output-dir $PWD/checkpoints/cls_ct_dinov2_vits14_${spec#chips_} \
  --config /home/gao/projects/solar_cls/configs/classifier/dinov2_vits14.json \
  --arch dinov2_vits14 --workers 4
```

`spec` is `chips_orig`, then `chips_nocap`. Logs: `train_orig.log`, `train_nocap.log` (the script appends `TRAIN_ALL_DONE` to the nocap log).

- `s5_eval.py` loads `best_cls.pth` for each variant and writes `cls_retrain_info.csv`, `cls_retrain_durban_metrics.csv`, `durban_blocks_cls_retrain.csv`, and `s5_eval.log`.

The per-epoch times in the train logs sum to 164.6 s (orig) and 210.9 s (nocap). That sum leaves out process startup.

## Cape Town validation

From `cls_retrain_info.csv` and the train logs. Balanced accuracy is the best val value. The checkpoint stores the unrounded value.

| run | best val balanced acc | where | val PV | val non-PV | val AUC | thr (95% PV kept) | non-PV killed at that thr |
|---|---:|---|---:|---:|---:|---:|---:|
| orig | 0.9524 | stage 2, epoch 12 | 569 | 236 | 0.987 | 0.9643 | 0.958 |
| nocap | 0.9590 | stage 2, epoch 2 | 825 | 261 | 0.987 | 0.5764 | 0.962 |

The two AUCs are on different val sets (805 images versus 1,086). Nocap's best epoch is epoch 2 of the 20 fine-tune epochs. Later epochs do not beat it.

Frozen embeddings, Cape Town only, 5-fold `StratifiedGroupKFold` by grid, one seed (`random_state=0`). Logistic regression: `C=0.01`, `class_weight=balanced`. MLP: one hidden layer of 256, `alpha=1e-2`, early stopping. Features are the frozen native in-polygon mean, the close-up CLS plus polygon mean, or both concatenated.
Backbones, from `s1_ct_embed.py`: DINOv2-S gate (`vit_small_patch14_dinov2.lvd142m`, 392 px) and DINOv3-SAT-L (`vit_large_patch16_dinov3.sat493m`, 448 px). Cache dirs: `~/zasolar_data/models/dinov2_floor/hf_cache` and `~/zasolar_data/models/dinov3_sat/hf_cache`.

`s2_eval.log` picks the head by Cape Town CV AUC, without looking at Durban: `ct_dinov2s_gate_both_mlp`, threshold 0.399. Cape Town CV AUC 0.977. That is the CT-best row below.

## Durban 343, blocks at or above 100 m2

Denominator: 65 false, 191 true. A flag is a block called non-PV. "TP flagged" is a true panel thrown out.

| rule | CSV name | false caught | true flagged | false-area recall | true area flagged m2 | dark/corrugated |
|---|---|---:|---:|---:|---:|---:|
| deployed-skip | cls[orig] orig deployment (clipped bypass) p<0.964 | 6/65 | 1/191 | 0.024 | 129 | 1/40 |
| forced | cls[orig] orig-chips classify-all p<0.964 | 62/65 | 23/191 | 0.938 | 23,326 | 39/40 |
| orig weights on full-roof chips | cls[orig] nocap-chips classify-all p<0.964 | 64/65 | 121/191 | 0.995 | 185,567 | 39/40 |
| nocap retrain | cls[nocap] nocap-chips classify-all p<0.576 | 47/65 | 8/191 | 0.450 | 8,334 | 28/40 |
| free rule | score_max<0.95 OR frac50<0.25 | 61/65 | 7/191 | 0.925 | 4,644 | 37/40 |
| DINOv3-SAT-L frozen + logreg | CT-dinov3l_sat-native-logreg p>=0.56 | 60/65 | 10/191 | 0.972 | 9,712 | 39/40 |
| DINOv3-SAT-L frozen + MLP | CT-dinov3l_sat-native-mlp p>=0.25 | 60/65 | 7/191 | 0.916 | 6,172 | 38/40 |

The rule names round the cut (three decimals for solar_cls, two decimals for the frozen heads). The counts come from comparing to the exact quantile. `cls_retrain_info.csv` records the classifier cuts as 0.9643 and 0.5764. The frozen-head table below records the three-decimal cuts.

Deployed-skip flags a block only when the orig chip is not clipped and P(PV) is below the exact orig quantile (0.9643 at four decimals). The six large false blocks it catches are the six unclipped ones. It also flags 1 of the 4 unclipped large true blocks. The other 59 large false blocks are kept as PV because the chip is clipped. False-area recall is 0.024.

Forced uses the same orig weights and the same orig chips, and scores the clipped chips too. It catches 62/65 false blocks and 0.938 of false area, and it flags 23 true blocks (23,326 m2).

The same orig weights on nocap chips (the whole roof, scaled to 224) catch 64/65 false blocks and flag 121/191 true blocks (185,567 m2). Durban AUC for that pairing is 0.802 on the large blocks (`cls_retrain_info.csv`). The orig model never trained on a roof that big, and the chip no longer looks like a 224 px window of a small array.

Nocap is the retrain that does see large Cape Town objects. At its own 95% Cape Town recall cut it catches 47/65 false blocks but only 0.450 of their area, so the misses are the bigger false roofs. It flags 8 true blocks (8,334 m2). Durban AUC on large blocks is 0.952, the best of the three fine-tune pairings, and the operating point still loses to the free rule on false area (0.450 versus 0.925) while flagging more true area (8,334 versus 4,644).

Other Durban AUCs for the fine-tunes (`cls_retrain_info.csv`): orig weights on orig chips, 0.936 all blocks and 0.946 on large blocks; orig weights on nocap chips, 0.838 and 0.802; nocap weights on nocap chips, 0.923 and 0.952.

The free rule is the comparison rule from `pregate_reverse_v1/summary_v1.md`. The two arms, from the same CSV:

| rule | false caught | true flagged | false-area recall | true area flagged m2 | dark/corrugated |
|---|---:|---:|---:|---:|---:|
| score_max<0.95 | 56/65 | 6/191 | 0.646 | 4,528 | 33/40 |
| frac50<0.25 | 56/65 | 4/191 | 0.895 | 3,006 | 36/40 |
| score_max<0.95 OR frac50<0.25 | 61/65 | 7/191 | 0.925 | 4,644 | 37/40 |

`pregate_reverse_v1/summary_v1.md` prints the same counts and rounds the area recall to 0.65, 0.90, and 0.93. The CSV values above are the ones used here. That summary also says the 0.95 and 0.25 cuts were picked on these same labels.

Nocap at other cuts (`cls_retrain_durban_metrics.csv`), same nocap chips:

| cut | false caught | true flagged | false-area recall | true area flagged m2 | dark/corrugated |
|---|---:|---:|---:|---:|---:|
| p<0.576 (Cape Town 95% recall) | 47/65 | 8/191 | 0.450 | 8,334 | 28/40 |
| p<0.5 | 46/65 | 6/191 | 0.444 | 4,484 | 28/40 |
| p<0.3 | 35/65 | 4/191 | 0.298 | 601 | 20/40 |
| p<0.1 | 20/65 | 2/191 | 0.135 | 257 | 10/40 |
| score<0.95 OR nocap | 59/65 | 11/191 | 0.722 | 9,973 | 36/40 |
| score<0.95 AND nocap | 44/65 | 3/191 | 0.374 | 2,890 | 25/40 |

Joining nocap to the score rule does not beat the free rule. The OR catches fewer false blocks (59 versus 61) and flags more true area (9,973 versus 4,644).

Below 100 m2 the free rule and the score rule flag nothing (0/8 false, 0/79 true), because both are defined on large blocks only. On those small blocks, deployed-skip and forced have the same counts in the CSV: 8/8 false caught and 14/79 true flagged. Nocap at p<0.576 catches 7/8 small false blocks and flags the same 14/79 true blocks.

Anchor `g0.25_021786` is a 1,275 m2 false block, dark/corrugated, detector `score_max` 0.954, `frac50` 0.259. The free rule lets it pass (0.954 is not below 0.95, and 0.259 is not below 0.25). Deployed-skip also lets it pass, because the orig chip is clipped, even though that model scores it P(PV)=0.09. Forced flags it. Both DINOv3-SAT native heads flag it (P(false) 0.97 logreg, 0.96 MLP).

## Frozen heads, all 12

Cape Town CV and the Durban large-block operating point. Thresholds are the Cape Town 95% true-pass cuts. Durban counts are from `durban_heldout_metrics.csv`. AUCs are from `ct_cv_and_durban_auc.csv`. The flag is applied only at or above 100 m2.

| model | view | head | CT CV AUC | CT CV AUC >=30 m2 | thr | CT false caught | Durban AUC all | Durban AUC >=100 | false caught | true flagged | false-area recall | true area m2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dinov2s_gate | native | logreg | 0.970 | 0.979 | 0.652 | 0.840 | 0.948 | 0.970 | 55/65 | 11/191 | 0.828 | 12,479 |
| dinov2s_gate | native | mlp | 0.973 | 0.979 | 0.368 | 0.881 | 0.950 | 0.967 | 58/65 | 13/191 | 0.868 | 12,520 |
| dinov2s_gate | close | logreg | 0.973 | 0.973 | 0.663 | 0.847 | 0.936 | 0.955 | 46/65 | 8/191 | 0.653 | 23,024 |
| dinov2s_gate | close | mlp | 0.971 | 0.970 | 0.440 | 0.852 | 0.946 | 0.964 | 55/65 | 14/191 | 0.823 | 33,027 |
| dinov2s_gate | both | logreg | 0.976 | 0.979 | 0.618 | 0.869 | 0.935 | 0.950 | 39/65 | 11/191 | 0.479 | 11,103 |
| dinov2s_gate | both | mlp | 0.977 | 0.982 | 0.399 | 0.874 | 0.947 | 0.962 | 51/65 | 14/191 | 0.726 | 31,023 |
| dinov3l_sat | native | logreg | 0.972 | 0.978 | 0.561 | 0.874 | 0.954 | 0.978 | 60/65 | 10/191 | 0.972 | 9,712 |
| dinov3l_sat | native | mlp | 0.974 | 0.978 | 0.247 | 0.892 | 0.954 | 0.982 | 60/65 | 7/191 | 0.916 | 6,172 |
| dinov3l_sat | close | logreg | 0.964 | 0.968 | 0.642 | 0.810 | 0.947 | 0.979 | 42/65 | 4/191 | 0.463 | 601 |
| dinov3l_sat | close | mlp | 0.957 | 0.961 | 0.626 | 0.776 | 0.930 | 0.977 | 36/65 | 4/191 | 0.259 | 601 |
| dinov3l_sat | both | logreg | 0.976 | 0.978 | 0.495 | 0.887 | 0.950 | 0.981 | 50/65 | 8/191 | 0.644 | 7,194 |
| dinov3l_sat | both | mlp | 0.975 | 0.976 | 0.351 | 0.870 | 0.947 | 0.978 | 33/65 | 2/191 | 0.278 | 244 |

The CT-best head (DINOv2, both views, MLP) catches 51/65 large false blocks and flags 14 true blocks (31,023 m2). Pairing it with the existing rules (`durban_heldout_metrics.csv`):

| rule | false caught | true flagged | false-area recall | true area flagged m2 |
|---|---:|---:|---:|---:|
| score<0.95 OR CT-best | 61/65 | 15/191 | 0.846 | 31,172 |
| score<0.95 AND CT-best | 46/65 | 5/191 | 0.527 | 4,379 |
| frac50<0.25 OR CT-best | 63/65 | 14/191 | 0.992 | 31,023 |

The DINOv3-SAT native rows are the strongest transfer rows on this Durban table. They were not the head Cape Town CV selected. Native MLP has the highest Durban large-block AUC of the 12 (0.982) and the same true-block count as the free rule (7/191), with more true area flagged (6,172 versus 4,644) and a lower false-area recall (0.916 versus 0.925).

A separate number in the same files is the Durban out-of-fold DINOv3-SAT native logreg at p>=0.5: 61/65 false, 10/191 true, false-area recall 0.967, 9,785 m2 of true area. That head was trained on these 343 labels. It is a ceiling from `pregate_dino_021786_v1`, not a Cape Town transfer, and it is not a clean test.

## Conclusion

Not adopted.

The original training set is entirely objects under 30 m2 (2,502 chips, max 29.98 m2). The full Cape Town review is mostly that small too (2,502 of 3,299), and it contains 5 false objects at or above 100 m2. The Durban false blocks are large roofs: 65 of 73 are at or above 100 m2, 55 of those are 100–1,000 m2, and 10 are larger (16,549 m2, 53.6% of the large-false area). A model trained on small Cape Town panels does not learn those roofs.

Read as a filter for the large Durban blocks, the Cape Town classifier loses to the free rule `score_max<0.95 OR frac50<0.25` (61/65 false, 7/191 true, 0.925 of false area, 4,644 m2 of true area):

- Deployed-skip leaves clipped chips as PV. It catches 6/65 false blocks and 0.024 of false area.
- Forcing the orig model to score clipped chips catches 62/65 and flags 23 true blocks (23,326 m2).
- Forcing those weights onto full-roof chips flags 121/191 true blocks (185,567 m2).
- The nocap retrain catches 47/65 false blocks and only 0.450 of false area, and flags 8 true blocks (8,334 m2).

The frozen DINOv3-SAT native heads come closest (60/65, and 7 or 10 true blocks). They are not adopted either. Cape Town CV would have shipped the DINOv2 both-view MLP, which flags 14 true blocks and 31,023 m2. The native DINOv3 numbers are the best DINOv3 rows on this same Durban set.

If this classifier is deployed anyway, one of two things happens. The production skip keeps almost every large false roof (59 of 65 are clipped). Turning the skip off throws out on the order of 23,000 m2 of true panels with the orig chips, or about 186,000 m2 if the full roof is scored with the orig weights. The nocap retrain still misses more than half of the large false area.

## What would be next

Train a small head on frozen DINOv3-SAT-L, using new Durban labels. The head in this run was logistic regression or a one-layer MLP. The 350 audit blocks (pilot 200 + val 150) stay as the test set only. 343 of them were scored here, and the free-rule cuts were already chosen on this set, so they cannot also be the training or calibration set.

The labels have to include large false roofs. Cape Town does not have them (5 false objects >= 100 m2). The labelling sketch in `RUN-durban-stitch-sam3-2026-10-09.md` (new Durban blocks, the 350 held out) is still the right shape. This retrain does not change that plan.

## Caveats

- One seed. Orig's best checkpoint is stage 2 epoch 12. Nocap's is stage 2 epoch 2. No repeat run.
- Orig and nocap validation AUCs use different images.
- The free-rule cuts 0.95 and 0.25 were chosen on these Durban labels (`pregate_reverse_v1/summary_v1.md`). The Cape Town thresholds were fixed on Cape Town. The two kinds of number are not the same evidence.
- Quoting the best DINOv3 row uses the Durban table to choose among 12 heads. The pre-registered choice, Cape Town CV, is the weaker DINOv2 row.
- The Durban out-of-fold DINOv3 number trains on the 343. It is not a held-out test.
- Scored labels are label_v2. A conservative label column exists on the same file (258 true / 85 false) and was not the label in these CSVs.
- The production area cutoff of 30 m2 would skip every large Durban block. The deployed-skip row is the milder rule: skip only clipped chips.
- Stage 2 does not use the layer-wise learning-rate decay named in the config note.
- Below 100 m2, forced classification flags 14 of 79 true blocks. The free rule does not look at those blocks.

## Weights

`s5_eval.py` loads `best_cls.pth`. `last_cls.pth` is the final epoch, not the best epoch.

- `checkpoints/cls_ct_dinov2_vits14_orig/best_cls.pth` (248M). Best val balanced accuracy 0.9524031157844568.
- `checkpoints/cls_ct_dinov2_vits14_orig/last_cls.pth` (248M).
- `checkpoints/cls_ct_dinov2_vits14_nocap/best_cls.pth` (248M). Best val balanced accuracy 0.9590247300592128.
- `checkpoints/cls_ct_dinov2_vits14_nocap/last_cls.pth` (248M).

Each directory also has `config.json` and `training_history.json`. Frozen-head weights were not written. Their Durban scores are in `durban_blocks_ct_scores.csv`. The DINOv2 and DINOv3-SAT backbone files stay in the two Hugging Face caches named above.

## Files (`du -sh` in the work dir)

| path | size |
|---|---:|
| `.` | 1.7G |
| `checkpoints/` | 992M |
| `checkpoints/cls_ct_dinov2_vits14_orig/` | 496M |
| `checkpoints/cls_ct_dinov2_vits14_nocap/` | 496M |
| `chips_nocap/` | 234M |
| `chips_nocap/train/` | 157M |
| `chips_nocap/val/` | 78M |
| `chips_orig/` | 170M |
| `chips_orig/train/` | 115M |
| `chips_orig/val/` | 55M |
| `ct_crops/` | 140M |
| `durban_chips/` | 67M |
| `durban_chips/orig/` | 34M |
| `durban_chips/nocap/` | 34M |
| `ct_emb_dinov3l_sat.npz` | 33M |
| `ct_emb_dinov2s_gate.npz` | 13M |
| `ct_labels.gpkg` | 9.7M |
| `durban_blocks_ct_scores.csv` | 472K |
| `durban_blocks_cls_retrain.csv` | 416K |
| `ct_chip_manifest.csv` | 384K |
| `ct_pregate.csv` | 160K |
| `durban_chip_manifest.csv` | 36K |
| `s1_ct_embed.py`, `s2_train_eval.py`, `s5_eval.py`, `s2_eval.log` | 8.0K each |
| `s0_ct_labels_fetch.py`, `s3_build_chips.py`, `s4_train.sh`, `s5_eval.log`, `summary_v1.md`, `fetch.log`, `train_orig.log`, `train_nocap.log`, the four result CSVs | 4.0K each |

Result CSVs: `cls_retrain_info.csv`, `cls_retrain_durban_metrics.csv`, `ct_cv_and_durban_auc.csv`, `durban_heldout_metrics.csv`.
