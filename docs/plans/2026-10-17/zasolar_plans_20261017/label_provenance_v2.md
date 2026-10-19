# ZAsolar label provenance v2 (Cape Town + Joburg/Li)
Written 2026-10-17 (NZDT) by Grok Bot, read-only check. Sources: Dropbox `RA_Solar/Gao/progress_doc/` (weekly logs and daily logs), `RA_Solar/RA_Working_Hours_Log.xlsx`, rog `~/zasolar_data/capetown_ye_manual_labels_original/{README.md,inventory.csv}`, `~/projects/zasolar_verification_20261017/{VERIFICATION.md,verification.csv}`. No files were changed.

## What 葉 said (2026-10-17)
- Only the FIRST batch was pure hand labels: 葉 drew them in QGIS with the SAM plugin. Every later batch was model predictions that 葉 judged right/wrong, then used SAM to add the FNs and fix boundaries.
- Li's labels never went into training. Joburg labels are clean for the Cape Town models.

## Table
| # | Source | Grids | Count | Date | Method | Evidence (doc : line) |
|---|---|---|---|---|---|---|
| 1 | Early SAM2 (葉) | G1189 G1190 G1238 | 475 polys (135/98/242) | 2026-03-20 | Pure hand: GeoOSAM + SAM2.1 in QGIS | `week_2026-03-11/weekly_report_0316-0321.md:105` "使用 GeoOSAM 插件 + SAM2.1 对 3 个 grid（G1238, G1189, G1190）重新标注 … 共 456 个多边形（T1 质量）"; verification.csv L3 = 475. **Not used in v2c/v2d/v2d1** (inventory: absent) |
| 2 | Batch 001 (葉) | G1240 G1243 G1244 G1245 G1293 G1294 G1297 G1298 G1299 G1300 G1349 G1354 G1410 G1411 (14) | 492 panels | 2026-03-22 | Pure hand: SAM2.1 / GeoSAM + QGIS | `week_2026-03-18/2026-03-22.md:39-60` "**标注** (SAM2.1 / GeoSAM+QGIS) … Batch 001 (14 grids) … 小计 492" |
| 3 | Batch 002 part 1 (葉) | G1466 G1467 G1516 G1520 G1521 G1522 G1523 G1524 G1569 G1570 G1571 G1572 (12) | 656 panels | 2026-03-22 | Pure hand, same as above | `2026-03-22.md:62-78`; inventory tool = "GeoSAM(QGIS) SAM2.1 clicks", with a timestamp on every click |
| 4 | Batch 002 remainder (葉; README calls it "Batch 003 260324") | G1578 (03-23), G1579 G1580 G1625–G1635 (03-24) (14) | ≈1,359 polys (inventory sum) | 2026-03-23/24 | Pure hand, GeoSAM click timestamps (e.g. G1630 2026-03-24 10:03–10:41) | `2026-03-22.md:103` "完成 batch 002 剩余 19 grids 标注"; rog README "Batch 003 (260324)" |
| **1–4 total** | **Core hand-SAM set = 葉's "first batch"** | **43 grids** | **2,982 polys, ≈51,645 m²** | 2026-03-20 → 03-24 | Pure hand | `week_2026-03-25/2026-03-25.md:11` "**数据**: 43 grids, 2982 polygons (SAM2 cleaned)"; VERIFICATION L1 match; folder `core_hand_sam_20260320_0324` |
| 5 | Batch 003 (葉, review) | G1636–G1847 range: 100 scanned, 27 kept (e.g. G1686, G1687, G1683, G1747, G1750) | G1686: 131 preds → 102 correct + 32 delete + 3 SAM = 105 | from ≈2026-03-25 | Review GUI over **V2** predictions + SAM FN | `2026-03-25.md:33-51` "Detection Review GUI … G1686 标注完成 131 个模型预测 → 102 correct + 32 delete + 3 SAM 补标"; `week_2026-03-25/2026-03-29.md:52` "batch 003 GT 来源于 V2 推理结果" |
| 6 | Batch 004 (葉, review) | 37 grids: G1855 G1856 G1862 G1863 G1864 G1909 G1910 G1911 G1917–G1921 G1966 G1970–G1977 G1979 G1981 G2025–G2035 G2037 G2038 | ~3,000+ preds reviewed; 205 SAM FN markers → 177 accepted | grid list 2026-03-29; review 2026-04-01 → 04-07 | Review GUI over **V3-C** predictions + `sam_fn_review.py` | `2026-03-29.md:55-67` "Batch 004 使用 V3-C 权重 … 入选 grid 列表"; `week_2026-04-01/weekly_summary_0401-0407.md:12-14` |
| 7 | "March-3" set = `batch004_260403_detector_reviewed_NOT_pure_hand` | G1683 G1747 G1750 (batch-003 range) + G1911 G1966 G1970 G1974 G1981 G2031 G2032 G2037 G2038 (12) | 143 polys, 3,444.9 m² | file date 260403 = **2026-04-03** | Detector output with `review_status=correct` | VERIFICATION L2; inventory "detector pred reviewed=correct"; `week_2026-04-01/2026-04-03.md:88-89` commit `31142e2` "batch004 GT", `6cec0a9` "82 cleaned GT + 21 reviewed prediction gpkg". Training split: 11 train, G2038 eval |
| 8 | Cape Town merged COCO (by 04-03) | 106 grids | 7,301 polys (111 from SAM FN) | 2026-04-03 | Mixed: rows 1–7 | `2026-04-03.md:38`; `weekly_summary_0401-0407.md:70` |
| 9 | Joburg CBD (葉, review) | 25 JHB CBD grids | 1,003 review decisions; 136 SAM recut polys | 2026-04-07 | Review of V4 predictions + SAM recut | `weekly_summary_0401-0407.md:23-24`. Clean for CT models (葉) |
| 10 | Joburg Sandton (葉, review) | 25 grids (G1110–G1114, G1144–G1148, G1179–G1183, G1214–G1218, G1250–G1254) | — | 2026-04-14 → 04-19 | Review GUI + SAM FN pass | `RA_Working_Hours_Log.xlsx` April rows 15–19/04 (Botao column). JHB grid IDs; they overlap CT IDs (`weekly_summary:51` "Joburg/CT 46 个 grid ID 重叠") |
| 11 | Li's hand GT (CT `L####`, JHB Alexandra/CBD) | CT L0208–L1954; JHB 25-grid CBD / Alexandra | — | JHB drawn on 2024-02 Google Earth imagery | Hand labels by Li | Never in training (葉). rog README: "Li 格全在 L0208–L1954"; `weekly_summary:127-132` vintage gap 2023 tiles vs 2024 GT |

## Inconsistencies / open points
1. **Where batch 003 starts.** The rog README labels G1579–G1635 (03-24) "Batch 003". The 03-25 weekly report gives batch 003 as G1636→G1847, labelled by **reviewing V2 predictions**. By the Dropbox logs, G1578–G1635 is the rest of batch 002 (hand-made). The 43-grid core set is still all hand-made, and it matches what 葉 means by "first batch". Fix the README label.
2. **The "March-3" name is wrong.** The folder date is 260403 = **3 April**, not 3 March. Labelling had not started on 3 March (QGIS/GeoSAM was installed 03-18).
3. **batch004 folder covers only part of batch 004.** It holds 12 grids, but batch 004 has 37. Three of the 12 (G1683, G1747, G1750) are in the batch-003 range, not batch 004. Its 143 polys are tiny next to ~3,000 reviewed predictions. It looks like a partial export of `correct` predictions only, possibly without the SAM FN additions and edits. The full reviewed batch 003/004 GT (106-grid / 7,301-poly COCO) needs a separate trace.
4. **Batch 002 counts.** On 03-22: 34 kept, 3 skipped for no buildings (G1517, G1573, **G1632**), leaving 19 to do. Only 26 batch-002 grids reached core (12 + 14). G1632 was skipped but later labelled (64 polys, 03-24), presumably after a full download. The 03-29 log reports "Batch 002, 26 grids, GT=2015"; the 03-22 log used 1,148 incomplete.
5. **Early 3 grids.** The 03-20 log says 456 polys; the files hold 475. They may have been touched up afterwards. These grids are not in any v2 split.
6. **G1630 seg 130 is a bad geometry** (already known; v2d replaced it).
7. **RA log grid IDs.** Jinze's April entries (G1188…, G1630–G1634, BFN…) are RA labelling. The IDs collide with CT core IDs (e.g. G1630–G1634), but they look like JHB/other-city grids. Before using them, check the region; do not mix them up with 葉's CT core.
8. **No logs before 03-16.** Dropbox has no weekly logs from before 03-16, so V1's 257-poly Google-Earth hand labels (and the legacy G0854… files) are covered only by the rog README.

---
# v2 addition (2026-10-17 NZDT): label source "Ye reviewed-prediction batches 003/004"
Read-only search of rog and the Dropbox mount (`~/Dropbox`). Nothing was modified.

## Answer: the complete set already exists and is already in training
- **Location (primary):** rog `~/projects/ZAsolar/data/annotations/Capetown/G*_SAM2_260403.gpkg`, **51 files / 51 grids / 4,070 polygons**. All mtimes are 2026-04-03 23:03–23:09 NZDT. Every feature has `review_status=correct`.
  - Batch 003 (V2 preds): G1682 G1683 G1685–G1693 G1743 G1744 G1747 G1749 G1750 G1798 G1800 G1806 G1807 (20 grids)
  - Batch 004 (V3-C preds): G1855 G1856 G1862 G1863 G1864 G1909 G1910 G1911 G1917 G1919 G1920 G1921 G1966 G1970–G1976 G1981 G2025–G2032 G2037 G2038 (31 grids; G1918 G1977 G1979 G2033–G2035 have no 260403 file)
- **Copies:**
  - Dropbox `RA_Solar/Gao/annotations/Capetown/`: 50 of 51 are md5-identical. **G1688 differs.** In Dropbox, `G1688_SAM2_260403.gpkg` was rewritten on 2026-10-14 14:36. The original is kept there as `G1688_SAM2_260403_orig_backup_20261014.gpkg`, same size and mtime as rog. The v2d README says the raw G1688 has no CRS and the builder reads `--g1688-fixed`.
  - rog `~/capetown_eval/ann/Capetown/`: 51 files.
  - rog `~/zasolar_data/geid_temporal/ct_labels_corrected_v2d1_20261016/Capetown/`: 51 files (the v2d1 corrected label copy).
- **SAM-added FNs:** they are traceable through the `source` column: `sam_fn_marker` (batch 003, Review GUI FN points) and `sam_fn_review` (batch 004, `sam_fn_review.py`). There are about 196 polygons with a SAM source in total (e.g. G1690 17, G1971 17, G1855 16, G1800 14). The other features have `source` NULL, meaning a detector polygon kept as correct.
  - Boundary edits do not have a separate flag. Only the final geometry remains, so how much of each polygon was edited cannot be counted.
- **Relation to the 12-grid folder** `capetown_ye_manual_labels_original/batch004_260403_detector_reviewed_NOT_pure_hand/`: its 12 files are byte-identical copies of 12 of these 51. They are **exactly the 12 files with no `source` column**, i.e. no SAM additions (G1683 G1747 G1750 G1911 G1966 G1970 G1974 G1981 G2031 G2032 G2037 G2038; 143 polys). It is a subset, not a different version.

## Use in training versions (grid in manifest.csv)
| Version | 260403 grids used | train / val / eval | absent |
|---|---|---|---|
| v2c (2026-10-15) | 50 | 39 / 7 / 4 | G1807 (`--exclude G1807`) |
| v2d (2026-10-16) | 50 | 39 / 7 / 4 | G1807 |
| v2d1 (2026-10-16) | 50 | 39 / 7 / 4 | G1807 |
| v2d2 (2026-10-17) | 50 | 39 / 7 / 4 | G1807 |
- val: G1689 G1691 G1743 G1798 G1863 G1920 G2028. eval/held-out: G1686 G1692 G1800 G2038.
- v2d replaced some human polygons with SAM3 ones in G1687 G1971 G2028 (thin_strip) and dropped 4 zero-area polygons in G1682.

## Inconsistencies found
1. **The READMEs say too little.** The README in `capetown_ye_manual_labels_original` and the v2d2 README describe only "batch004_260403 (12 grids, 143 polygons)" as being in training. In fact, all 50 reviewed-prediction grids (≈4,000 polygons) are in v2c to v2d2, including 7 val and 4 eval grids. So most of the Cape Town training and validation labels are model-reviewed rather than hand-drawn. This matters for the label-noise and fairness discussion, because val/eval labels come from V2/V3-C predictions.
2. **106 / 7,301 does not fully reconcile.** Core 43 grids / 2,982 polys plus these 51 grids / 4,070 polys gives 94 grids / 7,052 polys. The 04-03 log's 106 / 7,301 (and PROGRESS.md "Cape Town: 106 grids / 7367") probably also counts the 12 legacy Google Earth files (G0854 … G967, G1189/G1190/G1238.gpkg, all_annotations_cleaned). This was not verified.
3. **G1688 has diverged** between rog and Dropbox (see above). Training used the fixed copy, not the raw rog file.
