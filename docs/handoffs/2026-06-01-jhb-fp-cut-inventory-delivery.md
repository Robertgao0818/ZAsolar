# JHB Vexcel FP-cut inventory — delivery & backdating handoff

**日期:** 2026-06-01
**状态:** ✅ 完成。全量 362-grid JHB Vexcel 推理库存经 Gemini 两阶段 FP-review 清洗,合并为单层交付物。
**承接:** `docs/handoffs/2026-05-31-jhb-fp-cut-sweep-resume.md`(暂停/续跑记录)、
`docs/handoffs/2026-05-31-jhb-two-stage-fp-review-production.md`(pipeline 验证)。
**消费方:** `solar_backdating/scripts/temporal/build_inventory_chip_groups.py`。

---

## 一句话

JHB Vexcel 全量库存 **47,465 → 41,393**(丢弃 **6,072** 假阳,12.8%),9 个 sweep 单元闸门全 PASS、
abstain 0、0 冲突、0 完整性违例。交付物是原始 merge01_c0925 库存的**同 schema 同 layer 直替版**。

## 交付文件

| 用途 | 路径 |
|---|---|
| **FP-cut 库存(交付)** | `results/analysis/full382_merge01_2026-05-15/jhb_full382_unified_A_merge01_c0925_fpcut_2026-06-01.gpkg` |
| 原始库存(清洗前,保留) | `results/analysis/full382_merge01_2026-05-15/jhb_full382_unified_A_merge01_c0925.gpkg` |
| 每格保留多边形数 | `data/analysis/gemini_review_calib/prod_jhb/fpcut_per_grid_kept.csv` |
| 合并脚本(可复现) | `data/analysis/gemini_review_calib/prod_jhb/merge_fpcut_inventory.py` |
| 各 batch 闸门/丢弃 rollup | `data/analysis/gemini_review_calib/prod_jhb/sweep_rollup.tsv` |

- layer = `solar_predictions`,CRS = `EPSG:32735`,**41,393 features / 356 grids**
- schema 与原始库存完全一致:`[source_grid, confidence, score, area_m2, orig_area_m2, sam_score, n_merged, source_tile, geometry]`

## 给 backdating 的接法(直替)

`build_inventory_chip_groups.py` 默认 `--inventory` 仍指向**原始**(未清洗)库存。改用 FP-cut 版:

```bash
cd /home/gaosh/projects/solar_backdating && source scripts/activate_env.sh
python scripts/temporal/build_inventory_chip_groups.py \
  --inventory /home/gaosh/projects/ZAsolar/results/analysis/full382_merge01_2026-05-15/jhb_full382_unified_A_merge01_c0925_fpcut_2026-06-01.gpkg \
  --layer solar_predictions \
  --inventory-tag jhb_full382_unified_A_merge01_c0925_fpcut_2026-06-01 \
  --region-key johannesburg
```

builder 只强依赖 `geometry` + `source_grid`,其余列 `.get()` 兜底 —— schema 与原始一致,无需改 builder。
（如要长期默认走 FP-cut 版,可把 `DEFAULT_INVENTORY` 指过来,但建议显式传 `--inventory` 保留可追溯。）

## Provenance

- **推理源:** `results/johannesburg/unified_reviewall_A_perdet_sam_maskbox_vexcel_2024_full382_sam_maskbox/JNB*/predictions_metric_merge01_c0925.gpkg`(362 grid)
- **judge 模型:** `gemini-3-flash-agent`(底层 `gemini-3-flash-b` = Gemini 3.5 Flash),默认 thinking,多尺度双图(z20 tight + z48 wide)两阶段 FP-review。
- **置信分带:** HI = conf≥0.95(两阶段:stage1 + stage2 merge 复核);LO = 0.925–0.95(单阶段 stage1-as-drops)。
- **并发:** 30 workers / QPS 8 / routing-salt=target。(35 worker 试验超订 3×10 池 → ~9% 503,已回退 30。)
- **配额自治:** v3 driver 撞限额自动 sleep 1h + `--resume`;401/503 已纳入可重试信号。

## 物料平衡(清洗前→后)

| 分带 | 复审数 | — |
|---|---|---|
| HI (conf≥0.95) | 44,213 | |
| LO (0.925–0.95) | 3,252 | |
| **合计复审** | **47,465** | = 原始库存 |
| **丢弃(FP)** | **6,072** | 12.8% |
| **保留(交付)** | **41,393** | 356 grids |

**6 个格被全量 FP-cut 到 0**(原始仅 1–11 检测、全判假阳,共 23 个多边形):
`JNB0199, JNB0225, JNB0247, JNB0248, JNB0273, JNB0333`。

## 各 batch rollup(闸门全 PASS,abstain 全 0)

| 单元 | grids | HI | LO | drop | keep | review | gate |
|---|---|---|---|---|---|---|---|
| pilot (JNB0002–0025) | 24 | 2,568 | 173 | 361 | 2,380 | 0 | PASS |
| batch_00 | 40 | 9,034 | 607 | 925 | 8,716 | 0 | PASS |
| batch_01 | 40 | 9,976 | 559 | 864 | 9,671 | 0 | PASS |
| batch_02 | 40 | 6,821 | 475 | 834 | 6,462 | 0 | PASS |
| batch_03 | 40 | 3,371 | 368 | 690 | 3,049 | **38** | PASS |
| batch_04 | 40 | 2,019 | 217 | 509 | 1,727 | 0 | PASS |
| batch_05 | 40 | 1,877 | 201 | 556 | 1,522 | 0 | PASS |
| batch_06 | 40 | 2,936 | 226 | 575 | 2,587 | 0 | PASS |
| batch_07 | 40 | 3,385 | 259 | 478 | 3,166 | 0 | PASS |
| batch_08 | 18 | 2,226 | 167 | 280 | 2,113 | 0 | PASS |
| **合计** | **362** | **44,213** | **3,252** | **6,072** | **41,393** | **38** | **9/9 PASS** |

## Review queue

仅 batch_03 有 **38** 条 `review`(其余单元全 0)。这 38 条是 **keep 内被打「建议复核」flag 的子集 —— 保留在库存中,未丢弃**(fail-safe 方向),与 batch_03 stage2 那次 503 中断+重跑相关。
位置:`data/analysis/gemini_review_calib/prod_jhb/batch_03/filtered/review_queue.csv`。库存不受影响,可选做人工抽检。

## 完整性

- 9/9 batch fail-closed 闸门 **PASS**,abstain_rate 全 **0.0**。
- 各 apply:`n_conflicts=0`、`n_integrity_violations=0`、`n_undecided=0`。
- 物料平衡:复审 47,465 = drop 6,072 + keep 41,393,逐 grid 核对一致。

## 待提交代码

- `scripts/analysis/gemini_fp_review_multiscale.py` 的 **`--resume`**(保留 usable、只重跑 abstain/缺失;已单测)。
  续跑全程依赖,跑完应连同本交付提交。
- driver / helper(`sweep_driver_v3.sh`、`jsonl_health.py` 含 401/503 信号加固、`merge_fpcut_inventory.py`)
  在 gitignored 的 `data/analysis/.../prod_jhb/`(运行产物区),不进 git。
