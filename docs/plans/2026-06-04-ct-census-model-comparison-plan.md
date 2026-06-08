<!--
Generated 2026-06-04 via multi-agent workflow (ct-census-model-comparison-plan).
6 parallel investigators (GT inventory / leakage map / models / eval infra / V1.4 channels / imagery domain)
→ synthesis → adversarial critique (code-line-verified) → revision (rev2).
Run ID: wf_708319fc-e70
-->

# Cape Town Census 前模型比较方案 (终稿 rev2)

> 范围：在 CT 全量 census 推理前，用 leakage-safe 协议选出生产模型。成功指标遵循 V1.4：grid 级 area-aggregate 为主，polygon F1 降为诊断。本方案区分「已验证事实(✅)」与「推断(🔶)」。本 rev2 已吸收审稿全部 high/medium 意见，修正了 merge_mode 双模式执行机制、CT GT 发现路径、ranking postproc 一致性三处命令层硬伤。

---

## 0. 一句话结论

CT census 模型选型 = **2 模型对决：v3c (现役 baseline) vs exp_unified_reviewall_A (挑战者)**，二者都是 aerial_2025 in-domain；排名面 = `cape_town_independent_26` (26) **+ Li CT GT (17，独立 li scheme)** = **43 grid / ~2,640 polygon**；主裁判是 `area_aggregate_eval.py` 的 Tier-1 (σ_Bw + RMSE)，bulk∈[0.5,2.0] 仅作 sanity gate；每个候选必须双 merge_mode 都评（但不能靠 run_benchmark 自动 sweep）；现有磁盘结果不足排名，必须在 **RunPod** 上重跑全 43 grid × 2 模型 × 2 merge_mode。

## 决策状态 (2026-06-04 已确认)

| # | 决策 | 结论 |
|---|---|---|
| 1 | ranking postproc config | ✅ `v4_canonical.json`，只变 merge_mode |
| 2 | v3_cleaned (ep4/20 paused) | ✅ 排除，不烧推理 |
| 3 | unified_A exclude 列表 | ✅ 发布排名前必须从训练 config/handoff config-确认覆盖全 ranking grid |
| 4 | census merge_mode | ✅ 用排名 (σ_Bw+RMSE) 胜出的 mode，并在 v4_agg/v4_high 显式 pin `merge_mode` 键（不留 per-detection 默认） |
| 5 | Li CT GT 纳入 | ✅ 全量纳入主排名：先下载 16 grid aerial_2025 瓦片 + 注册 Li 独立 scheme + 规整 gpkg，再 43-grid 一次性出排名 |
| — | 计算环境 | ✅ 推理全部在 **RunPod** 上跑（tiles→/dev/shm，PAR=6 @5090，nohup/tmux，S3 传输，结果 `aws s3 cp` 拉回） |

---

## 1. 可用 CT GT 清单与质量/语义 caveat

### 1.1 CT GT universe (✅ fiona/disk-verified)

- **94 个 SAM2 grid** = 唯一 canonical evaluable CT GT，文件形如 `data/annotations/Capetown/Gxxxx_SAM2_YYMMDD.gpkg`。✅ disk 核实：`data/annotations/Capetown/` 下 **94 个 `G*_SAM2_*.gpkg`，0 个子目录，0 个 clean_gt 文件**（CT GT 是 **flat 布局**，这点对 Phase-3 命令至关重要）。
- 9 个 bare-only legacy grid (A3-G Google-Earth 弱监督) — 不在任何 suite，**不可作 gold GT**。
- 2 个聚合快照 (`all_annotations_cleaned.gpkg`, `solarpanel_g0001_g1190.gpkg`) — 非 per-grid，忽略。

### 1.2 语义级别 caveat (载荷性，必须写进结论)

- ✅ CT SAM2 GT 是 **sub-array 级，不是 installation 级**。`label_source=human_manual_sam_assisted` (Axis B = H)，spec 默认 **Axis A = A2**（"mostly installation-like, 未逐 polygon 核实 merge/boundary"）。
- ✅ regions.yaml 把 G1189/G1190/G1238 标 T1，但 spec 自己的定义要求 T1 必须有逐 polygon A1 复核记录——CT SAM2 grid 没有。**严格意义上全部 CT SAM2 GT 是 A2/T2 级**；不要在选型文档里声称 "CT T1 gold"。
- ✅ 大装机上 SAM 要么切碎成兄弟 polygon，要么吞屋顶 (memory `feedback_sam_tool_ceiling`)。

**推论 (🔶)**：sub-array 碎片化使 polygon-F1 结构性不公平（一个 installation 级预测 vs N 个 sub-array GT 会被判 FP/under-detection）。area-aggregate 对碎片化大体鲁棒（总覆盖面积守恒），但有已知偏置：碎/吞屋顶 GT 会奖励 over-painting（`feedback_pixel_union_with_fragmented_gt`），所以必须用 **pixel-level set-theoretic IoU/F1**（`area_aggregate_eval.py` 用 `unary_union` 而非 polygon-area-sum，✅ 代码核实 line 134），并以 σ_Bw/RMSE 为裁判。

### 1.3 数据卫生红旗 (✅)

- `annotation_manifest.csv` **stale**：只覆盖 3 grid，引用的 source 文件不是 pipeline 实际用的 gpkg，且对 G1189/G1190 的 tier 与 regions.yaml 矛盾。**不要把它当 inventory 引用。**
- regions.yaml 只注册 3/94 CT grid，其余 91 靠 `annotation_loader.py` 目录 glob fallback（无 provenance 元数据）。
- 休眠 bug：fallback 是 sorted-glob first-wins，未来任何「未注册 + 同时有 bare 和 SAM2」的 grid 会静默选到劣质 bare 文件（当前 3 个 dual-file grid 都已注册，故暂不触发）。

---

## 2. 训练泄漏判定与 leakage-safe 评估 suite (严格)

### 2.1 Scope correction (✅)

训练 scope **不是**由 manifest 决定的，而是由各 config 的 `exclude_grids` + `core.annotation_loader.discover_annotations()` 目录扫描决定。判 leakage 用 per-config exclude 列表，不用 manifest。

### 2.2 模型 × CT suite 泄漏矩阵 (✅ from config 行号)

| 模型 | `cape_town_t1_smoke` (G1189/90/1238) | `cape_town_independent_26` (26 grid) | `cape_town_batch003_diagnostic` (20 grid) |
|---|---|---|---|
| **v3c** (现役 best) | **leaked** (EarlySAM2 在训练池, 未排除) | **clean** (导出 HOLDOUT) | **leaked** (batch003 FP 是其 HN 源) |
| **exp_unified_reviewall_A** | **leaked** (EarlySAM2 在 CT positives) | **clean 🔶 (依据 note, 未 config 确认 — 见 §2.4)** | **leaked** (Batch003 positives + HN) |
| v4_1 | leaked | **clean** (yaml 排除 26) | leaked |
| v3_cleaned | leaked 🔶 | clean 🔶 (假设同 26-grid HOLDOUT, **未由专属 config 确认**) | clean 🔶 |
| v2 | **leaked** (只训过这 3 grid) | clean | clean |
| train20_val5_hn | **leaked** (G1238) | **leaked** (G1300/1411/1570/1572) | partial-leaked |
| jhb_phaseA | clean (0 CT grid) | clean | clean (但纯域外) |
| v4_2 (archived) | 🔶 unknown | 🔶 unknown | 🔶 unknown |

### 2.3 leakage-safe 评估 suite 设计 (✅ grid_ids 已核实)

**唯一可排名面 = `cape_town_independent_26`** (26 grid / 1148 polygon, `role: primary`, `leakage_risk: low`):
`G1240,G1243,G1244,G1245,G1293,G1294,G1297,G1298,G1299,G1300,G1349,G1354,G1410,G1411,G1466,G1467,G1516,G1520,G1521,G1522,G1523,G1524,G1569,G1570,G1571,G1572`

它对 4 个 CT-domain contender (v3c / unified_A / v4_1 / v3_cleaned) 全 clean，是项目声明的 primary suite，且 26/26 grid 都在 aerial_2025 coverage 且磁盘有 SAM2 GT。

**只可作 diagnostic / regression，禁止排名：**
- `cape_town_t1_smoke` (G1189/90/1238)：对**每个** CT-trained 模型都 leaked（=EarlySAM2 训练 grid）。post_train.yaml 标 `leakage_risk: low` 是指它作 regression smoke-test 的用途，**不是**说它适合排名——确保 orchestrator 不把 smoke 当低风险排名面。
- `cape_town_batch003_diagnostic` (20 grid)：`leakage_risk: high` by design（batch003 FP = v3c/v4_1/unified_A 的 HN 源）。

**硬排除出排名：**
- `train20_val5_hn`：双重泄漏（26 个 ranking grid 里 4 个在其训练集 + smoke），任何 independent_26 数字虚高不可比。已被 reject、不在 registry。
- `jhb_phaseA` / `v4_2`：0 或不明 CT 暴露 + sensor/vintage 域差，非 CT census 候选。
- `v3_cleaned`：**未收敛 paused checkpoint（ep4/20），从排名 + 推理矩阵移除**（见 §3）。

### 2.4 两个发布前必须 config-确认的 exclude 列表 (🔶)

1. **`exp_unified_reviewall_A` (主挑战者)**：其 independent_26 cleanliness 目前仅依据 regions.yaml model_run note「CT_INDEP_26 hardcoded exclude」+ unified handoff，**没有 training_sets.yaml exclude 列表**。叠加它的 CT eval 从未跑完，发布排名前必须从其训练 config/handoff 确认 26 grid 全被排除（与对 v3_cleaned 的同等审查）。在确认前，unified_A 的 independent_26 cleanliness 是 🔶。
2. **`v3_cleaned`**：假设用同样 26-grid HOLDOUT，但无专属 training_sets.yaml 条目确认。结合它是未收敛 paused checkpoint，本轮直接排除（不再列 reference）。

---

## 3. 候选模型 shortlist

> ✅ checkpoints/ 磁盘核实有 9 个 dir + checkpoints_cleaned；model_registry.yaml 只有 **v1/v2/v3c/v3_cleaned/v4_1/v4_2 共 6 个**（grep 核实，无 `exp_unified_reviewall_A`），**漏掉最关键的挑战者**。这是 Phase-0 硬 blocker：run_benchmark.py:83-84 对未注册 key 直接 `[WARN] ... skipping`。

| 等级 | 模型 | checkpoint | 训练影像域 | CT census 候选? | 理由 |
|---|---|---|---|---|---|
| **排名** | **v3c** (V3-C-HN) | `checkpoints/exp003_C_targeted_hn/best_model.pth` | CT aerial_2025 | **是，强制 baseline** | ROADMAP 现役 production，independent_26 P69.6/R72.2/F1 70.9% |
| **排名** | **exp_unified_reviewall_A** | `checkpoints/exp_unified_reviewall_A/best_model.pth` | CT aerial_2025 + JHB Vexcel 混合 | **是，主挑战者(带 gating)** | 2026-05-13 CT warm-start, mask-trusted halo fix；**CT independent_26 eval 在 2026-05-13 因 pod quota 失败、从未确认完成**；exclude 列表 🔶 未 config 确认 |
| reference | v4_1 (exp005) | `checkpoints/exp005_v4_1_hn/best_model.pth` | CT aerial_2025 | 诊断 | recall 较 V3-C 回退 ~6pp，ROADMAP "V3-C remains best" |
| floor | v2 / v1 | `checkpoints/v2_sam2_260320/`, `v1_ft_*/` | CT | floor only | 历史地板 |
| **排除** | **v3_cleaned** | `checkpoints_cleaned/best_model.pth` | CT aerial_2025 | **否** | ✅ 目录存在但 registry 标 "epoch 4/20, paused"，coco_ap50 0.788 是 paused 快照、**非收敛模型**；exclude 列表亦未 config 确认。**不烧推理**，除非训练完成 + exclude 确认 |
| **排除** | v4_2 (archived) | `exp006_v4_2_jhb_ft/` | JHB aerial_2023 | 否 | 已 archived，AP50 0.8371 是 Goodhart (deploy F1 0.587) |
| **排除/负控** | train20_val5_hn | `train20_val5_hn_20260508_v3c/` | JHB Vexcel | 否 | 5-grid CT 全输 V3-C + ranking 面泄漏 |
| **排除/负控** | jhb_phaseA | `jhb_phaseA_20260508_0953/` | JHB Vexcel | 否 | 三项 JHB pass criteria 全 fail |

**关键 caveat (🔶→需 gating)**：unified_A 被广泛引用的强数字 (xdomain60 / JHB-382 production / bulk 1.04) **全是 Vexcel 影像结果，不是 CT aerial_2025**，不能迁移。它在 CT 上目前只是"纸面挑战者"，直到其 independent_26 area-eval 真正跑完 + exclude 列表 config-确认。

---

## 4. 指标与评估协议

> 核心区分（不要混淆）：✅ `validation_strategy.md:117` 明言 benchmark suite 是 "internal detection benchmarks, **NOT validation**"。内部 benchmark 做**模型选型排名**；V1.4 四通道是对**选定模型 census 产物**的验证。

### 4.1 polygon F1 — 诊断层 (run_benchmark.py)

- ✅ 输出 micro/macro F1@IoU + P/R/TP/FP/FN + mean_iou，写 `summary.json/by_suite.csv/by_grid.csv`。
- 因 CT GT sub-array 碎片化，polygon F1 **结构性不公平**，仅作诊断 + per-grid 错误定位，**不作裁判**。

### 4.2 area-aggregate Tier-1 — 主裁判 (area_aggregate_eval.py) (✅ console 逐行核实)

console deploy view 输出正是项目指定的判官集：
`F1  pgF1  bulk  σ_Bw  log-σ  RMSE  thru0_β  R²`

- **主裁判：σ_Bw (std_ratio_Bw, B-weighted, paper-primary) + RMSE_m2**
- **sanity gate：bulk_pred_gt_ratio ∈ [0.5, 2.0]**（越界直接淘汰，但 bulk 本身不是单调好坏，见 `feedback_bulk_ratio_perverse_incentive`）
- 辅助：agg_area_F1、mean_per_grid_F1、log-σ、thru0_β、OLS R²
- ✅ 用 set-theoretic union 几何 (`unary_union`, pred∩gt)，不是 polygon-area-sum；丢 >20000 m² / 非有限 polygon。
- ✅ **收集机制**：`area_aggregate_eval._load_run_grids` (area_aggregate_eval.py:165-178) **直接从已注册 model_run 目录读 `predictions_metric.gpkg`**，无需任何 benchmark run。这与 run_benchmark 的 collect-only 限制是两套独立的 collector（见 §8 Risk #7），不要混为一谈。

### 4.3 cluster-level — 碎片化诊断 (cluster_level_eval.py)

- 回答"同一 installation 是否在 split/merge 容忍下被匹配"，对 sub-array GT 特别有用；作 polygon F1 失公平时的补充诊断。

### 4.4 merge_mode / postproc 成对评估 (一等杠杆) — **机制已修正** (✅ 代码核实)

> **审稿 high #1/#2 修正**：初版断言"每 checkpoint 双 merge_mode"是对的，但给的命令跑不出来。事实如下，必须按真实机制操作：

- ✅ `merge_mode` 是 **postproc config 的 JSON key**（`core/postproc.py:67` 声明），由 `finalize.py:525` 解析：`args.merge_mode = postproc_cfg.get("merge_mode", "per-detection")`。**unset 时默认 = `per-detection`，不是 pixel-or。**
- ✅ `detect_and_evaluate.py` **没有** `--merge-mode` 开关，只接 `--postproc-config`。`run_benchmark.py` 的 `geoai` 路径（line 222-258）构造命令时**不传** `--merge-mode`，且 benchmark 默认 postproc = `configs/postproc/batch003_best_f1.json`（无 merge_mode 键）→ 继承 `per-detection`。**所以 run_benchmark 一次只产单一 merge_mode（默认 per-detection），它自己不会 sweep merge_mode。**
- ✅ `run_benchmark --predictions-source direct`（line 987）走 detect_direct→finalize→evaluate，但**不在 run_benchmark 层暴露 merge_mode 开关**，merge_mode 仍只来自 suite/preset 的 postproc config。`direct` source **不给** CLI 级 per-detection 切换。
- ✅ 唯一正确的 `--merge-mode` 标志在 **`finalize.py:84`（repo root，不在 `scripts/postproc/`）**。

**双 mode 的两条可行实现路径（任选其一，全程一致即可）：**

- **路径 A（推荐，最省算力）— 共享 raw artifact + 两次 finalize：**
  对每个 (模型, grid) 只跑一次 `detect_direct.py` 产 `raw_detections.pkl`，再用**同一个** ranking postproc config（见 §4.5）跑两次 `finalize.py`，分别 `--merge-mode pixel-or` 和 `--merge-mode per-detection`，写两个独立 output-subdir，注册成两个 model_run。
- **路径 B — 两个 postproc config：**
  准备两份内容相同、仅 `merge_mode` 键不同的 ranking postproc config，各跑一遍 detect+finalize。

- **census 阶段配置（与排名分开）：**
  - census 聚合产物用 **`v4_agg.json`** (post_conf 0.65)
  - backdating seed 用 **`v4_high.json`** (post_conf 0.98)
  - ✅ **注意**：v4_agg/v4_high **当前都不设 merge_mode 键 → 继承 `per-detection` 默认**（finalize.py:525）。这是个隐式决策；既然 merge_mode 是 ~6pp F1 / 0.6 bulk 杠杆且与 bulk sanity gate 交互，census 阶段必须**显式 pin** merge_mode（在 v4_agg.json/v4_high.json 加 `"merge_mode": ...` 键，或 finalize 时显式传），用排名阶段胜出的 mode。
  - **不要把 census 默认成 v4_canonical**（那是 cross-experiment 可比性 config）
  - 🔶 caveat：v4_agg/v4_high **只在 JHB CBD 25-grid Vexcel-2024 上校准过，无 CT 侧校准**；census 前应在 CT 重新校准阈值。

### 4.5 RANKING postproc config 锁定 — **可比性 confound 已闭** (审稿 medium #4)

> 初版 Phase-1 用 `v4_canonical.json`（post_conf 0.85, merge_mode pixel-or）而 Phase-2 的 run_benchmark 继承 preset `batch003_best_f1.json`（post_conf 0.92, 无 merge_mode → per-detection），导致 polygon-F1 诊断与 area-aggregate 裁判在**不同 postproc/merge_mode** 上计算，不可比。

**修正规则：排名全程锁定单一 postproc config，polygon-F1 读取与 area-aggregate 读取共用它，只变 merge_mode。**

- **推荐 ranking config = `configs/postproc/v4_canonical.json`**（V1.4 cross-experiment 可比性基准，post_conf 0.85），但**显式覆盖 merge_mode**跑两遍（pixel-or / per-detection）。
- run_benchmark（polygon-F1 诊断）必须用**同一个** config + 同一 merge_mode 来读取，做法：要么用 `--collect-only` 收集由路径 A 的 finalize 产出的 predictions（同一 raw + 同一 config + 指定 merge_mode），要么覆盖 run_benchmark preset 的 `postproc_config` 使其指向 ranking config。**明确告知 reviewer：benchmark preset 默认是 `batch003_best_f1.json`（per-detection），若不覆盖则 Phase-2 polygon-F1 与 Phase-3 area 不可比。**
- 一旦锁定，两个 merge_mode 行的 polygon-F1 与 area-aggregate 都来自同一 raw + 同一 config，差异**只**归因于 merge_mode。

---

## 5. 推荐横评矩阵 (模型 × suite × 指标 × merge_mode)

排名面 = `cape_town_independent_26` (26 grid) **+ Li CT GT (17 grid, 独立 `li` scheme)** = 43 grid（见 §5.1 / Phase 0.5）；ranking postproc config 锁定 `v4_canonical.json`（§4.5），只变 merge_mode。下表对 43 grid 全集逐 (模型 × merge_mode) 跑，Li grid 与 independent_26 命令完全一致（只是 region/scheme 解析走 `li`）。

| 模型 | suite | merge_mode | polygon F1 (诊断) | area Tier-1 (裁判) |
|---|---|---|---|---|
| v3c | independent_26 | pixel-or | run_benchmark (同 config collect) | area_aggregate σ_Bw/RMSE/bulk-gate |
| v3c | independent_26 | per-detection | run_benchmark | area_aggregate |
| unified_A | independent_26 | pixel-or | run_benchmark | area_aggregate |
| unified_A | independent_26 | per-detection | run_benchmark | area_aggregate |
| (可选) v4_1 | independent_26 | both | 诊断 | reference |
| v2 | independent_26 | one | floor | floor |

- **v3_cleaned 从矩阵移除**（未收敛 paused checkpoint，不烧推理）。
- smoke (G1189/90/1238) 与 batch003 **只在主表外**作 regression / 错误剖析，明确标 diagnostic-only。

### 5.1 Li CT GT (✅ 已决定纳入主排名)

Li 的 CT GT (17 grid: G1842/43/44/46, G1895–1902, G1950–54, ~1,490 polygon)，2026-06-04 核实结论：
- ✅ **同 GT 语义类**：图层 `SAM_太阳能_…`、中位面积 13–49 m²、panel/sub-array 级、A2 —— 与现有 CT 排名 GT 同类，可直接进同一 area-aggregate 表。
- ✅ EPSG:4326；**在 aerial_2025 上标注**（on-disk `aerial_2025/G1895` 瓦片 bbox 与 Li G1895 GT bbox 精确吻合）→ vintage 对齐。
- ✅ **leakage-clean**：不在 Gao 94-grid SAM2 语料、不在任何 suite、不在 v3c/unified_A 训练集。
- ✅ 与 independent_26 零重叠；位于**东部 Cape Flats / False Bay**（lon ~18.82–18.86，Gao task-grid 范围 lon≤18.606 之外）→ 给排名加一块不同城市肌理的地理多样性。
- ⚠️ **独立 grid scheme**：Li 用自己的 KML（`cape_town_grid_Li_G0029_G1841.kml`），**Li 的 G1895 ≠ Gao 的 G1895**（Gao 无此 cell）。必须注册成单独 `li` annotation_scheme，**禁止并进 Gao grid 命名空间**（多 scheme 撞 ID 规则）。
- ⚠️ **瓦片只有 Li-G1895 在盘**；其余 16 grid 需从 CT WMS 下载（Phase 0.5）。源文件名含空格/全角括号、G1896 是 zip，入库需规整。

---

## 6. census 前的 CT GT 缺口 (vs JHB CBD 25-grid)

| 需求 | JHB CBD 25-grid | Cape Town | 缺口 |
|---|---|---|---|
| Ch1 stratified RA precision | 有 25-grid clean_gt 作 proxy | 无；`ra_precision_sample.py` 未建，无 `ra_precision_*.csv` | **HIGH** |
| Ch2 exhaustive-AOI recall | `data/annotations_channel2_clean/` (25 JHB grid, sub-array clean_gt 非 exhaustive-recall) | **无**（全 25 dir 是 G07/08/09，零 G1xxx） | **HIGH** |
| grid_strata.csv | 非正式 | **不存在** → 所有 grid 退化成单一 "default" stratum | **HIGH**（Ch1/2/3 的 "stratified" 前提塌掉） |
| Ch3 plausibility bound | JHB clean_gt 分布派生 | 复用 JHB-derived bound；CT V3-C aerial_2025 已跑 46 grid (5 high-sev small-fragment flag) | **MEDIUM**（bound 是 JHB 先验，CT 有独特碎片失败模式） |
| Ch3 屋顶分母 (Overture) | 有 joburg.parquet | **Overture 无 Cape Town** → adoption-rate bound 算不了 | **MEDIUM** |
| Ch4 外部 admin | 此处未映射 | **CT 最富**：`data/sseg_registration_geo.csv` ✅ **17,176 数据行 (17,177 行含 header)**，16,503 geocoded, ward/H3, 6,853 有 date_commissioned；`sseg_kw_calibration.py` 已 fit | **LOW (数据)**，但 `external_agreement_grid.py` 未建，且是 household-offset/urban-biased 仅 supporting |

**结论**：Ch3 (`grid_plausibility.py` 已建已跑) 与 Ch4 (CT SSEG，待建 `external_agreement_grid.py`) 现在可作 CT 证据包；**Ch1 + Ch2 是两个硬 blocker**（工具与 CT GT 双缺，v4_high 的 0.95 precision 目标与任何 CT recall 声明都无法证实）；grid_strata.csv 必须先落地。census GT 缺口不阻塞「v3c vs unified_A 排名」（那只需 independent_26 + 影像），但阻塞「选定模型的 census 验证」。

---

## 7. 分步执行计划

### Phase 0 — registry 卫生 (前置, 阻塞排名工具识别)
1. ✅ 把 `exp_unified_reviewall_A` 加进 `configs/model_registry.yaml`（checkpoint=`checkpoints/exp_unified_reviewall_A/best_model.pth`, region_scope ct+jhb）。**硬 blocker**：grep 核实 registry 当前无此 key，run_benchmark.py:83-84 会对未注册 key `[WARN] skipping`。
2. 修/退役 stale registry 条目：`unified_reviewall_A_aerial_2025` (grid_count=26 但磁盘仅 G1240/G1243，无 finalized gpkg) 与 `v3c_geid_experiment` (实为 JHB GEID tile 误标 CT, bulk 3.63 artifact)。
3. 确认 `exp_unified_reviewall_A` 训练 config/handoff 的 exclude 列表覆盖全 26 ranking grid（§2.4）；未确认前 unified_A 排名结果标 🔶。

### Phase 0.5 — Li CT GT 纳入 (前置, gating 43-grid ranking) — **新增 2026-06-04**
1. **注册 Li 独立 annotation_scheme**：在 regions.yaml `cape_town.annotation_schemes` 下加 `li:`（own `task_kml` = Li KML / own `coverage_grids` / own `annotations_dir`，如 `data/annotations/Capetown_Li/`），并给 aerial_2025 imagery_layer 的 coverage 补上 Li 的 17 grid（或单列 li-scheme coverage）。**不要把 Li 的 G18xx/G19xx 当 Gao cell**。
2. **规整 gpkg 入库**：把 Dropbox 源（`/mnt/c/Users/gaosh/Dropbox/RA_Solar/Li/capetown/`）复制到 scheme 目录，重命名为干净 grid ID（去空格/全角括号），解压 `G1896(202).zip`，记 `label_source=human_manual_sam_assisted` / A2 / T2。括号里的数 = polygon count，入库后核对。
3. **下载 16 grid 的 aerial_2025 瓦片**（仅 Li-G1895 已在盘）：按 Li KML grid 几何从 CT 市政 WMS 下载（免费、限速）。**先按 RunPod 规则 `ls /workspace` 查现成底图复用**；下载产物经 **S3** 推到 pod（rule 08，禁 scp 大文件）。
4. 下载后 spot-check 1–2 grid 瓦片与 Li 多边形目视对齐（确认 WMS 仍服务同一 2025 vintage）。

### Phase 1 — 重跑 43-grid 推理 (RunPod, 阻塞排名) — **命令已修正**
✅ 现状：26 个 independent_26 grid 只有 3 个 (G1570/71/72) 有 v3c finalized；unified_A 在 CT 上 0 个 finalized（其 CT model_run dir 仅 G1240/G1243 子目录、无 `predictions_metric.gpkg`）；17 个 Li grid 全 0。**必须在 RunPod 上重跑全 43 grid × 2 模型 × 2 merge_mode**（independent_26 缺 23 + Li 17）。重跑根因是**磁盘上没有 finalized predictions**，不是 collect-only 限制（见 §8 Risk #7）。

**RunPod 数据准备（rule 05/08）**：先 `ls /workspace/tiles/cape_town/aerial_2025/` 查现成底图复用；缺的瓦片（含 Phase 0.5 下载的 Li grid）经 S3 推到 pod；跑前把热瓦片 `cp` 到 `/dev/shm`（network volume IO 慢 10–50×）；`export SOLAR_TILES_ROOT=/dev/shm/tiles`；PARALLEL=6 @5090（每进程 ~3–4GB VRAM），长任务 `nohup`/tmux + `python3 -u`；结果用 `aws s3 cp`（≈12 MB/s）+ `scripts/pack_and_pull_pod_results.sh` 拉回。用**路径 A（共享 raw + 两次 finalize）**：

```bash
# (a) 每个 (模型, grid) 只跑一次 detect_direct.py 产 raw_detections.pkl
SOLAR_TILES_ROOT=/dev/shm/tiles python3 detect_direct.py \
  --grid-id <G> --region cape_town --imagery-layer aerial_2025 \
  --model-path checkpoints/exp003_C_targeted_hn/best_model.pth \
  --output-dir results/cape_town/_raw/v3c_indep26/<G> --force

# (b) 同一 raw + 同一 ranking config(v4_canonical) 跑两次 finalize, 仅变 --merge-mode
python3 finalize.py \
  --input results/cape_town/_raw/v3c_indep26/<G>/raw_detections.pkl \
  --postproc-config configs/postproc/v4_canonical.json \
  --merge-mode pixel-or \
  --output-dir results/cape_town/v3c_indep26_pixelor/<G>
python3 finalize.py \
  --input results/cape_town/_raw/v3c_indep26/<G>/raw_detections.pkl \
  --postproc-config configs/postproc/v4_canonical.json \
  --merge-mode per-detection \
  --output-dir results/cape_town/v3c_indep26_perdet/<G>
# 注: finalize.py:84 是真实 --merge-mode 标志; 它覆盖 config 里的 merge_mode 键(v4_canonical pin pixel-or),
#     所以 per-detection 那一遍必须显式 --merge-mode per-detection 才不会被 config 的 pixel-or 覆盖回去。
```
对 `unified_A` 重复（model-path 换成 unified_A checkpoint，output-dir 换名）。把 4 个新 batch（v3c×2mode, unified_A×2mode）各注册成正式 regions.yaml model_run，便于 area_aggregate_eval 与 run_benchmark --collect-only 收集。

> **不要**写 `detect_and_evaluate.py --merge-mode`（无此标志）或指望 `run_benchmark --predictions-source direct` 切 merge_mode（不暴露此开关）。merge_mode 的唯一 CLI 入口是 `finalize.py --merge-mode`。

### Phase 2 — polygon F1 诊断排名 (本地或 pod) — **config 一致性已修正**
```bash
# 用 --collect-only 收集 Phase 1 已 finalize 的 predictions(同一 v4_canonical config + 指定 merge_mode),
# 保证 polygon-F1 与 Phase-3 area 在同一 postproc/merge_mode 上可比。
python3 scripts/analysis/run_benchmark.py \
  --models v3c exp_unified_reviewall_A \
  --suite cape_town_independent_26 \
  --collect-only
# 读 results/benchmark/<run_id>/summary.json -> winner/model_verdicts; by_grid.csv 看 per-grid
```
- ✅ **若不用 --collect-only 而让 run_benchmark 自己推理**：它会用 preset 默认 `batch003_best_f1.json`（post_conf 0.92, 无 merge_mode → per-detection），**与 Phase-3 的 v4_canonical 不可比**。必须覆盖 preset 的 postproc_config 指向 ranking config，或坚持 --collect-only 路径。
- polygon F1 诊断用，不作最终裁判。

### Phase 3 — area-aggregate Tier-1 主裁判 (✅ flags 核实) — **CT GT 路径已修正**
```bash
# 注册 4 个 model_run 后, area_aggregate_eval 直接读各 run dir 的 predictions_metric.gpkg
# (无需 benchmark run)。CT GT 用 AUTO-DISCOVERY (默认偏好 _SAM2_), 不要传 --gt-root/--gt-pattern。
python3 scripts/analysis/area_aggregate_eval.py \
  --region cape_town \
  --run v3c_indep26_pixelor v3c_indep26_perdet \
        unifiedA_indep26_pixelor unifiedA_indep26_perdet \
  --skip-deprecated
# 读 per_run_summary.csv: 主看 σ_Bw + RMSE, bulk∈[0.5,2.0] gate, F1 sanity
```
> **审稿 high #3 修正**：初版的 `--gt-root data/annotations/Capetown --gt-pattern "{grid}/{grid}_SAM2_*.gpkg"` **对 CT 完全跑不通**——✅ CT GT 是 **flat 布局**（94 个 `Gxxxx_SAM2_*.gpkg` 直接在 `data/annotations/Capetown/` 下，0 子目录），而 `_gt_spec_for` (area_aggregate_eval.py:187-191) 对 override 做 `gt_root / gt_pattern.format(grid)` 后 **literal `.exists()`，不 glob**；pattern 含 `{grid}/` 子目录 + `*` 通配会让每个 grid 都 None→skip。该脚本默认 pattern 是 `{grid}/{grid}_clean_gt.gpkg`，而 CT **0 个 clean_gt**。
>
> **CT 唯一可行路径 = auto-discovery（不传 --gt-root）**：`_discover_gt` (line 141-162) glob `{grid}*.gpkg` 并按 `_GT_PRIORITY_SUFFIXES=("_SAM2_",...)` 优先选 SAM2，正确命中 flat 文件。
>
> 🔶 旁注：本批 26 ranking grid 实际共享 `_260322` 后缀（disk 核实 G1240/1243/1244/1245/1293/1294/1297/1298/1299 等均为 `_SAM2_260322.gpkg`），故理论上 flat-layout override `--gt-pattern "{grid}_SAM2_260322.gpkg"`（无子目录、无通配）能匹配此子集；但 auto-discovery 更稳健（不依赖后缀人工核对），仍是推荐做法。

判定：**bulk 越界先淘汰 → 剩下比 σ_Bw + RMSE → 平手看 thru0_β/R²**；对每个候选的两个 merge_mode 行都比。

### Phase 4 — 重跑旧 area_aggregate (补 Tier-1 列)
✅ 2026-04-22 的 v3c CT area_aggregate (39 grid, bulk 1.35, R² 0.926) **早于 σ_Bw/log-σ/RMSE/thru0 列**，必须用当前 area_aggregate_eval.py 重跑才有 paper-primary dispersion 指标。

### Phase 5 — 选型决定 + census 校准 (排名之后，独立)
- 选定模型后，census 聚合产物跑 `v4_agg.json`，backdating seed 跑 `v4_high.json`（**先在 CT 重新校准阈值**，因现有校准是 JHB-only）。
- ✅ **显式 pin census merge_mode（决策 4 已确认）**：v4_agg/v4_high 当前不设 merge_mode 键 → 默认 per-detection；用 Phase 3 排名 (σ_Bw+RMSE) 胜出的 mode 在 config 里显式写 `"merge_mode": ...`（或 finalize 时显式传），别留给默认。
- census 验证证据包：Ch3 (`grid_plausibility.py`) + Ch4 (待建 `external_agreement_grid.py` on CT SSEG)。
- 标注 Ch1/Ch2 为未交付 blocker。

### Phase 6 — census 扩量前置 (与模型选择正交)
✅ aerial_2025 当前只 tile **120 grid dir**（disk 核实 `~/zasolar_data/tiles/cape_town/aerial_2025/`）/ ~2214 task grid (~5%)；其余走 Cape Town 市政 WMS（免费、仅限速）下载，是全量 census 的 gating dependency。

---

## 8. 风险与坑

1. **不要把 internal benchmark suite 误称 V1.4 validation**（validation_strategy.md:117 明令）。benchmark 排名 ≠ census 验证。
2. **smoke 的 `leakage_risk: low` 是误导**——它对每个 CT-trained 模型都泄漏，那个 "low" 指 regression 用途，不是排名判断。
3. **不要只看 bulk_ratio**（Goodhart, 非单调；paper 真正怕 per-grid σ）；裁判是 σ_Bw + RMSE。
4. **不要把 polygon-area-sum 当 area metric**（SAM-supp 碎 GT 上偏置）；用 set-theoretic union（area_aggregate 已是 `unary_union`）。
5. **unified_A 的强数字都是 Vexcel 域**，不能迁到 CT aerial_2025；其 CT eval 从未跑完 + exclude 列表未 config 确认，是纸面挑战者。
6. **每个 checkpoint 必须双 merge_mode 都评，但 run_benchmark 自己不 sweep merge_mode**：`detect_and_evaluate.py` 无 `--merge-mode`，benchmark 默认 `batch003_best_f1.json` 无 merge_mode 键→per-detection 单 mode。双 mode 只能靠 `finalize.py --merge-mode`（路径 A：共享 raw 两次 finalize）或两个 postproc config。单 mode 结论无效。
7. **现有磁盘结果不足以排名**（26 grid 仅 G1570/71/72 有 v3c finalized；unified_A 0 个）；必须重跑——根因是**磁盘缺 finalized predictions**。**注意两套独立 collector**：`run_benchmark --collect-only` 只能收 prior benchmark run（polygon F1）；而 `area_aggregate_eval._load_run_grids` 能**直接读已注册 model_run dir 的 predictions_metric.gpkg**（area 裁判）。别把两者的限制混为一谈。
8. **unset merge_mode 默认 = per-detection**（finalize.py:525），不是 pixel-or。census config v4_agg/v4_high 当前都隐式落在 per-detection——必须显式 pin。
9. **CT GT 是 A2 sub-array，非 A1 gold**——不要在选型文档声称 CT T1 gold；这影响所有"gold GT"措辞。
10. **CT GT 是 flat 布局**：area_aggregate_eval 的 `--gt-root/--gt-pattern` override 做 literal `.exists()` 不 glob，含子目录或通配会全 skip；CT 必须用 auto-discovery。默认 pattern 找的是 CT 不存在的 clean_gt。
11. **registry stale**：model_registry.yaml 只有 6 个 model、漏 unified_A；用 regions.yaml model_runs + checkpoints/ 磁盘 listing 作权威 inventory。
12. **休眠 bug**：annotation_loader fallback sorted-glob first-wins，未来未注册 dual-file grid 会静默选劣质 bare 文件。
13. **v3_cleaned 是未收敛 paused checkpoint (ep4/20)**——已从排名 + 推理矩阵移除；不烧推理，除非训练完成 + exclude 列表 config 确认。
14. **ranking postproc config 必须全程一致**：polygon-F1（Phase 2）与 area-aggregate（Phase 3）若分别落在 batch003_best_f1.json 与 v4_canonical.json 就不可比。锁定单一 config（v4_canonical），只变 merge_mode。

---

## 9. 决策点 (✅ 全部已确认 2026-06-04 — 见顶部「决策状态」表)

1. **排名 ranking postproc config?** → ✅ **用 v4_canonical.json**。 — 选项：`v4_canonical.json`(推荐) / `batch003_best_f1.json`(现 preset 默认) / 新建 CT 专属。**建议**：用 `v4_canonical.json` 并显式覆盖 merge_mode 跑两遍；务必让 run_benchmark 走 `--collect-only` 读 Phase-1 finalize 产物，否则回落 batch003_best_f1 与 area 裁判不可比。

2. **v3_cleaned (ep4/20 paused) 是否进推理矩阵?** → ✅ **排除**，不烧推理。

3. **unified_A 排名是否需先 config-确认 exclude 列表?** → ✅ **是**。发布排名前必须从训练 config/handoff 确认全 ranking grid 被排除；确认前所有 unified_A 排名结果标 🔶。

4. **census 阶段 (v4_agg/v4_high) 用哪个 merge_mode?** → ✅ **用 Phase 3 排名胜出的 mode + 在 config 显式 pin `merge_mode` 键**（不留 per-detection 默认）。

5. **是否纳入 Li CT GT 扩展 ranking 面?** → ✅ **全量纳入主排名**（独立 `li` scheme + 下载 16 grid 瓦片 + 规整 gpkg），排名面 26 → **43 grid**。详见 §5.1 / Phase 0.5。

> 计算环境：✅ 全部推理在 **RunPod** 上跑（Phase 0.5 瓦片走 WMS→S3→pod；Phase 1 tiles→/dev/shm + PAR=6 + tmux + S3 拉回）。
