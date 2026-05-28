# ZAsolar 代码库优化方案

**日期**: 2026-05-29
**方法**: 9 个子系统并行代码评审 (覆盖全部 ~52K LOC / 154 个 Python 文件) → 20 条高严重度 finding 全部经对抗式复核 (0 条被推翻) → 综合排序。头部 7 条结论已由主 agent 亲自回读源码二次确认。
**统计**: 85 条 finding 存活 (复核后 severity: high 2 / medium 50 / low 33)。

> 本文是评审产出的快照。执行时勾选下方 checklist，finding id 可回溯到原始评审。

---

## 1. 总体评估

三条核心张力：

1. **frozen monolith `detect_and_evaluate.py`** (2154 行) 用 ~25 个 module global + `set_grid_context` 副作用驱动，导致后处理逻辑被复制 3–4 份、不可重入、且与新 direct pipeline 写出**互不兼容的 config.json**。
2. **35K LOC 的脚本沉积** — `scripts/analysis`/`scripts/training` 累积大量 one-off：spatial-NMS 5 份、union-merge 4 份、area-F1 3 份、Wilson CI 2 份，且 **12/35 个 analysis 脚本硬编码 EPSG:32734/32735**，直接威胁 multi-city census 目标。
3. **测试覆盖严重偏斜** — postproc 有 7 个测试文件，而 `train.py`/`detect_direct.py`/`get_metric_crs`/分类器防泄漏 split 全部零覆盖，且**无任何 CI/pre-commit gate**。

好消息：`core/` 包本身工程质量高，direct pipeline 干净，**没有被证实的灾难性 bug**。

---

## 2. 核心架构问题 (P0)

### P0-1 双推理路径 config.json schema 不兼容 — `infeval-4`
legacy 写 `{config, artifacts, evaluation_config}`，direct 写扁平 `{pipeline_version, stage_counts}`，零共享顶层 key；两路都写 `results/` 且 grid ID 跨 region 重叠 → census 量产时目录可能被另一路径误判 reuse/overwrite。
**文件**: `detect_and_evaluate.py:294-317,399`、`finalize.py:581,866-921`。
**路径**: 抽 `core/inference/run_config.py` 统一契约；至少让各自 guard **显式识别并拒绝**对方 schema 而非静默误判。
**约束**: 不改 legacy 检测语义，只统一 provenance 元数据层。**先写 characterization test 锁住两种 schema 再动。**

### P0-2 后处理逻辑散落 5 份 — `infeval-1` + `analysis-postproc-1/4`
`spatial_nms` 在 `detect_and_evaluate.py:414` 与 `core/postproc.py:115` 字节级重复，parity test 又快照第三份；`calibration_sweep.py`/`param_search.py`/`multi_grid_baseline.py`/`export_hints.py` 仍 `from detect_and_evaluate import spatial_nms`，让 dead inline code 保持 load-bearing。posthoc 脚本再加 3–4 份 union-find。
**路径**: `core/postproc.py` 设为唯一来源，monolith 改 `from core.postproc import spatial_nms`；新增 `union_merge_overlapping(gdf, iou, score_col)` 供 posthoc 复用；parity test 改 import live 函数。
**回归防护**: 先跑 parity test 确认数值不变再删 inline（larger-polygon-wins vs higher-score-wins 语义不同会改 inventory）。

### P0-3 multi-city CRS/region 硬编码系统性扩散 — `infeval-6, pipeline-3, anlyeval-2/3, cls-1/3, anno-gui-1/4, xhealth-7`
违反 rules/06 的最高杠杆问题，且**新增 untracked 脚本 (posthoc_*、combine_merge01、gemini chip builder) 仍在复制**。三种形态：硬编码 UTM zone、`lookup_region` 单匹配歧义、按经度阈值推断 region (`clean_annotations.py:124` `centroid_lon > 25`)。
**路径**: 统一走 `get_metric_crs(grid_id, region=)` + `lookup_regions` 显式消歧；`pipeline/hn_ops.py` 把已校验的 `hn_spec.region` 线程进 `extract_reviewed_fp_hn`（当前被丢弃重新按 grid_id 猜）。
**最优先**: 先修 4 个 untracked posthoc 脚本，别让违规进 git 历史。

### P0-4 分类器在 mosaic layout 上静默 no-op — `cls-1, cls-2`
`classify_predictions.py:152 _find_tile` 只 glob chunked `{grid}_*_*_geo.tif`，无 mosaic 分支、无 region — 而 v2 分类器恰恰在 `geid_2024_02` (mosaic) 上标定。结果：tile lookup 失败 → 全部静默当 PV 保留 (`cls_score=1.0`)。`thresholds_v2.json` 无生产 reader，per-layer 标定从未上线。
**路径**: 复用 `build_cls_dataset.extract_chip` 的 region-aware + `is_file()` mosaic 分支，加 `--region/--imagery-layer`，加载 `thresholds_v2.json` 按 layer 应用阈值；**零 chip 抽取时 raise/warn**。
> 注：分类器即将从主仓抽出（见独立方案），本项随抽出一并处理。
> ✅ **2026-05-29 DONE**：分类器已抽到兄弟子仓 `solar_cls`；P0-4 落地 —
> `classify_predictions._find_tile` 改用 `resolve_tiles_dir(grid_id, region=, imagery_layer=)`
> + `is_file()` mosaic 分支（镜像 `build_cls_dataset.extract_chip`），新增
> `--region/--imagery-layer`，`thresholds_v2.json` 按 `(arch, imagery_layer)` 查表消费
> （`--pv-threshold` 仍可覆盖）；`compare_results.py` 的 `RESULTS_DIR` 也改锚 `ZASOLAR_ROOT`
> （handoff 漏列，读的是主仓 detector 预测）。

### P0-5 train.py 无 seed/determinism — `train-1`
所有 A/B 模型选择依赖 sub-percent F1 delta (V3-C vs train20、phaseA pass/fail)，但 `train.py` 零 seeding，shuffle/augmentation/cuDNN 全随机 → run-to-run variance 被静默折进每个"这个 checkpoint 更好"的决策。
**路径**: `main()` 顶部加 `--seed`(默认 42) 设 torch/np/random + DataLoader `Generator`，可选 `--deterministic`，写入 `training_history.json`。**零风险，不碰 supervision 逻辑。**

---

## 3. 分主题优化项

### 架构 / 解耦
- **P0** 抽 `core/inference/run_config.py` 统一 config.json (M) — `infeval-4`
- **P1** 纯数学 eval helper (`iou_matching`/`compute_iou`/`evaluate_*`) 提到 `core/eval/matching.py`，monolith re-import 保兼容 (M) — `infeval-2`
- **P1** `pipeline/` 明确定位：production 走 `build_unified_reviewall.py`，把 `pipeline/` 降级或仅提取 `validate_equivalence`，写进 CLAUDE.md (L) — `pipeline-1`
- **P2** `classify_predictions.model_path_global` 改显式参数 (S) — `cls-7`

### 重复代码消除
- **P0** spatial_nms / union-merge 收敛到 `core/postproc.py` (M) — `infeval-1` + `analysis-postproc-1/4`
- **P1** HN merge + chip-extraction 5 份合一到 `pipeline/hn_ops.py` (M) — `pipeline-2`
- **P1** 3 个 posthoc_* 合成单 CLI (`--op {nms,merge} --area-f1`)，cbd25_ids 读 regions.yaml (M) — `anlyeval-4` + `analysis-postproc-2`
- **P1** 抽 `core/eval_stats.py`(wilson/set-theoretic PRF) + `core/training/transforms.py`(共享 aug + masks_to_boxes) (S/M) — `anlyeval-7` + `train-2`
- **P2** SAM segment 栈抽 `core/sam_segmenter.py` 供两个 review GUI 共用 (M) — `anno-gui-2`；WMS 配置进 regions.yaml (S) — `anno-gui-6`

### 正确性 bug
- **P0** `dissolve_hairline_gaps` 空/全 NaN 列 `max()` 崩溃 → guard (S，已核实未修) — `core-1`
- **P0** 分类器 mosaic no-op + thresholds_v2 未消费 (M) — `cls-1/2`
- **P1** `export_coco_dataset.split_tiles` 贪婪 val 分配可严重 overshoot val_fraction → 偏置 val 指标 (M) — `export-2`
- **P1** `clean_annotations.py` 经度阈值定 UTM zone (S) — `anno-gui-4`
- **P2** `_arcgis_fetch.py:133` f-string `[:120]` 在字面量内永不截断 (S) — `anno-gui-7`
- **P2** `train.py:379` 死/错的 area re-index 三元；`evaluate_coco` 硬编码 400px fallback (S) — `train-3/4`
- **P2** 收窄 `except (KeyError, Exception)`、imported helper 用 raise 替代 `sys.exit` (S) — `infeval-8`

### 性能 (census 量产)
- **P1** `evaluate_at_multiple_thresholds` 对同 gt/pred 跑 ~12 次 IoU matching → per-GT merged-IoU 算一次复用 (M) — `infeval-5`
- **P2** posthoc `greedy_nms` 每个 polygon 重建 STRtree O(n²logn) → 持久 sindex (S) — `analysis-postproc-7`
- **P2** `assign_annotations_to_tiles` O(ann×tiles) → STRtree/sjoin (S) — `export-3`

### 测试补强
- **P0** 加单个 GH Actions / pre-push：`pytest -q` + `validate_registry.py`(CPU ~21s) (S) — `xhealth-1`
- **P0** `get_metric_crs` table-driven 测试 (CT→32734 / JHB→32735 / 北半球边界) (S) — `xhealth-3`
- **P1** 分类器防泄漏 split + `threshold_at_pv_recall` 单测 (S) — `cls-5`
- **P1** detect_direct CPU smoke + grid_utils/annotation_loader CRS 启发式测试 (M) — `xhealth-3` + `core-8`
- **P1** 加 `pytest.ini` `testpaths = tests`，jhb_phaseA 测试移入或 mark slow (S) — `xhealth-9`

### 配置 / 可复现性
- **P0** 重生成 `requirements.lock.txt`(cu126→cu128、补 pytest、加硬件 header) (S，已核实仍 cu126) — `xhealth-2`
- **P0** `train.py` seed (S) — `train-1`
- **P1** postproc JSON 加 `extends` 继承（4 个 v4_canonical 变体只留 delta）(M) — `xhealth-5`
- **P1** 删 `building_filter.py`(368L 孤儿，零 importer) + monolith 内 buildings.gpkg dead block (S) — `infeval-3` / `xhealth-4`
- **P1** 删 `imagery_sources.yaml`(零 reader) + 退役 `benchmark_weights.py`，更新 architecture.md (S) — `xhealth-6` / `anlyeval-1`
- **P2** `start-*.sh/.bat` 加 .gitignore (S) — `xhealth-8`；manifest HN-shortlist 用 `_normalize_path` (S) — `pipeline-8`

---

## 4. 执行顺序 (Phased Roadmap)

> 关键纪律：**先建测试安全网，再动 frozen monolith。**

**Phase 0 — 零风险快赢 (~1 天)**: train seed、NaN guard、requirements.lock 重生、删 building_filter + dead block、arcgis 截断、start-* gitignore。无依赖，立即做。

**Phase 1 — 安全网先行**: CI gate + `get_metric_crs` 测试 + 在重构 P0-1/P0-2 **之前**写 characterization test（锁双 config schema、锁 postproc 数值 parity）。

**Phase 2 — multi-city 正确性 + 解耦**: P0-3（先修 untracked posthoc 再 commit）、P0-4 分类器 mosaic（随分类器抽出处理）、P0-1 config 统一、pipeline-3 region 线程。依赖 Phase 1 CRS 测试。

**Phase 3 — 去重整合**: 收敛 spatial_nms、HN merge、posthoc 合并、`core/eval_stats.py`。依赖 Phase 1 parity test。

**Phase 4 — 性能 + 收尾**: IoU 复用、STRtree、postproc `extends`、pipeline 定位决策。

> **并行进行的两项独立结构化工作** (见各自方案文档)：
> - **分类器抽出主仓** — cls 仍为探索性、未进生产，从主仓解耦。包含 P0-4 的修复。 ✅ **2026-05-29 DONE**（→ `/home/gaosh/projects/solar_cls/`）。
> - **训练集资源规范化** — 按正样本(边界可信/不可信) + 负样本(HN) 建 pool，并建立 per-run 参数账本。

---

## 5. 不要动的东西 / 风险清单

- **monolith V1.3 frozen 评估语义**: `evaluate_predictions.py` 复用是 intentional (decision #22)。重构只提取**纯函数**到 `core/eval/`、re-import 保兼容、**不改数值**。
- **installation-level GT 语义 + `installation` profile 的 pred-side many-to-one merge**: 不降级 panel-level，不静默切 profile。
- **COCO 空目标 chip = intentional hard negative**: 去重时禁止 drop。
- **SAM 边缘噪声**: cleanup 只修 merge/area，不动边缘噪声。
- **finalize.py `--parity-mode geoai`**: 是 frozen path 的忠实镜像，保留 (`infeval-7`)，只抽共享子步骤。
- **回归风险**:
  - (a) 收敛 spatial_nms 后必须 parity test 验证 keep-semantics。
  - (b) 统一 config.json 须保留对旧 schema 的兼容读，否则旧 results 被新 reuse-check 误判。
  - (c) seed 引入改变历史 checkpoint 的 bit-reproducibility，但不影响已发布权重。

---

## 附录 A — 分阶段 Checklist

### Phase 0 — 零风险快赢
- [ ] `train-1` train.py 加 `--seed` + determinism (S)
- [ ] `core-1` dissolve_hairline_gaps NaN `max()` guard (S)
- [ ] `xhealth-2` 重生成 requirements.lock.txt (cu128 + pytest) (S)
- [ ] `infeval-3` / `xhealth-4` 删 building_filter.py + monolith dead load block (S)
- [ ] `anno-gui-7` _arcgis_fetch f-string 截断修复 (S)
- [ ] `xhealth-8` start-*.sh/.bat 加 .gitignore (S)

### Phase 1 — 测试安全网
- [ ] `xhealth-1` CI / pre-push gate: pytest + validate_registry (S)
- [ ] `xhealth-3` get_metric_crs table-driven 测试 (S)
- [ ] characterization test: config.json 双 schema 读写
- [ ] characterization test: postproc spatial_nms 数值 parity

### Phase 2 — multi-city 正确性 + 解耦
- [ ] `infeval-4` 抽 core/inference/run_config.py 统一 config.json (M)
- [ ] `infeval-6` 移除 evaluator 默认 32734 fallback (S)
- [ ] `anlyeval-2` / `analysis-postproc-3` / `xhealth-7` analysis 脚本去硬编码 CRS (M)
- [ ] `anlyeval-3` repostprocess.py 去绝对路径 + results_joburg 引用 (M)
- [ ] `anno-gui-1/4/5` review/clean 脚本去 region 推断 (S/M)
- [ ] `pipeline-3` hn_ops 线程 region 替代 lookup_region (M)
- [ ] `core-2/3` 收敛 region-alias + 弃用 lookup_region (M)
- [ ] `cls-3` 分类器去硬编码 CRS（抽出已完成 2026-05-29 → `solar_cls`；CRS 去硬编码留作子仓内后续）

### Phase 3 — 去重整合
- [ ] `infeval-1` + `analysis-postproc-1` spatial_nms 收敛到 core/postproc (M)
- [ ] `analysis-postproc-4` + `anlyeval-4` posthoc union-merge 收敛 + 合成单 CLI (M)
- [ ] `pipeline-2` HN merge + chip 抽取合一 (M)
- [ ] `anlyeval-7` + `analysis-postproc-5` 抽 core/eval_stats.py (S)
- [ ] `train-2` 抽 core/training/transforms.py (M)
- [ ] `core-4` / `cls-6` 收敛 sliding-window chip 逻辑 (M)
- [ ] `anno-gui-2` 抽 core/sam_segmenter.py (M)
- [ ] `xhealth-5` postproc JSON extends 继承 (M)
- [ ] `xhealth-6` / `anlyeval-1` 删 imagery_sources.yaml + 退役 benchmark_weights.py (S)

### Phase 4 — 性能 + 收尾
- [ ] `infeval-5` 评估 IoU matching 复用 (M)
- [ ] `analysis-postproc-7` greedy_nms 持久 sindex (S)
- [ ] `export-3` assign_annotations_to_tiles STRtree (S)
- [ ] `infeval-2` set_grid_context module-global 解耦 (M)
- [ ] `export-1` build_base_coco god-function 拆分 (M)
- [ ] `anno-gui-3` review_detections.py build_html 878 行内联拆出 (L)
- [ ] `pipeline-1` 决定 pipeline/ 去留并写进 CLAUDE.md (L)

---

## 附录 B — 全部存活 finding（按 slice）

| id | sev | category | effort | 标题 |
|---|---|---|---|---|
| core-1 | M | correctness | S | dissolve_hairline_gaps all-NaN 列 max() 崩溃 |
| core-2 | M | duplication | M | region-alias 三份 map，short vs canonical 不一致 |
| core-3 | M | correctness | M | 弃用 lookup_region() 仍用于 core 路径解析 |
| core-4 | M | duplication | M | chip sliding-window 三份实现，edge-coverage 语义不同 |
| core-8 | M | testing | M | grid_utils/annotation_loader/training 子包测试薄弱 |
| core-5 | L | dead_code | S | jhb_phaseA 训练栈 live 但 memory 记为 abandoned，待澄清 |
| core-6 | L | maintainability | S | annotation_loader 按 grid-ID 前缀推断 schema/region |
| core-7 | L | correctness | M | boundary_aware_mask module-global 非 DDP/线程安全 |
| core-9 | L | maintainability | S | jhb_phaseA_dataset import 时 sys.path.insert 副作用 |
| infeval-1 | M | duplication | M | postproc 三份分叉，仅一份 canonical |
| infeval-2 | M | architecture | M | set_grid_context 改 ~25 module global，全靠副作用复用 |
| infeval-3 | M | dead_code | S | building_filter.py 孤儿，每次推理 load 但 unused |
| infeval-4 | M | reproducibility | M | 两 pipeline 写互不兼容 config.json 到同一 results/ |
| infeval-5 | M | performance | M | 评估对同 gt/pred 跑 12+ 次 IoU matching 无复用 |
| infeval-6 | M | correctness | S | evaluator module fallback 硬编码 EPSG:32734 |
| infeval-7 | L | duplication | S | finalize.py --parity-mode geoai 第四份 postproc 变体 |
| infeval-8 | L | correctness | S | 过宽 except、lib 内 sys.exit、冗余 tuple |
| train-1 | H | reproducibility | S | train.py 无 seed/determinism，训练不可复现 |
| train-2 | M | duplication | M | aug + masks_to_boxes 在 train.py 与 PhaseA transforms 重复 |
| train-3 | L | dead_code | S | TrainTransforms scale-jitter 死/错 area re-index 三元 |
| train-4 | L | correctness | S | evaluate_coco 空 val 图硬编码 400px fallback |
| train-5 | L | maintainability | S | early-stop 暴露 evaluate_coco 不产出的 area_f1/ch2_recall key |
| train-6 | L | config | S | --pretrained 默认是机器上多半不存在的相对路径 |
| train-7 | L | correctness | S | Stage-2 resume 按 per-epoch 假设推进 per-batch scheduler |
| train-8 | L | architecture | M | _BATCH_STATE/_BOX_LOSS_BUCKETS global 耦合 patched transform 与 loss patch |
| export-1 | M | maintainability | M | build_base_coco 280 行 god-function |
| export-2 | M | correctness | M | split_tiles 贪婪 val 分配在小 grid 上严重 overshoot |
| export-3 | L | performance | S | assign_annotations_to_tiles O(ann×tiles) 无空间索引 |
| pipeline-1 | M | architecture | L | 声明式 builder 孤儿，production 绕过 pipeline/ |
| pipeline-2 | M | duplication | M | HN merge + chip 抽取 copy 4 份 + hn_ops |
| pipeline-3 | M | correctness | M | hn_ops 调用路径上 lookup_region + 硬编码 32734 |
| pipeline-5 | M | testing | M | pipeline/ 零测试覆盖 |
| pipeline-6 | M | maintainability | M | validate_equivalence 按 magic ID range/前缀推断 |
| pipeline-4 | L | dead_code | S | export_v4_1_hn.py 死 no-op id-remap block |
| pipeline-7 | L | performance | S | build_dataset 两次读 train.json，冗余算 n_hn |
| pipeline-8 | L | reproducibility | S | HN shortlist CSV 写绝对路径，他处写归一化路径 |
| pipeline-9 | L | duplication | M | gemini chip builder 重复 solar_backdating 的聚类 |
| pipeline-10 | L | reproducibility | S | spec_to_dict asdict 与 resolved_spec 分开 resolve，build-id 可能分叉 |
| anlyeval-1 | M | duplication | S | benchmark_weights.py 是 run_benchmark.py 的陈旧副本 |
| anlyeval-2 | M | config | M | metric CRS 硬编码于 9+ analysis 脚本 |
| anlyeval-3 | M | config | M | repostprocess.py 硬编码绝对路径 + 已删 results_joburg |
| anlyeval-4 | M | duplication | M | 三份 merge/UnionFind + spatial_eval + inline CBD25 列表 |
| anlyeval-5 | M | correctness | M | posthoc 报 raw bulk_ratio，Tier-1 已 demote (Goodhart) |
| anlyeval-8 | M | architecture | M | param_search/multi_grid_baseline 硬编码 results/ + 依赖 module-global |
| anlyeval-9 | M | reproducibility | M | validate_checkpoint 硬编码 grid cohort/tiles/CRS/baseline 路径 |
| anlyeval-6 | L | dead_code | S | 死函数 determine_winner，winner 逻辑三份分叉 |
| anlyeval-7 | L | duplication | S | Wilson CI + area-F1 helper 在 ch1/ch2/posthoc 重复 |
| anlyeval-10 | L | performance | S | run_benchmark 并行 fan-out 无 per-grid GPU/VRAM guard |
| analysis-postproc-3 | M | config | S | analysis 脚本硬编码 UTM CRS |
| analysis-postproc-4 | M | reproducibility | M | 硬编码绝对数据路径，忽略 SOLAR_TILES_ROOT/grid_utils |
| analysis-postproc-5 | M | duplication | M | posthoc area-F1 重写 area_aggregate_eval，dispersion 数学分叉 |
| analysis-postproc-6 | M | duplication | M | postprocess_ablation 重写 core 已有的 filter + TP/FP/FN matching |
| analysis-postproc-7 | M | performance | S | greedy_nms 每个 polygon 重建 STRtree O(n²logn) |
| analysis-postproc-1 | L | duplication | M | 四份 spatial-NMS/merge，core/postproc 已提供 |
| analysis-postproc-2 | L | duplication | M | posthoc_nms/merge/combine_merge01 近同 untracked 一次性脚本 |
| analysis-postproc-8 | L | maintainability | M | trackers 按 path 子串推断 region/imagery layer |
| analysis-postproc-9 | L | performance | S | spatial_eval 每 grid 读两次 gpkg、逐次 reproject，无 CBD union 缓存 |
| cls-1 | H | correctness | M | 推理 chip 抽取非多城、不支持 mosaic（v2 标定层） |
| cls-2 | M | dead_code | M | thresholds_v2.json 产出但无推理路径读取 |
| cls-3 | M | config | S | 分类器/validation 脚本硬编码 region UTM CRS |
| cls-4 | M | reproducibility | S | train_cls.py config merge 静默覆盖等于默认值的 CLI flag |
| cls-5 | M | testing | S | 防泄漏 split + 阈值标定无单测 |
| cls-6 | M | duplication | M | tile-lookup + chip-extraction 三份分叉 |
| cls-7 | L | maintainability | S | model_path_global module-global 线程进 summary |
| cls-8 | L | correctness | S | compare_results 依赖位置 index==pred_id 未强校验 |
| cls-9 | L | correctness | S | build_large_polygon_review 按最近面积匹配可选错 polygon |
| anno-gui-1 | M | correctness | M | sam_fn_review 硬编码 UTM + 按 results-path 子串推断 region |
| anno-gui-2 | M | duplication | M | 两个 review GUI 重复 SAM 分割 + mask→polygon 栈 |
| anno-gui-3 | M | maintainability | L | review_detections build_html 878 行内联 HTML/CSS/JS |
| anno-gui-4 | M | correctness | S | clean_annotations 按经度阈值选 UTM zone |
| anno-gui-5 | M | config | M | review_detections 残留硬编码 JHB 路径 + results_joburg 推断 |
| anno-gui-6 | M | duplication | S | WMS 源/GetMap 参数在两脚本重复，CT 源硬编码于 regions.yaml 外 |
| anno-gui-9 | M | testing | M | 无测试覆盖任何 review GUI / 影像获取模块 |
| anno-gui-7 | L | correctness | S | _arcgis_fetch f-string [:120] 永不截断 |
| anno-gui-8 | L | dead_code | S | download_admin_imagery 引用不存在脚本 + 永久死 Vexcel 分支 |
| anno-gui-10 | L | config | S | expand_cut_polygons 硬编码 2 城 CRS dict，第三城 KeyError |
| xhealth-1 | M | testing | S | 无 CI/pre-commit gate 跑测试 |
| xhealth-2 | M | reproducibility | S | requirements.lock 错 CUDA build (cu126) + pytest 未 pin |
| xhealth-3 | M | testing | M | train.py/detect_direct.py/get_metric_crs 关键路径无测试 |
| xhealth-4 | M | dead_code | S | building_filter.py (368L) 孤儿 |
| xhealth-5 | M | config | M | 4 个 v4_canonical 变体重复 6 个基础字段，无继承 |
| xhealth-6 | M | config | M | imagery_sources.yaml 零 reader，三套影像 registry 并存 |
| xhealth-7 | M | config | M | 12/35 analysis 脚本硬编码 EPSG，新脚本延续 |
| xhealth-8 | L | config | S | start-*.sh/.bat 应 gitignore 而非提交 |
| xhealth-9 | L | testing | S | jhb_phaseA 下两个游离 test 被 pytest 自动收集跑前后向 |
