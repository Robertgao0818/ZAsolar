# Context Map — Agent 导向

Clean agent 接手时读此文件，快速了解关键文件和它们为什么重要。

## 核心流水线

| 文件 | 重要性 | 为什么 |
|------|--------|--------|
| `detect_and_evaluate.py` | 最高 | 主检测+评估入口；`installation` profile 定义在此；config.json 复用逻辑在此 |
| `export_coco_dataset.py` | 高 | COCO 数据集导出；auto-discover `Capetown/*_SAM2_*.gpkg`；empty-chip hard negative 逻辑在此。**当前仅 Cape Town**。 |
| `train.py` | 高 | Mask R-CNN 微调；两阶段训练；CUDA 强制验证 |
| `core/grid_utils.py` | 高 | Grid 路径/CRS 解析共享模块；内部委托 `region_registry`，外部函数签名稳定 |
| `core/region_registry.py` | 高 | 加载 `regions.yaml` 提供 `lookup_region()`, `get_region_config()`, `list_grids()` 等 API |

## 配置层

| 文件 | 重要性 | 为什么 |
|------|--------|--------|
| `configs/benchmarks/post_train.yaml` | 高 | Benchmark suite 定义（primary/diagnostic/secondary/smoke）+ auto-verdict 规则 |
| `configs/datasets/regions.yaml` | 高 | **权威区域注册表**：grid 列表、CRS、路径、grid_id_pattern。新城市先加此文件。 |
| `configs/datasets/training_sets.yaml` | 高 | 训练集 recipe provenance：region_scope、holdout、HN 来源。model_registry 引用此文件。 |
| `configs/model_registry.yaml` | 中 | 模型权重路径注册 + provenance（training_set_id, region_scope, postproc_config） |
| `configs/postproc/*.json` | 中 | 后处理阈值配置；影响评估结果 |

## 标注与数据

| 文件 | 重要性 | 为什么 |
|------|--------|--------|
| `data/annotations/ANNOTATION_SPEC.md` | 高 | GT 规范（installation-level）+ **Two-Axis Model**（A1-A3 × H/R/S/G）+ tier 规则 |
| `data/annotations/Capetown/` | 高 | Cape Town 清洗后 SAM2 标注，COCO export 的主数据源 |
| `data/annotations/Joburg/` | 高 | Johannesburg GT：JHB01-06 manual legacy + G07xx-G09xx CBD batch1 V4-reviewed |
| `data/annotations/annotation_manifest.csv` | 中 | 质量分层 T1/T2、review status、label_source、semantic_confidence |
| `data/annotations/PROGRESS.md` | 低 | 标注进度汇总（auto-updated by hook） |

## 分析与 Benchmark

| 文件 | 重要性 | 为什么 |
|------|--------|--------|
| `scripts/analysis/run_benchmark.py` | 高 | Canonical benchmark runner；`--checkpoint`（非 --weights）；输出到 `results/benchmark/{run_name}_{timestamp}/` |
| `scripts/analysis/batch_inference.sh` | 中 | 并行批量推理入口 |
| `scripts/training/export_targeted_hn.py` | 中 | Targeted hard negative 提取 |
| `scripts/validate_registry.py` | 中 | 注册表交叉验证（manifest ↔ training_sets ↔ model_registry ↔ regions） |

## 云端/RunPod

| 文件 | 重要性 | 为什么 |
|------|--------|--------|
| `scripts/runpod_init.sh` | 中 | Pod 初始化（rasterstats 是关键依赖！） |
| `scripts/sync_from_runpod.sh` | 中 | SSH rsync 下载；`^G[0-9]+` 模式不兼容 JHB（已知脆弱点，待修复） |
| `.env` | 高 | RunPod SSH/S3 凭证（gitignored） |

## 治理与规则

| 文件 | 重要性 | 为什么 |
|------|--------|--------|
| `.claude/rules/02-evaluation-semantics.md` | 高 | V1.3 语义守护；profile 不得静默切换 |
| `.claude/rules/03-doc-sync.md` | 高 | docs/ 是唯一事实源；入口文档三方同步 |
| `.claude/rules/06-multi-city.md` | 高 | 多城市路径/CRS 硬约束；禁止新增城市路径硬编码 |
| `.claude/rules/07-annotation-semantics.md` | 高 | 标注语义规则；Two-Axis Model 引用；T1=A1；tier 不可 auto-promote |
| `docs/governance/repo-rules.md` | 中 | Git 大文件保护 |

## CRS 速查

| 区域 | Metric CRS | 注意 |
|------|-----------|------|
| Cape Town | EPSG:32734 (UTM 34S) | 默认 |
| JHB | EPSG:32735 (UTM 35S) | 不同于 Cape Town！ |
| Tiles | EPSG:4326 | 瓦片原始坐标 |

使用 `core.grid_utils.get_metric_crs(grid_id, region=)` 查询，不要硬编码。

## Two-Axis Model 速查

| 轴 | 值 | 含义 |
|----|-----|------|
| A (语义符合度) | A1 | installation-spec compliant — 人工确认符合 merge/boundary 规则 |
| | A2 | mostly installation-like — 大致符合但未逐条核验 |
| | A3 | weak/fragmentary/noisy |
| B (来源) | H | human-from-scratch |
| | R | reviewed prediction |
| | S | SAM-refined review |
| | G | legacy weak annotation |

**T1 requires A1.** T2 includes A2/A3.

## New City Checklist

添加新城市时：
1. `configs/datasets/regions.yaml` — 添加 region 条目（CRS、paths、grids）
2. `data/{city}_task_grid.gpkg` — 放置 task grid
3. 下载 tiles 到 region paths 指定位置
4. `data/annotations/{CityName}/` — 创建标注目录
5. `configs/benchmarks/post_train.yaml` — 添加 benchmark suite
6. 运行 `python scripts/validate_registry.py` 验证
