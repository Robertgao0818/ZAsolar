# Context Map — Agent 导向

Clean agent 接手时读此文件，快速了解关键文件和它们为什么重要。

## 核心流水线

| 文件 | 重要性 | 为什么 |
|------|--------|--------|
| `detect_and_evaluate.py` | 最高 | 主检测+评估入口；`installation` profile 定义在此；config.json 复用逻辑在此 |
| `export_coco_dataset.py` | 高 | COCO 数据集导出；auto-discover `cleaned/*_SAM2_*.gpkg`；empty-chip hard negative 逻辑在此 |
| `train.py` | 高 | Mask R-CNN 微调；两阶段训练；CUDA 强制验证 |
| `core/grid_utils.py` | 中 | Grid 路径/CRS 解析共享模块；Cape Town vs JHB 差异在此体现 |

## 配置层

| 文件 | 重要性 | 为什么 |
|------|--------|--------|
| `configs/benchmarks/post_train.yaml` | 高 | Benchmark suite 定义（primary/diagnostic/secondary/smoke）+ auto-verdict 规则 |
| `configs/model_registry.yaml` | 中 | 模型权重路径注册；benchmark runner 从此读取 |
| `configs/datasets/regions.yaml` | 中 | Grid 注册表；baseline 指标、T1/T2 分层、CRS |
| `configs/postproc/*.json` | 中 | 后处理阈值配置；影响评估结果 |

## 标注与数据

| 文件 | 重要性 | 为什么 |
|------|--------|--------|
| `data/annotations/ANNOTATION_SPEC.md` | 高 | V1.3 标注规范（GT 仍为 installation-level） |
| `data/annotations/cleaned/` | 高 | 清洗后 SAM2 标注，COCO export 的主数据源 |
| `data/annotations/annotation_manifest.csv` | 中 | 质量分层 T1/T2 和 review status |
| `data/annotations/PROGRESS.md` | 低 | 标注进度汇总（auto-updated by hook） |

## 分析与 Benchmark

| 文件 | 重要性 | 为什么 |
|------|--------|--------|
| `scripts/analysis/run_benchmark.py` | 高 | Canonical benchmark runner；`--checkpoint`（非 --weights）；输出到 `results/benchmark/{run_name}_{timestamp}/` |
| `scripts/analysis/batch_inference.sh` | 中 | 并行批量推理入口 |
| `scripts/training/export_targeted_hn.py` | 中 | Targeted hard negative 提取 |

## 云端/RunPod

| 文件 | 重要性 | 为什么 |
|------|--------|--------|
| `scripts/runpod_init.sh` | 中 | Pod 初始化（rasterstats 是关键依赖！） |
| `scripts/sync_from_runpod.sh` | 中 | SSH rsync 下载；`^G[0-9]+` 模式不兼容 JHB（已知脆弱点） |
| `.env` | 高 | RunPod SSH/S3 凭证（gitignored） |

## 治理与规则

| 文件 | 重要性 | 为什么 |
|------|--------|--------|
| `.claude/rules/02-evaluation-semantics.md` | 高 | V1.3 语义守护；profile 不得静默切换 |
| `.claude/rules/03-doc-sync.md` | 高 | docs/ 是唯一事实源；入口文档三方同步 |
| `docs/governance/repo-rules.md` | 中 | Git 大文件保护 |

## CRS 速查

| 区域 | Metric CRS | 注意 |
|------|-----------|------|
| Cape Town | EPSG:32734 (UTM 34S) | 默认 |
| JHB | EPSG:32735 (UTM 35S) | 不同于 Cape Town！ |
| Tiles | EPSG:4326 | 瓦片原始坐标 |
