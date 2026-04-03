# Architecture — Cape Town Solar Panel Detection

> **V1.3 (2026-04-03)**: Task definition updated from installation-level footprint segmentation to reviewed prediction footprint segmentation. `installation` evaluation profile name preserved; GT annotations still follow installation-level rules. Some sections below may reference historical V1.2 conventions.

## Directory Structure

```
data/
  task_grid.gpkg              — Grid 编号集合
  annotations/                — 标注数据（详见 annotations/README.md）
    G1238_detailed.gpkg       — SAM2.1 精细切割 (248 polygons, layer SAM_Residential_merged)
    solarpanel_g0001_g1190.gpkg — Google Earth 标注（已校准, G1189=58, G1190=76, 其余少量）
    cleaned/                  — 清洗后 SAM2 标注 ({GridID}_SAM2_{YYMMDD}.gpkg)，export 主数据源
    ANNOTATION_SPEC.md        — V1.3 标注规范（GT 仍为 installation footprint，流水线输出为 reviewed predictions）
    PROGRESS.md               — 标注进度自动汇总（batch/grid/installation 统计）
    annotation_manifest.csv   — 标注 manifest (quality tier T1/T2, review status)
  coco/                       — COCO 格式训练数据（export_coco_dataset.py 生成）
tiles/                        — 符号链接或空目录（实际数据禁止放 WSL 项目目录）
                               所有 tiles 存放在 D:\ZAsolar\tiles (WSL: /mnt/d/ZAsolar/tiles)
                               环境变量: SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles
                               COCO 数据集同理: /mnt/d/ZAsolar/coco_*/
results/<GridID>/             — 检测结果、评估报告、图表（数据目录，禁止放源码）
  masks/                      — per-tile 检测掩膜
  vectors/                    — per-tile 矢量化结果
  presence_metrics.csv        — V1.3 installation presence P/R/F1
  footprint_metrics.csv       — V1.3 footprint IoU/Dice 分布
  area_error_metrics.csv      — V1.3 面积误差分桶
checkpoints/                  — 微调模型权重（数据目录，禁止放源码）
core/
  grid_utils.py               — Grid 路径/坐标工具函数（共享模块）
scripts/
  analysis/
    param_search.py            — 检测参数网格搜索
    calibration_sweep.py       — 后处理阈值校准扫描
    multi_grid_baseline.py     — 多 grid baseline/泛化对比
    benchmark_weights.py       — 训练后多权重 benchmark / delta 对比
  imagery/
    download_tiles.py          — WMS 瓦片下载 + 地理配准
    grid_preview_batch.py      — 低分辨率 grid 预览批量生成
    review_grid_previews.py    — 浏览器交互式 grid 预览审查
    build_vrt_g1238.py         — G1238 VRT 拼接（legacy helper）
  annotations/
    bootstrap_manifest.py      — 从 GPKG 生成初始 annotation manifest
    prepare_jhb_grids.py       — JHB grid 准备
configs/
  datasets/
    regions.yaml               — 区域/Grid 注册表（基线指标、标注源）
    imagery_sources.yaml       — 影像源参数（分辨率、CRS、下载脚本）
  benchmarks/
    post_train.yaml            — Benchmark preset (grid suites, verdict 规则)
  model_registry.yaml          — 模型注册表 (V1/V2/V3-C 等权重路径与元数据)
  postproc/                    — 后处理阈值配置 (calibration sweep 产物)
scripts/
  runpod_init.sh               — RunPod pod 初始化（补装 GIS 依赖，验证环境）
  upload_to_runpod.sh          — S3 上传（需 .env 凭证）
  sync_from_runpod.sh          — SSH rsync 下载 results/tiles（需 .env SSH 配置）
cloud_setup.sh                 — RunPod 训练启动器（stage COCO → local SSD）
docs/
  architecture.md              — 本文件（目录结构、路径映射）
  workflows.md                 — 工作流命令序列
  governance/repo-rules.md     — 仓库规则（Git 大文件保护、目录治理）
  experiment-archive/          — 实验日志归档
  session_history/             — 会话历史文档归档（agent / user 讨论记录）
```

## Scripts

| Script | Description |
|--------|-------------|
| `detect_and_evaluate.py` | 主流程：检测→过滤→评估→可视化。支持 `--model-path`、`--evaluation-profile`、`--data-scope` |
| `export_coco_dataset.py` | 标注→COCO 实例分割数据集导出（scan-then-write: 先扫描 metadata、balance 采样、再只写选中 chip）。支持 `--manifest`、`--tier-filter`、`--no-balance` |
| `scripts/training/export_targeted_hn.py` | 从审核 FP 预测提取 targeted hard negative chips，合并到 no-HN 基础数据集 |
| `scripts/analysis/run_benchmark.py` | 训练后按固定 benchmark suites 对多个模型权重做统一回归对比 (agent-first, summary.json 主输出) |
| `scripts/analysis/batch_inference.sh` | 并行批量推理 (canonical 入口，支持任意 grid list + 并行度) |
| `train.py` | Mask R-CNN 微调训练（两阶段：heads-only → full fine-tune），需要 CUDA GPU |
| `building_filter.py` | OSM+Microsoft 建筑轮廓 → buildings.gpkg + tile_manifest.csv |
| `core/grid_utils.py` | Grid 路径/坐标工具函数（共享模块） |
| `scripts/progress_tracker.py` | ROADMAP.md 自动更新 |

## CRS 约定

| 用途 | CRS |
|------|-----|
| QGIS 标注导出、人工交换格式 | `EPSG:4326` |
| 航测瓦片地理参考 | `EPSG:4326` |
| 检测后处理（面积/长度/buffer、IoU 评估） | 按区域动态确定（见 `core/grid_utils.py`） |
| Cape Town 米制 CRS | `EPSG:32734` (UTM 34S) |
| JHB 米制 CRS | `EPSG:32735` (UTM 35S) |
| QGIS 回看的导出结果 (`predictions.geojson`) | `EPSG:4326` |
| 米制计算结果 (`predictions_metric.gpkg`) | 与区域对应的 UTM CRS |

## Result Reuse Rules

- 每次检测在 `results/<GridID>/config.json` 记录运行参数和脚本指纹
- 已有结果仅在 `config.json` 与当前配置完全一致时复用
- 配置/代码已变化时使用 `--force` 重新检测
- 参数搜索同样在 `results/<GridID>/param_search/<experiment_id>/config.json` 记录实验配置
- config.json 包含 `evaluation_config` section 以提供可追溯性
