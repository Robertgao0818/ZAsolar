# Architecture — Cape Town Solar Panel Detection

> **V1.4 (2026-04-22)**: Success metric reframes from per-polygon F1 to aggregate installation inventory at grid level, suited for economic analysis. Sub-repo split landed 2026-05-05: install-date back-dating moved to **`solar_backdating`** at `/home/gaosh/projects/solar_backdating/` (plugin of this repo via shared venv + PYTHONPATH; old `geid_bbox` GEID free-detection prototype archived under `/home/gaosh/projects/_archive/geid_bbox_legacy_2026-05-05/`). Validation framework: four channels (stratified precision, exhaustive recall, plausibility, opportunistic external) with task grid as primary aggregation unit. See [`validation_strategy.md`](validation_strategy.md) for the full spec.
>
> Main-repo copies of `scripts/temporal/`, `scripts/validation/{probe_geid_vintages,parse_geid_probe_results,run_geid_vintage_probe}.*`, and `tests/temporal/` are frozen with deprecation headers; scheduled for removal after 2026-05-31. Bug fixes go to `solar_backdating` first.
>
> **V1.3 (2026-04-03)**: Task definition updated from installation-level footprint segmentation to reviewed prediction footprint segmentation. `installation` evaluation profile name preserved; GT annotations still follow installation-level rules. Some sections below may reference historical V1.2 conventions.

## Directory Structure

```
data/
  task_grid.gpkg              — Grid 编号集合
  annotations/                — 标注数据（详见 annotations/README.md）
    Capetown/                 — Cape Town 标注（{GridID}_SAM2_{YYMMDD}.gpkg），COCO export 主数据源
                                包含 SAM2 cleaned, 早期 legacy（G1023/G1134 等）, all_annotations_cleaned.gpkg
    Joburg/                   — Johannesburg 标注
                                JHB01-06.gpkg          — Li 手标 6-grid pilot (legacy)
                                G07xx-G09xx_V4_*.gpkg  — CBD batch1 25 grids (V4 推理 → review → SAM 重切, 2026-04-07)
    ANNOTATION_SPEC.md        — V1.3 标注规范：GT = installation footprint + Two-Axis Model (A1-A3 × H/R/S/G)
    PROGRESS.md               — 标注进度自动汇总（batch/grid/installation 统计）
    annotation_manifest.csv   — 标注 manifest (quality tier T1/T2, review status)
  coco/                       — COCO 格式训练数据（export_coco_dataset.py 生成）
tiles/                        — 项目目录占位符（实际数据禁止放 WSL 项目目录）
                               post-2026-04-26: 数据迁移到 ~/zasolar_data/ (WSL ext4, 替代 /mnt/d/ZAsolar/)
                               env: SOLAR_TILES_ROOT=/home/gaosh/zasolar_data/tiles
                                    SOLAR_ARTIFACT_ROOT=/home/gaosh/zasolar_data
                               COCO 数据集: ~/zasolar_data/coco/coco_v4_*/

# ~/zasolar_data/tiles/ 目录结构 (post-2026-04-19 重构, 2026-04-26 搬到 WSL):
#   cape_town/
#     aerial_2025/{MANIFEST.json, G1189/, ...}      — CT 航测 2025 (chunked, 120 grids incl. G1895)
#   johannesburg/
#     aerial_2023/{MANIFEST.json, G0772/, ...}      — JHB 航测 2023 (chunked, 100 grids)
#     geid_2024_02/{MANIFEST.json, G0772_mosaic.tif, ...}  — JHB GEID 2024-02 (mosaic, 100 files)
# /mnt/d/ZAsolar/tiles/johannesburg/aerial_legacy/  — JHB legacy pilot 6 grids (留在 NTFS, 仅 JHB01-06 用)
# /mnt/d/ZAsolar/annotations_inbox/                 — QGIS handoff 区, 留在 NTFS 供 Windows 工具直访
# 每层 MANIFEST.json 记录 source/vintage/file_layout/coverage_grids/crs。
# 解析 tile 路径: core.region_registry.get_imagery_layer_path(region, layer_id)

# WARNING: Grid IDs overlap between regions. G1189/G1190/G1293/G1513/G1570/
#   G1630 等在 CT 和 JHB 都有真实、独立的地理覆盖。**永远** 用
#   (region, imagery_layer) 定位,不要靠 grid_id 猜 region。

results/<region>/<model_run>/<GridID>/  — 检测结果按 region × model_run 分组
                                         每层 RUN_MANIFEST.json 记录 model_version/imagery_layer/inference_date
# 已注册的 model_runs (configs/datasets/regions.yaml):
#   cape_town/v3c_targeted_hn_aerial_2025/    — V3-C on CT aerial (45 grids)
#   cape_town/v3c_geid_experiment/            — 跨区实验 G1189/G1190 (可删)
#   johannesburg/v3c_geid_2024_02/            — V3-C on JHB GEID (85 grids)
#   johannesburg/v3c_targeted_hn_aerial_2023/ — V3-C on JHB aerial (100 grids)
#   johannesburg/v4_aerial_2023/              — V4 on JHB aerial (92 grids, canonical)
# Legacy results/<GridID>/ (直接挂 CT 根下) 仍为 CT 旧格式结果的后备位置,
#   待 PR8 清理 (checkpoints_cleaned 24 grids + 无 config 77 grids)。
# results_joburg/ 过渡 symlink 已于 2026-04-26 删除; 部分 analysis 脚本仍引用，待逐一更新
  masks/                      — per-tile 检测掩膜
  vectors/                    — per-tile 矢量化结果
  config.json                 — 包含 region/imagery_layer_id/model_run_id 字段 (post-2026-04-19)
  presence_metrics.csv        — V1.3 installation presence P/R/F1
  footprint_metrics.csv       — V1.3 footprint IoU/Dice 分布
  area_error_metrics.csv      — V1.3 面积误差分桶
checkpoints/                  — 微调模型权重（数据目录，禁止放源码）
core/
  grid_utils.py               — Grid 路径/坐标工具函数（共享模块，内部委托 region_registry）
  region_registry.py           — 加载 regions.yaml 提供 lookup_region/get_region_config 等 API
scripts/
  analysis/
    param_search.py            — 检测参数网格搜索
    calibration_sweep.py       — 后处理阈值校准扫描
    multi_grid_baseline.py     — 多 grid baseline/泛化对比
    benchmark_weights.py       — 训练后多权重 benchmark / delta 对比
    build_gt_heater_audit.py   — GT 加热器污染审计队列构建 + chip 导出
    label_gt_heater_audit.py   — GT 加热器审计 HTML 标注器生成
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
    regions.yaml               — 权威区域注册表（grids、CRS、paths、grid_id_pattern）
    training_sets.yaml         — 训练集 recipe provenance（region_scope、holdout、HN 来源）
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
| `detect_and_evaluate.py` | 主流程：检测→过滤→评估→可视化。支持 `--model-path`、`--evaluation-profile`、`--data-scope`、`--imagery-layer`、`--model-run` |
| `export_coco_dataset.py` | 标注→COCO 数据集导出。支持 `--neg-ratio`（neg:pos 比例）、`--exclude-grids`（benchmark holdout）、`--audit-csv`（热水器过滤）、`--manifest`、`--no-balance` |
| `scripts/training/export_targeted_hn.py` | Batch 003 审核 FP → targeted HN chips，合并到 base COCO |
| `scripts/training/export_v4_hn.py` | Batch 004 小目标 FP shortlist → HN chips（分层采样） |
| `scripts/training/export_v4_1_hn.py` | V4.1 合并 HN: batch 003 (ID 900000+) + batch 004 (ID 950000+) |
| `scripts/analysis/run_benchmark.py` | 标准化 benchmark（多 suite 对比）。`BENCHMARK_PARALLEL` 环境变量控制并行推理 |
| `scripts/runpod_pod.sh` | RunPod pod 生命周期管理（start/stop/status/ssh/init） |
| `configs/postproc/v4_canonical.json` | 标准后处理参数（post_conf=0.85 + tiered），确保跨实验可比 |
| `scripts/analysis/batch_inference.sh` | 并行批量推理 (canonical 入口，支持任意 grid list + 并行度) |
| `train.py` | Mask R-CNN 微调训练（两阶段：heads-only → full fine-tune），需要 CUDA GPU |
| `building_filter.py` | OSM+Microsoft 建筑轮廓 → buildings.gpkg + tile_manifest.csv |
| `core/grid_utils.py` | Grid 路径/坐标工具函数（共享模块，内部委托 region_registry） |
| `core/region_registry.py` | 加载 regions.yaml 提供 region/grid 查询 API |
| `scripts/validate_registry.py` | 注册表交叉验证（manifest ↔ training_sets ↔ model_registry ↔ regions） |
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
