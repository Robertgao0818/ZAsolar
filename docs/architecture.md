# Architecture — Cape Town Solar Panel Detection

## Directory Structure

```
data/
  task_grid.gpkg              — Grid 编号集合
  annotations/                — 弱监督标注（详见 annotations/README.md）
    G1238_detailed.gpkg       — SAM2.1 精细切割 (248 polygons, layer SAM_Residential_merged)
    solarpanel_g0001_g1190.gpkg — Google Earth 标注（已校准, G1189=58, G1190=76, 其余少量）
    ANNOTATION_SPEC.md        — V1.2 标注规范（installation footprint 定义）
    annotation_manifest.csv   — 标注 manifest (quality tier T1/T2, review status)
  coco/                       — COCO 格式训练数据（export_coco_dataset.py 生成）
tiles/<GridID>/               — 各 Grid 的航测瓦片（数据目录，禁止放源码）
                               本地存储: D:\ZAsolar\tiles (WSL: /mnt/d/ZAsolar/tiles)
                               环境变量: SOLAR_TILES_ROOT=/mnt/d/ZAsolar/tiles
results/<GridID>/             — 检测结果、评估报告、图表（数据目录，禁止放源码）
  masks/                      — per-tile 检测掩膜
  vectors/                    — per-tile 矢量化结果
  presence_metrics.csv        — V1.2 installation presence P/R/F1
  footprint_metrics.csv       — V1.2 footprint IoU/Dice 分布
  area_error_metrics.csv      — V1.2 面积误差分桶
checkpoints/                  — 微调模型权重（数据目录，禁止放源码）
core/
  grid_utils.py               — Grid 路径/坐标工具函数（共享模块）
scripts/
  analysis/
    param_search.py            — 检测参数网格搜索
    calibration_sweep.py       — 后处理阈值校准扫描
    multi_grid_baseline.py     — 多 grid baseline/泛化对比
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
docs/
  architecture.md              — 本文件（目录结构、路径映射）
  workflows.md                 — 工作流命令序列
  governance/repo-rules.md     — 仓库规则（Git 大文件保护、目录治理）
  experiment-archive/          — 实验日志归档
```

## Scripts

| Script | Description |
|--------|-------------|
| `detect_and_evaluate.py` | 主流程：检测→过滤→评估→可视化。支持 `--model-path`、`--evaluation-profile`、`--data-scope` |
| `export_coco_dataset.py` | 标注→COCO 实例分割数据集导出（chip 切分 + train/val 划分 + georeferenced chips）。支持 `--manifest`、`--tier-filter`、`--category-name` |
| `train.py` | Mask R-CNN 微调训练（两阶段：heads-only → full fine-tune），需要 CUDA GPU |
| `building_filter.py` | OSM+Microsoft 建筑轮廓 → buildings.gpkg + tile_manifest.csv |
| `core/grid_utils.py` | Grid 路径/坐标工具函数（共享模块） |
| `scripts/progress_tracker.py` | STATUS.md / ROADMAP.md 自动更新 |

## CRS 约定

| 用途 | CRS |
|------|-----|
| QGIS 标注导出、人工交换格式 | `EPSG:4326` |
| 航测瓦片地理参考 | `EPSG:4326` |
| 检测后处理（面积/长度/buffer、IoU 评估） | `EPSG:32734` |
| QGIS 回看的导出结果 (`predictions.geojson`) | `EPSG:4326` |
| 米制计算结果 (`predictions_metric.gpkg`) | `EPSG:32734` |

## Result Reuse Rules

- 每次检测在 `results/<GridID>/config.json` 记录运行参数和脚本指纹
- 已有结果仅在 `config.json` 与当前配置完全一致时复用
- 配置/代码已变化时使用 `--force` 重新检测
- 参数搜索同样在 `results/<GridID>/param_search/<experiment_id>/config.json` 记录实验配置
- V1.2 在 config.json 新增 `evaluation_config` section 以提供可追溯性
