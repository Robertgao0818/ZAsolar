# ZAsolar — 南非城市屋顶光伏检测

[English](README.md)

ZAsolar 是一个研究代码库，用于从高分辨率航拍影像中检测住宅屋顶光伏装置，
目标是构建覆盖南非的网格聚合光伏面板数据。当前覆盖开普敦和约翰内斯堡。

检测流程为 **Mask R-CNN (ResNet-50 + FPN) + SAM 2.1 mask-prompt 后处理精修**。
V1.4 验证框架以网格聚合光伏装机量作为主要成功指标，per-polygon F1 作为诊断
辅助。安装时间反推由兄弟仓库
[`solar_backdating`](https://github.com/Robertgao0818/solar_backdating)
单独处理。

## 当前指标

主 benchmark：**约翰内斯堡 CBD 25 grid，Vexcel 2024 航拍（约 6.7 cm GSD）**，
检测器 = V3-C-HN，后处理 = SAM 2.1 mask+box 精修。

| 通道 | 指标 | 结果 | 样本 |
|---|---|---|---|
| Ch1 — 分层精度 | P (V3-C, hit-table) | 0.749 [0.71, 0.78] | 25 grid × 分层屋顶样本 |
| Ch3 — 装机量准确度 | area F1 | 0.821 | JHB CBD 25-grid Vexcel |
| Ch3 — 装机量准确度 | 聚合 \|A\|/\|B\| | 0.992 | JHB CBD 25-grid Vexcel |

完整四通道框架、每个通道认证什么不认证什么、以及已知干扰因素
（如 SSEG building geocoding 不匹配、影像 vintage gap），见
[`docs/validation_strategy.md`](docs/validation_strategy.md)。

## 仓库结构

```
core/                     共享模块 (region_registry, postproc, models)
pipeline/                 声明式数据集构建器 (V1.2 spec)
configs/                  region / imagery / training / model 注册表
data/annotations/         开普敦 + 约堡标注 (本地 gitignore)
docs/                     architecture.md, validation_strategy.md, workflows.md
scripts/
  analysis/               benchmark / audit / 校准扫描
  imagery/                瓦片下载 / 预览 / VRT
  training/               COCO 导出, hard-negative 导出
  classifier/             PV vs 热水器二分类 pipeline
  annotations/            review GUI, SAM FN GUI, 批量 finalize
detect_and_evaluate.py    主推理 + 评估入口
detect_direct.py          直接 pipeline 第 1 阶段 (raw detections)
finalize.py               第 3 阶段: raw_detections -> predictions_metric.gpkg
train.py                  Mask R-CNN 微调
export_coco_dataset.py    标注 -> COCO 实例分割数据集
```

详细结构：[`docs/architecture.md`](docs/architecture.md)。
工作流命令序列：[`docs/workflows.md`](docs/workflows.md)。

## 快速上手

```bash
# 环境 (从 requirements.lock.txt 创建 ./.venv)
./scripts/bootstrap_env.sh && source scripts/activate_env.sh

# 验证 CUDA GPU + GIS 依赖
./scripts/check_env.sh

# 单 grid 推理 + 评估 (需要 CUDA)
python detect_and_evaluate.py \
  --grid-id G1688 \
  --model-path checkpoints/exp003_C_targeted_hn/best_model.pth \
  --postproc-config configs/postproc/v4_canonical.json \
  --force

# 主 benchmark
python scripts/analysis/run_benchmark.py --suite jhb_cbd_25_vexcel
```

大数据（tiles、COCO 数据集、模型权重）放在仓库外的 `~/zasolar_data/`。
`configs/datasets/regions.yaml` 是 imagery-layer 与 model-run 注册表的权威
来源。标注通过 Dropbox 同步，权重通过 RunPod S3 同步。

## 验证框架 (V1.4)

四个正交通道：

1. **分层精度** — benchmark grid 上的随机分层屋顶采样；认证按屋顶类型与
   目标尺寸条件化的检测精度。
2. **穷举召回** — 小规模 grid 集上的 clean GT（完整光伏清单）；衡量对
   installation-merged 参考的召回。
3. **合理性** — hex 聚合检测结果 vs admin 级装机数（SSEG, kW 校准）；
   作为 sanity check，不作为 benchmark。
4. **机会性外部对比** — 可用且 vintage 与覆盖匹配时，与第三方数据集
   （如开普敦 Li GT）对比。

任务 grid 是主聚合单元。Per-polygon F1 仅作诊断。Tier-1 指标体系使用
`area_aggregate_eval.py`（`agg_F1` + `pgF1` + `bulk` + `sigma_Bw` +
log-`sigma` + RMSE + `thru0_beta` + R²），以 `sigma_Bw` 和 RMSE 为主裁判，
`bulk in [0.5, 2.0]` 为 sanity gate。

## 兄弟仓库 — `solar_backdating`

安装时间反推（使用 Google Earth 历史影像逐 footprint 反推）单独放在兄弟仓库：
[Robertgao0818/solar_backdating](https://github.com/Robertgao0818/solar_backdating)。
它作为本仓库的插件运行 — 共享 `.venv`，import `core.region_registry`、
`core.annotation_loader`、`core.grid_utils`。任何新的 temporal /
GEHistoricalImagery / 安装时间相关代码都去那边，不进本仓库。

## 引用

论文撰写中。请引用为：

> Tao Yu Chen et al. (2026). *Grid-aggregate rooftop photovoltaic detection
> for South African cities.* [Manuscript in preparation].

## License

代码：MIT。
标注与人工审核预测：见
[`data/annotations/ANNOTATION_SPEC.md`](data/annotations/ANNOTATION_SPEC.md)。
