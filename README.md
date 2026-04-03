# Cape Town Rooftop Solar Panel Detection

基于航测影像的开普敦屋顶太阳能安装检测与评估流水线。

**V1.3 任务定义**: reviewed prediction footprint segmentation — 模型预测经人工审查后导出的多边形。标注标准仍遵循 installation-level 规则（见 `data/annotations/ANNOTATION_SPEC.md`），但流水线输出是审查后的预测结果，不要求 installation 级合并。

## 项目概况

- **模型**: geoai (Mask R-CNN ResNet50-FPN)，当前权重 V3-C（targeted hard negative fine-tuning）
- **GT 数据集**: 106 grids, 7301 polygons（含 SAM 2.1 FN 填充 111 个）
- **标注流程**: 模型预测 → review GUI 人工审查（correct/edit/delete） → SAM 2.1 Large 补标 FN → cleaned GPKG 导出
- **评估指标**: F1@IoU0.3，当前 baseline F1=0.873（8-grid sample）

### 标注批次

| Batch | Grids | Polygons | Date | 模型 |
|-------|-------|----------|------|------|
| Legacy + Early SAM2 | 12 | 497 | ≤2026-03-20 | — |
| Batch 001-002b | 40 | 2507 | 2026-03-22 ~ 03-24 | V2 |
| Batch 003 | 20 | 1729 | 2026-03-25 ~ 03-27 | V3-C |
| Batch 004 | 31 | 2346 | 2026-04-03 | V3-C |
| JHB | 6 | 191 | — | — |
| SAM FN fills | — | +111 | 2026-04-03 | SAM 2.1 Large |

详细进度见 [`data/annotations/PROGRESS.md`](data/annotations/PROGRESS.md)，日常工作记录见 [`docs/progress_log/`](docs/progress_log/)。

## 快速开始

```bash
./scripts/bootstrap_env.sh         # 首次创建/更新 .venv
source scripts/activate_env.sh
./scripts/check_env.sh             # 检查依赖 + CUDA
python detect_and_evaluate.py      # 检测 + 评估（需 GPU）

# 微调
python export_coco_dataset.py --output-dir /mnt/d/ZAsolar/coco
python train.py --coco-dir /mnt/d/ZAsolar/coco --output-dir checkpoints

# Review GUI
python scripts/annotations/review_detections.py --grid-id G1688 --port 8766

# SAM FN 交互审查
python scripts/annotations/sam_fn_review.py --port 8770
```

## 文档导航

| 文档 | 内容 |
|------|------|
| [`docs/architecture.md`](docs/architecture.md) | 目录结构、路径映射、CRS 约定 |
| [`docs/workflows.md`](docs/workflows.md) | 推理、微调、分析完整命令序列 |
| [`docs/governance/repo-rules.md`](docs/governance/repo-rules.md) | Git 大文件保护、目录治理 |
| [`data/annotations/ANNOTATION_SPEC.md`](data/annotations/ANNOTATION_SPEC.md) | V1.3 标注规范（GT 仍为 installation-level） |
| [`ROADMAP.md`](ROADMAP.md) | 版本里程碑 + 决策记录 |
| [`docs/progress_log/`](docs/progress_log/) | 日报（按周分目录，同步 Dropbox） |

## 分析工具

| 脚本 | 用途 |
|------|------|
| `scripts/analysis/postprocess_ablation.py` | 后处理参数对照实验（epsilon/elongation/confidence） |
| `scripts/analysis/sam_recut_experiment.py` | SAM 2.1 FN 重切 + TP 形状对比 |
| `scripts/annotations/sam_fn_review.py` | SAM FN 交互审查 GUI（加减提示点、实时重切） |

## 本地环境

- 虚拟环境固定在 `./.venv`
- 大数据在 D 盘：tiles → `/mnt/d/ZAsolar/tiles/`，COCO → `/mnt/d/ZAsolar/coco_*/`
- 运行时缓存固定在仓库内：`.cache/`、`.config/`、`.local/`、`.tmp/`
- `requirements.lock.txt` 为环境快照，重建时优先使用
- `train.py` 强制验证 CUDA；`./scripts/check_env.sh` 显示 `cuda_available=False` 时训练不会启动
