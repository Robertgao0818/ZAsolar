# Cape Town Rooftop Solar Panel Detection

基于航测影像的开普敦屋顶太阳能安装检测与评估流水线。

**V1.3 任务定义**: reviewed prediction footprint segmentation — 模型预测经人工审查后导出的多边形。标注标准仍遵循 installation-level 规则（见 `data/annotations/ANNOTATION_SPEC.md`），但流水线输出是审查后的预测结果，不要求 installation 级合并。

## 项目进度

| Grid | 底图 | 标注 | 检测 | 评估 | 备注 |
|------|------|------|------|------|------|
| G1238 | done | done (QGIS) | done | done | 首个完整流程 Grid |
| G1189 | done | done (Google Earth, 已校准) | done | done | Fine-tuned F1≈0.595 |
| G1190 | done | done (Google Earth, 已校准) | done | done | Fine-tuned F1≈0.649 |

详细进度见 `ROADMAP.md`，日常工作记录见 `docs/progress_log/`。

## 快速开始

```bash
./scripts/bootstrap_env.sh         # 首次创建/更新 .venv
source scripts/activate_env.sh
./scripts/check_env.sh             # 检查依赖 + CUDA
python building_filter.py          # 下载建筑轮廓
python detect_and_evaluate.py      # 检测 + 评估（需 GPU）
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

## 本地环境

- 虚拟环境固定在 `./.venv`
- 运行时缓存固定在仓库内：`.cache/`、`.config/`、`.local/`、`.tmp/`
- `requirements.lock.txt` 为环境快照，重建时优先使用
- `train.py` 强制验证 CUDA；`./scripts/check_env.sh` 显示 `cuda_available=False` 时训练不会启动
