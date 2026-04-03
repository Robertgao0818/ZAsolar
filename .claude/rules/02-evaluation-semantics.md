---
paths:
  - "detect_and_evaluate.py"
  - "scripts/analysis/**"
  - "export_coco_dataset.py"
---

# Evaluation Semantics Rules

## V1.3 reviewed prediction footprint 语义守护
任务定义是 reviewed prediction footprint segmentation（模型预测经人工审查后导出的多边形）。GT 标注仍遵循 installation-level 规则。不得将语义退化为 panel-level，除非用户明确要求改变任务定义。`installation` evaluation profile 名字保留，评估 reviewed predictions vs installation-level GT。

## 评估 profile 不得静默切换
`--evaluation-profile` 选项（`installation` / `legacy_instance`）必须保持显式。不得在代码中静默切换默认 profile 或忽略用户选择。

## 空 chip 不得删除
COCO 数据集中空标注 chip 是有意保留的 hard negatives，确保检测器学习 false positive 抑制。不得在导出或训练阶段丢弃空 chip，除非用户明确要求。

## 基线指标必须从 CSV 读取
基线指标（F1、precision、recall 等）必须从 `results/<GridID>/presence_metrics.csv` 等 CSV 文件动态读取，不得在代码中硬编码。如需 fallback 值，使用 `configs/datasets/regions.yaml` 作为结构化数据源。
