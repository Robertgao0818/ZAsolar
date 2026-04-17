# Code Review Checklist

审查代码变更对检测/评估/训练流水线的影响。

## P1 — Blockers

- [ ] `installation` / `legacy_instance` evaluation profile 切换是否仍显式（不允许静默改默认值）— 见 `.claude/rules/02-evaluation-semantics.md`
- [ ] `results/<GridID>/config.json` 复用安全是否保持
- [ ] empty-target chips 与 targeted hard negatives 是否被误删
- [ ] 是否有安全漏洞（命令注入、路径遍历）
- [ ] **路径必须通过 `core/grid_utils.py` / `core/region_registry.py` 获取**，直接构造城市路径 = blocker — 见 `.claude/rules/06-multi-city.md`
- [ ] CRS 是否通过 `get_metric_crs()` 查询（非硬编码 EPSG 号）
- [ ] COCO export 是否声明区域范围（`--regions` 参数或注释说明仅 Cape Town）— 见 `.claude/rules/07-annotation-semantics.md`
- [ ] **标注 tier 升级是否符合 Two-Axis Model**：T1 requires A1 (installation-spec compliant)。不允许仅凭 label_source 自动升 T1。

## P2 — Should-fix

- [ ] 远端 RunPod 相关改动是否仍通过 `.env` / `RUNPOD_SSH_*` 工作（非硬编码）
- [ ] 新增 region-specific 常量是否加到 `configs/datasets/regions.yaml` 而非硬编码
- [ ] 是否需要更新 CLAUDE.md stable skeleton
- [ ] `docs/workflows.md` 命令是否仍一致
- [ ] 新模型进入 `model_registry.yaml` 是否带 `training_set_id` 或等价 provenance
- [ ] `scripts/validate_registry.py` 是否通过（0 errors）

## P3 — Nice-to-have

- [ ] 遵循现有代码风格
- [ ] 新增配置有合理默认值
