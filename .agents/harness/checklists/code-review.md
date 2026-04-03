# Code Review Checklist

审查代码变更对检测/评估/训练流水线的影响。

## P1 — Blockers

- [ ] `installation` / `legacy_instance` evaluation profile 切换是否仍显式（不允许静默改默认值）
- [ ] `results/<GridID>/config.json` 复用安全是否保持
- [ ] `export_coco_dataset.py` 是否仍自动发现 `data/annotations/cleaned/*_SAM2_*.gpkg`
- [ ] empty-target chips 与 targeted hard negatives 是否被误删
- [ ] 是否有安全漏洞（命令注入、路径遍历）
- [ ] 是否把 Cape Town 假设写死到 JHB
  - JHB 使用 **EPSG:32735**、`jhb_task_grid.gpkg`、`johnberg/` 路径
  - Cape Town 使用 **EPSG:32734**、`task_grid.gpkg`、`G*` grid 命名
  - `scripts/sync_from_runpod.sh` 的 `^G[0-9]+` 模式不兼容 JHB（已知脆弱点）
  - Grid ID 模式、CRS、task_grid 文件名都不同

## P2 — Should-fix

- [ ] 远端 RunPod 相关改动是否仍通过 `.env` / `RUNPOD_SSH_*` 工作（非硬编码）
- [ ] CRS 处理是否正确
  - Cape Town: `EPSG:32734` (UTM 34S)
  - JHB: `EPSG:32735` (UTM 35S)
  - Tiles: `EPSG:4326`
- [ ] 是否需要更新 CLAUDE.md / AGENTS.md stable skeleton
- [ ] `docs/workflows.md` 命令是否仍一致

## P3 — Nice-to-have

- [ ] 遵循现有代码风格
- [ ] 新增配置有合理默认值
