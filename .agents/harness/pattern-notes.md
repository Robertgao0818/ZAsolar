# Pattern Notes — 探索笔记

供 clean agent 直接读取的关键发现，避免重新探索。

## 2026-04-03: Harness 设计过程中的发现

### 1. Leakage 边界不是 T1/T2

- T1/T2 是**质量分层**（reviewed vs original），不是 train/eval split
- Leakage 边界是 **suite/grid 独立性**：`cape_town_independent_26` 的 26 个 grid 不参与训练
- 定义在 `configs/benchmarks/post_train.yaml` 的 suite 配置中

### 2. Benchmark CLI 实际 flag

- 指定 checkpoint 用 `--checkpoint`，不是 `--weights`
- 输出路径格式：`results/benchmark/{run_name}_{timestamp}/summary.json`
- 见 `scripts/analysis/run_benchmark.py`

### 3. JHB 与 Cape Town 的关键差异

| 维度 | Cape Town | JHB |
|------|-----------|-----|
| Metric CRS | EPSG:32734 | EPSG:32735 |
| Grid 命名 | G{4位数字} | 非 G* 模式 |
| Task grid | task_grid.gpkg | jhb_task_grid.gpkg |
| Tile 目录 | tiles/G{id}/ | tiles/johnberg/ |
| sync_from_runpod.sh | `^G[0-9]+` 匹配 | **不兼容**（已知脆弱点） |

### 4. V1.2 → V1.3 任务定义迁移

- 旧定义：installation-level footprint segmentation（每个 polygon = 一个 installation）
- 新定义：reviewed prediction footprint segmentation（模型预测经人工审查后导出）
- 变化原因：batch_finalize_reviews.py 导出 `review_status==correct` 到 cleaned/，无 installation merge 步骤
- `installation` profile 名字保留，代码逻辑不改（选项 C）
- GT 标注标准不变（仍遵循 installation-level merge/boundary 规则）

### 5. Reviewer 隔离限制

- Claude Code settings.json 允许 Edit/Write，PreToolUse hook 仅拦截特定 git add
- Reviewer 只读约束是 **brief 指令约定**，非技术强制
- Codex 侧无 Claude Code hooks，完全靠 reviewer-brief 约束
