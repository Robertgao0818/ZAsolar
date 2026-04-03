# Experiment Review Checklist

审查训练实验的配置、结果解读和结论。

## P1 — Blockers

- [ ] 结论是否明确标注所用评估 profile（`installation` / `legacy_instance`）及其含义
- [ ] 训练/验证集有无数据泄露
  - Leakage 边界是 **suite/grid 独立性**：`cape_town_independent_26` 的 26 个 grid 不参与训练（见 `configs/benchmarks/post_train.yaml`）
  - T1/T2 是**质量分层**，不是 train/eval split
- [ ] 评估用的是哪个 suite？结论是否与 suite role 对应？
  - `primary` (`cape_town_independent_26`): 用于排名决策
  - `diagnostic` (`cape_town_batch003_diagnostic`): 仅分析用途，不用于排名
  - `secondary` (`jhb_transfer_6`): 跨城市 transfer，不参与主排名
  - `smoke` (`cape_town_t1_smoke`): 快速回归检测
- [ ] F1/Precision/Recall 数字是否与 `summary.json` 一致（reviewer 从远端独立核验）
- [ ] auto-verdict (`improved`/`regressed`/`flat`/`mixed`/`failed`) 是否完整且用法正确
  - `improved`: primary suite F1 delta >= +0.005
  - `regressed`: primary suite F1 delta <= -0.005
  - `flat`: delta 在 (-0.005, +0.005) 之间
  - `mixed`: F1 improves but Precision OR Recall drops > 0.02
  - `failed`: runtime errors or missing results
- [ ] 若引用 `jhb_transfer_6`，是否明确它是 secondary / cross_city，不参与主排名

## P2 — Should-fix

- [ ] 超参数选择有无合理解释
- [ ] 训练 loss 曲线是否收敛？过拟合迹象？
- [ ] hard negative 策略是否记录
- [ ] 结论是否过度声称（如基于 diagnostic suite 声称 generalization）
- [ ] 若结论基于 RunPod 远端，handoff 是否给出远端路径与命令证据
- [ ] 若使用 `legacy_instance` profile，结论是否显式标注并说明理由

## P3 — Nice-to-have

- [ ] 实验命名符合 `exp_NNN` 规范
- [ ] 结果归档到 `docs/experiment-archive/`
- [ ] `configs/model_registry.yaml` 是否更新
