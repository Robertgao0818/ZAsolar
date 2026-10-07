# 存在性门试点

方案：`docs/plans/2026-09-23-yolo-presence-gate-pilot.md`。数据写到 `~/zasolar_data/gate_pilot_2026-09/`，不进 git。

## 顺序

1. `python3 scripts/gate_pilot/step0_vector_feasibility.py`

   只拉矢量。用 `SlidingWindowDataset` 在几何载体 GeoTIFF 上枚举 chip（仿射与 `download_tiles.py` 相同，不是市政影像）。CPT1238 上若 `0.99 × P_pre > 20%`，脚本以退出码 2 停，不下载影像。

   超过几分钟就放进 tmux，日志在 `~/zasolar_data/gate_pilot_2026-09/logs/step0_vector_feasibility.log`。

```bash
tmux new -d -s yolo_gate_step0 'cd /home/gao/projects/ZAsolar && python3 -u scripts/gate_pilot/step0_vector_feasibility.py'
```

结果写到 `docs/experiments/2026-09-yolo-gate-pilot/RESULT.md`。步骤 0 若判停，就不做后面的步骤。

2. `python3 scripts/gate_pilot/step1_imagery_parity.py`

   只在步骤 0 的 `feasibility.json` 写着 `pass` 之后跑。下载 6 个格的市政航拍，用真 GeoTIFF 重算 chip，再对 4 个有光伏的格重跑 `detect_direct.py` 和 `finalize.py`。输出在 `parity/`，不覆盖普查结果。匹配率低于 95% 或数量差超过 5% 时退出码 2。

```bash
tmux new -d -s yolo_gate_step1 'cd /home/gao/projects/ZAsolar && python3 -u scripts/gate_pilot/step1_imagery_parity.py'
```

3. `python3 scripts/gate_pilot/step2_chips_and_labels.py`

   步骤 1 的 chip 窗口和影像匹配都通过之后才跑。给六个格写 chip PNG；YOLO 框只写四个训练格。留出格如果已经有 `labels/` 目录，脚本会停。

4. `/home/gao/venvs/yolo_gate/bin/python scripts/gate_pilot/step3_train.py`

   只在四个训练格上训。折外阈值写入 `~/zasolar_data/gate_pilot_2026-09/frozen_gate.json` 之后，才允许跑步骤 5。

5. `/home/gao/venvs/yolo_gate/bin/python scripts/gate_pilot/step5_holdout.py`

   用冻结阈值给 CPT1238 和 E_test 打一次分。退出码 2 表示预注册的三条没有同时过。

   步骤 1 的 chip 窗口和影像匹配都通过之后才跑。给六个格写 chip PNG；YOLO 框只写四个训练格。留出格如果已经有 `labels/` 目录，脚本会停。

## 留出格

CPT1238 和抽中的 E_test 可以下载影像，也可以做 Mask R-CNN 一致性重跑。冻结文件写好之前，门的任何版本都不能在这两格的 chip 上推理。

## 德班小批对照（2026-10-08，尚未开跑）

`python3 scripts/gate_pilot/pregate_pair.py print-plan --grids DBN#### --model-path <检测器权重>` 只打印命令。`run` 才做两次检测。它不下载底图，不删瓦片，也没有全城开关。说明见 `docs/experiments/2026-09-yolo-gate-pilot/PLAN-durban-pregate-pair-2026-10-08.md`。这不是留出格打分，也不写 frozen_tau。
