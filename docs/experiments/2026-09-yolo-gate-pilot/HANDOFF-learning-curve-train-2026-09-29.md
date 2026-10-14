# Handoff：存在性门学习曲线开训

写于：2026-09-29
目的：下一个会话直接开训。这份文档自洽，不需要翻上一轮对话。
设计真源：`docs/briefs/BRIEF-presence-gate-backbone-2026-09-24.md` 版本 B 的第 2 步。
抽样和取数记录：`docs/experiments/2026-09-yolo-gate-pilot/RUN-learning-curve-2026-09-28.md`。

存在性门是普查 Mask R-CNN 前面的小模型，只判断一张 400 像素小图里有没有光伏。这一轮不换 Mask R-CNN，不把门接进 `detect_direct.py`，不下载德班影像，不做生产机读图天花板（简报里的 S1）。

## 1. 现在停在哪

数据准备已完成，训练没有开始。

| 标记 | 内容 | 核对过的状态 |
|---|---|---|
| `~/zasolar_data/gate_pilot_2026-09/learning_curve_2026-09-28/fetch.done` | 39 格航拍 + 老师多边形 | `ok 39`（2026-09-29 01:53） |
| 同目录 `chips.done` | 小图和训练标签 | `ok 39`（2026-09-29 02:17） |
| 同目录 `selection.json` | 冻结抽样 | `learning_curve_selection_v1`，种子 `20260928` |
| GPU | 本机 RTX 4070 Laptop，8 GB | 开训前再看一眼 `nvidia-smi`。2026-09-29 交接时没有计算进程占卡 |
| 磁盘 | `/home` | 交接时约 65 GB 空闲 |

没有 `train/` 目录，没有新的 `best.pt`，没有这一轮的冻结阈值。

39 格都是 2,058 张小图。30 个训练格各有 2,058 个 YOLO 标签文件。9 个留出格没有 `labels/` 目录。过滤后有板的小图全部落在过滤前有板的小图里，`chip_summary.csv` 的 `post_not_in_pre` 合计是 0。

| 集合 | 过滤前有板小图 | 过滤后有板小图 | 过滤前框 |
|---|---:|---:|---:|
| 训练 10 格 | 2,369 | 1,780 | 6,355 |
| 训练 30 格 | 6,391 | 4,713 | 16,372 |
| 留出 9 格 | 2,174 | 1,741 | 5,898 |

上次失败的那一折只有 179 张过滤前有板小图。这 10 格已经包含 CPT2932（过滤前 778 张），所以不是那一折的重演。若 10 格已经整块不丢，只说明分层后的 10 格够了，不推翻「179 张太少」。

## 2. 这一轮要训什么

三个臂，空小图都不下采样（`selection.json` 里 `empty_chip_keep_ratio` 是 1）：

1. YOLOv11n，只用嵌套的 10 个训练格。
2. YOLOv11n，用全部 30 个训练格。10 格是 30 格的子集。
3. 冻住的 DINOv2 ViT-S/14，加一个 patch 头，只用 30 个训练格。chip 分数是所有 patch 分数的最大值，不用整图分类向量。本地权重缓存是 `~/zasolar_data/models/dinov2_floor/hf_cache/models--timm--vit_small_patch14_dinov2.lvd142m`，timm 名 `vit_small_patch14_dinov2.lvd142m`。

YOLO 超参沿用试点已在这张 4070 上探过的配置，不要重新发明：`yolo11n.pt`，`imgsz` 416，100 epoch，patience 20，batch 32，`flipud` 0.5，`fliplr` 0.5，`workers` 2，`single_cls`，训练种子 `20260923`。解释器是 `/home/gao/venvs/yolo_gate/bin/python`（torch 2.11.0+cu128，ultralytics 8.4.160）。把这些写进 `learning_curve_2026-09-28/train/train_config.json` 再开第一轮。不要读、不要写 `gate_pilot_2026-09/step3/`。

ViT 的 400 像素不能被 14 整除。开训前把输入写成 392×392（28 个 patch，双线性缩放，不裁切），写进同一份配置。ViT 的 batch 另探，不要假设 32 放得下。

## 3. 折和阈值：这一份交接才写死

简报只说「阈值用训练格的折外分数定」，没有写折数。30 次留一格会超出「本机大约一天」的预算。这里锁定 5 折，按 `selection.json` 里各档的顺序轮转。折号是 `下标 % 5`。这是 2026-09-29 的操作化，不是 2026-09-24 简报里的原文。不要重抽。

| 折 | 格 |
|---|---|
| 0 | CPT2932，CPT3695，CPT2373，CPT4235，CPT2216，CPT2729，CPT2330，CPT3865 |
| 1 | CPT3468，CPT3734，CPT2182，CPT2776，CPT2332，CPT2406，CPT3802 |
| 2 | CPT3635，CPT2174，CPT2718，CPT2997，CPT2040，CPT3021，CPT2207 |
| 3 | CPT3246，CPT1506，CPT2390，CPT2616 |
| 4 | CPT3638，CPT2496，CPT2167，CPT3194 |

10 格臂只用这 10 个格：CPT2932、CPT3468、CPT3635、CPT2373、CPT2182、CPT2718、CPT2216、CPT2332、CPT2330、CPT3802。它们落在折 0、1、2。折 3 和折 4 没有 10 格成员，10 格臂跳过这两折。每个训练格的分数必须来自没看过它的那一折。

阈值不再用 `scripts/gate_pilot/threshold.py` 的 chip 召回 99.5%。那个规则是试点的，正是它把阈值压到 0。这一轮对折外分数做如下选择：

- 小图分数大于等于阈值才放行。
- 安装用过滤后多边形，也就是 `vectors/<格>/predictions_metric_cls_filtered.gpkg`。
- 整块丢失：和这块多边形相交的小图一张都没放行。
- 部分丢失面积：多边形减去「放行且与它相交的小图」的并集，在 EPSG:32734 里算面积。算法就是事后脚本 `~/zasolar_data/gate_pilot_2026-09/posthoc_2026-09-24/eval_rules.py` 里的规则 A（只跑放行）。不要用规则 B、C、D。
- 合法阈值必须同时满足：整块丢失 = 0，并且面积 ≥ 30 m² 的安装上，部分丢失面积合计 = 0。
- 取满足这两条的最大阈值。若没有任何正阈值满足，阈值记 0，这一臂的丢失条件判失败。不要改标准再算一次。

`intersections.csv` 的 `poly_id` 是 `pyogrio.read_dataframe(..., fid_as_index=True)` 的 FID。评分时用同一种读法打开 GeoPackage。小图多边形用 `scripts/gate_pilot/chip_windows.py` 的 `chip_windows_from_tifs` 从航拍现算，不要另写一套窗口。

折外阈值写成 `learning_curve_2026-09-28/train/frozen_tau_<臂>.json` 之后，才允许对 9 个留出格预测一次。留出格是 CPT3247、CPT2119、CPT2714、CPT2308、CPT2957、CPT4369、CPT2035、CPT3577、CPT3758。最终权重是用该臂的全部训练格再训的一个模型，阈值仍用折外分数，不用最终模型在训练集上的分数。

留出报告写：整块丢失数、≥30 m² 的部分丢失面积、每格放行率、含读图的 chip/秒。本机 Mask R-CNN 大约 20.24 chip/秒、约 102 秒/格。可以报本机相对耗时，但不要用它宣布简报 S2 的速度条款通过。S2 的 0.5 / 0.2 是相对生产机大约 16.7 秒/格。那次读图天花板没做。

S4 的读法：10 格到 30 格，折外整块丢失数没有下降，才说「再从 10 加到 30 没有用」。两边都是 0，同样算没有下降，结论是 10 格已经够，不是「数据量从来无关」。

## 4. 不要做的事

- 不要跑 `scripts/gate_pilot/step3_train.py` 或 `step5_holdout.py`。它们锁在旧的 4 个训练格、chip 级阈值，以及已经看过的 CPT1238、CPT2172。
- 不要把 CPT1238、CPT2172 放进训练或留出。
- 不要给留出格写 `labels/`。训练路径里出现任何一个留出格编号就停。
- 不要把瓦片下到仓库的 `tiles/`。下载脚本要设置 `SOLAR_TILES_ROOT=/home/gao/zasolar_data/tiles/cape_town/aerial_2025`。这一轮不需要再下载。
- 不要重跑 Mask R-CNN，除非留出评分已经写完、并且用户明确要求做简报第 2 步的第 6 小步（3 个留出格的端到端）。那一步要给 `detect_direct.py` 加「只跑指定 chip」。
- 不要改开普敦普查锁，不要开始德班 10 格探针。德班决定在 `solar_backdating/docs/briefs/BRIEF-durban-census-goal-2026-09-28.md`。

CPT2776 的备份没有 `predictions.geojson`。两个 GeoPackage 都在，这个格留在 30 格和折 1 里。

## 5. 代码和数据位置

未提交。不要自行 commit。

ZAsolar 工作树里这次新加或改过的文件：

- `scripts/gate_pilot/learning_curve_select.py`
- `scripts/gate_pilot/learning_curve_fetch.py`
- `scripts/gate_pilot/learning_curve_chips.py`
- `scripts/gate_pilot/step2_chips_and_labels.py`（给写图、写标签加了可选输出目录，试点默认路径不变）
- `tests/gate_pilot/test_learning_curve_select.py`（2026-09-28 当时 3 项通过）
- `docs/experiments/2026-09-yolo-gate-pilot/RUN-learning-curve-2026-09-28.md`
- `docs/briefs/BRIEF-presence-gate-backbone-2026-09-24.md`（顶部加了 2026-09-28 重开说明）

数据只在 `~/zasolar_data/gate_pilot_2026-09/learning_curve_2026-09-28/`：

- `chips/<格>/images/*.png`
- `chips/<格>/labels/*.txt`，仅训练格
- `chips/<格>/manifest.csv`，列含 `y_pre`、`y_post`、`y_post_small`
- `chips/<格>/intersections.csv`
- `vectors/<格>/predictions_metric.gpkg` 和 `predictions_metric_cls_filtered.gpkg`
- `chip_summary.csv`

四个旧训练格的小图是指向 `~/zasolar_data/gate_pilot_2026-09/chips/` 的硬链接。不要删除那份试点小图。

航拍在 `~/zasolar_data/tiles/cape_town/aerial_2025/<源格号>/`。CPT 编号对应的目录是 G 编号，例如 CPT3468 的瓦片在 `G3468`。

## 6. 下一个会话的开工顺序

1. 读这份交接和抽样记录。确认两个 `.done` 仍是 `ok 39`，留出格仍然没有 `labels/`。
2. 新写 `scripts/gate_pilot/learning_curve_train.py`。没有 `chips.done` 就退出。输出只写到 `learning_curve_2026-09-28/train/`。已有 `best.pt` 的折跳过。
3. 用 tmux，会话名 `yolo_gate_lc_train`。不要把训练挂在会话进程上。

```bash
tmux new -d -s yolo_gate_lc_train 'cd /home/gao/projects/ZAsolar && /home/gao/venvs/yolo_gate/bin/python -u -m scripts.gate_pilot.learning_curve_train >> /home/gao/zasolar_data/gate_pilot_2026-09/learning_curve_2026-09-28/logs/train.tmux.log 2>&1'
```

4. 顺序：10 格的折外训练和折外打分，30 格的折外训练和折外打分，各自冻结阈值，再训两个最终 YOLO。ViT 用同一套折和同一条阈值规则。阈值文件落盘前，预测函数若收到留出格就退出。
5. 留出只评一次。结果写进 `docs/experiments/2026-09-yolo-gate-pilot/RESULT.md` 的新一节，并改抽样记录顶部的状态。没有留出结果就不要写 S2–S5 的判决。速度判决留空，写明 S1 未测。
6. 试点 4 格训练大约 1–3 小时量级（简报里的外推，不是这一轮的实测）。30 格、5 折会更长。先跑完 10 格的第一折，用日志里的耗时修正总时间，再决定是否把其余折留在同一个 tmux 里过夜。

开训前的检查命令：

```bash
test "$(cat ~/zasolar_data/gate_pilot_2026-09/learning_curve_2026-09-28/chips.done)" = "ok 39"
test -z "$(find ~/zasolar_data/gate_pilot_2026-09/learning_curve_2026-09-28/chips/CPT3247/labels \
  ~/zasolar_data/gate_pilot_2026-09/learning_curve_2026-09-28/chips/CPT2035/labels \
  -type d 2>/dev/null)"
cd /home/gao/projects/ZAsolar && python3 -m pytest tests/gate_pilot/test_learning_curve_select.py -q
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

第二条用留出里的一个密格和一个空格做哨兵。任何一个留出格出现 `labels/` 都停。
