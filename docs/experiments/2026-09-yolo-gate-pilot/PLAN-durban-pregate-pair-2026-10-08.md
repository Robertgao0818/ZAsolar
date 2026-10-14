# 德班存在性门小批对照 — 2026-10-08 用的代码

日期：2026-10-07 写好。状态：2026-10-07 已在 RTX 4090 上跑完重建后的 10 格，见 [`RUN-durban-pregate-pair-2026-10-07.md`](RUN-durban-pregate-pair-2026-10-07.md)。同一天负责人看过低分例子，认定对照轮廓是未经人工筛选的机器识别，阈值保持 0.45。接着决定下一步全城跳过 solar_cls、使用这扇门。全城循环已写在 `scripts/gate_pilot/durban_citywide.py`，本机测试通过，还没同步到 Pod，扫描还没开始。RunPod 模板还没改，配对脚本仍然没有全城开关。

存在性门是检测器前面的一道筛子：每张 400 像素小图先由冻住的 DINOv2 打一个存在分数，分数不够就不把这张小图送进 Mask R-CNN。小图是从航拍瓦片上切下来的窗口。DINOv2 是这道门用的视觉骨干，本次只使用已经训好的线性小头，不重新训练。

## 这次准备了什么

`detect_direct.py` 增加了可选参数 `--pregate-head`。不传它时，每一张小图仍进检测器，输出和原来一样，全城 RunPod 模板也没有这个参数。传入一个或多个 `head.pt` 时，小图按下面的规则决定是否进检测器，被拒绝的小图仍写进 `raw_detections.pkl`，只是检测列表为空，方便事后核对窗口。旁边会多两个文件：`stage_timing.json` 和 `pregate_decisions.csv`。

`scripts/gate_pilot/pregate_pair.py` 对同一批明确写出的德班格各跑两次。一次不加门，一次加门。两次都用锁定的后处理：`configs/postproc/v4_canonical.json`，合并方式 `per-detection`。后处理指 `finalize.py` 把检测掩膜收成多边形。solar_cls 不在这次对照里。门若把一块真光伏挡在检测器外，后面的分类器也看不到它。

默认输出在 `~/zasolar_data/gate_pilot_2026-09/durban_pregate_pair_20261008/`，不进 git。已有 `raw_detections.pkl` 时脚本拒绝覆盖，也不会自己删。

## 0.45 在这次试跑里怎么用

0.45 是负责人在开普敦 26 格上确认的起点：折外分数大于或等于 0.45 才送去检测。折外分数指该格没有参与训练的那一折模型打出的分数。盘上有四份头：`vit30_fold0` 到 `vit30_fold3` 的 `head.pt`。第五折和最终重训头没有训练，所以不用。

德班小图不在任何一折里，没有「该用哪一折」的指定。试跑因此把四份头都打上去，取最高分。最高分大于或等于 0.45 才进检测器；四份都低于 0.45 才跳过。这比任意单份头更不容易漏，也比开普敦「每张小图只用一份折外头」更松。它是试跑规则，不是把开普敦工作点原样搬到德班，更不是德班图上已经验证过的阈值。

每张小图的四份分数都留在 `pregate_decisions.csv`。那张表能回答「若只信某一份头，会多拒绝多少张小图」，但不能代替那套更严规则下的多边形。要看单份头实际漏掉哪些轮廓，需要另传一个 `--pregate-head` 再跑一次。

开普敦 26 格在 0.45 下跳过约 83.39% 的小图。那是张数比例。德班还没有这个数，也不能把它当成总工时会减少的比例。

## 时间怎么比

每次检测写出四段时间：

- `batch_wait_s`：主进程等下一批小图的时间。读图工人和显卡是重叠的，所以这段只是没能被计算盖住的等待，不是全部读图时间。
- `pregate_s`：DINOv2 打分时间。不开门时为 0。
- `detect_s`：检测器前向时间，只包括真正送进去的小图。
- `loop_wall_s`：这一圈的墙钟，已经包含读图、打分和检测，三段相加不等于它。

`finalize` 另记 `finalize_process_s`。检测进程本身的墙钟记为 `detect_process_s`，含模型加载。

对照报告里的 `work_s` 是 `loop_wall_s` 加后处理。`process_s` 再把模型加载算进去。要看省了多少时间，看 `work_s_ratio_on_over_off` 和 `process_s_ratio_on_over_off`。`chip_skip_fraction` 只说明跳过了多少张小图。报告里写明它不是省下的时间比例。忽略这一点，就会用开普敦的 83% 去估德班全城的机器时间。

## 轮廓怎么比

不开门的那次多边形是参照。对每一块参照多边形，计算开门结果的并集盖住了它多少面积。容差是 0.01 平方米。

- `missed`：盖住的面积不超过 0.01 平方米。这块安装在开门结果里等于没有。
- `reduced`：盖住了一部分，但露在外面的面积仍超过 0.01 平方米。轮廓变小或被切掉一块。
- `retained`：没盖住的只是容差内的碎边。

`gated_only_area_m2` 是开门结果里、参照并集没有盖到的面积。门本身不会造出新光伏。这个数若明显大于零，先查合并差异或实现问题，不要把它读成「门新发现了安装」。

这些都是检测器多边形之间的差别，不是相对李标注的召回，也不是德班 10 格面积门（bulk、σ_Bw、空格假阳性）的结论。面积门仍按 2026-09-28 的备忘，用不开存在性门的锁定链去判。

小批即使一块都没漏，也只说明这几个格在这套四头取最大规则下没有漏。它证明不了全城零漏检。全城启用要等这批评定可以接受之后另定配置。直接全城打开时，被门拒绝的小图不会进检测器，真光伏可能整块消失，而且没有不开门的对照可以回看。

## 底图

脚本不下载、不删除、不等待存在性门。`run` 在第一张卡启动前检查列出的格是否已有瓦片；缺任何一格就整批停下。瓦片要已经在注册图层 `vexcel_ortho_2025_12_2026_01` 的目录里，也就是 `~/zasolar_data/tiles/vexcel/durban/ortho/<格号>/` 下的 `*_geo.tif`。正式图选哪一种、坐标怎么改，沿用已经写下的记录；这次代码不改 `regions.yaml` 里的产品字段。

格号必须是现在瓦片目录用的 `DBN####`。四月样本号和 2026-10-06 重建网格不是同一套编号，不能靠文件名对上就当同一个格。

## 明天怎么开

先看命令，不占显卡：

```bash
cd /home/gao/projects/ZAsolar
source scripts/activate_env.sh
python scripts/gate_pilot/pregate_pair.py print-plan \
  --grids DBN0001 \
  --model-path /home/gao/zasolar_data/models/exp_unified_reviewall_A/best_model.pth
```

把 `DBN0001` 换成已经下好瓦片的格，可以列多格，默认不超过 10 个。检测器权重用锁定的 `exp_unified_reviewall_A`。不传 `--pregate-head` 时用上述四份折外头。

瓦片齐了再跑。一次对照包含两次检测加两次后处理，格子一多就会超过半小时，放进 tmux，不要挂在当前会话上：

```bash
tmux new -d -s durban_pregate_pair 'cd /home/gao/projects/ZAsolar && source scripts/activate_env.sh && python -u scripts/gate_pilot/pregate_pair.py run --grids DBN0001 --model-path /home/gao/zasolar_data/models/exp_unified_reviewall_A/best_model.pth'
```

看进度用 `tmux capture-pane -pt durban_pregate_pair:0 -S -80`。结果是输出目录里的 `pair_report.json`、`pair_coverage.csv` 和 `pair_timing.csv`。

## 还不能做的事

- 不在配对脚本里打开全城。单格推理已在 2026-10-07 01:10 写好，而且不调用 solar_cls。全城循环在 `scripts/gate_pilot/durban_citywide.py`。2026-10-07 05:16 UTC 已在 Pod `x6yhvn6rs3vuuu` 的 tmux `durban_citywide` 里开跑，06:12:53 UTC 因句柄泄漏自行退出（fail 719），06:43 UTC 修好后续跑，09:53:46 UTC 结束（fail 0，846 格齐）。产物已拷回 `~/zasolar_data/gate_pilot_2026-09/durban_citywide_pregate_20261008/`，Pod 于 10:05 UTC 停止。运行记录见 [`RUN-durban-pregate-pair-2026-10-07.md`](RUN-durban-pregate-pair-2026-10-07.md)。不把这个循环写进这一份配对计划。RunPod 模板到写这份状态时还没改。
- 不把这次小批写成全城零漏检。2026-10-07 的本机面积对照写在 [`RUN-durban-pregate-pair-2026-10-07.md`](RUN-durban-pregate-pair-2026-10-07.md)，那不是开普敦交并比 ≥ 0.5 的形状通过。
- 不把跳过的小图比例写成节省的总时间。
- 不在这套代码里下载或删底图。
- 不重训门，不跑第五折，不写 frozen_tau。
- 不用德班这批格去改 0.45。2026-10-07 对照多边形已经看过，负责人决定阈值不动，仍是四份折外头最高分 ≥ 0.45。
