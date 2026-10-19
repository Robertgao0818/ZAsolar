# 实验设计：train/val/eval 跨 split 共用 tile 去重（v1，2026-10-17，只是设计，未跑）

批准依据：葉 2026-10-17 批准 round-1 建议（第 5 题）。GPU 未批。

## 先做的 CPU 核查（已完成，2026-10-17）
脚本 `scripts/split_overlap_check.py`，结果 `scripts/split_overlap_check_20261017.json`（只读 v2d2 mask）：
- 490 张 tile 被不同 split 共用，组成 574 对跨 split mask（train-val 437、eval-train 86、eval-val 51）。
- 544 对有"双方都非 255"的重叠像素，复现了 pending 里的 574/544。
- 这些重叠像素共 391,919,805 个，其中 **99.77% 是双方都为背景**（391,004,813）。
- **72 对**含 PV：任一方为 PV 的像素 863,628 个（约 2,159 m²），双方都为 PV 655,908 个（约 1,640 m²），双方不一致 207,720 个。
结论：泄漏以背景为主，但确有 72 对、约 1,640 m² 的 PV 像素同时出现在 train 与 val/eval，val/eval 分数可能偏乐观，需实测。

## 假设
H：从 train 中去掉与 val/eval 共用的 tile 并加 100 m（1 tile）缓冲后，val/eval 的 object F1 会小幅下降（反映真实泛化）；若下降 < seed 噪声，则说明泄漏对已报数字影响可忽略。

## 设计
| 臂 | 训练集 | 说明 |
|---|---|---|
| D0 | v2d2 原 train | 按原配方重训，作为同条件基线 |
| D1 | train 去掉与 val/eval 共用的 tile，再去掉其 100 m 邻接 tile | val/eval 保持不变，历史 val 数字可比 |
其他全部相同（超参、步数、seed、阈值规则、zaeval-2 评分）。去重后的 tile 列表写入新目录并记 md5；原 manifest 不动。预计 train 减少约 437+86 对涉及的 tile 及缓冲，需先用 CPU 脚本统计确切数量与 PV 面积损失后再开跑。

## 数据
v2d2（manifest v3.2）；评估 val 全集 + eval，另报 Li 测试集 group A（按冻结规则，冻结后才看分）。Li 测试集与训练 0 重叠（最近 22,970 m），不受本实验影响。

## 指标（并排）
object P/R/F1 @0.3/0.5/0.7；小板召回 <4/4–10/≥10 m²；BF@0.25/0.5 m（主）、0.3/1.0 m（诊断）；pixel IoU；另单独报"受共用 tile 影响的 val 格"子集与其余子集的差。

## GPU 预算（T = 3.2 GPU-h）
D0、D1 各 1 seed = 2T ≈ 6.4 GPU-h；评估约 +1 GPU-h。差异在噪声附近时各补 2 seed：+4T ≈ 12.8 GPU-h。上限约 20 GPU-h（估计）。若与 exp_train_rasterisation 合并，D0 可与 R0 共用（省 1T）。

## 停止条件
- CPU 统计显示 D1 损失 train PV 面积 >10% → 先报葉，改用"只删含 PV 共享像素的 tile"变体再定。
- D1 与 D0 差 < seed 噪声 → 结论"泄漏影响可忽略"，停止补 seed。
- 训练异常或超出 GPU 上限即停。

## 支撑的决定
下一版训练划分是否按 tile 去重加缓冲；已报的 val/eval 数字（如 v2c 0.926、v2d1 0.908）是否需要加"可能偏乐观"的注释或重算。
