# CPU 端到端测试报告

2026-10-18（Pacific/Auckland）。首次端到端评估、渲染及独立复算全部通过，未发生需要修复后重试的运行失败。CPU 即中央处理器，本次全部计算均在 CPU 上进行，`CUDA_VISIBLE_DEVICES=''`，没有调用 CUDA 推断。

测试使用已有 `smoke_v5_v2f/mit_b2/best_objf1.pt`，其 SHA256 为 `a8a1c25b7243ab712cfb4fc38f66d4a421b69c5b8e3c13760f9d5fd65f599c9c`。检查点按原训练模型构造后严格加载，无缺失或多余权重。评分器提交为 `8044d1447a92cc100bbc01976bf66cb4245c89bb`，定义版本 `zaeval-2`，v2f 主清单 MD5 为 `f182b9c3c770e533aa7e629ca529550e`。

## 执行范围与命令

测试覆盖 val 的 G1689 和 eval 的 G1686，各选择一张 PV 较多及一张 SWH 较多的完整 2000×2000 瓦片。这既检查了跨窗口融合，也使两类真值都进入评分；不是随机抽样，也不是完整网格评分。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B eval_phase1.py \
  --ckpt /home/gao/capetown_eval/ds3m_ft/train2_v2e_20261017/smoke_v5_v2f/mit_b2/best_objf1.pt \
  --arch mit_b2 --out ./test_run_smoke --split both \
  --max-grids 1 --grids G1689,G1686 --max-tiles-per-grid 2 --device cpu --threads 4
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B render_overlays.py \
  --eval-dir ./test_run_smoke --out ./test_run_smoke/overlays --allow-partial
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B recompute_pixel_iou.py \
  --eval-dir ./test_run_smoke --out ./test_run_smoke/independent_pixel_iou.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B check_contracts.py
```

瓦片为：

- val：`E260800_N6220300`、`E260700_N6220600`。
- eval：`E258400_N6220600`、`E258200_N6221100`。

每张瓦片执行 16 个 640×640 滑窗，步长 480，共 64 个滑窗；使用 CPU float32、Hann 权重融合和训练源码的 `norm()`。运行环境 Python 3.14.8；依赖版本见 `requirements_pinned.txt`。

## 保存文件与独立复算

已保存 4 张地理配准的单波段 uint8 预测，实际类别值为 0 和 1，满足只允许 0/1/2 的约束。此早期 smoke 检查点没有预测出类别 2。输出保留原始 2000×2000 尺寸、EPSG:32734 坐标系与像素位置。

独立 NumPy 脚本没有导入流水线或 zaeval，只读取保存的预测和原始真值掩膜。IoU 指交集面积除以并集面积；255 真值像素不参与。结果如下：

- val：交集 25,297，并集 37,290，PV IoU = **0.6783856261732368**；TP/FP/FN 为 25,297 / 4,853 / 7,140。
- eval：交集 125,021，并集 145,032，PV IoU = **0.8620235534226929**；TP/FP/FN 为 125,021 / 5,056 / 14,955。
- 两个 split 的 TP/FP/FN、并集整数均与 `metrics.json` 完全一致，IoU 绝对误差均为 **0**，通过 `1e-12` 阈值。

这说明保存掩膜与报告采用的像素计算一致，而不是只验证内存中的中间预测。若跳过这一步，保存时的类别、尺寸或路径错误可能在汇总分数中被掩盖。证据：`test_run_smoke/independent_pixel_iou.json`、`independent_pixel_iou.log`。

## zaeval 指标与渲染

完整的并排 PV/SWH 指标、各 IoU 阈值面积组、匹配对象及边界计数保存在 `test_run_smoke/metrics.json` 与 `metrics.md`。以下仅用于证明流水线实际产生了非空评分：

- val 的 PV 真值对象 8 个、预测对象 13 个；IoU=0.5 时对象 P/R/F1 为 0.461538 / 0.750000 / 0.571429。0.25 m / 0.5 m 边界 F 为 0.517527 / 0.744436。
- eval 的 PV 真值对象 7 个、预测对象 14 个；IoU=0.5 时对象 P/R/F1 为 0.357143 / 0.714286 / 0.476190。0.25 m / 0.5 m 边界 F 为 0.484080 / 0.774508。
- 两个 split 各有 2 个可见确认 SWH 真值对象，真值像素分别为 18,645 和 19,930；模型没有预测 SWH，所以其像素 IoU、对象召回和边界 F 为 0。这是这个 smoke 模型在这些瓦片上的漏检结果，并非用“未标注”替代了 SWH 评估；不能据此判断完整训练后的 SWH 表现。
- 缺少邻瓦片时，zaeval 规则剔除距瓦片边缘 1 m 内的边界：PV 的 val/eval 分别剔除了 274 / 786 个边界像素。忽略这个子集效应，会把部分边界评分误认为完整网格的结果。

已生成两张 eval 叠图和 G1686 缩略图总览，逐张打开检查：原影像与轮廓对齐，PV 青色、SWH 橙色，真值用实线轮廓，预测用半透明填充；图例完整，忽略区域变暗。可查看：

- `test_run_smoke/overlays/G1686/E258200_N6221100.png`（同时可见 PV、SWH）。
- `test_run_smoke/overlays/G1686/E258400_N6220600.png`（大 PV 屋顶）。
- `test_run_smoke/overlays/G1686_contact_sheet.png`。

渲染入口在正式模式要求全部五个 eval 网格和所有 632 条 eval 预测，本次显式使用 `--allow-partial`；没有假造剩余四个网格的预测或图片。

## 其他验证

`contract_checks.json` 与 `contract_checks.log` 记录了以下通过项：

- 禁止的 test/Li Test 路径、坐标缓存路径及允许名单外网格，在打开前触发断言。负向检查仅使用不存在的合成路径，没有访问真实测试集。
- 已存在的输出目录被拒绝复用。
- 合成的一个多边形横跨两个相邻瓦片后，评分器正确合并为一个预测对象，在 0.3、0.5、0.7 IoU 下都一对一命中；面积正确落入 `<4 m²` 组。
- 非空完美边界在 0.25 m 和 0.5 m 下都得到 F=1。
- Python 文件语法检查、`bash -n run_phase1_eval.sh` 通过。

三臂 `mit_b2`、`segnext_b`、`convnext_t` 还分别通过 CPU 构建及 64×64 合成输入前向检查，输出尺寸均为 `[1,3,16,16]`，所有值有限。记录见 `architecture_checks.json`、`architecture_checks.log`。SegNeXt 使用已有本地 MSCAN 权重；ConvNeXt 不请求下载。只有 mit_b2 使用了本任务指定的实际训练检查点完成端到端测试，不能将构建检查说成另外两臂检查点评估已经完成。

磁盘清单检查确认：14 个 val 网格的 1,768 条原始记录和 5 个 eval 网格的 632 条记录，全部影像与掩膜路径存在；共 2,320 张不同影像。默认进入评估的是 1,703 条合格 val 记录及全部 632 条 eval 记录。`grids_and_tiles.csv` 保留逐行路径及资格。

## 耗时、限制与失败记录

- 评估内部计时：**164.303 秒**，其中四张影像推断合计 **159.508 秒**。
- 单张推断分别约 44.396、43.518、36.298、35.297 秒。
- 渲染内部计时：**1.245 秒**；独立 NumPy 复算：**0.255 秒**；语义/安全契约检查：约 **0.048 秒**。
- 上述内部计时不包含启动解释器及导入第三方包的时间，也不包含预先检查源码与编写文档的时间。依据本次单张推断时间粗略外推，完整 2,335 条记录单臂仅推断约需 26 小时；不同架构和机器会改变耗时。这是安排 CPU 运行的数量级参考，不是三臂实测吞吐量。

本次是非随机、四瓦片的功能测试，没有执行三臂完整评估，也没有用本次分数进行架构选择。若忽略这些限制，将会把瓦片选择和样本规模造成的差异误当成模型优劣。

实际评估、渲染、独立复算和架构检查均首次通过；预期的安全拒绝属于通过项，没有连续失败两次的情况。训练源码、标签、v2f 数据和坐标缓存未修改；没有访问 Li 测试数据，没有运行 GPU 推断、终止进程或推送 Git。唯一有意保留的接口差异是：原样导入 `load_bundle()`、`prep()`、`_validate_body()` 但不执行其缓存/测试集读取或 CUDA 路径，原因和替代范围已在 README.md、RUN.md 记录。
