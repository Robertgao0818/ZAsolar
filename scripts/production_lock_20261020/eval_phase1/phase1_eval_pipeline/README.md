# Cape Town phase-1 checkpoint evaluation

本目录提供 CPU 评估与叠图流水线。CPU 即中央处理器，负责本次全部推断与评分。PV 指光伏板；SWH 指太阳能热水器及泳池加热器。val 指训练验证网格，eval 指本任务允许评估的五个保留网格。只允许冻结的 14 个 val 网格和 5 个 eval 网格，不读取 Li 测试数据，也不读取或修改任何 `coord_cache*.npz`。

开发、测试与日志全部写在本目录。脚本按要求为未来 pod 使用提供默认输出位置；本次没有在该外部位置执行或创建输出。既有输出目录一律拒绝复用。没有安装依赖、修改外部源码、修改标签或训练集、终止进程或推送 Git。

## 环境与来源

在本目录执行以下命令。`requirements_pinned.txt` 是本机 `pip freeze` 中实际导入的软件包子集；zaeval 不安装、不复制，直接从指定工作树导入。

```bash
cd ~/capetown_eval/ds3m_ft/phase1_eval_pipeline_20261018
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
```

需要已有的软件环境及以下只读文件：

- 训练源码：`~/capetown_eval/ds3m_ft/train2_v2e_20261017/train2_v2e.py`。
- MiT 模型构建还需要 `~/capetown_eval/deepsolar3m/infer.py` 及 `~/capetown_eval/deepsolar3m/weights/segformer.pt`；这是原始 `build_model()` 的初始化依赖，之后会严格加载完整评估检查点。
- 评分器：`~/projects/zasolar_evalfix_8044d14_wt`，HEAD 必须为 `8044d1447a92cc100bbc01976bf66cb4245c89bb`，且已跟踪的 `zaeval/` 源码不能有改动。指标定义版本必须是 `zaeval-2`。
- v2f 数据：`~/data/ZAsolar/geid_temporal/ct_seg_trainset_v2f_20261017/data/` 下的 `manifest.csv`、`grid_manifest_v2e.csv`、`collectors.gpkg` 及 CSV 指向的 val/eval 掩膜。主清单 MD5 必须为 `f182b9c3c770e533aa7e629ca529550e`。
- PV 原始多边形使用网格清单的 `label_file`，逐个核验其 MD5；当前路径在 `~/data/ZAsolar/geid_temporal/ct_labels_corrected_v2e_20261017/Capetown/`。PV 去重及类别路由由评分器导入 `~/projects/ZAsolar_segv1_wt/scripts/training/build_ct_seg_trainset_v1.py` 和 `build_ct_seg_trainset_v2.py` 的已有函数完成。
- 运行协议：`~/projects/zasolar_plans_20261017/phase1_run_protocol_v2.md`。
- RGB 影像：见 `grids_and_tiles.csv` 的 `image` 列。所有路径保持清单中的绝对路径；pod 的 `/home/gao` 必须镜像这些文件。

Python 的字节码写入已关闭；临时目录和模型缓存位置固定在本目录 `.runtime/`，模型下载关闭。CUDA 即图形处理器计算接口，在这些脚本中始终禁用；`--device` 仅接受 `cpu`。ConvNeXt 构建使用 `pretrained_convnext=False`，SegNeXt 不下载初始化文件，随后严格加载评估检查点。

导入并复用原始 `ARCHS`、`build_model()`、`norm()`、`object_counts()`。**安全约束优先：**`load_bundle()` 也原样导入，但不调用，因为它会打开坐标缓存并读取 Li 测试清单；`prep()` 和 `_validate_body()` 原样导入但不调用，因为内部写死 CUDA。没有复制、编辑或临时替换这些函数。数据选择直接读取冻结 CSV；滑窗的步长、Hann 加权及归一化遵循原验证实现。导入的 `object_counts()` 仅提供单独的诊断计数，不冒充 zaeval 对象分数或训练选点分数。

## 正式运行命令

协议 v2 要求每个架构一个种子，但没有指定数字或目录命名；训练 CLI 默认种子与现有 smoke 配置均为 `0`。因此按本任务给定的 `<arch>_s<seed>` 命名使用 `_s0`。协议要求主检查点为 `best_objf1.pt`；所有三臂训练结束后再比较，测试集评分仍在暂停状态。本流水线没有新增测试集路径。

在 pod 上，每个架构一条命令，按协议顺序执行：

```bash
HOME=/home/gao bash ./run_phase1_eval.sh mit_b2 0
HOME=/home/gao bash ./run_phase1_eval.sh segnext_b 0
HOME=/home/gao bash ./run_phase1_eval.sh convnext_t 0
```

这些命令分别读取：

```text
/workspace/ct_phase1_20261017/runs/mit_b2_s0/best_objf1.pt
/workspace/ct_phase1_20261017/runs/segnext_b_s0/best_objf1.pt
/workspace/ct_phase1_20261017/runs/convnext_t_s0/best_objf1.pt
```

默认新建输出目录分别为：

```text
/home/gao/capetown_eval/ds3m_ft/phase1_runpod_20261017/eval/mit_b2_s0/
/home/gao/capetown_eval/ds3m_ft/phase1_runpod_20261017/eval/segnext_b_s0/
/home/gao/capetown_eval/ds3m_ft/phase1_runpod_20261017/eval/convnext_t_s0/
```

可通过 `PYTHON`、`CKPT_ROOT`、`OUT_ROOT` 设置解释器和两个根目录；架构子目录名不变。输出存在时停止，不自动覆盖、清理或续写。长时间正式 CPU 运行请放在已有 `tmux` 会话中（tmux 是让终端任务在断线后继续运行的工具），把标准输出重定向到自己新建的日志；这些脚本不会终止进程。本机四瓦片测试的推断时间粗略外推，完整单臂仅推断约需 26 小时，因此应预先安排运行时间；这不是其他机器或架构的实测耗时。`--max-grids` 等限制参数用于直接调用下方 Python 命令做检查，正式 shell 命令默认评估全部网格并渲染五个 eval 网格。

也可分别调用：

```bash
python -B eval_phase1.py \
  --ckpt /workspace/ct_phase1_20261017/runs/mit_b2_s0/best_objf1.pt \
  --arch mit_b2 --split both --device cpu \
  --out ./eval_mit_b2_s0
python -B render_overlays.py --eval-dir ./eval_mit_b2_s0 --out ./eval_mit_b2_s0/overlays
python -B recompute_pixel_iou.py --eval-dir ./eval_mit_b2_s0 --out ./eval_mit_b2_s0/independent_pixel_iou.json
```

`--arch` 使用导入的 `ARCHS`，包括 `mit_b2`、`mit_b3`、`segnext_b`、`convnext_t`；协议正式三臂不包含 `mit_b3`。默认 `--split both`，另支持 `val` 和 `eval`。`--max-grids N` 对每个 split 分别生效，按网格 ID 排序取前 N 个；`--grids` 可显式选择允许名单内的网格。`--max-tiles-per-grid N` 仅供 smoke：交替选取 PV 像素最多、SWH 像素最多的瓦片，去重，路径作为最终排序依据。这种非随机小子集用于检查连通性，不能估计总体表现。任一子集参数都会将结果标为 `partial: true`。

## 网格、路径及完整性

`grids_and_tiles.csv` 列出所有 2,400 条 val/eval 网格-瓦片记录及影像/掩膜绝对路径，共 2,320 张不同影像。检查时全部影像与掩膜均存在。`image_exists`、`mask_exists` 为逐行检查结果；`included_default` 说明默认是否进入推断与评分。

- val：G1293、G1354、G1410、G1523、G1629、G1630、G1689、G1691、G1743、G1798、G1863、G1920、G1979、G2028。
- eval：G1349、G1686、G1692、G1800、G2038。主瓦片清单写作 `eval`，网格清单写作 `eval_orig10`；只使用 v2f 实际包含的这五个，不扩展到名字暗示的另外五个。

默认 val 使用原训练 `load_bundle()` 的资格规则：有效像素大于 100,000，且影像不出现在 eval 中，共 1,703 条。65 条不进入默认评估，其中训练逻辑剔除 47 条与 eval 共享影像的合格记录，其余 18 条未达到有效像素门槛。eval 使用全部 632 条。这里是完整合格 val 池，**不是训练时固定的 300 张选点子集**；忽略这个区别会把评估范围变化误当成模型性能变化。同一 split 内若影像被相邻网格引用，保留各网格独立掩膜作为独立观察，不把全 split 当作去重后的地理像素总体。

## 输出和指标解释

- `preds/<split>/<grid>/<tile>.tif`：GeoTIFF，即保留地理位置的栅格影像；单波段 `uint8`，仅含 0=背景、1=PV、2=SWH。保留原尺寸、坐标变换及 EPSG:32734 坐标系，不把背景 0 标为缺失值。
- `predictions.json`：预测、真值、影像路径、SWH 资格、滑窗数及每张推断时间。
- `metrics.json` 与 `metrics.md`：PV/SWH 并排显示对象 P/R/F1、面积组召回、边界 F 与像素 IoU。JSON 保留评分器原始对象计数、分组、每网格结果、匹配及边界排除计数，同时记录检查点 SHA256、源码/清单摘要、评分器提交和软件版本。
- `overlays/<grid>/<tile>.png`：影像上叠加半透明预测填充及原始 GT 多边形轮廓。GT 即人工真值。PV 为青色，SWH 为橙色，忽略区域变暗；每张图含图例。另有每网格 contact sheet（缩略图总览）及 `render_manifest.json`。默认要求完整五个 eval 网格；小子集必须显式传 `--allow-partial`。

推断默认 640×640 窗口、步长 480；若检查点旁 `config.json` 记录其他合法 chip，则采用该尺寸。先双线性放大各窗口的类别输出、转为概率，再按训练验证的 Hann 权重平滑融合，最终取最大概率类别。整个过程使用 CPU float32；SegNeXt 随机分解采用训练验证的固定种子 12345，可重复运行。

像素 IoU 是有效像素上“交集除以并集”，真值来自 v2f 掩膜，255 不计分；没有前景且没有误报时记为 null，避免将无证据误报成完美成绩。PV 与 SWH 分别计算。SWH 只在 `swh_labelled=True` 的网格评分；未标注网格返回 `not_labelled`，不能解释为没有热水器。

对象指标直接调用指定 zaeval-2 的 `ObjectAccumulator`：PV 使用已去除热水器、去重和修复的原始多边形；SWH 使用 v2f collectors 表里 `mask_value=2` 的确认多边形；原始多边形按像素中心栅格化。`exclude-pixels` 策略排除 255，预测的八邻接连通块跨瓦片边缘合并后，去除小于 0.25 m² 的预测对象，以 0.3/0.5/0.7 IoU 作一对一匹配。精确率 P 表示预测中有多少被正确匹配，召回率 R 表示真值中有多少被找到，F1 综合二者。面积分组为评分器默认的 `<4`、`4–10`、`>=10 m²`，以真实多边形面积而非腐蚀后的掩膜像素数分组；保留各阈值的命中数和分母，JSON 另有兼容旧版的面积组。

边界 F 直接调用 `BoundaryAccumulator`，容差为 0.25 m 和 0.5 m。使用原始多边形边界，保留对象边缘的忽略带，排除远处忽略区域；拼接同网格已预测邻瓦片的 halo（即邻瓦片边缘像素），缺少任一正交邻居时按评分器规则排除瓦片边缘 1 m 内的边界。小子集会缺少更多邻居，排除量随结果保存，不能把它等同于完整网格的边界评估。原评分器在双方都无边界时返回 F=1；Markdown 显示 N/A 并解释该约定，JSON 原样保存计数及原始分数。

所有跨瓦片结果先累加计数再求比率，避免空瓦片与有对象瓦片获得同样权重。训练的 `best_objf1` 使用覆盖率命中规则、PV 概率阈值和每瓦片对象；本流水线使用保存的三分类 argmax 掩膜及 zaeval 一对一 IoU 匹配，所以两者不能直接相减。忽略这个区别会错误地归因于评分实现或模型退步。

## 本机端到端测试

首次执行以下命令生成 `test_run_smoke/`；目录已存在时应使用新的目录名，保留原结果。

```bash
CUDA_VISIBLE_DEVICES='' python -B eval_phase1.py \
  --ckpt ~/capetown_eval/ds3m_ft/train2_v2e_20261017/smoke_v5_v2f/mit_b2/best_objf1.pt \
  --arch mit_b2 --out ./test_run_smoke --split both \
  --max-grids 1 --grids G1689,G1686 --max-tiles-per-grid 2 --device cpu --threads 4
CUDA_VISIBLE_DEVICES='' python -B render_overlays.py \
  --eval-dir ./test_run_smoke --out ./test_run_smoke/overlays --allow-partial
CUDA_VISIBLE_DEVICES='' python -B recompute_pixel_iou.py \
  --eval-dir ./test_run_smoke --out ./test_run_smoke/independent_pixel_iou.json
```

独立复算脚本只导入 NumPy 和 rasterio，不导入流水线或 zaeval；直接读取保存的预测和原掩膜，用布尔运算重算 PV 交集、并集、TP/FP/FN，逐 split 与 `metrics.json` 核对整数计数和 IoU，允许浮点误差上限 `1e-12`。测试结果、耗时和局限见 `TEST_REPORT.md`；过程及失败记录见 `LOG.md`。同一步骤若连续失败两次，停止并记录，不自动重试。
