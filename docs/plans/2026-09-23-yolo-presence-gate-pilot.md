# 执行方案 — YOLO 存在性门试点（开普敦，6 个网格）

日期：2026-09-23
状态：READY，交给执行 agent
上游决策：`docs/briefs/BRIEF-census-detector-speed-2026-09-23.md` §5「最小探针」
仓库：ZAsolar（census 引擎）。本试点**不改** `detect_direct.py` / `finalize.py` / solar_cls，也不碰已交付库存。

---

## 0. 这轮要回答的一个问题

一个在开普敦航拍 chip 上微调过的单类 YOLOv11n，能不能在**没训练过的网格**上做到：凡是含最终光伏的 chip 一张不丢，同时把该格至少 80% 的 chip 挡在 Mask R-CNN 外面。

能，才有资格接到 Mask R-CNN 前面（那是下一轮，不在本方案内）。不能，就按 §7 止损，不调到“差不多”为止。

## 1. 通过标准（预注册，执行中不得修改）

只在**留出格**（CPT1238 + 一个留出空格）上判定。三个数必须同时成立：

| 编号 | 指标 | 门槛 | 标签来源 |
|---|---|---|---|
| G1 | 过滤后光伏 chip 召回 | **= 100%**（漏 0 张） | `predictions_metric_cls_filtered.gpkg` |
| G2 | 过滤前检测 chip 召回 | **≥ 99%** | `predictions_metric.gpkg` |
| G3 | 门放行率 | **≤ 20%**，每个留出格各自单独算 | 门的输出 |

G1、G2 在 CPT1238 上算（空格没有正样本）。G3 两个留出格都要过。

「相交」的定义：chip 的有效像素矩形（`ChipMeta.valid_window` 对应的地理范围，不含零填充）和多边形的交集面积 > 0。一条细缝也算。这是负责人要的严口径，不许加面积下限。

必须同时报告、但**不参与**判定的诊断数（写进结果文档）：

- 小于 30 m² 单独报：(a) 与任一过滤后 <30 m² 多边形相交的 chip 的召回；(b) 多边形口径——过滤后 <30 m² 多边形里，有多少块的**全部**相交 chip 被放行、有多少块**至少一张**被放行（后者为 0 的块 = 整块安装从普查消失）。≥30 m² 同样报一份。
- G1 若有漏的 chip：逐张列出 chip id、相交多边形 id、面积、交集面积占 chip 与占多边形的比例、门分数，并存 chip 图。只描述，不因此改判。
- 速度：同一张本机 GPU 上，门的 chip/秒（含读图解码）和 `detect_direct.py` 的 chip/秒；按 `相对耗时 = 门单价/MaskRCNN单价 + 放行率` 算出预期倍数。brief 的「低于 2 倍就停」在接入生产那一轮使用，本轮只报。
- 两个留出格上完整的召回–放行率曲线（阈值从 0 扫到 1）。只作图，不许用它回头挑阈值。

## 2. 已核对的事实（2026-09-23，执行前请勿再假设）

- 备份：`dropbox:RA_Solar/Gao/ct_census/results/unifiedA_census_perdet/<CPT id>/`，每格 4 个文件：`config.json`、`predictions.geojson`、`predictions_metric.gpkg`、`predictions_metric_cls_filtered.gpkg`。两个 gpkg 都是 EPSG:32734，带 `chip_index`、`source_tile`、`area_m2`、`score` 列。
- **单格文件与交付库存的计数对不上**，执行时必须处理：

  | 网格 | 过滤前（单格 gpkg） | 过滤后（单格 gpkg） | 交付合并库存 |
  |---|---:|---:|---:|
  | CPT1238 | 253 | 218 | 183 |
  | CPT2932 | 949 | 796 | 584 |

  合并库存（`results/analysis/ct_census_output_table/ct_full_inventory_2026-06-21_merged.gpkg`）应当做过跨格去重或按 CPT 网格边界裁剪。**本轮 G1 用单格过滤后文件**（它是合并库存的超集，更严），另外再按「落在交付库存里的那部分多边形」报一遍 G1，作为对照。步骤 1 要查清差异来源，写一句话进结果文档。
- 生产切 chip：`core/inference/tile_dataset.py::SlidingWindowDataset`，400 像素、overlap 0.25、stride 300、`window_origin_mode="anchored"`、边缘零填充；每张 2000 像素瓦片 49 个 chip，每个 CPT 格 7×6 = 42 张瓦片、2,058 个 chip。**chip 枚举必须直接调用这个类**，不许自己重写窗口规则，否则门跳过的 chip 和 Mask R-CNN 实际看的 chip 对不上。
- 影像：`scripts/imagery/download_tiles.py --grid-id <CPT id> --region ct`，市政 WMS 层 *Aerial Imagery 2025Jan*，JPEG95 GeoTIFF，一格约 50–80 MB。落盘在 `~/zasolar_data/tiles/cape_town/aerial_2025/G####/`（目录用源 G 号，文件名用 CPT 号；工具内部自动处理，别手动改名）。本机该目录目前是空的。
- 生产检测参数的唯一来源：`scripts/lib/census_caliber.sh`（`CENSUS_CALIBER_DETECT_ARGS` / `CENSUS_CALIBER_FINALIZE_ARGS`），调用方式照抄 `scripts/ct_census_setup.sh` 第 155–165 行。检测器权重在本机：`checkpoints/exp_unified_reviewall_A/best_model.pth`，备份 config 记录的 sha256 是 `c32ebf26…696e`，用前核对。
- 本机 GPU：RTX 4070 Laptop 8 GB。YOLOv11n 训练和 Mask R-CNN 单格重跑都够用，不需要开 pod。
- 本机没有装 `ultralytics`。

## 3. 网格分配（负责人已定，不得挪用）

| 角色 | 网格 | 说明 |
|---|---|---|
| 训练 | CPT2932 | 密格（交付库存 584 块，451 块 <30 m²） |
| 训练 | CPT2373 | 全小目标（35 块，全部 <30 m²） |
| 训练 | CPT2182 | 中位格 |
| 训练 | 空格 E_train | 交付库存里 0 块，选法见下 |
| **留出** | **CPT1238** | 183 块，163 块 <30 m²；2026-06-16 正控 |
| **留出** | **空格 E_test** | 选法见下 |

空格选法（写成脚本、固定随机种子 20260923，结果写进结果文档）：候选 = `data/task_grid_cpt.gpkg` 里 census 范围内（`data/ct_grid_crosswalk_g_to_cpt.csv` 中 `kept==True`）、在交付库存中 0 块的格。先拉这些格的 `predictions_metric.gpkg`（只几 KB），按过滤前多边形数排序，只保留过滤前 ≥1 块的格（有屋顶、检测器被热水器之类骗过、再被 solar_cls 清掉——这才是有意义的负样本；全是海和山的格会让放行率白白好看）。在其中过滤前多边形数的上半区里按种子随机抽 2 个，第一个为 E_train、第二个为 E_test。若上半区不足 2 个，退到全部候选里抽，并在结果里注明。

留出纪律：留出格的影像可以下载，也可以在步骤 1 用 Mask R-CNN 做一致性检查（这不接触门）；但在 §6 的冻结文件写好并记录哈希之前，**门的任何版本都不许在留出格 chip 上推理**，留出格 chip 也不许出现在训练、验证、调参、挑阈值的任何环节。

## 4. 执行步骤

输出根目录：`~/zasolar_data/gate_pilot_2026-09/`（数据不进 git）。代码放 `ZAsolar/scripts/gate_pilot/`。结果文档：`ZAsolar/docs/experiments/2026-09-yolo-gate-pilot/RESULT.md`。每一步写日志，脚本可重跑可续跑。

### 步骤 0 — 只用矢量的可行性检查（不下影像，先做）

1. 拉 6 个网格的备份矢量到 `vectors/<CPT id>/`，记录每个文件的 sha256。
2. 用 `download_tiles.py --dry` 或瓦片常量（0.0016°、2000 像素）得到每格 42 张瓦片的地理范围，再按 `SlidingWindowDataset` 的规则算出 2,058 个 chip 的范围。若只能用常量推算，步骤 1 下载后必须用真实 GeoTIFF 重算一次并核对完全一致。
3. 每格统计：chip 总数；与过滤前多边形相交的 chip 数和占比 P_pre；与过滤后多边形相交的 chip 数和占比 P_post；<30 m² 子集同样统计。
4. **可行性判断**：若 CPT1238 上 `0.99 × P_pre > 20%`，那么 G2 和 G3 在数学上不能同时成立，门再好也过不了。此时停在这里，把表交回负责人，不下载、不训练。
5. 这一步的表就是 brief 里欠的「抽样网格上 raw 阳性 chip 的真实比例」，无论后面结果如何都要写进结果文档。

### 步骤 1 — 下载影像并确认影像没换过

1. 用 `download_tiles.py` 下 6 个格。检查每格正好 42 张瓦片、能被 rasterio 打开、CRS 和范围与步骤 0 一致。
2. **影像一致性检查**：市政 WMS 的影像如果在 6 月之后被替换过，备份多边形就不再对应现在下到的像素，整轮标签作废。在 4 个有光伏的格（CPT2932、CPT2373、CPT2182、CPT1238）上，用生产参数重跑 `detect_direct.py` + `finalize.py`（照抄 `ct_census_setup.sh`，本机 `--batch-size 4`），输出到 `parity/<CPT id>/`，**不覆盖**任何已有结果。把重跑的过滤前多边形和备份的 `predictions_metric.gpkg` 做一对一匹配（IoU ≥ 0.5）。
   - 两个方向的匹配率都 ≥ 95%、总数差 ≤ 5%：通过。
   - 否则停，报告匹配率和几张典型差异截图，交负责人决定（可能要改用重跑结果作标签，这是负责人的决定，执行者不许自行替换）。
3. 顺手记录这次重跑的 chip/秒（`detect_direct` 日志里 `chips → detections … chips/s` 那一行），作为 §1 速度诊断的 Mask R-CNN 单价。
4. 查清 §2 里单格与合并库存计数差异的原因，一句话写进结果文档。

### 步骤 2 — 生成 chip 与标签

1. 对 6 个格，用 `SlidingWindowDataset(tif_paths, chip_size=400, overlap=0.25)` 枚举 chip，把每张 chip 的有效像素存成 PNG（零填充部分保留为黑边，和生产输入一致），并写一份 chip 清单 CSV：`grid_id, chip_index, source_tile, window, valid_window, 地理范围(瓦片 CRS)`。chip_index 必须与生产 `detect_direct` 的编号一致——用步骤 1 重跑结果里多边形的 `chip_index` 抽查 20 个对上。
2. 标签（每张 chip）：
   - `y_pre` = 是否与任一过滤前多边形相交；`y_post` = 是否与任一过滤后多边形相交；`y_post_small` = 是否与任一过滤后 <30 m² 多边形相交；记录相交多边形 id 列表和交集面积。
   - 相交计算在瓦片 CRS 里做：把 EPSG:32734 的多边形投到瓦片 CRS，不要反过来把 chip 投到米制再算。
3. YOLO 训练框（只给训练格生成）：每块与 chip 相交的**过滤前**多边形，裁到 chip 有效范围后取外接矩形，换成 chip 像素坐标，单类 `pv`。宽或高不足 2 像素的补到 2 像素，不丢。无相交的 chip 写空标签文件（背景图）。
4. 自检：`y_post ⊆ y_pre` 应当逐张成立；不成立的 chip 数要报告（这意味着过滤后多边形不是过滤前多边形的子集，需要解释）。

### 步骤 3 — 训练和挑阈值（只用 4 个训练格）

环境：别动系统 torch。`python3 -m venv --system-site-packages ~/venvs/yolo_gate` 后在里面 `pip install ultralytics`；安装前后各记一次 `python -c "import torch;print(torch.__version__, torch.version.cuda)"`，不一致就回滚。版本号写进结果文档。

模型与超参（写进配置文件，全程不改）：
- `yolo11n.pt`（COCO 预训练）起步，任务 = detect，单类，`imgsz=416`，epochs 100（早停 patience 20），默认增强外加上下、左右翻转；batch 按 8 GB 显存取最大的 2 的幂。
- chip 分数 = 该 chip 上所有框置信度的最大值；推理 `conf=0.001`；没有框记 0。

阈值用**留一格交叉验证**来挑，不碰留出格：
1. 3 折：每折留出一个有光伏的训练格（CPT2932 / CPT2373 / CPT2182 各一次），另外两个有光伏格 + E_train 训练，对留出的那个格打分。三折合起来得到每张训练 chip 的「折外分数」。
2. 阈值 τ = 在折外分数上**同时**满足以下两条的最大值：`y_post` chip 召回 = 100%；`y_pre` chip 召回 ≥ 99.5%（比 G2 多 0.5 个百分点余量）。
3. 再用同样超参在全部 4 个训练格上训最终模型；最终模型在训练 chip 上、阈值 τ 下的 `y_post` 召回应为 100%，若不是，报告并停（说明训练本身没学会）。
4. 若 3 折里任何一折在「`y_post` 召回 100%」处的放行率已经 > 50%，提前报告：门在没见过的格上分不开有无光伏，G3 基本无望。仍可继续跑完留出测试，但结果文档要先写这条。

### 步骤 4 — 冻结

写 `frozen_gate.json`：最终权重路径和 sha256、τ、imgsz、conf、chip 枚举参数、ultralytics 与 torch 版本、训练格与 E_train/E_test 的 id、步骤 0 到 3 的关键数。算出该文件的 sha256，写进结果文档并提交这一版结果文档（只提交文档和代码，不提交数据）。**提交之后**才进入步骤 5。

### 步骤 5 — 留出测试（只跑一次）

1. 用冻结的模型和 τ 对 CPT1238、E_test 的全部 2,058 张 chip 打分。
2. 算 G1、G2、G3 与 §1 的全部诊断数，含「只算交付库存内多边形」的 G1 对照。
3. 在同一台机器上测门的 chip/秒（从读 GeoTIFF 窗口开始计，和 `detect_direct` 可比），算预期加速倍数。
4. 不许重训、不许换阈值、不许换网格后再测一次。若发现代码 bug 必须重跑，结果文档里写明原因、改了什么、前后两次的数都保留。

## 5. 判定

- 三个数都过：写「试点通过」。下一轮（另开方案）才是把门接进 `detect_direct.py`，用锁定口径重跑 σ_Bw 和 RMSE，差于 brief 表 A 的锁定点就不转产。本轮不做这件事。
- 任何一个不过：写「试点失败」，说明是哪个数、差多少。不要自行加训练格、换大模型或调阈值再测——那是负责人的下一步决定，而且会把 CPT1238 变成被看过的网格，下次需要另选留出格。

## 6. 交付物

1. `ZAsolar/scripts/gate_pilot/`：步骤 0–5 的脚本（每步一个入口，可续跑），加一个 `README.md` 说明运行顺序。
2. `~/zasolar_data/gate_pilot_2026-09/`：vectors、tiles 清单、chips、labels、parity、runs、`frozen_gate.json`、各步日志。
3. `ZAsolar/docs/experiments/2026-09-yolo-gate-pilot/RESULT.md`，结构固定：
   - 结论一句话（通过 / 失败 / 步骤 0 或步骤 1 提前止损）
   - G1/G2/G3 表（CPT1238、E_test 各一行），另加交付库存口径的 G1
   - <30 m² 与 ≥30 m² 的 chip 口径和多边形口径表
   - 步骤 0 的 P_pre / P_post 表（6 个格）
   - 影像一致性匹配率；计数差异原因
   - 速度：两种单价、放行率、预期倍数
   - G1 漏掉的 chip 逐张清单和截图（若有）
   - 空格抽取记录、ultralytics/torch 版本、`frozen_gate.json` 哈希
4. 在 brief（`docs/briefs/BRIEF-census-detector-speed-2026-09-23.md`）末尾追加一行结果指针；有 `ROADMAP.md` 相关条目的，同步更新状态。文档没更新 = 任务没完成。

## 7. 止损汇总

| 何时 | 条件 | 动作 |
|---|---|---|
| 步骤 0 | CPT1238 上 0.99 × P_pre > 20% | 停，交表，不下载不训练 |
| 步骤 1 | 影像一致性匹配率 < 95% 或数量差 > 5% | 停，交负责人决定标签来源 |
| 步骤 2 | chip_index 与生产编号对不上 | 停，先修枚举，不许带着错位训练 |
| 步骤 3 | 最终模型在训练 chip 上 `y_post` 召回 < 100% | 停，报告 |
| 步骤 5 | G1/G2/G3 任一不过 | 判失败，不重测 |

## 8. 执行分工建议

- 步骤 0–5 的实现和运行：机械活，交 codex（gpt-5.6-sol）。步骤 3 训练约 4 次 YOLOv11n × 100 epoch，本机大约 1–3 小时；超过 30 分钟的步骤放 tmux，进度看日志。
- 步骤 4 冻结前、步骤 5 结果出来后，各做一次独立复核（opus 或 fable）：重点查 chip 枚举是否直接复用了 `SlidingWindowDataset`、相交是否在瓦片 CRS 里算、留出格是否在冻结前被门碰过、G1 分母是否用了单格过滤后文件。
- 不许用 Haiku。
