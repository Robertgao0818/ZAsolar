# ZAsolar 数据附录

这份 `README` 面向希望理解数据、方法和当前证据基础的研究读者，而不是面向日常开发者。代码仍然保留在仓库中，但本文档尽量用非技术语言说明：这个项目想解决什么问题，模型怎么工作，训练和验证是如何组织的，目前已经覆盖到哪里，以及这些结果与现有文献相比处在什么位置。

## 1. 研究目标

我们开发了一套计算机视觉流水线，利用高分辨率航拍影像自动识别屋顶光伏太阳能板的位置和空间轮廓。算法输出每个检测结果的地理参考多边形，然后可以把这些多边形汇总到网格单元层面，构建南非城市屋顶光伏采纳的空间面板数据。需要特别说明的是，当前 V1.3 工作流产出的不是“完全原始的模型输出”，而是**经人工审查后的预测足迹数据**：模型先给出候选多边形，人工删除明显误检、标记需要修正的多边形，并在必要时补充漏检目标，最后形成可用于训练和汇总的清洗结果。

## 2. 算法描述

### 2.1 底层模型是什么

本项目的底层检测器是 **Mask R-CNN（ResNet-50 + FPN 骨干网络）**。在 [`train.py`](train.py) 中，模型通过 `maskrcnn_resnet50_fpn` 构建，然后默认加载公开的 `geoai` 太阳能板检测权重作为起点。按照 GeoAI 公开模型卡的说法，这个起始权重是一个**在美国 RGB 影像上训练出的太阳能板检测器**，然后 ZAsolar 再在自己的南非数据上做微调。

当前仓库里需要区分两个模型名：

- **V3-C-HN**：当前生产中使用的模型
- **V4.1-HN**：更新但尚未取代 V3-C 的实验模型

有一点需要诚实说明：仓库本地代码只明确写出了默认加载的 `geoai` 检查点文件名和来源地址，但**没有完整记录这个上游权重到底用了哪一套训练数据和验证协议**。因此，关于“上游预训练数据集”的说法，目前只能以公开模型卡为准，不能完全由本仓库独立复现。

可追溯来源：

- 本地训练脚本：[`train.py`](train.py)
- 模型注册表：[`configs/model_registry.yaml`](configs/model_registry.yaml)
- GeoAI 模型卡：<https://huggingface.co/giswqs/geoai/blob/main/README.md>

### 2.2 输入是什么

开普敦主数据源是 **City of Cape Town 的航拍正射影像 WMS 图层 “Aerial Imagery 2025Jan”**。仓库配置把这套影像记录为大约 **0.08 米/像素**（约 8 厘米 GSD）。原始下载瓦片为 **2000 × 2000 像素** GeoTIFF；按照当前网格和瓦片设置，一张原始瓦片在地面上大约覆盖 **162 米 × 162 米**。模型训练时并不是直接吃整张大瓦片，而是把它切成有重叠的小图块：标准训练 chip 为 **400 × 400 像素**，重叠率 **25%**，在开普敦影像上大约对应 **32 米 × 32 米**。

这里有两个补充说明：

- 最早期的 `G1189` 和 `G1190` 标注还使用过人工地理配准的 **Google Earth** 影像，仓库中把它记录为大约 **0.3 米/像素**。
- 约翰内斯堡试点使用的是另一套影像：**City of Johannesburg 2023 航拍影像**，分辨率约 **0.15 米/像素**。

可追溯来源：

- 影像源配置：[`configs/datasets/imagery_sources.yaml`](configs/datasets/imagery_sources.yaml)
- 开普敦瓦片下载脚本：[`scripts/imagery/download_tiles.py`](scripts/imagery/download_tiles.py)
- 网格与瓦片几何设置：[`core/grid_utils.py`](core/grid_utils.py)
- Johannesburg 影像说明：[`docs/joburg_batch1_plan.md`](docs/joburg_batch1_plan.md)

### 2.3 模型具体做了什么

用人话说，流程可以概括为五步：

1. 模型读取一张航拍影像 chip 作为输入。
2. Mask R-CNN 在这张图里找出可能是屋顶光伏的候选区域，并给出每个候选区域的分割掩膜。
3. 每个候选区域都会得到一个置信度分数。
4. 我们再用一套显式的后处理规则去掉明显误检。
5. 保留下来的结果被导出为地理参考多边形。

当前标准 benchmark 使用的后处理规则主要包括：

- **最小面积阈值（`>= 5 m²`）**：去掉面积太小的碎片和噪声点。
- **后处理置信度阈值（`>= 0.85`）**：这是当前压低误检最重要的一道过滤。
- **形状细长度过滤（`<= 8.0`）**：去掉过于细长的条状目标，这类目标往往是屋檐、阴影或其他线性伪影。
- **阴影阈值（`shadow_rgb_thresh = 60`）**：尽量排除很暗、很像屋顶阴影的目标。
- **可选的形状与分层规则**：代码里还保留了 solidity 过滤和针对大商业阵列的分层阈值，但标准 benchmark 使用的是固定的 canonical 配置 [`configs/postproc/v4_canonical.json`](configs/postproc/v4_canonical.json)。

这些后处理之所以重要，是因为本项目的主要困难不是“纯背景噪声”，而是**看起来很像光伏的屋顶物体**，例如太阳能热水器、天窗、屋顶设备、阴影，以及被切碎的大面积商业阵列。

可追溯来源：

- 标准后处理配置：[`configs/postproc/v4_canonical.json`](configs/postproc/v4_canonical.json)
- 主检测与评估脚本：[`detect_and_evaluate.py`](detect_and_evaluate.py)

### 2.4 模型是怎么训练的

训练数据通过 [`export_coco_dataset.py`](export_coco_dataset.py) 导出为 COCO 实例分割格式。做法是把原始航拍瓦片切成带重叠的 **400 × 400** 图块，并把落在这些图块里的标注多边形写入 `train.json` 和 `val.json`。**不含任何标注的空白 chip 不会被自动丢弃**，而是作为 hard negatives 保留下来。

对当前生产模型 **V3-C-HN**，仓库中的实验文档给出的训练规模是：

- **5,623 张训练 chip**
- **7,720 个光伏实例标注**
- 正样本来自 **63 个开普敦网格**
- 额外加入 **883 张 targeted hard-negative chips**，来源是 batch 003 人工审查后确认的误检位置

“**V3-C hard negative fine-tuning**” 的非技术解释是：模型不是只看“有光伏”和“完全空白”两类样本，而是额外学习一类**最容易把模型骗错的非光伏样本**。这些 hard negatives 不是随机抽来的空地，而是模型以前已经高置信度误判过的地方，比如热水器、天窗、屋顶设备或阴影。换句话说，V3-C 是在“真实正样本 + 模型自己的过去错误”上继续训练出来的。

仓库中更新但尚未成为生产模型的 **V4.1-HN** 使用了更晚期的数据导出方案，其训练规模为：

- **8,495 张训练 chip**
- **6,227 张正样本 chip**
- **934 张随机负样本 chip**
- **1,334 张 targeted hard-negative chips**
- 训练导出时明确排除了 **26 个 benchmark holdout grids**

### 2.4.1 训练数据的一个已知偏差：几乎没有大板样本

对当前 V3-C COCO 训练集（共 7,720 个标注实例）做面积分布审计，结果如下：

| 面积区间 | 实例数 | 占比 |
|---------|------:|-----:|
| `< 10 m²` | 3,483 | 45.1% |
| `10–50 m²` | 3,988 | 51.7% |
| `50–100 m²` | 148 | 1.9% |
| `100–200 m²` | 54 | 0.7% |
| `≥ 200 m²` | 47 | **0.6%** |
| `≥ 500 m²`（其中） | 8 | 0.1% |

也就是说，**约 96.8% 的训练标注是 50 m² 以下的住宅级小板**，工业级 / 商业级大阵列在训练集中几乎不存在。这一点对解读后面的跨城市迁移结果很重要：当模型在约翰内斯堡 CBD 工业屋顶上漏掉成片大板时，主要原因不是后处理过滤太严，而是**模型在训练阶段就几乎没见过这种目标**。下一轮训练计划把 JHB01–JHB06 中已有的 76 个 ≥200 m² 大板正样本补进去。

可追溯来源：

- COCO 导出脚本：[`export_coco_dataset.py`](export_coco_dataset.py)
- V3-C 实验说明：[`docs/experiments/exp_003_hard_negatives.md`](docs/experiments/exp_003_hard_negatives.md)
- V4.1 训练日志：[`docs/progress_log/week_2026-03-31/2026-04-05.md`](docs/progress_log/week_2026-03-31/2026-04-05.md)
- 大板缺失审计：[`docs/progress_log/week_2026-04-07/2026-04-07.md`](docs/progress_log/week_2026-04-07/2026-04-07.md)

### 2.5 验证流程是什么，以及目前最大的空白在哪里

当前仓库里实际上有**两层不同的验证**。

第一层是**训练内部验证**。在 [`export_coco_dataset.py`](export_coco_dataset.py) 中，每个被纳入训练导出的 grid，都会把自己的 source tiles 切成 **80% 训练 / 20% 验证**。这个切分是按 tile 做的，不是按 grid 做的；也就是说，同一个 grid 往往会同时出现在 train 和 val，只是 tile 不重叠。[`train.py`](train.py) 再用这个 `val.json` 的 **segmentation AP50** 来选最优 checkpoint。这里不是 k 折交叉验证，而是**单次切分**。

第二层是**全网格 benchmark**。仓库后期引入了固定的开普敦主 benchmark `cape_town_independent_26`，由 **26 个整 grid** 组成；在较新的流程中，这 26 个 grid 会通过 `--exclude-grids` 从训练导出中剔除，然后再跑完整的 full-grid benchmark。

但这里也正是当前文档中最大的空白：

- V3-C 的原始实验文档写的是“使用截至 batch 003 的全部正样本”，并明确写出 **63 个 grid** 的训练语料。
- 后来的 benchmark 文档又把 `cape_town_independent_26` 描述成“**不参与 V3 训练**”。

这两句话在当前仓库快照中**还没有被完全对齐**。也就是说：

- `train.py` 和 `detect_and_evaluate.py` 在代码层面确实是分开的；
- 但是 **training / validation / testing 的治理还没有完全硬编码成一个不可歧义的制度**；
- 对 **V4.1 及更晚流程** 来说，26-grid holdout 是明确执行的；
- 对 **原始 V3-C** 来说，26-grid 主 benchmark 是否完全无泄漏，目前还不能仅凭仓库里的不可变证据彻底审计清楚。

还有一个额外的口径冲突也需要说明：

- [`results/benchmark/README.md`](results/benchmark/README.md) 把主 benchmark 写成 **IoU = 0.5**。
- 但当前 `detect_and_evaluate.py` 写出的 `presence_metrics.csv` 来自 **IoU = 0.1** 的 installation matching，而 `run_benchmark.py` 又直接读取这个 CSV 做汇总。

因此，最稳妥的结论是：

- 当前没有 k 折交叉验证。
- 当前也没有一个完全不可歧义、全局统一的 train/val/test 注册表。
- 现有流程是“**同 grid 的 tile-level train/val split + 较新模型使用的 26-grid 外部 holdout**”。
- **V3-C 的泄漏状态**以及**主 benchmark 最终对应的 IoU 阈值**，都还需要进一步审计。

可追溯来源：

- 训练切分逻辑：[`export_coco_dataset.py`](export_coco_dataset.py)
- checkpoint 选择方式：[`train.py`](train.py)
- benchmark 预设：[`configs/benchmarks/post_train.yaml`](configs/benchmarks/post_train.yaml)
- V3-C 实验文档：[`docs/experiments/exp_003_hard_negatives.md`](docs/experiments/exp_003_hard_negatives.md)
- V4.1 holdout 工作流：[`docs/workflows.md`](docs/workflows.md)、[`ROADMAP.md`](ROADMAP.md)

### 2.6 人工审查步骤是什么，当前报告的 F1 到底是不是“审查后”的精度

人工审查是这个项目的核心环节之一。当前流程是：

1. 模型先对一个 grid 输出预测多边形。
2. 审查员在浏览器 GUI 中打开这些预测。
3. 每个预测多边形会被标记为 **`correct`**、**`delete`** 或 **`edit`**。
4. 如果模型漏掉了目标，审查员还可以放置 **false-negative markers**。
5. 这些漏检点可以再通过 **SAM 2.1 Large** 做交互式分割。
6. 最终保留下来的多边形导出到 `data/annotations/cleaned/`。

因此，当前仓库里同时存在两类完全不同的数字：

- **正式 benchmark 的 F1**：这是**模型原始输出经过自动后处理后**，与人工标注数据比较得到的指标；**不是人工把预测修正以后再算的精度**。
- **旧 README 里的 8-grid F1 = 0.873**：这不是正式 holdout benchmark，而是一个**review-based calibration sample**。它用的是“人工接受的预测”构造出的较窄 GT 口径，因此不能拿来当作 headline model accuracy。

换句话说，**当前报告的 26-grid benchmark F1 代表模型原始性能**；而人工审查步骤主要用于生成 cleaned 数据、改进训练集和构建 reviewed corpus，而不是用来把 benchmark 分数“修高”。

### 2.6.1 GT 标注本身也在被审计：太阳能热水器污染

人工审查的一个副产品是发现了**ground truth 自身的系统性误标**。在最近一轮针对 batch 003 高优先级 chip 的污染审计中，**671 个被人工逐个 relabel 的 chip 里有 80 个（12.8%）实际上是太阳能热水器，而不是光伏板**。pollution 主要集中在 batch 003 的少数 grid（G1687 占 30 个，G1686 占 17 个）。这 86 条已确认的“heater 排除清单”会在下一轮训练时自动从正样本中剔除。

这一发现对解读模型 precision 数字也很关键：项目过去把“看起来像板子的小目标 high-confidence 误检”几乎全部归因到模型，但其中**至少有一部分实际上是 GT 错误**——也就是说模型是“对的”，被人工标错的 GT 反过来扣了它一次 false positive。这是接下来要专门解决的方向之一，目前的工作思路是再训练一个 **PV vs 太阳能热水器二分类器**作为 Mask R-CNN 的后处理插件，相关训练集（1,933 个 train chip + 234 个 val chip）已经准备好。

可追溯来源：

- Review GUI：[`scripts/annotations/review_detections.py`](scripts/annotations/review_detections.py)
- SAM FN GUI：[`scripts/annotations/sam_fn_review.py`](scripts/annotations/sam_fn_review.py)
- 批量导出脚本：[`scripts/annotations/batch_finalize_reviews.py`](scripts/annotations/batch_finalize_reviews.py)
- 半自动标注流程说明：[`docs/semi_auto_annotation_workflow.md`](docs/semi_auto_annotation_workflow.md)
- 8-grid calibration sample：[`results/analysis/postprocess_ablation/evaluation_full.csv`](results/analysis/postprocess_ablation/evaluation_full.csv)、[`scripts/analysis/postprocess_ablation.py`](scripts/analysis/postprocess_ablation.py)
- GT 热水器审计 pipeline：[`scripts/analysis/build_gt_heater_audit.py`](scripts/analysis/build_gt_heater_audit.py)、[`scripts/analysis/label_gt_heater_audit.py`](scripts/analysis/label_gt_heater_audit.py)

## 3. 当前应用与结果

### 3.1 地理覆盖范围

按当前**去重后的 cleaned canonical 语料**计算，本项目已经覆盖：

- **开普敦：103 个 grid，约 103 km²**
- **约翰内斯堡：6 个 cleaned pilot grids，约 6 km²**

不过到 2026-04-08 为止，约翰内斯堡之外**还存在一层正在进行中的跨城市工作**，它已经超出 6-grid 的最初 pilot 但还没有被合并到 canonical 语料里：

- 用 V4 模型在约翰内斯堡 4 类城区（CBD / Sandton / Alexandra / Midrand）共 **100 个 grid** 上跑了完整推理。
- 其中 **CBD 25 个 grid 已 100% 人工 review 完成**，共审查了 **1,003 个预测**，并放置了 180 个 false-negative markers。
- 对 179 个 FN markers 用 SAM 2.1 Large 做了三类有针对性的重切，生成了 **136 个候选多边形**，等待清理后并入下一轮训练正样本。

这些 reviewed JHB CBD 数据**目前仍存放在 `results_joburg/` 工作区**，还没有被提升到 `data/annotations/cleaned/` 的 canonical 归档；因此本节的总数仍按原 6 个 cleaned pilot grid 报告，但读者应该知道实际"被人工触碰过的 JHB 区域"已经远不止 6 个 grid。

可追溯来源：

- JHB CBD batch1 review 日志：[`docs/progress_log/week_2026-04-07/2026-04-07.md`](docs/progress_log/week_2026-04-07/2026-04-07.md)
- 周报汇总：[`docs/progress_log/week_2026-04-07/weekly_summary_0401-0407.md`](docs/progress_log/week_2026-04-07/weekly_summary_0401-0407.md)

### 3.2 当前最重要的性能数字

目前仓库里最接近“正式 benchmark”的结果是开普敦 **26-grid primary benchmark**，但它的 IoU 阈值口径仍需最后审计，因为 benchmark 叙述和当前自动汇总代码尚未完全对齐。旧 README 里最醒目的 `F1 = 0.873` 则来自一个 narrower、review-based 的 calibration sample，不应继续当作主结果引用。

| 模型 | 样本 | IoU | Precision | Recall | F1 | 说明 |
|------|------|-----|-----------|--------|----|------|
| **V3-C-HN** | 8-grid 后处理校准样本 | 0.3 | 78.0% | 99.0% | 87.3% | 基于 review 接受样本的 calibration sample，可用于说明后处理口径，不是正式 headline benchmark |
| **V3-C-HN** | 26-grid 开普敦主 benchmark | 文档写作 0.5；代码路径仍为 IoU=0.1 + installation merge | 69.6% | 72.2% | **70.9%** | 当前生产模型的主 benchmark 数字 |
| **V4.1-HN** | 26-grid 开普敦主 benchmark | 同上 | 71.0% | 65.8% | 68.3% | 最新实验模型，precision 略高（FP 较 V3-C 减少 14.6%），但 recall 退步 6.4 pp |
| **V4-HN** | 25-grid Joburg CBD（首次跨城市 review） | 直接基于 reviewer 的 accept/edit/delete 决策 | 68.4% | 82.7% | **74.8%** | 这是 review-acceptance 口径，不是 GT-polygon 匹配口径，不能与上面两行严格直接比较 |

到当前仓库快照为止，并没有一份与这套 26-grid holdout 完全对齐、同时又以 **IoU = 0.3** 汇总好的正式总表。因此，这里只能把“8-grid calibration sample 的 IoU=0.3 结果”和“26-grid 主 benchmark 的 headline 数字”并列展示，而不能假装它们来自同一套、完全同口径的评估。

关于约翰内斯堡 CBD 的 **F1 = 74.8%** 这一行，需要补三句话：

1. 这个数字的算法是“**reviewer 把多少个预测打上了 correct 或 edit 标签**”，分母是模型预测总数 + reviewer 放置的 FN markers，而不是先有一份独立 GT 多边形再做 IoU 匹配。这种算法对 precision 的衡量很直接，但对 recall 的衡量受 reviewer 主观漏放 FN marker 的程度影响。
2. 仓库里**确实存在**另一份独立的 Joburg ground truth：李同学手工在 Google Earth 历史影像（**2024-02**, Vexcel/ArcGIS 来源）上画的标注。但本项目的瓦片来源是 **City of Johannesburg 2023 航拍**，两者之间存在大约一年的 vintage gap。在 G0772–G0857 的抽样比对中，按这份 Li GT 算出的 per-polygon F1 约为 0.08，而其中大量"漏检"位置在 2023 瓦片上**根本没有面板**——它们是 2024 年才装上的新光伏。换句话说，**不能直接拿 Li 的 Joburg GPKG 去给当前模型算 recall**，否则会把"影像比 GT 早一年"误读成"模型很差"。
3. CBD 25 grids 里 FP 极不均匀：G0922 precision 88%，但 G0776 / G0853 / G0891 大部分 FP 是**天窗（skylight）和屋顶设备架的阴影**，不是热水器。这是开普敦数据集里被泳池热水器遮盖、未被针对性挖掘的另一类系统性误检。

可追溯来源：

- 26-grid benchmark 注册表：[`results/benchmark/README.md`](results/benchmark/README.md)
- Joburg CBD review + vintage gap：[`docs/progress_log/week_2026-04-07/2026-04-07.md`](docs/progress_log/week_2026-04-07/2026-04-07.md)
- Joburg 影像源说明：[`docs/joburg_batch1_plan.md`](docs/joburg_batch1_plan.md)

### 3.3 目前总共识别了多少个太阳能安装点

如果严格问“模型在开普敦总共识别了多少个太阳能安装点”，当前仓库里**没有一个单独存放、全城范围、纯 raw-model、且无二义性的总数**。最稳定、最适合写进论文附录的数字其实是当前 reviewed archive 的规模。按这个口径，开普敦 cleaned corpus 目前包含：

- **7,074 个 reviewed polygons**
- 分布在 **103 个开普敦 grids**

这个数字应理解为“**当前人工审查后的城市级归档规模**”，而不是“模型在完全无人干预状态下的原始全城检测数”。约翰内斯堡 CBD 25 grids 在过去一周新增的 **686 个 reviewer-confirmed predictions**（662 个 correct + 24 个 edit），目前还在 `results_joburg/` 工作区内，**未并入上述 7,074 这个数字**。

### 3.4 标注数据汇总

下表使用的是 `data/annotations/cleaned/` 中**去重后的当前文件版本**：每个 grid 只保留一个当前版本，并排除汇总文件 `all_annotations_cleaned.gpkg`。

| 语料部分 | Grids | Polygons | 主要日期 |
|---------|------:|---------:|---------|
| Legacy pilot | 9 | 22 | 2026-03-20 之前 |
| Early SAM2 | 3 | 475 | 2026-03-20 |
| Batch 001-002 | 26 | 1,148 | 2026-03-22 |
| Batch 002b | 14 | 1,359 | 2026-03-23 至 2026-03-24 |
| Batch 003 | 20 | 1,731 | 2026-03-25 至 2026-03-27 审查，2026-04-03 导出 |
| Batch 004 | 31 | 2,339 | 2026-04-03 |
| Johannesburg cleaned pilot | 6 | 191 | 初步试点 |
| **去重后 archive 总计** | **109** | **7,265** | 当前仓库快照 |

之所以这里的总数与部分旧文档中的 **7,301**、**7,367** 或 **7,447** 不完全一致，是因为历史日志混合使用了不同口径：有时把 superseded 早期文件一起算入，有时把 Joburg pilot 和 SAM false-negative fills 单独累计。上表使用的是**当前去重后的文件库存**，这比旧版 README 里的总数更适合做论文式汇报。

可追溯来源：

- benchmark 结果注册表：[`results/benchmark/README.md`](results/benchmark/README.md)
- V4.1 benchmark 工作日志：[`docs/progress_log/week_2026-03-31/2026-04-05.md`](docs/progress_log/week_2026-03-31/2026-04-05.md)
- 周报汇总：[`docs/progress_log/week_2026-04-07/weekly_summary_0401-0407.md`](docs/progress_log/week_2026-04-07/weekly_summary_0401-0407.md)
- 标注规范：[`data/annotations/ANNOTATION_SPEC.md`](data/annotations/ANNOTATION_SPEC.md)

## 4. 与现有文献的比较

遥感与屋顶光伏文献中经常能看到比这里更高的 F1，但很多研究做的是**更容易的任务**：例如图像级分类、像素级语义分割，或者先识别屋顶再估算潜力，而不是直接做“城市级、全网格、地理参考、实例级屋顶光伏检测”。下表给出三个可对照的例子。

| 研究 | 地理范围与影像 | 任务与评估 | 论文报告结果 | 与 ZAsolar 的关系 |
|------|---------------|-----------|-------------|------------------|
| **ZAsolar（本仓库）** | 开普敦航拍正射影像，约 8 cm GSD | 全 grid 屋顶光伏检测，输出地理参考多边形；主 benchmark 在仓库文档中按 **IoU = 0.5** 叙述，但当前自动汇总代码仍需最终审计 | **P = 69.6%, R = 72.2%, F1 = 70.9%**（V3-C-HN） | 任务更难：南非城市高分影像、强混淆物、实例级匹配，而不是纯像素分类 |
| **DeepSolar**（[Yu et al., 2018](https://www.sciencedirect.com/science/article/pii/S2542435118305701)） | 美国全国，高分辨率卫星影像，测试集 93,500 张图像 | 住宅/非住宅太阳能检测，并进一步做规模估计 | **住宅：P = 93.1%, R = 88.5%；非住宅：P = 93.7%, R = 90.5%** | DeepSolar 的地理范围远大于本项目，但它报告的不是安装级 polygon IoU benchmark，不能与本项目一对一对比 |
| **Automated Rooftop Solar Panel Detection Through CNNs**（[Pereira et al., 2024](https://research.tudelft.nl/en/publications/automated-rooftop-solar-panel-detection-through-convolutional-neu/)） | 10 cm 高分辨率航拍影像 | 基于 U-Net 的屋顶光伏语义分割 | **F1 最高可达 91.75%** | 分数更高，但任务是语义分割，且场景更受控，与开普敦 full-grid reviewed-footprint benchmark 不完全可比 |
| **Building Rooftop Extraction for PV Potential Estimation**（[Muhammed et al., 2023](https://www.mdpi.com/2071-1050/15/14/11004)） | 埃及 Madinaty City，卫星影像 | **屋顶提取**，再估算光伏潜力 | **P = 91.2%, R = 98.6%, F1 = 94.7%** | 这是一个很有价值的非洲对照，但它识别的是**屋顶**而不是**屋顶光伏系统**，任务本身更容易 |

整体上看，ZAsolar 的绝对 F1 低于最“干净”的公开 benchmark 并不意外，因为它面对的是一个更难的应用问题：在高密度南非城市影像中做屋顶光伏实例检测，并处理热水器、天窗、阴影和商业大阵列碎片化等混淆因素。从这个角度说，当前结果更适合被解读为一个**保守但可信的应用型基线**，而不是一个简单的 leaderboard 分数。

## 5. 仓库说明

这份 `README` 有意避免写成“让 AI 或开发者看代码”的操作手册。真正的实现细节仍然保留在仓库中，尤其是：

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/workflows.md`](docs/workflows.md)
- [`ROADMAP.md`](ROADMAP.md)
- [`data/annotations/ANNOTATION_SPEC.md`](data/annotations/ANNOTATION_SPEC.md)
