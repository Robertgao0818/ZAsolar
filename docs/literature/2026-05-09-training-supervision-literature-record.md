# 2026-05-09 晚间文献讨论记录 — 训练监督分层 / PV 分割检测

**时间窗口**: 2026-05-09 晚间 NZST，主要为 21:51-23:12。  
**来源**: Claude Code 与 Codex 本地对话记录。  
**用途**: 记录两轮文献讨论后的稳定结论，尤其记录 Claude / Codex 容易混淆或反复说错的细节。  
**星号说明**: `★` 表示对话中提到的、直接涉及 PV / solar array / rooftop photovoltaic 分割、检测或 PV 标注数据集的文献。未标星的多为通用 instance segmentation、noisy label、remote sensing segmentation、active learning 或理论文献。  
**引用提醒**: 本文整理的是昨晚对话里的文献名与用途，尚未逐篇核对 DOI / 正式题名 / venue。正式写论文引用前要再查原文。

## 1. 一页版结论

昨晚两轮讨论的核心结论不是“换一个新模型”，而是：

> ZAsolar 的训练失败主要来自 reviewed-prediction / SAM-assisted 标签的边界与粒度噪声。不能继续把机器候选整张 mask 当 hard GT 回灌 BCE。下一轮应把监督按来源和几何责任分层：H 类干净锚点强监督，S 类中等权重并加 boundary ignore，R 类主要用于 box/cls 或低权重/core 弱监督；mask head 先隔离训练，所有改动独立 ablate。

最重要的工程落点：

1. `train.py --freeze-mask-head` 先做，Stage 1 不默认训练 mask head。
2. `label_source` 必须进入 dataset/targets，并进入 loss reduction。
3. H/S/R 的 loss weight 与 boundary ignore band 是两个独立 lever，不能打成一个包。
4. H 类 clean anchor pool 要常驻，不能被新一轮 R 类 reviewed prediction 替换。
5. R 类数量要受控，讨论中采用 `R <= 4 * H` 作为硬约束起点。
6. batch 配比先按 `25-40% H + 60-75% S/R` 试。
7. 不做 sub-array -> installation 训练前大合并；训练保留细粒度几何，统计 / post-process 再聚合。
8. 56x56 dense mask head 不是 halo 的解药；要先控标签噪声，再考虑 PointRend / DynaMask / SAMRefiner 类 refinement。
9. Phase A 的 `area_R` 高不代表模型好：它是 “area coverage 高 / instance separation 崩” 的典型失败。
10. 文献对很多点只是间接支撑；ZAsolar 的决定必须以 05-08 失败实验和 V1.4 grid-level inventory 目标为准。

## 2. 对话时间线

- 21:51-22:21 Codex 第一段：把第一轮 PV / SAM / weak supervision 文献按“可落地 / 中期 / 不建议”分类。
- 21:58-22:24 Claude 第一段：复核 Codex 结论，强调 Phase A 失败背景，区分 boundary-aware loss 与 boundary ignore band，并生成给 Claude 网页版的自包含文献调研 brief。
- 22:43-22:53 Claude + Codex 第二段：拆解 Q1-Q7 两轮文献调研，形成“source-aware loss / boundary ignore / pseudo-label degradation / staged training / high-res mask / granularity / gold+silver”综合判断。
- 23:04-23:12 Claude：复盘 Phase A `area_R` 与 Ch2 recall 分裂的评估逻辑，并把行动清单写入 `docs/plans/2026-05-09-training-supervision-layering.md`。

关联输出：

- `docs/plans/2026-05-09-training-supervision-layering.md`
- `docs/progress_log/week_2026-05-06/2026-05-09.md`

## 3. 必须记住的细节 / agent 容易搞错点

### 3.1 Boundary-aware loss != boundary ignore band

Phase A 失败的是一个 bundle：boundary-aware loss patch + dissolve_hairline_gaps + tiered mask threshold 等多项一起上，结果 `area_F1 0.608 / Ch2 recall 0.300 / bulk_ratio 1.897`。

不要让 agent 从这里推出“boundary ignore band 已经失败”。

- boundary-aware loss：改 loss 形式。
- boundary ignore band：在 label/target 层把 SAM/R 类边界 ±N px 设为 ignore，不参与 BCE。

这两者要拆成独立模块、独立 commit、独立 ablation。

### 3.2 Source weighting 和 boundary ignore 是正交杠杆

agent 很容易把“中强监督 = boundary ignore”混成一句话。正确拆法：

- source weighting：H/S/R 在 loss reduction 时乘不同权重，例如 `H=1.0, S=0.5, R=0.2`。
- boundary ignore：某个 polygon 的边界带不参与 mask loss，core 仍可监督。

一个 S 类样本可以同时有中等 weight + boundary ignore，也可以只做其中一项。不要再像 Phase A 一样把多个杠杆打包上线。

### 3.3 SAM-assisted 不自动等于 H 类 gold

只有当人工真正对最终几何负责时，SAM-assisted 才接近“人工辅助标注”。如果人工只是确认“这里有 PV”，边界仍由 SAM 给出，那就是 S 类弱/中等监督，不是 H 类。

尤其大面积 commercial roof 上，SAM 有结构性 ceiling：要么切碎，要么 roof-swallow。不要把 SAM mask+box 结果自动当干净 dense mask GT。

### 3.4 Reviewed prediction 不是 gold GT

R 类 / reviewed_prediction 的语义更接近“存在性确认”或“模型候选被人工接受”，不是边界精修真值。

训练上更稳的处理：

- box/cls 可以用；
- mask BCE 默认低权重或 0；
- 如要用 mask，只能 core-mask 弱监督或 boundary band ignore；
- 不要整张 R mask hard BCE。

### 3.5 当前阶段不应该默认训练 mask head

短期推荐：

- Stage A：freeze mask head，只训 RPN / box / classifier，目标是提升“找得到”。
- Stage B：只用 H anchors + 可信 S 类低权重解冻 mask head，目标是边界校正。
- Stage C：只有前两阶段不推高 bulk_ratio，才小学习率联合微调。

注意一个常被漏掉的 caveat：即使 mask head 参数冻结，如果 backbone/FPN 继续训练，mask 输出仍可能变，因为 mask head 吃的是 backbone feature。最保守实验是 freeze backbone/FPN + mask head，只训检测相关头。

### 3.6 H anchor 的作用不是“多一点干净数据”，而是打断 cycle

昨晚讨论中把这个现象对齐到：

- confirmation bias under iterative pseudo-labeling
- spatial pseudo-label drift
- pseudo-label degradation
- 更宽泛类比：model collapse

关键不是随机多标一点 gold，而是：

- H 类历史数据每轮保留，不能被新 R 类替换；
- R 类数量受控，起点 `R <= 4 * H`；
- 每个 batch 要有 H anchor；
- H pool 覆盖失败模式比数量更重要。

建议 H pool 覆盖：大 installation、密集小片、阴影边、浅色屋顶、SAM roof-swallow。

### 3.7 不要把 sub-array 训练前合成 installation blob

这是昨晚最容易被 Codex / Claude 说反的地方之一。

正确区分：

- 训练阶段：保留 tight PV geometry，宁可 sub-array-first。
- 统计 / finalizer 阶段：再聚合到 installation / grid-level inventory。

为什么不建议 pre-merge：

- sub-array 之间的屋顶缝隙会被 union/dissolve 学成 PV；
- reviewed-prediction halo 会被继承；
- SAM roof-swallow 会放大；
- 对 bulk_ratio 是系统性 dilation noise。

例外：如果确认为同一 GT-cluster 的 split_within_gt，可以做 GT-side spatial-merge 兄弟 polygon；这不是把所有 sub-array 升级成 installation blob，也不是训练前 convex hull / roof envelope。

### 3.8 “一个大 pred 盖住 N 个小 GT”不是 N 个 TP

`detect_and_evaluate.py` 的 `iou_matching(merge_preds=True)` 只支持多 pred -> 1 GT，不支持 1 pred -> 多 GT。

所以一个大 mask 覆盖 5 个小 GT 时：

- IoU 可能是 `small_gt_area / huge_pred_area`，通常低于 0.3，进不了 candidate；
- 即使过阈值，大 pred 被第一个 GT 贪心匹配后会被消耗，后面 GT 没可用 pred；
- 最好也是 1 TP + 4 FN，最坏 0 TP + 5 FN。

这不是 bug，而是 Ch2 polygon recall 在衡量 instance separation。若只问“面积有没有覆盖”，那是 Ch3 area-aggregate 视角；Phase A 在 Ch3 有 `area_R 0.881` 的功劳，但也被 `bulk_ratio 1.9x` 重罚。

### 3.9 56x56 dense mask head 不解决 systematic halo

高分辨率 mask head主要改善 boundary AP / 视觉边界，不会自动纠正标签本身的系统性外鼓。

优先级应是：

1. 控制训练标签噪声；
2. source-aware weight + boundary ignore；
3. H anchors / selective relabelling；
4. 再考虑 PointRend / DynaMask / Mask Transfiner；
5. 不优先做统一 56x56 dense head。

### 3.10 SAMRefiner 路线要和 SAM-as-GT-generator 分开

可以保留：V3-C raw -> SAM mask+box refinement -> final polygon，这是 SAM-as-inference-refiner。

需要谨慎：人工发现 FN -> SAM 生成 mask -> 进训练集当 hard GT，这是 SAM-as-GT-generator，容易把 SAM ceiling 写进训练目标。

这两个不能混在一起讨论。

### 3.11 NeurIPS 2022 那篇体积偏差理论要单独记

`On Image Segmentation With Noisy Labels` 的要点：边界有噪声时，体积型 loss 的最优解本身可能产生系统性体积偏差。也就是说，就算只训一轮，模型也可能因为 loss/label 组合而偏大；cycle 只是继续放大。

后续如果 agent 提议换 Dice / IoU / volume-style loss，要先对照这篇，不要只看 chip F1 或 mask AP。

## 4. 更新后的行动清单

### 4.1 立即可做

1. `train.py` 加 `--freeze-mask-head`。
2. 训练数据 build pipeline 加硬约束：H 类每轮保留；R 类数量 <= H 类 4 倍。
3. batch sampler 改成 25-40% H + 60-75% S/R。
4. mask loss 做最低成本的子任务分拆：前景/背景 BCE 全权重，形状/边界部分按 source weight。

### 4.2 下周可做

5. 写面积自适应 boundary ignore band：小目标 1 px，中目标 2 px，大目标或 S 类 3 px；独立模块，不塞进 `boundary_aware_mask`。
6. 从 2083 clean_gt 抽 100-200 个常驻 H anchors，覆盖五类失败模式。
7. 写 selective relabelling pipeline：每轮训练后按 bulk_ratio 贡献 / 高风险类型挑样本送人工精修。

### 4.3 中长期

8. Mirikharaji-style pixel-level meta-reweighting：子任务分拆跑稳后再做。
9. PointRend / Mask Transfiner：需要进一步边界精修时再上。
10. DynaMask：当小目标和大目标两端都需要不同 mask resolution 时再上。
11. SAMRefiner：补 production pipeline 文档，明确 SAM 是 inference refinement，不是训练 GT 真值生成器。

### 4.4 不进 sprint

- 统一 56x56 dense mask head。
- Sub-array -> installation pre-merge。
- 自定义 PVSAM 变体。
- PENet 点监督作为主线。
- size-aware backbone 整体替换。
- pseudo-label 整张 mask 回灌 BCE。

## 5. 文献清单（★ = PV / solar 分割检测或数据集）

| 标记 | 文献 / 线索 | 类型 | 昨晚讨论中的作用 | 不要误读 |
|---|---|---|---|---|
| ★ | Applied Energy 2024 rooftop PV + SAM/CAM + boundary-aware loss（对话未记录精确题名） | rooftop PV / weak supervision | 支持“伪标签边界是关键问题，不是继续加伪标签” | 不等于 Phase A 失败后还继续堆 boundary-aware loss；更重要是边界不确定性处理 |
| ★ | PVSAM / PV-SAM 2026（multi-scale prompt / edge-pyramid prompt） | PV + SAM adaptation | 说明 SAM 原样用于 PV/遥感不够，需要 domain prompt / edge prompt | 不代表现在要自定义 PVSAM；SAM 在大连片屋顶仍有 ceiling |
| ★ | Rooftop PV Segmenter (RPS) 2023 | rooftop PV segmentation / size-aware | 提醒小住宅 PV 与大阵列同头处理有风险 | 不建议马上换 size-aware backbone；短期用分层评估、classifier v2、postproc 分流 |
| ★ | Kasmi et al., “A Crowdsourced Dataset of Aerial Images with Annotated Solar PV Arrays and Installation Metadata”, 2023, Nature Scientific Data | PV aerial dataset | 提供 solar PV arrays + installation metadata，说明 sub-array / installation consistency 是真实问题 | 没有直接证明 pre-merge 更好；不能当成 union-to-installation 的依据 |
| ★ | “A harmonized dataset of ground-mounted solar energy in the US with enhanced metadata”, 2025, Scientific Data | solar dataset / ground-mounted | 说明 solar 数据里 array boundary 与 panel-row/sub-array metadata 可以分层存在 | 不是 rooftop PV，不可直接外推到屋顶实例训练 |
| ★ | “Labeled photovoltaic installations for orthographic aerial imagery in Queens, New York”, 2026, Scientific Data | rooftop PV aerial dataset | rooftop PV installation 数据资源，对 installation-level 数据定义有参考价值 | 数据集建设 ≠ 训练粒度 ablation；不能推出 training target 必须 installation blob |
| ★ | PV mapping literature 一般线索（RPS / Garcia / Li 等 2024-2025 semantic seg） | PV semantic segmentation | 说明多数 PV 文献是 semantic PV/non-PV，不区分 instance | ZAsolar 是 instance + grid inventory，不能直接照搬 semantic metrics |
|  | SAMST 2025 | semi-supervised remote sensing segmentation | 支持过滤、修正、pseudo loss 单独加权 | 只借鉴 weighting/filtering，不复刻完整 SAM self-training |
|  | PENet, 2025, Scientific Reports | point-supervised remote sensing segmentation | 说明稀疏监督可行 | ZAsolar 当前瓶颈不是标注成本，而是 halo / granularity；不进主线 |
|  | WACV 2026 GeoCV point-to-dense SAM remote sensing segmentation | point-to-dense / SAM | 支持点/框 + refinement 的弱监督思路 | semantic/remote-sensing setting，不能直接替换 Mask R-CNN pipeline |
|  | Ren et al., “Learning to Reweight Examples for Robust Deep Learning”, 2018, ICML | meta-reweighting | small clean validation set 引导 noisy sample weights | 是分类起源；用于 mask loss 是改造适配 |
|  | Mirikharaji et al., “Learning to Segment Skin Lesions from Noisy Annotations”, 2019, MICCAI-DART | pixel-level meta-reweighting | 把 reweighting 扩到 segmentation，每个 pixel loss 可自适应加权 | 医学分割；工程量大，先做手写 source weights |
|  | Hu et al., “Learning with Noisy Class Labels for Instance Segmentation”, 2020, ECCV | instance segmentation noisy labels | 支持 instance seg 子任务分拆，不是一锅 BCE | 处理 class noise，不是 mask-shape halo；只借鉴 task-specific loss 思路 |
|  | “Task-Specific Loss for Robust Instance Segmentation With Noisy Class Labels”, 2023, IEEE TCSVT | instance segmentation noisy labels | 进一步说明 foreground/background 与 foreground-instance/classification 噪声影响不同 | 仍主要是 class noise；不要说它直接解决 PV halo |
|  | “Benchmarking Label Noise in Instance Segmentation: Spatial Noise Matters” / COCO-WAN, 2024, arXiv | spatial label noise benchmark | 给 halo 更准确命名：spatial label noise / weak-annotation noise | 主要是 benchmark，不给现成 ZAsolar recipe |
|  | Kimhi et al., COCO-N / COCO-WAN noisy annotations, 2024 | segmentation label-noise benchmark | 用 dilation/erosion/scale 噪声帮助估计 ignore band 起点 | COCO 噪声尺度不能机械搬到 6.7 cm GSD；要面积自适应 |
|  | Holtz et al. / “Revisiting Meta-Learning with Noisy Labels”, 2025 arXiv | meta-reweighting theory | 说明 clean subset 会把 noisy weights 压低；clean set 约 5-10% 的 heuristic | 理论支持，不是 Mask R-CNN PV 实验 |
|  | “Incorporating Boundary Uncertainty into Loss Functions for Biomedical Image Segmentation”, 2021 | boundary uncertainty | 支持边界不确定性应局部化，而不是整 mask 软化 | 医学分割；不给 PV 固定 px 宽度 |
|  | “Label Uncertainty for Ultrasound Segmentation”, 2025 | pixel confidence / uncertainty | 支持只学 confident core，放弃 uncertain boundary | 专家 confidence 不等于自动 boundary band；只能借鉴思想 |
|  | Hao et al., “Uncertainty-aware Iterative Learning for Noisy-labeled Medical Image Segmentation”, 2023 | noisy boundary segmentation | 借鉴面积/尺度自适应 boundary uncertainty | 不需要照搬双网络 |
|  | NSegment+, 2025, AAAI | label-noise augmentation | label-only deformation 可打断系统性 noise | semantic seg；仅作 augmentation 参考 |
|  | “On Image Segmentation With Noisy Labels”, 2022, NeurIPS | segmentation theory | 解释 bulk_ratio / volume bias；提醒谨慎使用 Dice/IoU loss | 这是理论背景，不是直接工程 recipe |
|  | Shumailov et al., “AI Models Collapse When Trained on Recursively Generated Data”, 2024, Nature | model collapse theory | 把“自生成数据递归训练退化”作为广义类比 | 不是 segmentation 专文；报告中更精确术语应是 pseudo-label drift/degradation |
|  | Gerstgrasser et al., “Is Model Collapse Inevitable?”, 2024, COLM | accumulation principle | 支持 H 类 real data 不替换、R 类比例受控 | 1:4 是启发式起点，不是 PV 定律 |
|  | Zhu et al., “Learning from Future”, 2022, NeurIPS | self-training / confirmation bias | future-guided correction 可作为 cycle-breaker 思路 | 不直接落地；只启发 Stage 2 回检 R 类 |
|  | “A Review of Pseudo-Labeling for Semi-Supervised Learning”, 2024 | review | 给 confirmation bias 术语背景 | 太泛，不提供具体 ZAsolar 工程细节 |
|  | “A Self-Reinforcing Prototype Framework to Mitigate Pseudo-Label Degradation in Semi-Supervised Remote Sensing Segmentation”, 2026 | remote sensing SSL | `pseudo-label degradation` 是更贴切命名 | semantic segmentation，不直接替换 instance pipeline |
|  | HDVS semi-supervised semantic segmentation, 2026 | SSL / noisy pseudo-label | 提到 inaccurate data accumulation / confirmation bias | 题名不完整，正式引用前必须核对 |
|  | CRAAC, “Consistency Regularised Active Learning with Automatic Corrections for Real-Life Road Image Annotations”, 2025, WACV | active learning / Mask R-CNN corrections | 支持 selective relabelling，而不是只调 confidence threshold | road images，不是 PV；借鉴 loop 设计 |
|  | AmbiSSL, “Annotation Ambiguity Aware Semi-Supervised Medical Image Segmentation”, 2025, CVPR | ambiguous labels | 支持 H anchors 作为 ambiguity-calibrating anchors | 医学分割；不要直接套数据规模结论 |
|  | “Reciprocal Teaching: Dynamic Multi-Model Teacher-Student Learning for Multiple Noisy Annotations”, 2026, WACV | multiple noisy annotations | 支持保留 H/S/R 来源差异，不先压成单一 GT | 不必短期上 multi-teacher；先做 source weight |
|  | Mask Frozen-DETR, 2023/2024 | decoupled detection / segmentation | 给 freeze / staged training 背书 | 方向相反也可类比：它冻 detector 训 mask；ZAsolar 先冻 mask 训 detector |
|  | Incremental Few-Shot Instance Segmentation, 2021 | staged instance segmentation | 说明 head-level staged fine-tuning 常见 | few-shot setting，不等于 noisy mask 专门解法 |
|  | MS-DETR, “Efficient DETR Training with Mixed Supervision”, 2024, CVPR | mixed supervision / detection candidates | 支持先改善 detection candidate，再改善 mask | DETR 系，不是 Mask R-CNN 直接 recipe |
|  | “Robust Joint Instance-Semantic Segmentation for Roof Part Parsing ...”, 2025, ISPRS Archives | roof remote sensing / noisy multi-stage | 屋顶遥感近邻任务，支持 confidence-guided pseudo-label + selective relabel | roof parts 不是 PV；只借鉴 pipeline |
|  | PointRend, 2020, CVPR | mask boundary refinement | 比纯升分辨率更对症，关注 uncertain points | 先控标签噪声，再考虑上 |
|  | Mask Transfiner, 2022, CVPR | high-quality mask refinement | error-prone region refinement，适合边界细化 | 工程复杂，非 sprint |
|  | EffSeg, 2023, OpenReview | efficient refinement | 更轻的 sparse/structure-preserving refinement 线索 | 仅作部署/速度备选 |
|  | DynaMask, 2023, CVPR | dynamic mask resolution | 小目标/大目标动态选择 mask resolution，与 ZAsolar size-bimodal 匹配 | 工程量大，中长期；不先做统一 56x56 |
|  | Mask R-CNN, He et al., 2017 | baseline instance segmentation | 28->56 mask head ablation 给边界收益基线 | +Boundary AP 不等于解决 systematic halo |
|  | QTPR-Net, 2024, Mathematics | high-res remote sensing instance segmentation | 说明 high-res RS 仍重视 edge uncertainty point refinement | 不换架构，只借鉴方向 |
|  | SAMRefiner, 2025, arXiv | universal mask refinement | 给 SAM-as-inference-refiner 正名 | 不等于 SAM-as-GT-generator 可靠 |
|  | “How to Efficiently Annotate Images for Best-Performing Deep Learning Based Segmentation Models ... SAM”, 2023/2024 | annotation strategy | 支持 data-centric、cost-effective annotation，不是盲扩弱标签 | 题名/年份需核对；不是 PV 专文 |
|  | DeepMerge, 2024/2025 | remote sensing region merging | 支持把 complete geo-object 放到后合并/region merging，而非训练前大 union | 不是 PV；只支撑 post-process/aggregation 思路 |

## 6. 以后给 agent 的短提示

如果以后让 Claude / Codex 继续做这条线，先贴这段：

> 请不要把 boundary-aware loss 和 boundary ignore band 混为一谈。不要建议把 sub-array 训练前 union 成 installation blob。不要默认训练 mask head，也不要把 SAM-assisted / reviewed-prediction mask 当 H 类 hard GT。ZAsolar 当前主目标是 V1.4 grid-level inventory，训练侧先做 freeze-mask-head、source-aware loss weighting、H-anchor accumulation、面积自适应 boundary ignore、selective relabelling；所有改动独立 commit / ablation。★标星文献是 PV/solar 直接相关，其他文献多是间接方法支撑，不能硬说已有 PV rooftop instance-seg 直接验证。
