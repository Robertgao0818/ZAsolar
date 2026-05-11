# 训练监督分层方案 — 行动清单 (2026-05-09)

## 背景

紧接 [`2026-05-08-jhb-phaseA-retrain.md`](2026-05-08-jhb-phaseA-retrain.md) 失败之后的下一轮训练计划。

直接触发：

- **2026-05-08 JHB Phase A boundary-aware retrain 三项 pass criteria 全 fail**：area_F1 0.608 / Ch2 recall 0.300 / bulk_ratio 1.897，输 V3-C raw。boundary-aware loss 形态本身在 reviewed-prediction GT cycle 不破的前提下救不了 halo。
- **2026-05-08 train20_val5_hn V3-C 续训失败**：chip F1@85 0.737 看似涨，5-grid v4_canonical Ch2 recall −12pp / bulk_ratio +49pp 过冲。SAM-supp GT 体积偏大被 pixel-BCE 当 hard label 学进去。

两轮文献调研（Q1-Q7：source-aware loss weighting / boundary ignore band / iterative pseudo-label collapse / staged training / high-res mask head / GT granularity / mixed gold+silver supervision）合并后的工程行动清单。

## 工作假设

halo 是 reviewed-prediction GT cycle 里的结构性问题，靠任何 loss/postproc 重训都解决不了，必须同时三件事破环：

1. mask head 隔离训练（Stage 1 freeze、Stage 2 解冻）
2. 训练监督按 label_source 分层 (H/S/R) + batch 配比硬约束
3. 边界 ignore band 与 loss weighting 解耦、独立 ablate

---

## 立即可做（这周内）

### 1. train.py 加 `--freeze-mask-head`

最便宜的 unblock。文献支撑：Mask Frozen-DETR 2023 (ICLR 2024 submission)、Incremental Few-Shot Instance Segmentation 2021。

- 把 `model.roi_heads.mask_*` 的 `requires_grad` 都设 False
- Stage 2 解冻时建议 mask head 学习率 = base lr × 0.1
- 实现路径参考 `core.training.boundary_aware_mask.install_patch`

### 2. 训练数据 build pipeline 加两条硬约束

直接对应 train20_val5_hn 失败的根因。文献支撑：Gerstgrasser 2024 COLM "Is Model Collapse Inevitable" 的 accumulation principle。

- **H 类历史数据每轮全部保留**（accumulation 原则，不允许被 R 类替换）
- **R 类数量 ≤ H 类的 4 倍**（real:synthetic ≥ 1:4 经验比例）

落点：`scripts/training/build_*.py` 系列加比例 assertion。

### 3. Batch 配比改成 25-40% H + 60-75% S/R

文献支撑：CRAAC 2025 WACV、AmbiSSL 2025 CVPR、Reciprocal Teaching 2026 WACV。

实现：dataset sampler / collate 阶段按 `label_source` 分层抽样，保证每个 batch 都有 H 锚点 instance。

### 4. mask loss 子任务分拆 + per-source weight

文献支撑：Hu et al. 2020 ECCV "Learning with Noisy Class Labels for Instance Segmentation"、TCSVT 2023 扩展版。是 meta-reweighting 之前的入门版本，工程量小得多。

- 把 mask head 的 BCE loss 拆成两块：前景/背景判断 vs 形状边界
- 前景/背景部分对所有 source 全权重
- 形状边界部分对 R/S 类降权：`H=1.0 / S=0.5 / R=0.2`
- 前置依赖：dataset 层把 `label_source` 透传到 `targets`

---

## 下周可做

### 5. 面积自适应 boundary ignore band

独立模块，**不塞进 `core.training.boundary_aware_mask`**（Phase A 已经证明把多个 lever 打成一个包 debug 不掉）。建议路径 `core.training.boundary_ignore_band`。

文献支撑：Kimhi 2024 COCO-N benchmark、Bridge 2021 biomedical boundary uncertainty、Hao 2023 IET Image Processing。

宽度起点（mask head 分辨率）：

| 类别 | 宽度 |
|------|------|
| 小目标 | 1 px |
| 中目标 | 2 px |
| 大目标 / S 类（sam_added、sam_refined） | 3 px |
| R 类（reviewed_prediction） | 边界 band ignore、core 仍监督 |

逻辑：让 ignore 宽度 roughly 对应 0.07–0.20 m 实际空间误差（GSD 6.7 cm）。

### 6. 常驻 H 锚点池（100-200 个 polygon）

从 2083 clean_gt 抽出来作为 cycle breaker，不参与 reviewed-pred 污染。**覆盖比数量重要** —— 优先保证以下 5 类失败模式都有代表样本：

- 大 installation（>500 m²，SAM roof-swallow 风险点）
- 密集小片（多个 sub-array 紧凑）
- 阴影边（光照边界引发 halo）
- 浅色屋顶（panel/roof 对比度低）
- SAM roof-swallow 已知失败 case（从 G0925 类 outlier 反推）

文献支撑：Ren 2018 ICML / Mirikharaji 2019 MICCAI-DART 的 small clean validation set 做 meta-anchor 思路。

### 7. Selective relabelling pipeline

每轮训练完后挑出对 bulk_ratio 贡献最大的 N 个样本，送 `results/analysis/jhb_phaseA_boundary_refine_webapp/` 人工精修，下一轮纳入 H 池。

文献支撑：CRAAC 2025 WACV "selective relabelling"。比单纯调 confidence threshold 主动得多。

实现：训练后跑 5-grid validation → 按 grid 内 polygon 对 bulk_ratio 偏移贡献排序 → top-N 送精修队列。

---

## 中长期备选（先不进 sprint）

- **Mirikharaji 2019 pixel-level meta-reweighting 完整实现** — 子任务分拆（第 4 条）跑稳后再上。每 step 多一次小金集 forward，开销约 +30%。
- **PointRend mask head**（CVPR 2020）— 边界精度需要再压一档时。比直接换 56×56 dense 性价比高，对小目标尤其好。
- **DynaMask 动态 mask 分辨率**（CVPR 2023）— 项目同时有 <50 px 和 >500 px instance，按 size 动态选 28/56/112 是最对症方案，但工程量大。
- **SAMRefiner 用法正名 + production pipeline 文档化** — production 已经在做（V3-C raw + SAM mask+box refinement on inference 输出），只需补理论命名。明确区分 SAM-as-inference-refiner（保留）vs SAM-as-GT-generator（受限）。

---

## 不进 sprint（已评估、不要再 challenge）

- **56×56 dense mask head** — 文献明确说对 systematic halo 帮助最小，只改善 Boundary AP；PointRend / DynaMask 更对症
- **Sub-array → installation pre-merge** — 两轮文献都反对，等效引入 dilation noise（inter-array gap 进 mask）
- **自定义 PVSAM 变体** — SAM 在大连片屋顶是结构性 ceiling，换 prompt 设计救不了
- **Boundary-aware loss for instance seg** — Phase A 实测失败
- **点监督 (PENet)** — 标注成本不是瓶颈
- **Size-aware backbone (RPS)** — classifier v2 路线已覆盖小目标精度
- **Pseudo-label 整张回灌 BCE 硬训** — 双实验已证伪

---

## 工程纪律：每条独立 commit

不要再犯 Phase A 的错误（boundary-aware loss + dissolve_hairline_gaps + tier-aware mask threshold 三件事一起上、失败后归因不清）。

第 1-7 条**每一条都是独立 PR / commit**，跑完 baseline 才能加下一条。

---

## 决策提醒

1. **今天 / 明天就能让 codex 改的最有价值的事是第 2 条 + 第 3 条** —— 训练数据 build pipeline 的两条硬约束 + batch 配比。代码改动小，但直接 backing 了 train20_val5_hn 失败的根因。

2. **第 4 条子任务分拆是这一轮新挖到的最便宜的杠杆**。之前打算上的 meta-reweighting (Mirikharaji 2019) 工程量大得多；先做子任务分拆作为入门版本，跑通再考虑要不要升级。

3. **NeurIPS 2022 "On Image Segmentation With Noisy Labels" 这篇关于体积偏差的理论文献建议自己读一遍**（不交给 agent 看）。它从理论上解释了 bulk_ratio 过冲的根因：用体积型 loss（Dice 等）在边界有噪声时，最优解本身就跟真实体积有系统性偏差。哪怕只训一轮，只要 GT 边界有噪声 + loss 是体积型，模型最优解就已经偏了；cycle 只是再放大一次。对未来要不要换 loss 形态、要不要在体积层面单独加约束是关键背景。其他文献基本是工程 recipe 级别的。

---

## 关联文档

- 上一轮失败：[`2026-05-08-jhb-phaseA-retrain.md`](2026-05-08-jhb-phaseA-retrain.md)
- 训练实验记录：[`../experiments/exp_train20_val5_hn_negative_result.md`](../experiments/exp_train20_val5_hn_negative_result.md)
- V1.4 评估框架：[`../validation_strategy.md`](../validation_strategy.md)
