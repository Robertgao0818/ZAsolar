# 2026-05-10 文献调研 — PV 检测 HN 比例与 negative pool 设计

**触发**: 2026-05-10 train20 视觉 review 发现 unique FP = 屋顶纹理 + 天窗 (`project_train20_lookalike_fp_visual.md`)。诊断结论：JHB-only sub-array 训练丢失 lookalike negative archetype 多样性 (`feedback_hn_breadth_dominates_size.md`)。本次调研验证用户记忆"别的 PV 论文负样本比正样本多很多"是否成立，以及主流 HN 设计 recipe。

**关联记忆**:
- `feedback_hn_breadth_dominates_size.md` — HN archetype 广度 ≥ 正样本广度（单次训练原则）
- `feedback_negative_pool_persistent.md` — HN pool 跨训练只增不减（持续性原则）
- `project_v4_hn_strategy.md` — V4 era 已有 `safe_true_fp` corpus（455, 77% 水热器）
- `project_cls_v2_protocol.md` — multi-subtype FP suppressor 计划

## 1. 关键发现一句话

ZAsolar 现状 `neg_ratio=0.15 ≈ 0.18:1` 是 PV 检测文献里几乎最低的，**只比一篇训练时 0% 负样本的 Stanford 学生作业高**。文献区间 1:1 ~ 32:1，且每个 dedicated PV mapping benchmark 都 > 1:1。

## 2. 论文比较表

| Paper | neg:pos | 负样本来源 | 处理 lookalike 的策略 | 备注 |
|---|---|---|---|---|
| Stanford CS231n 2024 (Elbl & Nicollier Sanchez) | **0** | — | 后处理调超参收 FP | 失败案例：把数据集里 no-PV 图删了，模型从未见过负样本 |
| **ZAsolar 现状 (2026-05-10)** | **~0.18:1** | empty-target chip | 无独立 lookalike pool | 倒数第二，与文献规模差 1-2 个数量级 |
| Kasmi BDAPPV 2023 | ~1.2:1 (Google subset 28,807 thumbnails 13,303 with masks) | 众包 no-PV thumbnail | dataset-only，无训练 recipe | dataset paper |
| HyperionSolarNet 2022 (Parhar et al.) | 1.94:1 (训练) / 5.99:1 (Berkeley test) | random + **手动 curate 命名 lookalike pool** | 见 §3 | **最直接 ZAsolar 命中** |
| Camilo / Malof / Bradbury 2018 (SegNet on Fresno) | 3:1 (1.5M neg + 0.5M pos) | 像素 stride 采样（避开 PV 像素）的 urban 背景 | 无 | 纯 random，无 semantic curation |
| SolarDK 2022 (Khomiakov et al.) | **17–32:1** (Gentofte 32:1, Herlev 17.7:1) | 城市瓦片 | (a) class-weighted loss `\|N\|/\|P\|` 或 (b) **BBR 数据集 oversample 正样本至 1:1** | 两种策略 ablate，BBR oversample 完胜 (Cohen's κ 0.21 → 0.67) |
| DeepSolar-Germany Mayer 2020 (IEEE SEST) | 未公布数字 | iterative FP harvest | "efficient dataset creation methodology"，方法 lift recall +8pp | recipe 风格已确认；具体比例藏 IEEE 付费 |
| DeepSolar Yu 2018 (NeurIPS / Joule) | 未公布 | 50+ 美国城市随机抽 366,467 tile | 无 | classifier baseline，不报 ratio 也不做 HN mining |
| Castello et al. 2021 (EPFL U-Net) | 不适用（semantic seg） | — | weighted BCE, weight=4 for minority | 无 HN mining |

## 3. HyperionSolarNet 的关键 quote — ZAsolar 直接命中

> *"Early trained models revealed a number of false positives with images of objects that resemble solar panels. In subsequent image downloads, we collected no_solar images containing objects that could potentially be misclassified as solar panels, such as **skylights, crosswalks, and sides of tall buildings**."*
> — Parhar et al., HyperionSolarNet, NeurIPS CCAI 2022, [arxiv 2201.02107](https://arxiv.org/pdf/2201.02107)

train20 的水热器 / 天窗 / 金属屋顶纹理 FP 与 HyperionSolarNet 早期 FP 在物理类别上几乎相同。他们的 fix 不是炼 loss，是**手动 curate 命名 lookalike pool**作为单独 stream，迭代加入。

## 4. SolarDK 的关键发现 — 验证 cross-region HN 必要性

DeepSolarDE → fine-tune 到 DK：Cohen's κ 0.21 → **0.67**。意味着即便 DE/DK 同属 Europe，跨 region fine-tune 已能解锁巨大 lookalike robustness。直接背书 train21 必须 cross-region HN（CT informal settlement + JHB Soweto/Alex 同进 pool）。

## 5. 反向 anchor

- **没有任何 PV 检测论文用 focal loss / OHEM 解 lookalike FP**（focal-loss 论文都是 PV 缺陷检测，非 aerial mapping）
- **没有任何 PV 论文用体积型 loss 处理边界噪声**（NeurIPS 2022 已警告，见 supervision record §3.11）
- → 进一步实锤 lookalike FP 是 **data-coverage 问题**，不是 loss 问题

## 6. 关于 Mask R-CNN chip-level neg_ratio 的 caveat

SolarDK 32:1 / Camilo 3:1 是 **classification / patch-level** ratio，不能直接 1:1 类比 ZAsolar 的 Mask R-CNN chip-level `neg_ratio`。Mask R-CNN 的 RPN 在每张 chip 内已经采 ~1:3 正负 anchor，所以单看 anchor-level 负样本数不少。但是：

- **空 chip 的独特价值在于"整张图都是 lookalike 负样本"的信号** —— 非空 chip 里模型注意力会被 PV 抓走，无法学到"这整片金属屋顶都不是 PV"的判断
- 0.15 偏低不是无意义，而是**纯 lookalike 背景曝光严重不足**

所以 ZAsolar 的 `neg_ratio` 提升不必上 SolarDK 的 17:1，但应进入 Camilo / BDAPPV 的 1:1 ~ 3:1 区间。

## 7. train21 literature-anchored recipe

1. **`neg_ratio` 0.15 → 0.5–0.75**（进入 Camilo / BDAPPV 同一数量级）
2. **HyperionSolarNet stream**: 单独维护 `data/negative_pool/{heater, skylight, metal_roof_informal, rust_corrugated, parking_canopy, glass_curtain}`，每类 ≥ 数百 chip，与 random empty 并行注入。详见 `feedback_negative_pool_persistent.md`
3. **Mayer 2020 iterative loop 形式化**: 每次 benchmark 后 top-confidence FP → subtype 标 → 写回 pool。V4 era 的 `safe_true_fp` 是已有起点，必须制度化继承
4. **SolarDK cross-region 强制**: CT informal settlement (Khayelitsha / Gugulethu / Mitchell's Plain) + JHB Soweto/Alex 同进 pool；ablate cross-city 共享性
5. **不要换 loss** — 文献无先例支持 focal/Dice/IoU 解 lookalike FP，supervision record §3.11 已警告体积偏差

## 8. 引用链接

- [DeepSolar NeurIPS 2018 PDF](https://aiforsocialgood.github.io/2018/pdfs/track1/78_aisg_neurips2018.pdf)
- [HyperionSolarNet (Parhar et al. 2022) arxiv 2201.02107](https://arxiv.org/pdf/2201.02107)
- [Camilo / Malof / Bradbury 2018 arxiv 1801.04018](https://arxiv.org/pdf/1801.04018)
- [Kasmi BDAPPV Scientific Data 2023](https://www.nature.com/articles/s41597-023-01951-4) / [arxiv 2209.03726](https://arxiv.org/abs/2209.03726)
- [SolarDK Khomiakov 2022 PDF](https://backend.orbit.dtu.dk/ws/files/338834163/solardk_paper.pdf)
- [Castello et al. EPFL 2021](https://infoscience.epfl.ch/server/api/core/bitstreams/474cb251-bc11-4ac3-a26c-bc39f770cc15/content)
- [DeepSolar-Germany Mayer 2020 / GitHub](https://github.com/kdmayer/PV_Pipeline)
- [Stanford CS231n 2024 student paper](https://cs231n.stanford.edu/2024/papers/solar-panel-detection-on-satellite-images-from-faster-r-cnn-to-y.pdf)

## 9. 引用提醒

本文 paper 数字与 quote 来自 2026-05-10 web search agent 调研，已对照公开 PDF / arxiv 抽取，未逐篇核对完整原文。正式写论文引用前要再查 IEEE 付费版（DeepSolar-Germany 实际比例）和最新 venue 版本。
