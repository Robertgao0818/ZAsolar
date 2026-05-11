# Admin-data-poor PV census validation — 探索备忘 (2026-05-10)

## 背景

V1.4 four-channel framework ([`docs/validation_strategy.md`](../validation_strategy.md)) 已落地 Ch1–Ch4：
stratified precision / exhaustive recall / plausibility / opportunistic external。
Ch4 在 DeepSolar (US Tracking the Sun) / DeepPVMapper (FR BDPV+registry) /
Mayer 2025 (FR national registry) 等标杆里是主验证支柱，
**但发展中国家上下文下这一支柱基本塌陷**：

- SA 无国家级 PV 装机注册表（仅 NERSA >100 kVA generation licenses + 部分城市 SSEG dashboard）
- 已有 admin 数据：地址点而非面板坐标，urban-biased，时延数月到数年
  （见 memory `feedback_sa_admin_data_limits.md`）
- DeepSolar 范式假设的 "registry as ground-truth-of-totals" 不成立

**论点**：缺 registry 不是单纯的工程缺陷，而是 methodology gap——
**developing-country PV remote sensing 缺一个标准化的 admin-data-poor validation framework**。
本文档记录探索方向，等数据库 validation 阶段再决定哪几条进 V1.5。

## 设计原则

1. **不试图替代单一 registry**——没有任何单通道能模拟 registry 的 ground-truth-of-totals 角色。
2. **复合多通道有界估计**——每个通道捕捉一类失败模式，组合后给出 inventory 的 confidence interval 而非点估计。
3. **优先内生通道**（用现有 imagery / 模型 / 公开数据），其次低成本外部采集（Street View / 抽样核查），
   admin 数据降为辅证。
4. **每个通道必须明确声明 spatial aggregation level**（grid / suburb / municipality / national），
   遵循 Hu/Malof/Bradbury 2022 关于 aggregation level 不可混的警告。

## 候选通道（按优先级）

### Ch5: Multi-vintage temporal monotonicity 【高 / 内生 / 立刻可做】

**核心**：PV 装机时间维近单调增长（拆除率 <1%/yr）。
T-1 检到的装机，T 时刻必须仍可检；不可则为 false negative 信号。

**数据**：
- CT: aerial_2023 + aerial_2025
- JHB: aerial_legacy + GEID 2024-02 + Vexcel 2024-08
- 同一 grid 跨 vintage 检测对齐

**度量**：
- `install_persistence_rate` = 旧 vintage 检到中、新 vintage 仍检到的比例（监督值 ≥ 0.95）
- `new_install_rate` = 新 vintage 出现而旧 vintage 缺失的比例（应大致符合 SAPVIA 增速）
- 偏离监督值 → detector 跨 vintage 不稳定

**协同**：和 sibling subrepo `solar_backdating` 的 install-date 工作天然耦合。
两个 repo evaluation 共享此通道，避免重复实现。

**风险**：
- imagery vintage 不同 sensor 时，FN 可能是 imagery 难度变化而非真漏检
- 需配合 Ch6 的 cross-imagery 校正

### Ch6: Cross-imagery agreement triangulation 【高 / 内生 / 立刻可做】

**核心**：同一物理装机被多种 imagery 独立观察 = 准独立观测。

**数据**：JHB CBD 25 grid 同时有 aerial / GEID / Vexcel 三套（已部分做过 G0816/G0925 比较）。

**度量**：
- `cross_imagery_IoU@parcel`：同一 parcel 上跨 imagery 的预测 polygon IoU
- `detection_agreement_rate`：≥2/3 imagery 都检到的装机比例
- 分两层：high-confidence inventory (≥2/3 agree) vs uncertain (only 1/3)
- 两层的 area_F1 差 = imagery-induced noise budget

**文献空白**：multi-source PV 文献（如 [Remote Sensing 2022 14:6296](https://www.mdpi.com/2072-4292/14/24/6296)）
主要把 multi-source 用在训练增广，**没人系统地把 imagery diversity 当 validation channel**。
Developing-country 的 imagery diversity 通常比 admin diversity 高，是天然的方法论窗口。

**风险**：vintage 跨度过大时 cross-imagery IoU 会被真增长稀释，需配 Ch5 的时间维校正。

### Ch7: Roof-conditional plausibility prior 【中 / 需要建模 / 1–2 周】

**核心**：装机数应在屋顶面积 / 朝向 / 区域社经 上呈可预测分布；偏离先验 = 异常 stratum。

**数据栈**（SA 全公开）：
- Microsoft Global Building Footprints（覆盖 SA 完整）
- OSM 屋顶轮廓（urban 区域较密）
- StatsSA 2022 census（suburb-level income / household density）
- Global Solar Atlas（GHI/DNI per pixel）
- CoCT/Joburg 公开房产估值（property24 等可补）

**方法**：
- GLM 或 Bayesian small-area-estimation 拟合
  P(install_density | roof_area, irradiance, suburb_income, building_age)
- 每个 stratum 检查 detection density 是否落在 95% 先验置信带
- **严重 over-detection 的 stratum** 大概率是 FP class 富集区
  （e.g. 低收入区 PV 密度异常高 → 可能是水暖热水器，接 cls_v2 工作）

**统计学先例**：wildlife census / disease mapping 的 small-area-estimation
（Cochran 1977；Rao & Molina 2015）。PV 普查文献里没人引这条线。

**风险**：先验本身的训练数据从哪来？
最初一轮可以只看 stratum 内分布的 outlier 而不需要绝对 ground truth。

### Ch8: Stratified site verification via Street View 【中 / 需要采集 / 一次性 8–10h】

**核心**：sample-survey 思路替代 census-vs-census。
SA 绝大部分街道 Google Street View 覆盖良好；SV 还能直接区分 PV vs solar water heater。

**抽样设计**：
- probability-proportional-to-error，分层 by (suburb_income × detector_confidence × building_density)
- n=200–500 站点足够推断 95% CI of total inventory（survey statistics 经典理论）
- 每站点标注 ~1 min on Street View

**输出**：
- unbiased estimator of total PV installations within the sampled region
- water heater FP rate 的独立校准（比 CBD 25 grid 自查更外部）
- per-stratum precision/recall 校准

**先例**：森林普查、wildlife population 估计的标准做法（Särndal et al. 1992）。
PV 文献几乎只见 Bradbury 2016 / BDPV 用 crowdsourcing，
**结构化 stratified sampling 是没人做过的角度**。

### Ch9: Partial registry as bound-check 【低 / 仅辅证】

**核心**：CoCT SSEG dashboard 是地址点形式 grid-tied registry，
**不当 primary truth**——它本身只覆盖 permitted grid-tied 安装，
未注册的逃过它，所以它是 inventory 的 **下界**。

**用法**：
- 若 detection_count_per_suburb < SSEG_count_per_suburb → 必有漏检（confident lower bound 被打破）
- 若 detection_count >> SSEG_count → 多出的部分要么是 unpermitted（合理）要么是 FP（需查）
- 时空趋势比绝对值更有用：`detection_to_sseg_ratio` 应在 suburb 间稳定，离群 suburb 是异常信号

**JHB**：City Power 类似但更稀疏，价值低于 CoCT。

### Ch10: Grid-side signals 【低-中 / 部分公开 / 需要仔细处理 off-grid bias】

**核心**：电网侧聚合数据可作为容量上下界的独立 proxy，但 off-grid bias 必须显式建模。

**A. SARS 模块进口量 (HS 8541.43)**
- SA 公开海关进口数据，HS code 8541.43 = solar cells assembled in modules
- 年度进口 → MW equivalent（avg 400W/module，500–700W 越来越多）
- **国家级 inventory 的强容量上界**（部分会再出口或商用，所以是过估上界）
- 时间序列已在公开数据：2023 ~5 GW，2024 H1 ~3 GW
- 用法：detection_total ≤ cumulative_SARS_imports × residential_share × utilization_factor

**B. Eskom 国家负荷曲线 midday dip**
- Eskom 公开 30-min RDM (Residual Demand Model) 数据
- 2022 后 SA 国家级 midday load suppression 在文献里已被刻画为 behind-the-meter PV 渗透的直接信号
- 用法：suppression_MW(t) ≈ aggregate_PV_capacity × capacity_factor(t)
- **限制**：只能在国家 / 省级用，无 suburb 粒度；loadshedding 期间扰动严重

**C. SAPVIA 季度报告**
- 行业协会 survey-based 估计，国家 + 部分省级 breakdown
- 作为 sanity-check 而非细粒度 GT
- 公开度有限，需要订阅或引用其官方发布

**D. NERSA 注册（>100 kVA）**
- 商用 / 工业 PV 的 license 数据库
- residential <100 kVA 不需 license，所以**对住宅普查无直接价值**
- 用作 commercial 部分的独立校准

**Off-grid bias 子问题**（关键 caveat）：
- **2020–2022 era**：SA SSEG 框架不成熟 + 限电恐慌，大量 unpermitted / off-grid / hybrid 安装。
  这部分被 imagery 检到但**不在任何电网侧信号里**——电网信号系统性低估总装机。
- **2023+**：CoCT 等城市的 feed-in tariff + NERSA 简化注册，加速 grid-tie 化。
  电网信号的覆盖率持续提升，但仍非全集。
- **Hybrid 系统**（battery + inverter，SA 主流）：可孤岛运行也可并网；
  即使不上网也降低 grid import → midday load curve 仍能捕到，但 SSEG 注册可能漏。
- **纯 off-grid**（无并网点）：urban 罕见，多见于 rural / informal settlements；
  imagery 仍可检到，电网数据完全捕不到。
- **结论**：电网侧信号 = "grid-tied subset 的下界" + "hybrid via load-curve 的中界"，
  **永远不是总 inventory 的无偏估计**。在 2020–2022 era 的 SA 上下文下偏差尤其严重。

**用法定位**：
- 国家级 sanity-check（SARS imports + Eskom midday dip）✓
- 区域分布 ground-truth ✗（urban-biased）
- 总 inventory 无偏估计 ✗（off-grid + unpermitted leakage）
- 时间趋势校准 ✓（imports growth rate vs detection growth rate 应大致吻合）

### Ch11: Off-grid bias correction 【探索 / 需要先有 Ch8 数据】

**核心**：用 Ch8 的 site verification 结果直接量化 grid-tied vs off-grid 比例，
作为 Ch9/Ch10 数据的修正系数。

**思路**：
- Ch8 抽样时附加问题：通过 SV 推断该装机是否 grid-tied（屋顶电表布线 / 是否有 NERSA 标识 / SSEG sticker）
- 得 grid_tied_ratio 的分层估计
- Ch10 的电网信号 ÷ grid_tied_ratio = 修正后 total inventory bound

**风险**：SV 判定 grid-tied 准确率有限，需要小规模实地校准。

## 实施优先级建议

**立刻（无新数据成本）**：
- Ch5 multi-vintage monotonicity → 加进 V1.4 的 plausibility 通道
- Ch6 cross-imagery agreement → 加进 stratified precision / exhaustive recall 通道

**等 V1.5 / database validation 阶段**：
- Ch7 roof-conditional plausibility → 需要建模 + 多源数据栈
- Ch8 stratified SV verification → 一次性 ~10h 标注成本，但产出 unbiased estimator

**辅证 / sanity-check**：
- Ch9 CoCT SSEG dashboard
- Ch10 SARS imports + Eskom load curve（仅国家 / 省级）
- Ch11 Off-grid 校正（依赖 Ch8）

## 研究叙事 / publishability hook

把上面打包成 **"Six-channel admin-data-poor PV census validation framework"**
（Ch1–Ch4 已有 + Ch5–Ch10 新增），论文叙事大致：

> Existing PV census validation pipelines (DeepSolar / DeepPVMapper / Mayer 2025)
> all hinge on a national or sub-national registry. In Global South contexts where
> such registries are absent or partial, this gold-standard collapses. We propose
> a **composite validation framework** combining (a) internal temporal monotonicity,
> (b) cross-imagery agreement, (c) socioeconomic-conditional plausibility priors,
> (d) probability-proportional stratified site verification, and (e) opportunistic
> partial-registry / grid-side bounds. We argue that no single channel substitutes
> for a registry, but their composition bounds the inventory error within a
> publishable confidence interval.

**这本身就是发展中国家 PV 遥感的方法论 contribution**，比单纯 detector 性能更有价值。

## 文献参考

- Hu, Malof, Bradbury et al. 2022, "What you get is not always what you see—pitfalls
  in solar array assessment using overhead imagery", _Applied Energy_
  ([arxiv:1902.10895](https://arxiv.org/abs/1902.10895))
- Bradbury et al. 2016, "Distributed solar PV array location and extent dataset",
  _Scientific Data_ ([nature](https://www.nature.com/articles/sdata2016106))
- Yu et al. 2018, "DeepSolar", _Joule_
- Kasmi 2022, DeepPVMapper / DeepSolar tracker (PhD)
- Mayer et al. 2025, "comprehensive building-wise rooftop PV detection — French
  territories", _Applied Energy_
- Cochran 1977, _Sampling Techniques_（Ch8 stratified survey 的统计学基础）
- Rao & Molina 2015, _Small Area Estimation_ 2nd ed.（Ch7 plausibility prior 的方法学）
- Särndal, Swensson, Wretman 1992, _Model Assisted Survey Sampling_

## 状态

**草稿 / exploration only。** 等数据库 validation 阶段（V1.5）再选 1–2 条试点实施。
现阶段 V1.4 仍按 [`docs/validation_strategy.md`](../validation_strategy.md) 既定四通道执行。
