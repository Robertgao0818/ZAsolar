# ZAsolar 周工作总结
## 2026-04-01 ~ 2026-04-07（共 4 个有效工作日：04-01 Wed / 04-03 Fri / 04-05 Sun / 04-07 Tue）

---

## Slide 1 — 标注 / Review 工作量

### Cape Town（batch 003 + batch 004）

| 项目 | 数量 | 备注 |
|------|------|------|
| Batch 004 review 完成 grids | **37 / 37 (全量)** | 04-01 起跨多批次完成 |
| Batch 004 review 累计 predictions | ~3,000+ | 5 批次全部完成 |
| Cape Town SAM FN 审查 | **205 markers → 177 accepted** | sam_fn_review.py 新工具 |
| 太阳能热水器 GT 污染审计 (Phase 1) | **671 chips 标注** | tier A 队列；85% PV / 12% 热水器 / 1% 不确定 |
| GT 加热器污染率 | **12.8%** (80/671) | 集中在 batch 003，G1687 (30 个) / G1686 (17 个) 最重 |

### Johannesburg（首次跨城市）

| 项目 | 数量 | 备注 |
|------|------|------|
| Joburg batch1 推理 grids | **100** | 4 类区域 × 25 grids (CBD/Sandton/Alexandra/Midrand) |
| Joburg CBD 25 grids review | **1003 / 1003 (100%)** | 04-07 完成 |
| Joburg SAM FN 重切 | **136 候选多边形 / 179 markers** | EDIT_INNER 19 + NEAR 49 + HARD 68 |

**累计**: 37 CT (batch 004 全量) + 25 JHB CBD = **62 grids review 完成**，3 个新增标注资产（GT 审计、SAM FN 库、Joburg seeds）。

---

## Slide 2 — 模型训练 / 算法改进

### 训练实验

| 模型 | 数据集 | 关键指标 | 结论 |
|------|--------|----------|------|
| **V4** (exp004) | 14842 chips, HN 1.5% | AP50=0.7635 | HN 占比过低被淹没，FP 反而增多 |
| **V4.1** (exp004_v4_1) | 8495 chips, HN 15.7% | AP50=0.7592 | HN 配比修复，但 recall -6.4pp |
| **V3-C** (production) | — | F1=70.9% (primary benchmark) | **仍为当前最优** |

### Primary Benchmark `cape_town_independent_26`

| Model | Precision | Recall | F1 | Verdict |
|-------|-----------|--------|-----|---------|
| **V3-C** | 69.6% | **72.2%** | **70.9%** | baseline |
| **V4.1** | **71.0%** | 65.8% | 68.3% | mixed (P+1.4 / R-6.4) |

### 算法 / 后处理改进

- **分面积段后处理阈值** (04-01): Confidence ≥200m²→0.70, ≥100m²→0.65；Elongation ≥100m²→≤15. **恢复 +48 个被误过滤的大商业板**（G2030 31→50）
- **`v4_canonical.json`** (04-05): 统一 V3/V4 后处理参数，所有 benchmark 必须用此 config
- **`get_grid_record` region 参数** (04-07): 修 Joburg/CT 46 个 grid ID 重叠导致的 spec 错误
- **SAM FN Review GUI 性能改造** (04-07): embedding cache 3-4x 提速，lazy startup 几分钟→25s，merge-based save 修严重数据丢失 bug

### 工具链新增/重构

- `sam_fn_review.py` (04-03 新增, 04-07 重构) — 浏览器 GUI 交互式 SAM FN 审查
- `sam_recut_joburg.py` (04-07 新增) — 三类 FN 策略 SAM 重切
- `build_gt_heater_audit.py` + `label_gt_heater_audit.py` (04-05 新增) — GT 污染审计 pipeline
- `analyze_small_fp.py` + `label_small_fp_taxonomy.py` (04-03 新增) — 小目标 FP 分类
- `export_v4_hn.py` / `export_v4_1_hn.py` (04-05 新增) — V4 HN 导出
- `run_benchmark.py` 并行化 (04-05)
- `RunPod` 自动化（pod 管理脚本 + inference 最佳实践规则文档）

---

## Slide 3 — 主要产出 / 数据资产

### 训练资产

- **Cape Town COCO**: 106 grids, 7301 polygons (含 111 个 SAM FN 新增)
- **V4.1 base COCO**: 68 grids (排除 26 个 benchmark holdout), 6227 pos + 934 neg + 1334 HN (15.7%)
- **GT heater audit canonical CSV**: 86 条热水器排除清单，下次训练自动过滤
- **PV vs 热水器二分类器数据集**: 1933 train + 234 val chips, 模型加载/训练 pipeline 已就绪
- **小 FP HN 候选池**: 455 safe_true_fp（排除 OOB / large-array / near-panel / seg_error）
- **Joburg CBD reviewed predictions**: 25 grids × ~40 polygons，含 1003 review decisions
- **Joburg SAM 重切候选**: 136 polygons (待清理后纳入 V4.2 训练)

### 发布

- **HuggingFace**: `botao0818/zasolar-exp004-v4-hn` 持续发布（V4.1 已上传）
- **GitHub**: 5+ commits 推送，cross-review harness 落地，submodule 改造完成 `Robertgao0818/SA_Solar`

### 文档 / 流程

- **Cross-review harness** (04-03) — Claude Code ↔ Codex 协同审查机制
- **Daily-log skill** + Dropbox 自动同步
- **`02-evaluation-semantics.md` / `04-runpod-ssh.md` / `05-runpod-inference.md`** — 三条新规则文档
- **PV vs 热水器文献调研**: 确认 DeepSolar 系列均未解决此问题，**ZAsolar 双轨方案是原创贡献**

---

## Slide 4a — 关键发现（上）

### 1. FN 根因不是单一原因，而是三类混合

| 类型 | 占比 (Joburg CBD) | 修复策略 |
|------|------------------|---------|
| **Inside existing pred (噪声)** | 19.6% | review GUI 跨 tile 渲染 bug |
| **Near boundary / fragmentation** | 41.3% | SAM bbox + 多点扩展 |
| **Hard miss (模型无响应)** | 39.1% | 训练数据补充 / 单点 SAM 救 |

> 04-03 "97% 后处理损失" 的结论部分修正：raw mask 与 polygon 完美对应，损失发生在跨 chip merge 而非 polygonization。

### 2. V3-C 训练数据严重缺乏大板样本

| 面积区间 | 占比 |
|---------|------|
| <50m² (住宅) | **96.8%** |
| ≥200m² (商业) | **0.6%** |
| ≥500m² (工业) | 0.1% |

→ Joburg 工业大板 hard miss 是必然结果。**JHB01-06 GT 提供 76 个大板正样本可立即补充。**

### 3. Cape Town vs Joburg FP 类型对比

- **Cape Town**: 泳池/太阳能热水器（小目标 high-conf FP），77% taxonomy 占比
- **Joburg CBD**: skylights + rack-shadow（工业屋顶规则几何），90% FP 集中在 G0776/G0891
- **结论**: 两类结构不同需独立 HN 采样；skylight+rack 在 CT 也存在但被泳池加热器掩盖

---

## Slide 4b — 关键发现（下）

### 4. Joburg 影像 vintage gap（重要！04-07 发现）

- **我们的 tiles**: CoJ ArcGIS `AerialPhotography/2023/ImageServer`，**2023 年拍摄**
- **Li 手工 GT**: 在 Google Earth 历史影像 **2024-02** 上画的（Vexcel/Arcgis）
- **~1 年时间差** → Joburg 屋顶 PV 在这一年增长很快

**症状**: G0772–G0857 上 Li GT vs 我们 reviewed+SAM GT 的 per-polygon F1≈0.08, area recall≈0.43。抽样发现大量"漏检"位置在 2023 tiles 上**根本没有面板**——是 2024 年的新装机。

**结论**: 不能直接用 Li 的 Joburg gpkg 做 recall/FN 评估。Joburg 诚实评估必须用同 2023 tiles 上做的标注。

### 5. 跨城市迁移效果（V4 on Joburg CBD）

| 指标 | 值 | 与 CT V3-C 对比 |
|------|-----|----------------|
| Precision | 68.4% | 接近（69.6%） |
| Recall | 82.7% | **优于** (72.2%) |
| F1 | **74.8%** | **优于** (70.9%) |

→ 反直觉：V4 在 Joburg CBD F1 比 V3-C 在 CT 还高。但 GSD 差异 (CT 0.05m / JHB 0.15m) 和 reviewer 主观差异需要对照实验确认。

---

## Slide 5 — 当前主要问题 / 下周计划

### 主要问题

1. **V4.1 recall 退步 -6.4pp** — HN 抑制过强，下一步 HN 比例调到 10-12%
2. **大板样本严重不足** — 训练集 ≥200m² 仅 0.6%
3. **mask head 容量限制（推测）** — 单纯加正样本可能不够，需架构改动
4. **Skylight + rack-shadow FP** — CT/JHB 都存在，无针对性 HN
5. **Review 数据安全** — sam_fn_review save 曾误删 60 个多边形（已修）
6. **目录结构碎片化** — 待重构 type/city/grid 三级
7. **Joburg GT vintage 不匹配** — Li GT (2024) vs tiles (2023)，需 vintage-aware 流程

### 下周计划

| 优先级 | 任务 |
|-------|------|
| **P0** | V4.2 实验：HN 10-12% + JHB 大板正样本 + 排除 86 条热水器，目标 F1 ≥ V3-C |
| **P0** | Joburg 4 dup grids 96 markers 重做 + SAM 重切结果合并到训练集 |
| P1 | Joburg batch1 第 2 批 review (Sandton 25 grids) |
| P1 | PV vs 热水器二分类器训练（数据集已就绪） |
| P1 | GT 审计 Phase 2：扩展到 tier B/C |
| P2 | Mask head 架构实验 (CBAM / MoE head) |
| P2 | 目录重构 + Review GUI 跨 tile 渲染修复 |

---

## Slide 6 — 高分辨率影像来源调研（forward-looking）

> 当前 Cape Town 0.05m / Joburg 0.15m，3x GSD 差距是跨城市迁移的主要障碍之一。后续要扩展到其他城市/国家时，需要稳定的高分辨率影像供给方案。

| 厂商 | 影像类型 | 典型分辨率 | 时间序列能力 | 历史档案 | 学术获取路径 | 适合本项目的点 |
|------|---------|-----------|-------------|---------|-------------|----------------|
| **Maxar** | 商业卫星 | 30 cm 原生，15 cm HD 派生 | 高，但偏"按需 / 高价值刷新"，不是 Planet 那种近每日全球序列 | 很强，官方 125+ PB，档案可追到 1999 | 未找到公开通用学术免费计划；官方主路径是订阅 / 销售。公开免费主要是灾害 Open Data | 最适合"高分细节、历史回看、屋顶级判读" |
| **Planet** | 商业卫星 | PlanetScope 3 m；SkySat 50 cm | 很强。PlanetScope 近每日全球陆地；SkySat 可对任意地点高频重访（最高 10x/day） | 中到强。PSScene 最早 2017-01-01；RapidEye 5 m 库 (2009–2020)；SkySat 自 2016 起 | 有官方 Science Programs / Education & Research。E&R Basic **不含 SkySat**，主要给 PlanetScope / RapidEye | 最适合"时间序列监测"；高分历史看 SkySat，但学术免费通常拿不到 |
| **Vexcel** | 航测 / 航空影像（非卫星） | **7.5 cm 到 15 cm** | 有周期性刷新和历史库，非卫星式高频时序 | 强，官方有 current + historical timestamped imagery | 官方 **University Access Program**，对合格学生 / 研究者免费开放部分数据与 API | 如果接受非卫星，对屋顶检测通常**最强** |

### 评估结论

- **Vexcel** 是 ZAsolar 用例的最佳匹配：屋顶级 GSD（7.5–15cm）跟 Cape Town 当前数据接近，且学术免费路径明确。**建议优先申请 University Access Program**。
- **Maxar** 适合补充历史回看和无 Vexcel 覆盖区域，但缺学术免费渠道，需要找其他经费/合作。
- **Planet** 的 PlanetScope (3m) GSD 太粗不适合屋顶检测；SkySat (50cm) 可用但学术计划基本拿不到。可以作为时间序列监测的辅助层。

### 后续 action

- [ ] 申请 Vexcel University Access Program（覆盖区域、API 调用配额、可发表条款）
- [ ] 联系 Maxar 学术销售/合作渠道询价或申请研究 grant
- [ ] 评估当前 Joburg 0.15m 影像是否来自 Vexcel；若是则验证 University Access 是否可拿同样数据

---

## 一句话总结

**本周从 Cape Town 推断/标注/HN 优化扩展到首次 Joburg 跨城市落地：完成 52 grids review、识别 GT 12.8% 热水器污染、跑出 V4/V4.1 训练实验、Joburg V4 在 CBD 工业区 F1=74.8% 反而高于 CT，但暴露大板样本缺失和 V4.1 recall 退步两个待解问题。**
