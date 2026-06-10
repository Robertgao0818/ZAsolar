# gtnoise_t1_ceiling 标注战役 — RA 协议(2026-06-10)

> 目的:实测「当前 A2 级 SAM 标注 GT 的可测 F1 天花板」。同一份冻结模型预测,
> 分别对(a)你产出的 T1 gold 标注、(b)现行 A2 标注重打分,差值 = GT 噪声效应。
> 这是 F1-gap 计划(`docs/plans/2026-06-10-rcnn-f1-gap-review.md` §A4/C9)的
> 测量战役;预算 30–60 标注小时。**产出同时解锁 D1 (PointRend) Gate①。**

## 0. 你拿到的包

| 文件 | 说明 |
|---|---|
| `results/analysis/gtnoise_t1_ceiling/sampling/windows_selected_{jhb,ct}.geojson` | 待标注窗口(QGIS 直接加载,EPSG:4326),共 ~58 个 200m 窗口,`stratum` 字段标明池别 |
| `windows_candidates_{jhb,ct}.gpkg` | 全候选窗口(供补选 shadow / light-roof 两个人工 stratum,各选 3–5 个:阴影覆盖明显的窗口 / 浅色屋顶密集的窗口;选中后把 `stratum` 字段改成 `arch_shadow_MANUAL` / `arch_lightroof_MANUAL`) |
| `configs/eval/gtnoise_t1_ceiling.yaml` | 战役配置(冻结预测、schema、交付落点) |
| 影像 | JHB = Vexcel 2024(6.7cm);CT = aerial 2025。沿用现行 QGIS 工程的底图配置 |

窗口数量按 60% 代表性 / 40% archetype 设计;**只有 `stratum=representative`
的窗口进入天花板 headline**,archetype 窗口只用于分层诊断——两类都要标,
但不要为了"多发现问题"在代表性窗口里挑屋顶,代表性窗口必须穷尽。

## 1. 标注规则(每窗口 exhaustive)

1. **窗口内穷尽标注**:窗口范围内每一个真实光伏装机都画(F1 重打分需要有效
   FP 计数——漏标会把模型的正确检测错判成 FP)。装机中心在窗口内则属于该
   窗口;跨窗口边界的装机整体画完。
2. **installation 级语义**:严格按 `data/annotations/ANNOTATION_SPEC.md` 的
   merge/boundary 规则——同一屋顶上同一装机的多块阵列合并为一个 polygon
   (间距 >1m 的独立阵列分立);边界贴板面外缘,不含屋顶、阴影、走道。
3. **工具**:允许 QGIS + GeoSAM(`label_source=human_manual_sam_assisted`,
   比纯 freehand 快 2–3×),也可纯手画(`human_manual`)。**SAM 只是画笔**:
   产出的每个 polygon 必须逐个过 A1 checklist(下节)后才算 T1。
4. **太阳能热水器不画**(PV-only;拿不准时标 `uncertain` 字段备注,照画并
   注明,评分侧会单独处理)。

## 2. A1 checklist(每个 polygon 必过)

- [ ] merge 规则:同装机阵列已合并;非同装机(间距>1m / 不同屋顶)未误并
- [ ] boundary:边界在板面外缘 ±1 像素内;无 SAM 吞屋顶 / 吞阴影
- [ ] 完整性:该屋顶上所有板面都已覆盖(没有漏掉小阵列)
- [ ] PV 判定:是光伏不是热水器/天窗(存疑 → uncertain 备注)

全部勾完 → `a1_checked=true`。**没有 a1_checked=true 的 polygon 不算 T1。**

## 3. 边界审计字段(D1 Gate① 需要,别漏)

每个窗口标注完后,对照窗口内的**冻结预测**(评分侧会提供 overlay 图层),
给每个与预测重叠的 GT polygon 填 `boundary_audit`:

- `sam_halo` — 预测边界比真实板面外扩(halo/oversize)
- `sam_undersize` — 预测边界内缩 / 漏掉阵列一部分
- `clean` — 预测边界基本贴合(±1–2 像素)
- `uncertain` — 影像不足以判断

## 4. 交付

- 文件:`/mnt/d/ZAsolar/annotations_inbox/gtnoise_t1_ceiling/t1_windows.gpkg`
- 必带字段:`window_id, label_source, a1_checked, boundary_audit, geometry`
  (window_id 从窗口图层属性复制)
- 交付后运行:`python scripts/analysis/gtnoise_t1_score.py --t1-gpkg <path>`
  即出 per-window paired delta + bootstrap CI。

## 5. 种子重审(战役第一项,~2 小时)

manifest 记录 G1238 有 248 张 T1/A1(source_file=`G1238_detailed.gpkg`),但:

1. `G1238_detailed.gpkg` **不在盘上**(data/annotations/、/mnt/d、
   ~/zasolar_data 都没有)— 先从 Dropbox/RA_Solar 备份确认是否找得回。
2. 盘上最接近的是 `G1238.gpkg`(124 个 human polygon,中位 26 m²,
   installation 形态)与 `G1238_SAM2_260320.gpkg`(242 个 SAM2 sub-array)。
   124 ≠ 248:需对账 manifest ↔ 盘面,确认 124 行文件是否为 T1 的部分
   re-export、还是 detailed 文件丢失。
3. 对账后:对 124(或找回的 248)张 human polygon 跑一遍 §2 的 A1 checklist
   抽查(抽 30 张),确认可作 T1 种子;G1189/G1190 的 134 张是 T2,不在
   种子范围(manifest 已核实)。

## 6. 与 CT Ch2 exhaustive-GT 缺口的工时复用

CT 侧 13 个代表性窗口全部落在 independent_26 内 —— 这批 exhaustive 窗口
标注同时填补 CT Ch2(exhaustive recall)长期缺口,一份工时两用。标注完成后
这些窗口会同时注册为 CT 的 Ch2 微型 AOI(独立 channel,不动 clean_gt 锁)。

## 7. 红线

- 本战役所有产出带 `gt_source=t1_gold`,**只进 `gtnoise_t1_ceiling` 诊断
  channel**;不改写 `data/annotations/Capetown/` 现行标注,不触 JHB CBD25
  clean_gt(评估锁),不进模型排名主表。
- 天花板 headline 只从 representative 池计算;archetype 池(含你人工补选的
  shadow/light-roof)只出分层行。
