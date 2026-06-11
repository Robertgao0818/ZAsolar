# Evaluation Protocol — 口径与工作点纪律

> Status: ACTIVE (2026-06-10, F1-gap Tier A 落地; plan:
> [`plans/2026-06-10-rcnn-f1-gap-review.md`](plans/2026-06-10-rcnn-f1-gap-review.md) §4)
> 历史背景:口径 artifact 在本项目移动过 5–13pp,导致 train20 / v4_2 两次错误裁决。
> 本文档与 `.claude/rules/02-evaluation-semantics.md` 互补:rules 文件管语义守护,
> 本文档管**数字怎么产生与怎么比较**。

## 0. 全局判分规则(摘自 F1-gap 计划 §4,此处为执行口径)

1. **Tier-1 全套**判分:agg_F1 / polygon F1 / σ_Bw / RMSE / R²;σ_Bw+RMSE 主裁判,
   bulk∈[0.5,2.0] 仅 sanity gate。
2. **双 merge-mode 强制**(pixel-or + per-detection),禁止 max-over-modes /
   per-grid best-of headline。
3. clean_gt 锁不动;新 GT 一律新增诊断 channel,禁入排名主表。
4. baseline 对照必须同口径直管线重跑,不引用历史 presence@0.1 数字。
5. chip 级 F1@85 / AP50 可选候选、不可下判。

## 1. IoU 口径(presence / polygon 指标)

### 1.1 双口径输出

- presence 指标(`presence_metrics.csv` 及其消费者)**同时输出 IoU 0.1 与 0.3
  两个口径**,每行带显式 `iou_caliber` 与 `merge_mode` 字段。
- 历史背景:`detect_and_evaluate.py` 的 `evaluate_presence` 链 hardcode
  IoU=0.1,而 `evaluate_predictions.py` 默认 0.3 —— 同名指标两套口径并存多月。

### 1.2 Go-forward 预声明

- **自 2026-06-10 起,新报告/新裁决的 presence 口径标准 = IoU 0.3**(direct
  路径既有默认)。IoU 0.1 列继续输出,仅用于与历史数字的连续性对照。
- polygon F1@{0.1, 0.3, 0.5} 多口径列从每个 results 目录现成的
  `iou_threshold_metrics.csv` 读出 —— additive,零重算,历史数字保持可比。
- **不原地翻转任何默认参数**:所有现有调用方默认行为不变;新口径以新增字段/
  新增列方式并存。

### 1.3 跨口径比较拒绝

- `run_benchmark.py` 在 diff 两个数字时校验 `iou_caliber` 与 `merge_mode`
  一致;不一致时报错退出,不静默比较。

### 1.4 Grandfather 条款

- **2026-06-08 CT census CLS-only 锁定 baseline** 与 **JHB CBD25 clean_gt
  suite** 的全部历史数字保持原口径,不回溯改判。
- 新协议(双口径/双 mode)重推的数值作为 **labeled secondary row** 并排展示,
  显式标 `protocol=2026-06-10`,不替换历史行。

## 2. 工作点(polygon-conf 阈值)锁定协议 — leakage-free

### 2.1 问题

在报告 suite 上 sweep 阈值再在同一 suite 上报数 = oracle 泄漏。2026-06-07
wave1 实测:排名随 polygon-conf 口径在「固定阈 v3c 赢 / 最优阈 unifiedA 赢」
之间翻转。

### 2.2 规则

1. 工作点 per (region, imagery_layer, model lineage) 锁定,登记于
   `configs/eval/locked_operating_points.json`(由
   `scripts/analysis/lock_operating_point.py --update-registry` 写入)。
2. 拟合只在**校准 grid 集**上进行;校准集必须与全部报告 suite 不相交
   (脚本自动校验,交集非空直接报错)。校准集登记于
   `configs/eval/operating_point_calibration.yaml`。
3. 排序规则 = `min(σ_Bw + RMSE/1e5) s.t. bulk∈[0.5,2.0]`
   (poly_conf_sweep 先例,`feedback_tier1_metric_system`)。
4. **验收**:锁定点 agg_area_F1 距 oracle-sweep 最大值 ≤1pp,对**每个**声明的
   报告 suite 成立;否则该 (lock × suite) 组合 FAIL,**禁止把 swept 数字当
   headline 报**(fail-closed)。
5. **校准链与报告链必须同 merge-mode**(merge-mode 是 first-class 口径维度,
   锁定条目必须声明 `merge_mode`;`lock_operating_point.py` 在 lock 声明了
   `merge_mode` 时强制 `validate_on[].merge_mode` 全部一致,否则报错退出)。
   2026-06-10 首次锁定把 pixel-or 校准预测同时验收 pixel-or + per-det 两个
   报告链,per-det 链 gap=5.22pp FAIL —— 当时归因为 merge-mode 不匹配。
   **2026-06-11 决策2 执行更正**:把 36 个校准 grid 用 finalize
   `--merge-mode per-detection` 重推
   (`v3c_targeted_hn_aerial_2025_perdet`),拆出独立 per-det 锁
   `ct_aerial_2025_v3c_perdet`(校准链=报告链=per-det,完全合规)。
   **重推后 gap 仍 = 5.22pp**:merge-mode 修复是必要的合规步,但**不是** gap
   的根因。详见 §2.3 诊断行与 §2.6。

### 2.3 现有锁与已声明缺口(2026-06-11)

| lock_id | merge_mode | 状态 |
|---|---|---|
| `ct_aerial_2025_v3c` | pixel-or | 锁定 t*=0.97;pixel-or 报告链 PASS(0.00pp)。校准链=报告链=pixel-or,合规。 |
| `ct_aerial_2025_v3c_perdet` | per-detection | 锁定 t*=0.97;per-det 报告链 **FAIL(5.22pp)**。校准链已用 per-det 重推(`v3c_targeted_hn_aerial_2025_perdet`),merge-mode 完全合规 —— **gap 不是 merge-mode artifact,是结构性**(§2.6)。fail-closed:per-det 链不得用 swept headline。**2026-06-11 已裁决走 lever (iii)(§2.5):fail-closed 为正式处置,报数用预声明阈值。** |
| `ct_aerial_2025_unifiedA_perdet` | per-detection | **校准集已登记(2026-06-11,选项 b)**:53 个 Li KML grid(`_ct_unifiedA_li_calibration_grids`),物理泄漏审计通过(距全部 Gao 标注格 ≥23.5 km,与 Li held-out 报告格 cell 重叠 0 m²)。**fit 待推理**(L-cell tiles 未下载);fit 完成前 unified_A 报告维持 fail-closed/预声明阈值。 |
| JHB vexcel_2024(任何 lineage) | — | **缺口 fail-closed**:唯一 exhaustive GT = clean_gt CBD25(报告锁) |

### 2.4 Platt / 温度缩放的定位

- **不作为生产 lever**(单调变换,层内 ranking 零影响 —— 2026-06-10 校验否决)。
- 保留为协议内部 ablation:`lock_operating_point.py --platt` 在薄校准集
  (n≤300 检测,TP 标签 = IoP≥0.5)上拟合 2-param Platt 并报 Brier/logloss。
- 2026-06-10 实测(ct_aerial_2025_v3c, n=300):Brier 0.296→0.152 ——
  confidence 确实未校准为概率,但层内单调 ⇒ 阈值放置与全部集合级指标不变;
  跨层共享概率阈值的迁移收益在只有一个 leakage-free 层锁时不可测。
  **裁决:未赢验收,不保留**(fail-closed)。第二个层锁落地后可重开。
- **与 solar_cls 阈值校准不叠加**:solar_cls `calibrate_v2_thresholds.py`
  校准的是分类器 PV-prob(另一个分数轴,在其自己的 per-layer val chips 上),
  与 detector polygon-conf 正交。链序固定:detector conf 锁定(上游)→
  solar_cls per-layer 阈值(下游);**禁止在 cls 过滤之后回头在报告 suite 上
  重调 detector 阈值**。报告的链配置(有无 cls attach)必须与锁定时声明一致。

### 2.5 用户裁决状态

- **已裁决(2026-06-11):unified_A 的 CT 校准缺口走选项 (b)**,校准集从 Li KML
  批次(L0208..L1841,57 grid,2026-06-10 入库)中拆出 **53 个 grid**
  (`configs/eval/operating_point_calibration.yaml`
  `_ct_unifiedA_li_calibration_grids`,lock `ct_aerial_2025_unifiedA_perdet`)。
  - **泄漏审计**(`scripts/analysis/_li_kml_calib_overlap_check.py`,证据
    `results/analysis/operating_point_lock/li_kml_calib_split/`):全部候选距
    Gao 标注格 ≥23.5 km(一次性排除 unified_A 训练泄漏 + G 侧报告面);真
    cell 几何(`data/task_grid_li.gpkg`,已扩至 73 cell)与 Li held-out 报告格
    重叠恰好 0 m²。namespace 论证(`project_li_grid_namespace`)由此获得物理
    实证。
  - **排除 4 个**:L1787(890 m² GT 越界进报告格 L1843 + installation 级口径
    异常)、L1841(41 m² GT 越界进报告格 L1897)、L0264(唯一 GT polygon 72%
    在自身 cell 外,测量无效)、L1520(KML 无 cell 几何,无法定义推理面)。
  - **不留 reserve**:新 GT 按全局规则 3 禁入排名主表,校准是唯一消费方,
    53 全划校准以最大化拟合样本。`ct_li_heldout_16` 同步从 pattern 改为显式
    16 名单(L1895 维持 unassigned)。
  - **fit 待推理**:53 个 L-cell 的 aerial_2025 tiles 未下载(WMS);步骤与
    contingency 见 calibration yaml 该 lock 条目注释。若 fit 后 per-det 链
    同样触发 §2.6 的结构性 FAIL,按 lever (iii) 回退(预声明阈值)。
- **已裁决(2026-06-11):per-det 链 5.22pp gap 走 lever (iii)** —— 接受
  fail-closed 为**正式处置**(不再是临时状态):
  - **CT census per-det 报告永不带 swept headline**;报数用**预声明阈值**,
    出处 = 2026-06-07 threshold-tradeoff 分析(unified_A per-det:下界 0.85 /
    拐点 0.90–0.92)。v3c per-det 本次 sweep 的 oracle t=0.925 落在同一拐点
    区间,互为佐证。报告必须显式标注「阈值为预声明,非报告 suite sweep 所得」,
    且链配置(有无 cls attach)与 2026-06-08 CLS-only 锁定声明一致(§2.4)。
  - **lever (i) 正式否决**(per-det 链单独换 aggF1-consistent ranking rule):
    即便 aggF1-argmax-on-calib,calib↔reporting bulk 漂移仍留 ≥1.55pp > 1pp
    bar —— 治标且不达标。
  - **lever (ii) 标为原则性长期修法**(验收 bar 改 σ_Bw,与 ranking rule 同轴,
    亦即与 Tier-1 主裁判 σ_Bw+RMSE 对齐):属验收语义变更,须专门 session
    重审全 §2 后才可执行;在此之前 aggF1 验收 bar 维持现状。
  - **重开条件**:若 unified_A 校准缺口走选项 (b) 新标校准 grid,应同时修
    校准集 bulk 代表性(对齐 CT census 全量分布;**禁止**照报告 suite 挑
    校准 grid —— 构成间接泄漏),届时可重开 per-det 锁定尝试。

### 2.6 per-det 链 5.22pp gap 的结构性诊断(2026-06-11 决策2)

决策2 已执行「校准链=报告链同 merge-mode」的合规修复(36 grid per-det 重推
→ 独立 per-det 锁),**gap 仍 = 5.22pp**。逐阈值 sweep(两链均跑
[0.85…0.99] + [0.90…0.95] 细网格)定位根因 —— **不是 merge-mode artifact,
而是 ranking rule 与验收 metric 在 per-det 链上系统性反向**:

1. **ranking rule(σ_Bw+RMSE)在 per-det 报告链上单调偏好高阈值**:
   rank 从 t=0.85 的 0.670 单调降到 t=0.97 的 0.311,**无内部极小**;细网格
   [0.90,0.95] 同样单调。锁因此选 t*=0.97。
2. **但验收 metric aggF1 在 per-det 报告链上 t=0.925 见顶(0.7441)**,
   0.925→0.97 区间 σ_Bw 几乎平(0.394→0.308)而 aggF1 掉 5.2pp ——
   过了 0.925 再抬阈值是在删 TP 面积,不是修过涂。
3. **机制 = pixel-or 有过涂、per-det 没有**。pixel-or 报告链 bulk 1.58→0.86,
   抬阈值同时改善 σ_Bw 和 aggF1,两者在 t=0.97 共同见底/见顶 ⇒ 0.00pp PASS。
   per-det 报告链 bulk 在 t=0.85 已是 1.36,预测体积已接近 GT,σ_Bw 的「越紧
   越好」与 aggF1 的「适中最好」从此分道。
4. **叠加 calib↔reporting 的 bulk 分布漂移**:同一阈值下 36 个校准 grid
   bulk≈0.93–0.99,26 个报告 grid bulk≈1.16–1.24(报告 suite 比校准集系统性
   多过涂 ~25%)。即便把 ranking rule 换成 aggF1-argmax-on-calib(t≈0.85–0.90),
   迁移到报告链仍有 1.55pp+ gap > 1pp。校准集本身比报告集「更紧」。

**结论**:per-det 链的 ≤1pp 锁定在「σ_Bw+RMSE ranking rule + aggF1 验收 bar +
当前 36 grid 校准集」三者组合下不可达,且与 merge-mode 无关。证据矩阵(两链
全阈值 + 细网格 sweep)在
`results/analysis/operating_point_lock/ct_aerial_2025_v3c_perdet/`。处置已裁决
(2026-06-11,lever (iii),见 §2.5);lever (ii) 的协议语义改动留待专门 session
重审,在那之前不变更 ranking rule 或验收 metric。

## 3. 双 merge-mode 输出纪律

- `validate_checkpoint.py` 每 checkpoint 输出 Tier-1 全套 + polygon@0.5,
  **双 merge-mode**(pixel-or + per-detection,finalize.py 层切换;同一份
  raw_detections.pkl finalize 两次)。
- summary 两 mode 并排;禁止 max-over-modes 单数字 headline。
- per channel production mode 预声明:Ch3 = pixel-or + SAM;Ch2 = per-det + SAM
  (2026-05-10 audit)。
- `configs/postproc/v4_poly_diag.json` 为 **optional gallery**
  (`validate_checkpoint.py --poly-diag`),diagnostic-only,禁入排名主表。
  2026-06-10 裁决:harness Ch2 默认 = per-det+SAM(已有 25-grid audit 证据;
  stage3b hybrid 从未 head-to-head 赢过 per-det+SAM);原计划的 unified_A
  stage3b re-validate 本地 BLOCKED(clean_gt 三 grid Vexcel tiles 不在本地、
  Tier A 不开 pod),重开条件见该配置 `_meta`。
- 新 checkpoint 转产(adoption ≠ experiment)需走完整链:双 mode + SAM-refined
  variant + per-layer poly_conf re-sweep(按 §2 锁定协议)+ solar_cls per-layer
  阈值重校准;动 CT census 须显式重开 2026-06-08 CLS-only 锁定 baseline 决定。

## 4. installation_sym 诊断 profile(GT 侧兄弟碎片 dissolve)

> 仅诊断 channel,禁入模型排名主表(全局规则 3)。
> 实现:`scripts/analysis/installation_sym_eval.py`(step0 / sweep / sym)。

- **语义**:GT polygons 以 buffer(+gap/2)→union→buffer(−gap/2) dissolve 成
  cluster 后,用现有 installation merge profile(pred 侧 many-to-one)@IoU0.5
  重匹配。`installation` profile 本身**仍然不是** GT-side clustering
  (`.claude/rules/07-annotation-semantics.md` 的例外注记)。
- **gap = 1.0 m**(2026-06-10 sweep 选定):clean_gt25 曲线在 1→2 m 出现最大
  跳变(fragments/cluster 1.44→1.92,2 m 起跨屋顶);Li module 级 over-merge
  audit @1 m 干净(PTA0292:287→275 cluster,最大 cluster 3 成员 / 9.6 m²,
  全部为同屋顶模块兄弟)。SolarMapper 3 m 仅为「proximity 定义评估单元」的
  先例锚(对**预测**像素分组),不是 GT-side merge 先例;3 m 在 clean_gt 上
  产生 47 成员 cluster,过并。
- **实测 fragments-per-cluster @1 m**:clean_gt25 = 1.44;CT SAM2(97 grid,
  3 grid 因 layer 名漂移跳过)= 1.25;xdomain60 Li = 1.34。
  (旧引用「1.49 fragments/installation」无出处,已废,以本表为准。)
- **输出**:installation_sym F1@0.5 + 两个 flip counter
  (`flip_fn_cluster_to_tp` = 切分 artifact 回收;`flip_tp_to_fn` = 部分检出
  installation 暴露),行带 `eval_profile=installation_sym` + `iou_caliber`。
- **GT-merge 不动结构性 FP**:xdomain60 strict FP=3,002(P=0.288);pred 侧
  merge 吸收后剩 ~1.3k;GT dissolve 进一步只把邻接兄弟旁的少量 pred 并入
  match(1,289→1,018),lookalike FP 主体不动 —— precision 修复仍归
  solar_cls,与本 profile 正交。

### 4.1 与 `cluster_level_eval.py` 的和解

两套语义**并存且分工**,不引入第三套:

| | `installation_sym_eval.py` | `cluster_level_eval.py` |
|---|---|---|
| 评估单元 | **pred-independent** GT-side dissolve(固定 gap) | prediction-bridged 连通分量(GT+pred 互桥) |
| 适用 | 「GT 切分 artifact 吃掉多少 F1」诊断;跨模型可比(单元不随预测变) | split/merge 容忍的整体一致性巡检 |
| 风险 | gap 选择需 over-merge audit(§4) | 过涂预测会桥接出大 cluster 而被奖励(单元随预测漂移) |

模型间诊断对比一律用 installation_sym(单元固定);cluster_level_eval 保留
为单模型形态巡检工具,不用于跨模型 delta。

## 5. 指针

- 工作点锁定脚本:`scripts/analysis/lock_operating_point.py`
- 校准集注册:`configs/eval/operating_point_calibration.yaml`
- 锁定注册表:`configs/eval/locked_operating_points.json`
- installation_sym 诊断:`scripts/analysis/installation_sym_eval.py`
  (结果:`results/analysis/installation_sym/`)
- 四通道框架:[`validation_strategy.md`](validation_strategy.md)
- legacy conf 过滤语义(fall-through vs first-match)的 pin:
  `tests/postproc/test_legacy_conf_filter.py`
