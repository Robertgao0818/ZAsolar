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
5. **校准链与报告链必须同 merge-mode**(2026-06-10 首次锁定实测:
   ct_aerial_2025_v3c 在 pixel-or 报告链上 gap=0.00pp PASS,在 per-detection
   报告链上 gap=5.22pp FAIL —— 校准预测来自 legacy 管线,其 merge 语义与
   pixel-or 同族。merge-mode 是 first-class 口径维度,锁定条目必须声明)。

### 2.3 现有锁与已声明缺口(2026-06-10)

| lock_id | 状态 |
|---|---|
| `ct_aerial_2025_v3c` | 锁定 t*=0.97;pixel-or 链 PASS(0.00pp);per-det 链 FAIL(5.22pp,merge-mode 不匹配,需 per-det 校准预测后重拟合) |
| unified_A × CT aerial_2025 | **缺口 fail-closed**:reviewall 训练集吞掉全部非 suite 标注 grid,不存在 leakage-free 校准集。选项(待用户决定):(a) Li held-out 16 拆 校准/报告 两半;(b) 新标校准 grid |
| JHB vexcel_2024(任何 lineage) | **缺口 fail-closed**:唯一 exhaustive GT = clean_gt CBD25(报告锁) |

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

### 2.5 待用户决定

- unified_A 的 CT 校准缺口走 (a) Li16 拆半 还是 (b) 新标校准 grid。
- per-det 校准预测(36 个校准 grid 的 direct+finalize per-det 重推,约 2–3h
  本地 GPU)是否排期,以解锁 per-det 链的合规锁定点。

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
