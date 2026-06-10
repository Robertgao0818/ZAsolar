# F1-gap Tier A 执行报告(2026-06-10)

执行 [`2026-06-10-rcnn-f1-gap-review.md`](2026-06-10-rcnn-f1-gap-review.md) §4 Tier A
全部四项。零 retrain;本地 GPU 仅用于 A2 等价性验证重跑(G1189)与 A4 种子
先行版补跑(G1238)。所有对照数字同口径直管线重跑,无历史 presence@0.1 引用。

## 1. 逐项状态

| 项 | 子任务 | 状态 | commit |
|---|---|---|---|
| A2 | legacy post_conf/conf_tiered 修复 + 调用方普查 + re-pin + 回归测试 | done | d356724 |
| A2 | leakage-free 工作点协议(脚本+注册+文档)+ Platt ablation | done | 1b58dad |
| A1 | presence 双口径 + iou_caliber/merge_mode 字段 + run_benchmark 跨口径拒绝 | done | eb84e61 |
| A1 | validate_checkpoint 双 merge-mode + Tier-1 全套 + polygon@0.5 | done | f104856 |
| A1 | v4_poly_diag.json + Ch2 默认裁决 | done(re-validate **blocked**,见 §3.3) | f104856 |
| A1 | grandfather 条款 | done(evaluation_protocol.md §1.4) | 1b58dad |
| A3 | Step 0 / dissolve sweep / installation_sym + flip counters / 和解 / 规则注记 | done | 13d4fe7 |
| A4 | 抽样脚本 + 战役配置 + 评分 harness + RA 协议 + 种子重审清单/先行版 | done(接线;**RA 标注 30–60h 待做**) | 2e46c80 |

## 2. 关键数字

### A3 Step 0(strict 1:1 vs pred-merge,@IoU0.5,同一份预测)

| suite | strict F1 | merge F1 | 回收 | strict P / FP |
|---|---:|---:|---:|---|
| xdomain60 (unified_A+SAM c0.925) | 0.3586 | 0.4850 | **+12.64pp** | 0.288 / 3,002 |
| ct26 v3c_wave1_perdet | 0.6261 | 0.6548 | +2.87pp | 0.516 / 860 |
| ct26 v3c_wave1_pixelor | 0.5609 | 0.5717 | +1.08pp | 0.506 / 707 |
| ct26 unifiedA_wave1_perdet | 0.6108 | 0.6247 | +1.39pp | 0.491 / 965 |
| ct26 unifiedA_wave1_pixelor | 0.5522 | 0.5621 | +0.99pp | 0.483 / 792 |

xdomain60 的 0.36 锚点复现(0.3586,P=0.288≈文档 0.291,FP 3,002≈2,969);
0.36→0.763(area F1)差距中 pred 侧 merge 占 ~12.6pp,其余为 GT 切分 +
polygon↔area 口径差。CT SAM2 GT 上切分效应小一个量级。

### A3 dissolve sweep(fragments-per-cluster 实测;废除无出处的「1.49」)

| face | @0.5m | @1m | @2m | @3m |
|---|---:|---:|---:|---:|
| clean_gt25 (2,083 GT) | 1.17 | **1.44** | 1.92 | 2.16 |
| CT SAM2 (97 grid, 6,598 GT) | 1.15 | **1.25** | 1.39 | 1.47 |
| xdomain60 Li (50 grid, 2,564 GT) | 1.21 | **1.34** | 1.50 | 1.62 |

**gap = 1.0 m 选定**:clean_gt 曲线 1→2m 出现最大跳变(+0.48,2m 起跨屋顶);
Li module 级 over-merge audit @1m 干净(PTA0292:287→275 cluster,最大
cluster 3 成员 / 9.6 m²,全部同屋顶模块兄弟)。SolarMapper 3m 仅为
prediction-pixel proximity 先例锚;3m 在 clean_gt 上产生 47 成员 cluster(过并)。
(CT SAM2 有 3 grid 因 gpkg layer 名漂移跳过:G1189/G1190/G1238 —— 与 A4
种子对账同根,见 §3.4。)

### A3 installation_sym @gap1.0m(vs pred-merge baseline)

| suite | base F1 | sym F1 | Δ | 回收 cluster | 暴露 | P 变化 |
|---|---:|---:|---:|---:|---:|---|
| xdomain60 | 0.4881 | 0.5512 | +6.31pp | 154 | 17 | 0.491→0.523 |
| ct26 unifiedA perdet | 0.6247 | 0.6907 | +6.60pp | 92 | 8 | 0.534→0.558 |
| ct26 unifiedA pixelor | 0.5621 | 0.6598 | +9.77pp | 117 | 8 | 0.495→0.531 |
| ct26 v3c perdet | 0.6548 | 0.7082 | +5.34pp | 77 | 13 | 0.579→0.590 |
| ct26 v3c pixelor | 0.5717 | 0.6827 | +11.10pp | 116 | 5 | 0.519→0.567 |

方向全面为正(与「unified_A 类预期升」一致;pixel-or 链增益更大);回收≫暴露。
**GT-merge 不动结构性 FP**:xdomain FP 仅 1,289→1,018(邻接兄弟旁少量 pred
并入 match),lookalike FP 主体留给 solar_cls。仅诊断 channel,不入排名主表。

### A2 调用方普查与等价性验证

- 活跃 legacy 检测路径用户:`run_benchmark.py`(pipeline=geoai 默认,
  post_train.yaml 正传 batch003_best_f1.json → 修复前 conf_tiered 不可注入、
  post_conf=0.92 是死配置)、`batch_inference.sh`、`benchmark_weights.py`、
  CLAUDE.md quick command;`repostprocess.py` import CONF_TIERED 常量(不受
  影响);`param_search.py`/`multi_grid_baseline.py`/`export_hints.py`/
  `review_detections.py` 不传 conf 配置(行为不变)。→ 修复而非 deprecate。
- 等价性:G1189 全管线重跑(修复前 main@0b6147e worktree vs 修复后,同
  re-pinned 配置):127→72 个多边形,geometry+confidence **逐多边形一致**。
- **新发现(pin 进测试)**:legacy tier 迭代是 fall-through(`~keep_mask`,
  area≥200 有效阈值 0.65,(200,0.70) tier 实际是死的),direct 路径
  first-match-wins(0.70)——两路径在 area≥200 & conf∈[0.65,0.70) 历史性
  分叉。修复保留 legacy 语义(行为保全);统一属口径翻转,须另立显式决定。

### A2 工作点锁定(首锁)

- `ct_aerial_2025_v3c`:36 个 suite-free + V3C-train-free 校准 grid,
  t\*=0.97(σ_Bw+RMSE 排序,bulk gate)。迁移验收:pixel-or 报告链
  gap=0.00pp **PASS**;per-detection 链 gap=5.22pp **FAIL** —— 校准预测来自
  legacy 管线(pixel-or 同族),实证**校准链必须与报告链同 merge-mode**
  (已立为协议规则 §2.2-5)。
- 已声明缺口(fail-closed):unified_A×CT(reviewall 训练吞掉全部非 suite
  标注 grid)与 JHB vexcel(唯一 exhaustive GT = clean_gt 报告锁)。
- Platt ablation(n=300,TP 率 0.643):Brier 0.296→0.152(confidence 确实
  非概率),但层内单调 ⇒ 阈值放置不变,单层锁下不可能赢验收 → **不保留**。
- 与 solar_cls 校准不叠加:不同分数轴(detector conf vs cls PV-prob),
  链序固定,禁止 cls 后回调 detector 阈值(§2.4)。

### A1 v4_poly_diag re-validate 裁决

unified_A 上 stage3b-vs-per-det+SAM 的一次性 re-validate **本地 BLOCKED**:
clean_gt 三 grid(G0816/G0817/G0925)的 Vexcel tiles 已不在本地,Tier A 约束
不开 pod;stage3b 脚本本身也已随 ablation 删除(可从 commit 5d85297 恢复)。
按计划的条件分支 + 既有证据裁决:**harness Ch2 默认 = per-det+SAM**
(2026-05-10 25-grid audit:train20 per-det+SAM poly@0.5=79.6%/F1 0.806;
stage3b 从未在任何面上 head-to-head 赢过 per-det+SAM),v4_poly_diag 留
optional gallery(`validate_checkpoint.py --poly-diag`)。重开条件记录于
config `_meta`。

### A4 种子与先行版

- **事实修正**:manifest 的 248×T1 全在 G1238(G1189/G1190 是 T2,58+76);
  T1 源文件 `G1238_detailed.gpkg` 不在盘上,盘上 human 文件 124 行 ≠ 248
  —— manifest↔盘面对账列为种子重审第一项(RA 协议 §5)。
- 种子先行版(G1238,unified_A per-det 本地补跑,同 lineage 同 mode):
  merge-profile F1 对 GT 版本几乎不动(A2-SAM2 0.672 → T1 候选 0.664,
  −0.8pp);strict 1:1 大动(0.687→0.371);area F1 0.812→0.730。n=1、
  T1 候选未过 A1 复核 —— 仅作先行参考,正式天花板待窗口战役。

## 3. 附带发现(本次执行新增)

1. legacy vs direct 的 conf tier 迭代语义分叉(fall-through vs
   first-match-wins)——已 pin 测试,统一需显式决定(§2 A2)。
2. `validate_checkpoint.stage_ch3` 与 `summarize()` schema 漂移(缺
   inter_m2/area_F1 必 KeyError)+ 面积口径(多边形和 vs union)与 baseline
   不可比 —— 已随 A1 修复(union 口径)。
3. CT GT auto-discovery 对 G1189/G1190/G1238 的 layer 名解析失败(sweep
   告警)——与 A4 种子对账同修。
4. `.claude/` 整目录在 .gitignore 中:规则文件(07-annotation-semantics.md
   的 installation_sym 注记)只能落盘、无法入 git。

## 4. 待用户决定

1. unified_A 的 CT 工作点校准缺口:Li held-out 16 拆 校准/报告 两半,还是
   新标校准 grid?(evaluation_protocol.md §2.5)
2. 是否排期 36 个校准 grid 的 per-det 重推(~2–3h 本地 GPU)以解锁
   per-det 链的合规锁定点。
3. RA 标注战役排期(30–60h;包已就绪,见
   `docs/handoffs/2026-06-10-gtnoise-t1-ceiling-ra-protocol.md`)。
4. 是否恢复 5d85297 的 stage3b 脚本以重开 v4_poly_diag re-validate
   (需 Vexcel tiles 回传或 pod)。

## 5. 产物索引

- 协议:`docs/evaluation_protocol.md`(新)
- 脚本:`scripts/analysis/{lock_operating_point,installation_sym_eval,gtnoise_t1_sampling,gtnoise_t1_score}.py`
- 配置:`configs/eval/{operating_point_calibration.yaml,locked_operating_points.json,gtnoise_t1_ceiling.yaml}`、`configs/postproc/v4_poly_diag.json`、batch003 双 preset re-pin
- 测试:`tests/postproc/test_legacy_conf_filter.py`、`tests/analysis/test_benchmark_caliber_guard.py`
- 数据产物(gitignored):`results/analysis/{operating_point_lock,installation_sym,gtnoise_t1_ceiling}/`
