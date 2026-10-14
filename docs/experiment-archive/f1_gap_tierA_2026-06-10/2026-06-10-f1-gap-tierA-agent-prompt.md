<!--
Handoff prompt for executing Tier A of docs/plans/2026-06-10-rcnn-f1-gap-review.md.
Written 2026-06-10. All file:line anchors re-verified against working tree at commit 0b6147e.
Usage: paste this entire document as the opening prompt of a fresh agent session
(or split into four sessions, one per work package — A1/A2/A3 are mutually independent).
-->

# Agent Prompt: F1-Gap 计划 Tier A 执行（度量诚实化,零 GPU retrain）

## 你的任务

执行 `docs/plans/2026-06-10-rcnn-f1-gap-review.md` 的 **Tier A 全部四项**(A1 协议硬化 / A2 真 bug 修复 / A3 installation_sym 诊断 profile / A4 gtnoise_t1_ceiling 接线)。这是 2026-06-10 40-agent review 定下的第一阶段:**先把度量做诚实,再谈模型改进**——历史上口径 artifact 移动过 5–13pp、导致过两次错误裁决(train20/v4_2)。

Tier A 不含任何 retrain。唯一可能用到本地 GPU 的是 A1 的 stage3b diag 调 SAM;不需要 RunPod pod。

## 必读上下文(开工前按序读完)

1. `docs/plans/2026-06-10-rcnn-f1-gap-review.md` — 主计划。重点:§4 开头的 **7 条全局判分规则**、Tier A 四节、§6 附带发现表、§7 局限。
2. `docs/plans/2026-06-10-rcnn-f1-gap-review.html` — 证据层。每个候选(A1=C12, A2=C11, A3=C3, A4=C9)有两份对抗校验意见逐字记录;实现细节有疑问时以校验全文为准。
3. `docs/experiments/exp_finalizer_pixel_or_vs_per_detection.md` — A1 的 `v4_poly_diag.json` 参数出处(finalizer 文档推荐项)。
4. `.claude/rules/07-annotation-semantics.md` — A3 要修订的规则文件(line 52 "is NOT GT-side clustering")。
5. `docs/governance/repo-rules.md` + `.claude/rules/03-doc-sync.md` — 提交纪律。

## 硬约束(违反任何一条 = 返工)

1. **clean_gt 评估锁不动**:JHB CBD25 评估 GT 锁 clean_gt,任何新 GT/新口径一律新增诊断 channel,禁入模型排名主表(`feedback_eval_gt_lock_clean`)。
2. **不原地翻转任何默认口径**。presence IoU 0.1→0.3 只做"双口径输出 + 显式字段 + 预声明 go-forward",不改任何现有调用方的默认行为。
3. **Grandfather 条款**:2026-06-08 CT census CLS-only 锁定 baseline 与 clean_gt suite 历史数字不回溯改判;新协议重推数值只作 labeled secondary row。
4. **一项一 commit**(A1 内部的独立子项也分 commit);分支命名建议:`fix/legacy-postconf-honor`(A2)、`feat/eval-protocol-hardening`(A1)、`feat/installation-sym-profile`(A3)、`feat/gtnoise-t1-ceiling-wiring`(A4)。
5. 大二进制(tiles/checkpoints/results)不进 git;文档移动/新目录须同 commit 更新 `docs/architecture.md`。
6. 下文给出的 file:line 为 2026-06-10 快照,动手前先 grep 重新定位,代码可能已漂移。

## 推荐执行顺序

**A2 → A1 → A3 → A4(接线部分)**。理由:A2 修的 legacy post_conf bug 改变 legacy 路径行为,A1 的协议输出会读到这条路径的结果,先修 bug 再在其上硬化协议;A3 完全独立可随时插队;A4 接线最后做,其产出(标注战役包)交还人类后本 session 即收尾。

---

## A2 = C11 · 修真 bug + leakage-free 阈值协议 〔~1–1.5 天〕

### 已核实事实(2026-06-10 快照)

- `detect_and_evaluate.py:74-79`:hardcoded `CONF_TIERED = [(200, 0.70), (100, 0.65), (0, 0.85)]`。
- `detect_and_evaluate.py:796-808`(legacy 路径):只要 `area_m2` 列存在就无条件用 CONF_TIERED 过滤,`--postproc-config` 传入的 `post_conf`/`conf_tiered` 被静默忽略;`_post_conf_threshold` 仅在无 `area_m2` 列的 fallback 分支生效。direct 路径(`core/postproc.py:1062-1066` 附近)行为正确。
- `configs/postproc/batch003_best_f1.json` 现写 `"post_conf_threshold": 0.92` —— 在 legacy 路径上是**死配置**,实际生效的一直是 CONF_TIERED 分层阈值。

### 任务

- [ ] 先做调用方普查:grep 找出所有走 legacy 路径的入口(run_benchmark `--parity-mode`、旧脚本等),列清单写进 commit message——修复影响面必须先知道。
- [ ] 修复:legacy 路径 honor `--postproc-config` 的 `post_conf`/`conf_tiered`(语义对齐 `core/postproc.py`),或(若调用方普查显示已无活跃使用者)正式 deprecate 该路径并在入口报错指引。二选一,在报告中说明理由。
- [ ] **同步 re-pin**(与修复同 commit,这是行为保全不是顺手改):`batch003_best_f1.json` 改为显式 `"conf_tiered": [[200, 0.70], [100, 0.65], [0, 0.85]]`——即把该 preset 此前**实际生效**的行为写成显式配置,使修复后行为 = 修复前行为(byte-identical 验证见验收)。CHANGELOG/commit message 注明「依赖 post_conf 的 preset 此前在 legacy 路径上是死配置」。
- [ ] leakage-free 工作点协议(写成脚本 + 协议文档,不是只写文档):per-imagery-layer 锁定阈值,在与**所有报告 suite 不相交**的小校准 grid 集上拟合;验收标准 =「锁定点 F1 距 oracle-sweep ≤1pp / 每个报告 suite」。
- [ ] Platt/温度缩放降为协议内部 ablation(薄校准集 n≈100–300 检测,2-param Platt vs raw argmax,赢了验收才保留);实施前核查与 solar_cls per-imagery-layer 阈值校准的重叠,**同一分数上不堆两层校准**。

### 禁止

- 不引入任何「+pp」声明——本项交付 = 协议完整性,0pp。
- 不做 per-layer 温度缩放作为生产 lever(单调变换,层内 ranking 零影响,已被校验否决)。
- 不做 MaskIoU head(已删)。

### 验收

- 修复后用 `batch003_best_f1.json` 在任一历史 grid 上重跑 legacy 路径,polygon 计数与修复前 hardcode 行为 **逐多边形一致**(re-pin 正确性的直接证据)。
- 新增最小回归测试:一个带 `conf_tiered` 的 config 在 legacy 路径上确实生效(修复前会 fail)。

---

## A1 = C12 · 验证协议硬化 〔~2–3 天〕

### 已核实事实

- `scripts/analysis/validate_checkpoint.py`(注意在 `scripts/analysis/` 不是 training/):V1.4 四通道 harness,10 个 stage;`grep -c merge_mode` = **0 hits**——当前完全没有双 merge-mode 概念。
- `configs/postproc/` 下**没有** `v4_poly_diag.json`(缺失已核实);参数出处在 finalizer 实验文档。
- presence 口径分裂:`detect_and_evaluate.py:2098` hardcode `iou_threshold=0.1`(`evaluate_presence` 链);`evaluate_predictions.py:51` default `--iou-threshold 0.3`。同名指标两个 IoU 口径。
- `evaluate_predictions.py:107` 已产出 `iou_threshold_metrics.csv`(multi-IoU sweep)——A1 的多口径列**直接从每个 results 目录这份现成 CSV 读出**,additive、零重算。
- `cape_town_t1_smoke` suite 已在 `scripts/analysis/run_benchmark.py` + `configs/benchmarks/post_train.yaml` 注册(A4 会用到)。

### 任务

- [ ] `validate_checkpoint.py`:每 checkpoint 输出 **Tier-1 全套(agg_F1/polygon F1/σ_Bw/RMSE/R²)+ polygon@0.5,双 merge-mode(pixel-or + per-detection)**。merge mode 在 finalize.py 层切换(`feedback_merge_mode_first_class`);summary.md 两 mode 并排,**禁止 max-over-modes 单数字 headline**。
- [ ] 落地 `configs/postproc/v4_poly_diag.json`(参数按 finalizer 实验文档推荐项);`_meta.notes` 必须标 *diagnostic-only / derived-from-V3-C-3-grid-ablation*。**接为 harness Ch2 默认之前**,先在 unified_A 上做一次性 re-validate vs plain per-det+SAM:若 per-det+SAM 赢 polygon@0.5,则 harness Ch2 默认 = per-det+SAM,v4_poly_diag 留 optional gallery。把 re-validate 结果(表格)写进报告。
- [ ] IoU 口径硬化(全部 additive):presence 指标**双口径(0.1 与 0.3)同时输出**,每行带显式 `iou_caliber` + `merge_mode` 字段;协议文档预声明 **0.3 为 go-forward 标准**;`run_benchmark.py` 加跨口径 diff 拒绝(两个数字 iou_caliber 不同时报错而非静默比较);F1@{0.1,0.3,0.5}-merge 列从 `iou_threshold_metrics.csv` 读出。
- [ ] Grandfather 落地:协议文档明确 2026-06-08 CT census CLS-only baseline 与 clean_gt suite 的历史数字保持原口径;新协议数值作 labeled secondary row 并排展示,不替换。

### 验收

- 对任一份现有 results 目录跑新 harness,输出含双 mode × 双 caliber 的完整表,字段齐全。
- 故意混口径 diff 一次,确认 run_benchmark 拒绝。

---

## A3 = C3 · installation_sym 诊断 profile 〔~1–1.5 天〕

### 已核实事实与定位

- 机制:`detect_and_evaluate.py:1056-1115` 的 `iou_matching` 原谅 pred 侧碎片(merge_preds)但贪心消耗 pred,**GT 侧兄弟碎片变 FN**。
- 合法性:这是 train20 否决记录明文保留的「允许例外」(GT-side spatial merge of `split_within_gt` siblings,见 `docs/experiments/exp_train20_val5_hn_negative_result.md:141`),与被删除的**训练侧** pre-merge 无关(`feedback_no_pretraining_installation_blob_merge` 不适用——那是训练数据,这是评估诊断)。
- `scripts/analysis/cluster_level_eval.py` 已存在,prediction-bridged clustering 语义,须和解。

### 任务(按序)

- [ ] **Step 0(零代码,先跑先报)**:用现有 installation merge profile @IoU0.5 重打 xdomain60 + CT independent_26,量化「0.36 → 0.763 的差距已被 pred 侧 merge 回收多少」。结果写进报告——这一个数字就值回本项成本。明示 GT-merge 不动 precision(xdomain60 P=0.291 / 2,969 lookalike FP 不受影响)。
- [ ] dissolve 工具第一交付物 = **实测 fragments-per-cluster 随 dissolve gap 的曲线**(sweep 0.5/1/2/3m;SolarMapper 3m 为上锚)于三个面:CT SAM2 94-grid / clean_gt 25 / xdomain60 Li GT。<1m gap 的参数选择验收 = 对 Li module 级切分(PTA0292 类 grid)的 over-merge audit。
- [ ] 实现 `installation_sym` 诊断 profile:GT 侧兄弟碎片 dissolve 后重匹配;输出 = installation_sym F1@0.5 + **两个 flip counter**(FN-cluster→TP 转换 = 切分 artifact 回收;TP→FN+FP = 部分检出 installation 暴露)。
- [ ] 与 `cluster_level_eval.py` 和解:extend/supersede 二选一,或在 suite 文档说明为何 pred-independent GT-side dissolve 优于其 prediction-bridged clustering(后者会因桥接奖励过涂)。**不留第三套无引用的匹配语义。**
- [ ] 同 commit 更新 `.claude/rules/07-annotation-semantics.md`(line 52 附近):installation_sym 作为显式命名的 GT-side-merge 例外注记,保留原「installation profile 本身 is NOT GT-side clustering」表述。

### 禁止

- **不承诺数值提升**:方向 model-dependent(unified_A 类预期升、V3-C sparse-correct 类预期平/降),两种结果都是诊断收益,照实报。
- 不引用「1.49 fragments/installation」——该数字 repo 中无出处,已从计划删除;本项 sweep 给出真实值。
- SolarMapper 引用口径:arXiv:1902.10895 的 3m dilation 是对**预测**像素分组,只能引为「proximity 定义评估单元」先例,不是 GT-side merge 先例。
- installation_sym 只进诊断 channel,不进排名主表(硬约束 1)。

---

## A4 = C9 · gtnoise_t1_ceiling 接线 〔~2–3 天;仅接线,标注是人类的活〕

### Agent 范围界定

本项的 30–60 RA 标注小时**不在你的范围内**。你交付的是「战役包」:窗口抽样 + 评分 harness + 种子重审工具,做到 RA 拿到包即可开标、标完即可出数。

### 已核实事实

- 种子:**已有 248 张 A1/T1 polygons**(G1189/G1190/G1238,`cape_town_t1_smoke`,已在 run_benchmark + `configs/benchmarks/post_train.yaml` 注册)——复用并重审,不是从零。
- 命名约束:**不叫 ch2_***(clean_gt 锁);独立诊断 channel 名 `gtnoise_t1_ceiling`,所有输出带 `gt_source=t1_gold` 字段;不触 `compute_ch2_recall` / `area_aggregate_eval` 的默认行为。

### 任务

- [ ] 窗口抽样脚本:**~60% 代表性核心**(按检测密度比例采样,JHB CBD25 + CT independent_26 范围)→ 全域 GT 噪声天花板 headline;**~40% failure-archetype 过采样**(大 >500m² / 稠密 multi-array / 阴影 / 浅色屋顶 / V3C-halo / SAM-wobble)→ 仅分层诊断。两池输出显式 `stratum` 字段,天花板 headline **只**从代表性池计算(防止过采样池抬高估计)。
- [ ] 标注协议文档(给 RA):每窗口 **exhaustive** installation 级标注(F1 重打分需要有效 FP 计数);允许 `human_manual_sam_assisted` + 显式 A1 checklist 复核(spec 合规 T1,比纯 freehand 快 2–3×);引用 `data/annotations/ANNOTATION_SPEC.md` 的 merge/boundary 规则。与 CT Ch2 exhaustive-GT 缺口协调窗口选择,一份 RA 工时两用。
- [ ] 评分 harness:**同一份冻结预测、换 GT 重打分**;输出 strict 1:1 polygon F1@0.5 + installation merge profile + Tier-1 area,双 merge-mode;**per-window delta 的 paired bootstrap CI**(绝不报独立 F1 CI)。冻结预测集 = unified_A per-det+SAM(production)——把预测来源 pin 进 harness config。
- [ ] 种子流程:248 张 T1 种子的重审清单 + 与新窗口的 paired 打分先行版(战役完成前就能给出首个 ceiling 粗估)。

### 禁止

- 不做训练侧 fold-in(anchor pool / mask_weight 2.0 —— 无 per-instance weight plumbing,`boundary_trust_rules.yaml` 是二值的;该方向已移出本项,gate 在实测 delta >~3pp 后另立计划)。
- 不引用 SolarMapper 0.45→0.67 作 pp 预估(registry-匹配审计,口径不可搬);delta 以实测为准。

### 备注

本项产出同时是 **D1 (PointRend) Gate① 的解锁条件**——边界审计字段(是否 SAM 风格 halo/oversize)要进标注协议,别漏。

---

## 报告要求(收尾时一并交付)

1. 逐项状态表(A1–A4 × 子任务 checkbox,done/blocked/skipped + 理由)。
2. 关键数字:A3 Step-0 回收量、dissolve 曲线拐点、两个 flip counter;A1 的 v4_poly_diag vs per-det+SAM re-validate 裁决;A2 的 legacy 调用方清单与逐多边形一致性验证结果。
3. 所有声明遵守全局规则 7:对照数字必须同口径直管线重跑,不引用历史 presence@0.1 数字。
4. 计划文档回写:把 `docs/plans/2026-06-10-rcnn-f1-gap-review.md` §4 Tier A 的 checkbox 勾掉(完成项),Status 行加注 Tier A 执行日期。
5. 提交遵守 doc-sync 规则;commit co-author 用 canonical 形式 `Claude <noreply@anthropic.com>`。
