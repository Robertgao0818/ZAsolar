<!--
Generated 2026-06-10 via multi-agent workflow (rcnn-f1-gap-review).
7 parallel code auditors (training / data / eval / results / failed-experiments / zerov2 / GT-quality)
+ 4 literature researchers (SOTA metrics / Mask R-CNN upgrades / foundation models / label noise)
→ synthesis (14 candidates) → per-candidate 2-lens adversarial verify (repo-history + evidence-transfer, 28 verdicts).
Run ID: wf_247ef3ce-c4c · 40 agents · ~3.5M tokens · 2 readers lost to API errors (read:gt-quality, lit:label-noise-metrics).
Companion full-text report (all 28 verify opinions verbatim): 2026-06-10-rcnn-f1-gap-review.html
Machine-readable source: 2026-06-10-rcnn-f1-gap-review.result.json
-->

# RCNN F1 差距拆解与赶超路线 (2026-06-10)

> **性质**：review 产物 → 行动计划。**Status: TIER A EXECUTED 2026-06-10**（A1/A2/A3 完成、A4 接线完成待 RA 标注；Tier B–D 待执行，gate 见各节）。执行报告：[`2026-06-10-f1-gap-tierA-execution.md`](2026-06-10-f1-gap-tierA-execution.md)；协议落点：[`../evaluation_protocol.md`](../evaluation_protocol.md)。
> 触发问题：「目前的 RCNN F1 结果和文献主流仍然有较大差距，该如何赶上？」
> 本文档区分「已验证事实(✅，校验 agent 在 repo/文献中逐条核实)」与「推断/估计(🔶)」。
> 所有建议**不触碰 JHB CBD25 clean_gt 评估锁**（`feedback_eval_gt_lock_clean`）。

---

## 0. 一句话结论

~12pp 名义差距中 **60–75% 是口径不可比 + A2 GT 天花板，真实模型短板约 4–6pp** 且已定位到四个测量点；赶超路径 = **度量诚实化（Tier A，零 GPU）→ 廉价证伪 probe（Tier B，≤1 GPU-day）→ 单 lever retrain（Tier C）→ 硬 gate 长线（Tier D）**；14 条候选经两道独立对抗校验后 11 条存活（全部带修正）、3 条否决（C4/C8/C10，理由入档防止重提）。在当前 GT 上把 polygon F1@0.5 推到 0.90+ 是追指标幻影——**可达目标 0.80–0.83，GT 可测天花板 ~0.85–0.88**（C9 战役实测此值）。

---

## 1. 差距归因：12pp 里有多少是真的

### 1.1 三层修正

**修正一：自家口径混用** ✅（eval 审计逐行核实）

| 被引用的数字 | 实际口径 | 出处 |
|---|---|---|
| 0.800 (unified_reviewall_A headline) | **pixel 级 area F1**，5 个 JHB val grid，swept 工作点 — 不是 polygon F1@0.5 | `area_aggregate_eval.py` unary_union |
| 0.709 (CT independent_26, V3-C) | **presence F1 @ IoU≥0.1** + pred 侧 many-to-one merge（极宽松定位） | `detect_and_evaluate.py:2097-2101` |
| 0.749 (Ch1) | 人工复核 precision，非 IoU 口径 | Ch1 stratified |
| 真实 polygon recall@IoU0.5 | **68.8%**（V3-C+SAM）— **79.6%**（train20 per-det+SAM），JHB CBD25 clean_gt | results 审计 |
| 0.36 vs 0.763 | 同一份跨域预测的 strict 1:1 polygon F1 vs area F1 — 差异主体是 GT 切分 artifact 不是检测失败 | xdomain60 |

**修正二：文献口径不可比** ✅（两个文献 agent 独立得到同一结论）

- 0.85–0.95 几乎全是 **curated、PV-dense、in-domain chip 数据集上的 pixel 级语义分割 F1**（USGS-CA 0.94 / PV01 0.91 / Heilbronn 0.96）或 image 级分类（DeepSolar P93/R89）。**没有重要论文报 instance/polygon F1@IoU0.5。**
- 唯一 instance 级锚点：**SolarMapper (Duke) array-wise AP@IoU0.5 = 0.71–0.82**（同一系统双口径：pixel IoU 0.73/0.60 ↔ array AP 0.82/0.71）。已发表的 Mask R-CNN instance 级 PV F1 仅 **0.73（卫星）/ 0.68（UAV）**。
- 真航拍正射上文献 SOTA 自己也崩：S3Former 在 IGN 法国航拍 **pixel F1 0.74**（全部 baseline IoU 0.39–0.59）；SolarDK 最好 pixel IoU 0.72。**ZAsolar 同口径 pixel area_F1 0.80–0.85 in-domain 持平或更好。**
- 跨域文献掉幅大于 ZAsolar：跨 provider 分类 F1 0.98→0.46（Kasmi）；DE→DK zero-shot 0.238；DeepSolar-Germany 跨传感器 P 92→64。ZAsolar 保住聚合校准 bulk 1.04 / R² 0.97。

**修正三：A2 GT 天花板** 🔶（GT 专项 reader 死亡，由 eval/results 审计旁证；C9 实测）

- 训练与评估 GT 100% 是 A2 级 SAM 画 sub-array（manifest 382 行无一 `human_manual`；Capetown_Li 17 grid 也全是 T2 sam_assisted）。
- matched 对平均 IoU 仅 0.73–0.74（边界噪声把真匹配压在 0.5 线附近）；clean_gt 内嵌已接受的 V3-C 预测（循环性）。
- 文献量化过同级 GT 噪声效应：SolarMapper 审计后 precision 0.45→0.67（42% 的「FP」其实是对的）；BDAPPV 共识前面积相关仅 0.61–0.68。
- **估计可测天花板：当前 A2 GT 上 polygon F1@0.5（merge profile）~0.85–0.88，即使模型近完美。**

### 1.2 归因结论与现实目标

| 成分 | 占比 | 数量级 |
|---|---|---|
| 口径不可比（pixel vs polygon、curated vs 全域、in-domain vs 跨域） | ~35–45% | — |
| GT 噪声/碎片天花板 | ~25–30% | — |
| **真实可改进的模型短板** | **~30–40%** | **4–6pp** |

**现实目标**：in-domain polygon F1@0.5（merge profile，当前 A2 GT）**0.80–0.83 可达**；strict 1:1 口径 0.72–0.78；对外可比口径 = pixel area_F1 对标 S3Former/IGN 0.74 与 SolarDK 0.72，polygon 口径对标 SolarMapper 0.71–0.82。V1.4 把 per-polygon F1 降级为诊断、以 grid 级聚合校准为主，与 DeepSolar/SolarMapper/DeepPVMapper 的 national-scale 结论一致——**方向不需要改**。

---

## 2. 真实短板定位（全部有测量）

| # | 短板 | 测量 | 定性 | 对应 lever |
|---|---|---|---|---|
| D1 | multi-array 屋顶 sub-array recall | Ch2 SAM_supp recall **0.30**@IoU0.5；30–100m² 段 **0.33** | detector-bounded：raw box hint rate 已 97.3–100%@score0.05，瓶颈是 **proposal IoU 质量 / mask undersizing** 而非存在性；postproc 救不了 | C1 / C6 / C5(b) |
| D2 | 单一 stratum 精度塌陷 | s_xs × c_lo P=**0.286**（其余 0.78–1.00）；77% 小 FP = 5–20m² 热水器 | 结构性 lookalike；in-domain 已被 solar_cls adaptive_v1 大部分吸收（kill 0.959） | C2 |
| D3 | 边界 halo | raw bulk 1.17；matched IoU 0.73–0.74 | SAM 风格 GT + pixel-BCE 体积偏置；production 靠 SAM refinement 收缩抵消 | C9→C14 / C5(b) |
| D4 | 跨域 per-grid 离散度 | σ_Bw **8×** 退化（聚合校准完好） | 两模式：lookalike 过涂（durban/bfn，cls 靶）+ 稠密 recall 崩（pta）；**同 provider Vexcel ⇒ 语义/地理 shift 非外观 shift**（C10 否决依据） | C2 + cls attach |

---

## 3. 否决台账（do-not-re-propose；重提须先推翻本节事实）

### ❌ C4 — staged ≤600m² SAM mask-trust retrain（双镜头 kill）

前提为假：halo audit 的赢家池 **sam_added_browser（1,699 polys）在 production unified_reviewall_A 中已经 mask_trusted=true / boundary_w=1.0** ✅（`boundary_trust_rules.yaml:41` + `build_unified_reviewall.py:200` + handoff doc 2026-05-12:12）。2026-05-08 的 600m² staged 设计已被 2026-05-11 的 per-source binary trust gate 实质性取代。按原方案翻转规则的真实 delta：`sam_refined_review` = **0 个 instance**（training set 中不存在）+ CT `sam_added_true_fn` ~184 个（**~5%**，不是宣称的 3.4k→9k）。untrusted ~5.6k 池 **95% 是 reviewed_prediction（=V3C_correct mask）** —— 恰是 audit 判定 99.1% 输掉的类、Phase A（bulk 1.897）证明 pixel-BCE 会学进去的 halo 源。候选自己引用的证据反对信任这个池。唯一真未接线的零头：untrusting >600m² SAM 尾（n=15，audit 自评伤害极小）。

### ❌ C8 — anchor/proposal-budget 小目标重调（hist=modify / evid=kill）

仓库内算术直接否决 ✅：推理时 400px chip 被 `GeneralizedRCNNTransform(min_size=800)` 内部 2× 上采样，工作分辨率下抽样 GT（n=392, CT+JHB）sqrt-area **p1=33–39px、中位 137–164px，仅 0.0–0.3% 低于最小 anchor 32px**；「P5/P6 nearly dead」亦假（p75–p99=290–1060px）；`MIN_OBJECT_AREA=5m²` postproc 地板本来就删掉 ~45–67px 以下目标。「预算截断」：`detections_per_img=300` 自 commit 856ddd0 就是 direct/production 默认（100 只在 `--parity-mode` geoai legacy 路径）；按分截断一个 c=0.925 TP 需同 chip 100 个更高分检测，旗舰失败案例 ~20 instances/roof，距 cap 5 倍。clean_gt 中位 instance ~26.8m² ≈ 工作分辨率 ~150px——是 sub-array 不是 12×20px panel。**残值**：1 CPU-day audit 把负结果归档；若 legacy 100 在稠密 chip 上 bind，作为 run_benchmark 可比性 bug 归档，不是 recall lever。

### ❌ C10 — 外观域增广 blur/wavelet/JPEG（hist=modify / evid=kill）

瞄错靶 ✅：xdomain60 六城全部 Vexcel 2025-26 ortho ~6.9–7.4cm —— **同 provider 同 GSD = 地理/语义 shift**，而候选唯一定量锚点 Kasmi 恰好测得外观增广对地理 shift 增益 ≈0（其 −0.11 F1 是 provider shift）；项目自己的失败归因 = lookalike 过涂（solar_cls 靶）+ 稠密 recall 崩；PMB/ELS 在同 imagery 上近 in-domain 泛化，直接证伪外观解释。**残值（挂起）**：唯一真实外观 gap 是 GEID 面（SSIM 0.21）—— re-scope 形态 = geid_2024_02 × JHB CBD25 × 同 clean_gt 量 provider-shift delta，但 GEID 已不在 census 主路径，优先级排不进前十。

> **C10 校验的附带发现（独立于 C10 本身，应修）**：`train.py:363` scale-jitter **area re-index bug** + augmentation 重复注册问题 —— 任何新增广落地前先修这两处。✅

---

## 4. 行动清单（修正后终稿；「修正要求」均出自两道校验全文，见配套 HTML）

### 全局判分规则（每一项 retrain/adoption 共用，不再逐条重复）

1. **Tier-1 全套**判分：agg_F1 / polygon F1 / σ_Bw / RMSE / R²，σ_Bw+RMSE 主裁判，bulk∈[0.5,2.0] 仅 sanity gate（`feedback_tier1_metric_system`）。
2. **双 merge-mode 强制**（pixel-or + per-detection）；per channel 预声明 production mode（Ch3=pixel-or+SAM，Ch2=per-det+SAM，2026-05-10 audit）；**禁止 per-grid / max-over-modes best-of headline**（operating-point shopping，同 bulk_ratio Goodhart 陷阱）。
3. clean_gt 锁不动；新 GT 一律新增诊断 channel，禁入排名主表。
4. **一 lever 一 commit 一 retrain**（Phase A 教训）；retrain 基于 unified_reviewall recipe（mask_trusted gating + freeze-mask-head staging + `--seed`），不做 naive V3-C warm start。
5. chip 级 F1@85/AP50 可选候选、不可下判（train20/v4_2 两次 Goodhart 失败在案）。
6. **adoption ≠ experiment**：任何新 checkpoint 转产前需走完整链 —— 双 mode + SAM-refined variant + per-layer poly_conf re-sweep + solar_cls per-imagery-layer 阈值重校准；动 CT census 须显式重开 2026-06-08 CLS-only 锁定 baseline 的决定。
7. baseline 对照必须**同口径直管线重跑**，不引用历史 presence@0.1 数字（0.1/0.3 IoU + hardcoded CONF_TIERED 双重 confound）。

---

### Tier A — 零 GPU，立即（把度量做诚实；历史口径 artifact 移动过 5–13pp、两次错误裁决）

#### A1 = C12 · 验证协议硬化 〔~2–3 天，本地 GPU only（stage3b diag 调 SAM），无 pod〕

- [x] `validate_checkpoint.py` 每 checkpoint 输出 Tier-1 + polygon@0.5 **双 merge-mode**（当前 grep merge_mode = 0 hits ✅）。〔2026-06-10 done,commit f104856;附带修复 stage_ch3 与 summarize 的 schema 漂移〕
- [x] 落地 `configs/postproc/v4_poly_diag.json`（finalizer 文档推荐项，缺失已核实 ✅）；notes 标 *diagnostic-only / derived-from-V3-C-3-grid-ablation*；接为 harness Ch2 默认前先在 unified_A 上一次性 re-validate vs plain per-det+SAM —— 若 per-det+SAM 赢 polygon@0.5，则 harness Ch2 默认 = per-det+SAM，v4_poly_diag 留 optional gallery。〔2026-06-10:config 落地;re-validate 本地 BLOCKED(clean_gt 三 grid 无本地 Vexcel tiles、不开 pod)→ 按既有 25-grid audit 证据裁决 Ch2 默认 = per-det+SAM,v4_poly_diag 留 --poly-diag gallery;重开条件见 config _meta〕
- [x] IoU 口径：**不原地改任何默认**。presence 指标双口径（0.1 与 0.3）输出 + 显式 `iou_caliber`、`merge_mode` 字段；预声明 0.3 为 go-forward 标准；run_benchmark **拒绝跨口径 diff**；F1@{0.1,0.3,0.5}-merge 列直接从每个 results 目录已有的 `iou_threshold_metrics.csv` 读出（additive、零重算、历史数字保持可比）。〔2026-06-10 done,commit eb84e61〕
- [x] held-out operating-point 规则只 forward-looking：**grandfather** 2026-06-08 CT census CLS-only baseline 与 clean_gt suite，新协议重推数值作 labeled secondary row，不回溯改判。〔2026-06-10 done,docs/evaluation_protocol.md §1.4〕

#### A2 = C11 · 修真 bug + leakage-free 阈值协议 〔~1–1.5 天，零 GPU〕

- [x] 修 `detect_and_evaluate.py:794-808`：legacy 路径 honor `--postproc-config` 的 `post_conf`/`conf_tiered`（与 `core/postproc.py:1062-1066` 一致），或正式 deprecate 该路径。〔2026-06-10 done,commit d356724:修复(活跃调用方=run_benchmark geoai/batch_inference/benchmark_weights);G1189 全管线重跑逐多边形一致;**新发现**:legacy tier fall-through vs direct first-match-wins 在 area≥200&conf∈[0.65,0.70) 分叉,已 pin 测试,统一须另立决定〕
- [x] 同步 re-pin `configs/postproc/batch003_best_f1.json` 为显式 `conf_tiered [[200,0.70],[100,0.65],[0,0.85]]`；CHANGELOG 注明「依赖 post_conf 的 preset 此前在 legacy 路径上是死配置」。〔2026-06-10 done(含 batch003_recall95),commit message 即 CHANGELOG 注记〕
- [x] leakage-free 工作点协议：per-imagery-layer 锁定阈值在与所有报告 suite 不相交的小校准 grid 集上拟合；验收 =「锁定点 F1 距 oracle-sweep ≤1pp / 每个报告 suite」。Platt/温度缩放降为协议内部 ablation（n≈100–300 检测的薄校准集上 2-param Platt vs raw argmax，赢了验收才保留）。实施前核查与 solar_cls per-layer 校准的重叠，避免同一分数上堆两层校准。〔2026-06-10 done,commit 1b58dad:首锁 ct_aerial_2025_v3c t*=0.97,pixel-or 链 PASS 0.00pp / per-det 链 FAIL 5.22pp ⇒ 新增协议规则「校准链与报告链必须同 merge-mode」;Platt 实测 Brier 0.296→0.152 但层内单调未赢验收→不保留;与 solar_cls 不同分数轴已核查并写入 §2.4〕
- 已删：per-layer 温度缩放作为生产 lever（单调变换，层内 ranking 零影响 ✅）；MaskIoU head；一切「+pp」声明（本项交付 = 协议完整性，0pp）。

#### A3 = C3 · installation_sym 诊断 profile（GT 侧兄弟碎片 dissolve）〔~1–1.5 天，零 GPU〕

机制已核实 ✅：`iou_matching`（`detect_and_evaluate.py:1056-1115`）原谅 pred 侧碎片但贪心消耗 pred，GT 侧兄弟碎片变 FN。本项是 train20 否决记录明文保留的「允许例外」（GT-side spatial merge of `split_within_gt` siblings，`exp_train20_val5_hn_negative_result.md:141`），与被删除的训练侧 pre-merge 无关。

- [x] **Step 0（零代码）**：用现有 installation merge profile @IoU0.5 重打 xdomain60 + CT26，量 0.36→0.763 已被 pred 侧 merge 回收多少；明示 GT-merge 不动 precision（xdomain60 P=0.291 / 2,969 lookalike FP 不受影响）。〔2026-06-10 done:xdomain60 strict 0.3586(复现锚点)→merge 0.4850,**回收 12.64pp**;CT26 wave1 四链仅 0.99–2.87pp〕
- [x] dissolve 工具第一交付物 = **实测** per-suite fragments-per-cluster 随 dissolve gap 的曲线（sweep 0.5/1/2/3m，SolarMapper 3m 为上锚）于 CT SAM2 94-grid / clean_gt 25 / xdomain60 Li GT；<1m gap 参数验收 = 对 Li module 级切分（PTA0292 类 grid）的 over-merge audit。**删除未经证实的「1.49 fragments/installation」引用**（repo 中 grep 不到，疑与 Li GT ~1,490 polygon 混淆）。〔2026-06-10 done:@1m clean_gt25=1.44 / CT SAM2=1.25 / Li=1.34;gap=1.0m 选定(1→2m 跳变 1.44→1.92=跨屋顶);PTA0292 audit 干净(287→275,max cluster 3 成员/9.6m²)〕
- [x] 与已有 `scripts/analysis/cluster_level_eval.py` 和解：extend/supersede 或在 suite 文档说明为何 pred-independent GT-side dissolve 优于其 prediction-bridged clustering（后者会因桥接奖励过涂）——不落第三套无引用的匹配语义。〔2026-06-10 done:evaluation_protocol.md §4.1 分工表(跨模型诊断=installation_sym;单模型形态巡检=cluster_level_eval)〕
- [x] 输出 = installation_sym F1@0.5 + 两个 flip counter（FN-cluster→TP 转换 = 切分 artifact 回收；TP→FN+FP = 部分检出 installation 暴露）。**不承诺数值提升**：方向 model-dependent（unified_A 类预期升、V3-C sparse-correct 类预期平/降），两种结果都是诊断收益。〔2026-06-10 done @gap1.0m:xdomain60 0.488→0.551(+6.31pp,回收154/暴露17);CT26 +5.3~+11.1pp,pixel-or 链增益更大;结构性 FP 主体不动(1,289→1,018)〕
- [x] 同步更新 `.claude/rules/07-annotation-semantics.md`：installation_sym 作为显式命名的 GT-side-merge 例外注记（该文件现写「is NOT GT-side clustering」）。〔2026-06-10 done(落盘;.claude/ 在 .gitignore,不入 git)〕
- SolarMapper 引用口径纠正：arXiv:1902.10895 的 3m dilation 是对**预测**像素分组、GT 本就 array 级 —— 引为「proximity 定义评估单元」先例，不是 GT-side merge 先例。

#### A4 = C9 · `gtnoise_t1_ceiling` 测量战役（T1 gold paired re-score）〔30–60 RA 小时 + 2–3 天 wiring；GPU 仅重打分〕

- [x] 命名/定位：**不叫 ch2_***（clean_gt 锁）；独立诊断 channel `gtnoise_t1_ceiling`，输出带 `gt_source=t1_gold` 字段，禁入模型排名主表，不触 `compute_ch2_recall`/`area_aggregate_eval` 默认。〔2026-06-10 done,commit 2e46c80〕
- [x] 标注预算拆分：**~60% 代表性核心**（按检测密度比例采样的窗口，JHB CBD25 + CT independent_26 范围）→ 全域 GT 噪声天花板 headline；**~40% failure-archetype 过采样**（大 >500m² / 稠密 multi-array / 阴影 / 浅色屋顶 / V3C-halo / SAM-wobble 类）→ 仅作分层诊断（防止天花板估计被偏置抬高）。〔2026-06-10 抽样已出:36 代表性 + 22 archetype(large/dense/wobble 自动;shadow/light-roof 留 RA 人工补选),显式 stratum〕
- [ ] 每个窗口内 **exhaustive** installation 级标注（F1 重打分需要有效 FP 计数）。允许 `human_manual_sam_assisted` + 显式 A1 checklist 复核（spec 合规 T1，比纯 freehand 快 2–3×）。〔**RA 标注待做**(30–60h);协议 = docs/handoffs/2026-06-10-gtnoise-t1-ceiling-ra-protocol.md,含边界审计字段(D1 Gate① 要件)〕
- [x] 评分：同一份冻结预测、换 GT 重打分；strict 1:1 polygon F1@0.5 + installation merge profile + Tier-1 area，双 merge-mode；报 **per-window delta 的 paired bootstrap CI**（绝不报独立 F1 CI）。〔2026-06-10 harness done(gtnoise_t1_score.py);冻结预测 pin 于 configs/eval/gtnoise_t1_ceiling.yaml〕
- [x] 种子纠正 ✅：复用并重审**已有的 248 张 A1/T1**（G1189/G1190/G1238，cape_town_t1_smoke）——不是从零；并与 CT Ch2 exhaustive-GT 缺口协调 RA 工时，一份工时两用。〔2026-06-10 接线 done + **事实修正**:248×T1 全在 G1238(G1189/G1190 是 T2 58+76);其源文件 G1238_detailed.gpkg 不在盘上、盘上 human 文件 124 行 ≠ 248 —— 对账列为种子重审第一项。种子先行版已跑:G1238 上 merge-profile F1 对 GT 版本几乎不动(A2 0.672→T1候选 0.664),strict 1:1 大动(0.687→0.371),n=1 仅作参考〕
- 已移出本项：训练侧 anchor pool / mask_weight 2.0 fold-in（无 per-instance weight plumbing —— `boundary_trust_rules.yaml` 是二值 1.0/0.0；且贴近被否决的 pre-merge 路线）。若日后要做：按 Phase B 原文（clean-boundary、sub-array 语义），gate 在本项实测 delta >~3pp，且拆三个独立 commit（binary trusted fold-in / H-presence sampler / weight plumbing），稠密屋顶 anchor 遵守 >1m gap 分立规则、同屋顶不得混粒度监督。
- 量级提示：SolarMapper 0.45→0.67 是 registry-匹配审计、不可直接搬作 pp 预估；本项 delta 以实测为准。**本项同时是 D 层 C14 的解锁条件之一。**

---

### Tier B — 廉价证伪 probe（先测机制再投工程；NMS-relax probe 半天杀一条死路的纪律）

#### B1 = C1 · TTA 证伪 pilot 〔~0.5 天;通过后集成实价 1–2 周〕

依据：TTA 是唯一 sanctioned 推理时 recall lever（`feedback_recall_recovery_constraints`），grep 证实零代码 ✅。但两道校验各砍半个原方案：

- 视图集 ✅：train.py TrainTransforms 已含 hflip/vflip/rot90 + 0.8–1.2× scale jitter → 原方案视图（flip/rot90/0.9/1.25×）几乎全在**已训练等变包络内**，输出相关、新 proposal 近零。**只用 out-of-envelope 放大视图：1.5× / 2.0× chip 重采样**（400px 窗口重叠处理）。
- 机制改写 ✅：raw_box_hint_rate 已 97.3–100%（G0816/G0817/G0925, score 0.05）→ headroom 不在 proposal 存在性（~3%）而在 **proposal IoU 质量 / mask undersizing**；claim = scale 驱动的 mask-quality/score 恢复。
- [ ] **Pilot**：预重采样 G0925、G0817、2–3 个最低 Ch2-recall clean_gt grid（+PTA0292 若 tiles 在本地）的 tiles；`detect_direct.py` 原样按视图各跑一遍；量 (i) 漏检 clean_gt polygon 在任意视图获得 conf≥0.3 raw proposal 的比例，(ii) per-missed-polygon **best-proposal IoU**（mask hint @0.3/0.5，复用 exp_finalizer raw-hint audit，不量 existence）。
- [ ] **Kill bar：<10–15% 漏检 polygon 转化为 IoU≥0.5 可恢复 proposal → 整条 TTA 放弃**（不做 merge/inverse-transform/重校准级联）。
- 若通过：集成实价 1–2 周 end-to-end（含 poly_conf_sweep redo + solar_cls per-layer 重校准），推理预算 2–3×（两个 scale 视图，非 4–8×）；预期改述 **+0.5–1.5pp** in-domain @re-swept 工作点（删「跨域更大」声明——跨域无 FP 吸收层）；pixel-or 模式专门盯 bulk（近重复 mask 在 NMS 前加肥 OR-union；NMS 只保护 per-det）；TTA 参数写入 raw artifact manifest。

#### B2 = C13 · solar_zerov2 probe 1b + 1a′ 〔~1 GPU-day, <$20 pod〕

Rejected #11 只覆盖 frozen-SAT 最差 cell；文献三重预测其失败（PANGAEA frozen GFM≈U-Net；GEO-Bench-2 <10m GSD 下 web 预训练 > SAT；DINO Soars unfreeze 值至 +11.9 mIoU），同样三重指向 web 权重 + unfreeze 是该测的 cell。**校验确认 Phase 0 decoder 是 random-init**（`phase_0_stdout.log:9` vs config 写明 pretrained-M2F 意图）= 第六个 confound ✅。

- [ ] 烧 GPU 前免费检查：复核 decoder random-init 对 Phase 0 anchor 有效性的影响（所有 probe 都要和它比）。
- [ ] **probe 1b 先行**（frozen web-DINOv3-L cached，native GSD，`m2f_dinov3_l.yaml` 已存在，零新代码，~6h）。
- [ ] **probe 1a′**（SAT backbone + pretrained Mask2Former decoder 权重，~3h cached）：把 decoder-init 效应单独定量；**「双双 <0.78」不得触发 Stage-2 直到 1a′ 排除 decoder 因素**。
- [ ] probe 1c（SAT@13.4cm）严格条件于 1b 不结论或显示 pretrain 域非主因；实价为文档所述「较大窗口 re-cache」而非小 flag。
- [ ] kill 判定前置条件：zerov2 `infer.py` 目前只输出 per_detection —— 补 **pixel-or merge 路径**（instance mask rasterize-OR 后矢量化，镜像 finalize.py 语义），任何 fail 判定双 mode 打分；revive 分支可 per-det vs per-det（与 0.817 floor 同口径）。
- [ ] Gate 不变：F1≥0.817 **且** σ_Bw≤0.248（JHB CBD25 clean_gt，area_aggregate_eval + poly_conf_sweep）→ revive frozen；否则 Stage-2 last-4-block unfreeze（5 处代码：backbone param group / scoped no_grad / grad ckpt / lr_backbone 1e-5 + layer-wise decay + EMA）成为 make-or-break，带 CT26 overfit guard。
- 预期改述：交付物 = **gate 决策本身**；revive 后 in-domain 上行低个位数封顶（A2 GT capped）；战略上行 = **cross-domain σ_Bw（Rein/PEFT 路线）**—— Mask R-CNN 系所有候选都不覆盖这一项。（RSPrompter +4 mask AP 类比已删。）

---

### Tier C — retrain 杠杆（一次一个 lever；判分按全局规则）

#### C-1 = C2 · 负样本池真正落地 〔~1 周工具 + 1 retrain (1–2 GPU-days) + 1–2 天 cls 重校准;零新标注〕 ← **retrain 类第一优先**

硬事实 ✅：`data/negative_pool` manifest **678/678 行无 geometry → 今天导出 0 个 chip**；production unified_reviewall_A 捆绑 **0 个 targeted HN chip**（`build_unified_reviewall.py:426` `hard_negatives_config=[]`），easy-neg ratio 0.18:1 为项目自家文献综述认定的「PV 文献几乎最低」（文献带 1:1–32:1，chip 级推荐 0.5–0.75）。FP 抑制现完全靠 V3-C warm-start 惯性。

- [ ] 建 `scripts/training/negative_pool/ingest_fp_audit.py` + geometry backfill（经 `source_pred_id` join `results/johannesburg/v3c_geid_2024_02/<G>/predictions_metric.gpkg`，可行性已验证 ✅）。GEID 行 backfill 仅作 provenance；geid_2024_02 chip 入训练包 gate 在 imagery-layer balance（CT-aerial 与 Vexcel HN 流量级相当才捆，避免「负样本独占第三外观域」）。
- [ ] 摄入源统一过 **human/cls-agreement filter**：
  - Gemini FP-cut（JHB 6,072 drops）：取 cls/人工一致子集（Gemini 在 CT 错杀过 107–119 真 PV）。
  - 跨域源限 **verified-non-PV**：10 个 confirmed-zero-PV empty-grid probe FP（33 polygons）+ durban/bfn 过涂多边形的人审/cls 一致子集；**禁止把 BFN0126/DBN0044 当「纯 FP」整体摄入**（xdomain60 文档自己警告可能是 Li 欠标注 = 真 PV；池子只增不减，污染不可逆）。
  - **eval-leakage 防护**：被挖 HN 的 grid 必须退出 xdomain60 评估面（或先定义 held-out split），否则任何跨域改进声明无效。
- [ ] `neg_ratio 0.15→0.5`；验收 = HN archetype 广度 ≥ 正样本广度 per region（`feedback_hn_breadth_dominates_size`）。
- [ ] Gate：Ch2 recall@0.3 非回归（locked clean_gt）+ Tier-1 全套双 mode。σ_Bw 声明改为「**在 solar_cls-attached baseline 之上的边际**，held-out 跨域 grid 上量」（cls 本就是 Mode-1 的 documented fix；原「σ_Bw 减半」无证据锚点，删）。
- [ ] 若与 B1 (TTA) 同轮评估：分开报 with/without TTA（单 lever 归因）。
- 🔶 预期 +1–3pp（precision 驱动，无 classifier 的 suite 上）：按 exp003 **孤立 HN delta ~+7pp（P=0.35–0.52 低基数）** 保守折算，不是 headline +12.7（后者混入了训练集刷新）。

#### C-2 = C7 · warmup + EMA 搭车 〔~1 天 flags，0 额外 GPU〕

现状核实 ✅：Stage-2 在 optimizer 热切换后 cosine 冷启动（`train.py:1413-1423`）；全仓库 warmup/EMA/SWA 零命中；`core/models/maskrcnn.py` 用 torchvision 默认 min_size=800。

- [ ] `train.py` 加 linear warmup（500–1000 iter）+ weight EMA（decay 0.999，EMA 权重选 checkpoint）两个 flag，**搭 C-1 retrain 顺风车**（无专属 retrain）；同一 job 产出 raw-best / EMA-best 双 checkpoint 保归因；训练用 byte-identical dataset manifest + seed（commit 8d93473 run-ledger），使唯一 delta = recipe。
- 已删：**SWA**（在 ~10k 噪声 SAM GT chips 上加 epoch 平均有 bulk 过冲反指征——train20 教训；GT 清理后再议）；**multi-scale min_size list**（torchvision eval 取 min_size[-1]，[640,800,960] 会静默把推理分辨率改成 960 破坏全部历史可比性——若未来做，与 anchor audit 合并且 pin 推理 min_size=800）；「EMA 修 chip→grid 分歧」的理由（那是 GT volume bias 的系统性 Goodhart，非选择方差）。
- 🔶 预期 0~+1pp（选择降噪，非 AP 式增益）。

#### C-3 = C5 · Ignore 监督（拆两半，独立 gate）

**(b) 边界 ignore-band 先行** 〔~3–4 天 + 1 retrain〕——supervision-layering **行动 5**，Phase A 尸检明文留下的唯一未消融正交 lever ✅（Phase A 的 ignore 只到 mask loss；`boundary_aware_mask.py` docstring 自证「仍贡献 box+cls」）：

- [ ] 实现 `core/training/boundary_ignore_band.py` 按 action-5 原 spec：area-adaptive 带宽（小 1px / 中 2px / 大 3px）+ R 类 band-ignore-core-supervised，替换固定 `band_iters=2`（`train.py:64-83`）。
- [ ] 单 lever 单 retrain；成功标准 = **bulk / σ_Bw / area_F1（不得 book 任何 polygon-F1 recall 收益）**，vs unified_A on locked clean_gt 双 mode。

**(a) RPN/box-cls ignore 先测前提** 〔Phase 0 测量 ~2 天，gate 过了再 ~1 周 + 1 retrain〕：

- [ ] Phase 0：从当前 production 训练池抽 150–200 chips，production detector 低 conf 扫 + 人/Gemini 审计背景区，量 **unlabeled-real-PV-as-background 率**。**≥5% chips 受影响才做，否则杀 (a)**。审计输出本身即 ignore 语料。
- [ ] 若做：ignore 语料修正——fn_markers **不作 ignore**（已被转正为 `sam_added_true_fn` positive ✅；确认真 PV 一律转正）；Gemini-abstain 源删除（max_tokens 截断 artifact，现 0%，且 prediction-conditioned 看不到 detector 也漏的 panel）；RA 确认 lookalike（天窗/热水器）→ `data/negative_pool/`，**绝不入 ignore**（精度瓶颈项目里移除 HN 违反广度约束）；ignore 只留给无人裁决区域（unreviewed-chip margins），per-chip ignore-area cap。
- [ ] 工程面纠正 ✅：torchvision 0.25 不消费 iscrowd —— 实现在 `RPN.assign_targets_to_anchors` + `RoIHeads.assign_targets_to_proposals` 两个新 patch 点（label=-1，对 ignore 区 intersection-over-foreground 高者豁免），**不走 maskrcnn_loss**；沿用现有 `wrap_select_training_samples` 模式。
- 🔶 (a) 预期 0~+2pp，conditional on 实测缺标率（稀疏标注文献的增益在 30–50% 缺标率才显现，低缺标率下消失）。

#### C-4 = C6 · Copy-paste 稠密配置合成 pilot 〔2–3 天 transform + 1 retrain〕 ← **硬依赖 C-1**

memory 中 sanctioned 的 training-time recall 路线 ✅（`feedback_recall_recovery_constraints`:36-41 与杀掉四条推理时路线同一份备忘录）；grep 证实零 copy-paste/mosaic 代码。

- [ ] **Blocking prerequisite = C-1**（chip exporter/ingest 工具）；**若 C-1 retrain 显示 trusted mask 仍 bulk 过冲 → C6 不跑直接杀**（train20 反证：真实稠密 SAM_supp 监督曾使 Ch2 recall −12pp / bulk +49pp，C6 无独立预期、条件于 mask-volume 根因已修）。
- [ ] 已删：对称负对象 paste 与 lookalike-FP 主张（FP 抑制留给 solar_cls）；负池屋顶仅作可选 canvas。
- [ ] paste 源**只取 volume-trusted 池**（`data/training_pool/positive_trusted_manifest.csv`，3,124 instances ✅；≤600m² halo-audit 类），**source 与 canvas 同 imagery layer**（外观/provider 匹配，不止 GSD）；paste 到负池 chip **副本**上作为新增正样本，原 empty-target HN chips 原比例保留。
- [ ] 实现为 **dataset-level transform**（source-instance bank），不塞 `TrainTransforms.__call__`；pasted instance 带 mask_trusted=true + 独立 provenance tag；透传 mask_pixel_weights/mask_weights/label_sources（参照 `train.py:330-383` 的 bookkeeping）；pilot 阶段 paste 真实感止于 alpha-blend，不迭代。
- [ ] 判分：主指标 = **稠密 multi-array stratum recall**（Ch2 recall@{0.3,0.5}，baseline 0.443）预注册涨幅；总 polygon F1 预期 ±1pp（大概率不可分辨）；Tier-1 area 须在 C-1 baseline 噪声内；显式 kill gate。

---

### Tier D — 长线（硬 gate 不满足不开工）

#### D1 = C14 · PointRend 类边界 head 〔gate 满足后 ~2–3 天 glue + 1 retrain〕

机制成立（matched IoU 0.73–0.74 = 边界质量卡 0.5 线；supervision plan 否决 56×56 dense head 时自己点名 PointRend「更对靶」✅），但有两道硬 gate：

- [ ] **Gate ①（GT）**：C9/A4 产出经人工边界审计确认**非 SAM 风格**（halo/oversize 显著低于 sam_added）的边界 GT，且该 GT 同时用于训练与刷新后的 referee。原因：增益随标注边界质量放大（LVIS +2.1 vs COCO +1.1），在 SAM 风格 GT 上训练 + SAM 风格 referee 上裁判,高保真拟合 SAM 边界的 head 不可能「边界上赢 SAM」（train20 已示范 bulk +49pp）。**C4 类「更多 SAM MRR-fill mask」不解锁本项。**
- [ ] **Gate ②（headroom oracle probe，便宜先跑）**：固定检测集，把 matched 预测几何替换为 GT 几何，重算 σ_Bw/bulk/area_F1 —— **perfect-boundary ceiling 相对当前 SAM 链增益 <0.05 σ_Bw → 杀 C14**（说明 binding constraint 是 recall 不是边界）。
- [ ] 实现：off-the-shelf **mmdet/detectron2 PointRend 侧分支**，导出 `predictions_metric.gpkg` 进现有 finalize/eval harness（不做 torchvision port，不做「distill」fallback）。⚠️ 两镜头冲突记录：history 镜头担忧 mmdet 侧分支丢失 warm-start lineage 与 mask_trusted/boundary_w plumbing —— 缓解：Gate ① 满足时训练 GT 是 clean-boundary T1，本就不需要 boundary-trust 降权；若仍需混入 A2 监督，则该 plumbing 成本须计入再评估。
- [ ] Baseline 纠正 ✅：要打的是 **unified_A per-det+SAM @c=0.925**（production）与 **train20 per-det+SAM @c=0.915**（已知最优：aggF1 0.849 / σ_Bw 0.148 / bulk 1.002），CT 侧对 unified_A per-det + solar_cls（锁定 CLS-only 链无 SAM 可退役）；决策表必须含 Li GT 独立 cross-check + dense-roof failure set（G0817/G0925 类，GT 经人工修正）——只在 clean_gt 上判会结构性偏袒 SAM 在位者（其 0.992 bulk 本是过冲×收缩抵消 artifact）。
- 预期改述：删 +1–2pp@0.5（PointRend 增益在 AP75+/Boundary-AP，AP50 delta 近零）；defensible win = **去掉 SAM stage**（管线简化 + 摆脱大连片 PV roof-swallow 结构性天花板），不是 headline 涨分。

#### D2 · zerov2 Stage-2 unfreeze —— 由 B2 gate 决定（见 B2）。

---

## 5. 依赖与时间线（约 4–6 周）

```
A1 C12 协议硬化 ─┐
A2 C11 bug+协议 ─┼─ 互相独立, 第 1 周并行          B1 C1 TTA pilot (0.5d) ─→ 过 bar 才集成
A3 C3 dissolve ──┘                                  B2 C13 probe 1b+1a′ (1 GPU-day) ─→ gate → Stage-2
A4 C9 T1 战役 (RA 并行, 跨 2-3 周) ──────────────→ 解锁 D1 Gate①
C-1 C2 负样本池 (第 2 周起) ──→ retrain (搭 C-2 warmup+EMA) ──→ C-4 copy-paste (若 bulk 不过冲)
C-3(b) ignore-band 单独 retrain (C-1 之后排队)     C-3(a) ← 2d 前提测量 gate (≥5%)
D1 C14 ← Gate① (A4 产出) + Gate② (oracle probe, 随时可先跑)
```

| 阶段 | 内容 | 产出 |
|---|---|---|
| 第 1 周 | A1+A2+A3（协议侧）∥ B1 TTA pilot ∥ B2 zerov2 probes；可顺手跑 D1 Gate② oracle probe | 可比的真实 polygon F1@0.5 基线；TTA / zerov2 / 边界 headroom 三个生死判定 |
| 第 2 周起 | A4 标注战役（RA 并行）+ C-1 负样本池 → retrain（搭 C-2） | GT 天花板实测值；HN-breadth checkpoint（Tier-1 双 mode 判分） |
| 之后 | C-3(b) ignore-band retrain → 视 C-1 定 C-4 → 视 A4+Gate② 定 D1 → 视 B2 定 D2 | 每步单 lever、归因干净的 Tier-1 对照 |

---

## 6. 附带发现（review 副产品，独立于候选成败,均应处理）

| 发现 | 位置 | 处理归属 |
|---|---|---|
| legacy 路径 `post_conf` 被 hardcoded CONF_TIERED 静默覆盖 | `detect_and_evaluate.py:794-808` ✅ | A2 |
| scale-jitter **area re-index bug** + augmentation 重复注册 | `train.py:363` 等 ✅ | 任何新增广前先修（C10 校验发现） |
| zerov2 Phase 0 decoder random-init（config 意图 pretrained） | `phase_0_stdout.log:9` ✅ | B2 |
| negative_pool 678 行零 geometry → 导出 0 chip | `data/negative_pool` manifest ✅ | C-1 |
| presence-F1 0.1（legacy hardcode）/ 0.3（direct default）口径分裂 | `detect_and_evaluate.py:2098` vs `evaluate_predictions.py:51` ✅ | A1 |
| legacy benchmark 路径 detections_per_img=100（production=300） | `--parity-mode` ✅ | C8 残值：若 bind 则作可比性 bug 归档 |

---

## 7. 已知局限

1. 两个 reader（`read:gt-quality`、`lit:label-noise-metrics`）因 API 中断死亡——GT 天花板 0.85–0.88 为旁证推断 🔶，**A4 战役即其验证手段**；噪声标签训练文献面（ignore-band 增益量级、soft label 等）未独立扫过，C-3 的预期区间相应保守。
2. 「1.49 fragments/installation」未能在 repo 中找到出处（已从所有依据中删除）；A3 的 dissolve sweep 与 A4 的 paired 测量给出真实值。
3. 所有 🔶 pp 预估都落在本项目 ~6pp 的 merge-mode/工作点噪声带内——判分一律以 Tier-1 全套 + 双 merge-mode 为准，不单看任何 F1。
4. C7/C14 存在两镜头意见冲突（SWA post-hoc 是否保留 / mmdet 侧分支 vs torchvision port），本文档各取更严格一侧并记录冲突；执行时若有新证据可重开。

## 8. 指针

- 全文证据层（14 候选原文 + 28 份校验意见逐字 + 指标对照图）：[`2026-06-10-rcnn-f1-gap-review.html`](2026-06-10-rcnn-f1-gap-review.html)
- 机读数据：[`2026-06-10-rcnn-f1-gap-review.result.json`](2026-06-10-rcnn-f1-gap-review.result.json)
- 关联 plan：[`2026-05-09-training-supervision-layering.md`](2026-05-09-training-supervision-layering.md)（行动 3/5/6/7 仍 pending，本计划 C-3(b)=行动 5、A4=行动 6 的测量化形态）
- 关联否决记录：[`2026-05-08-jhb-phaseA-retrain.md`](2026-05-08-jhb-phaseA-retrain.md)、[`../experiments/exp_train20_val5_hn_negative_result.md`](../experiments/exp_train20_val5_hn_negative_result.md)、[`../experiments/exp_finalizer_pixel_or_vs_per_detection.md`](../experiments/exp_finalizer_pixel_or_vs_per_detection.md)
- zerov2 侧：`/home/gaosh/projects/solar_zerov2/docs/r1_backbone_domain_ablation.md`
