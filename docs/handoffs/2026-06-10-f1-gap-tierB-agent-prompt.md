<!--
Handoff prompt for executing Tier B of docs/plans/2026-06-10-rcnn-f1-gap-review.md.
Written 2026-06-10. All file:line anchors re-verified against working trees at commit 0b6147e
(ZAsolar) and current solar_zerov2 HEAD.
Usage: paste this entire document as the opening prompt of a fresh agent session.
B1 and B2 are mutually independent — can also split into two sessions.
-->

# Agent Prompt: F1-Gap 计划 Tier B 执行（廉价证伪 probe,≤1 GPU-day）

## 你的任务

执行 `docs/plans/2026-06-10-rcnn-f1-gap-review.md` 的 **Tier B 两项**:B1(TTA 证伪 pilot,~0.5 天)与 B2(solar_zerov2 probe 1b + 1a′,~1 GPU-day)。

**定性必须先想清楚:这两项的交付物是生死判定(go/no-go verdict),不是模型改进。** 项目纪律来源是 NMS-relax probe——半天测量杀掉一条死路,省下数周工程(`feedback_recall_recovery_constraints`)。kill bar 在本 prompt 中预注册,**见到数据之后不许移动门槛**。两个 probe 任何一个判 kill 都是合格交付。

## 必读上下文(开工前按序读完)

1. `docs/plans/2026-06-10-rcnn-f1-gap-review.md` — 主计划:§4 全局判分规则 7 条、Tier B 两节、§3 否决台账(C8/C10 为何死——你的 probe 设计不许复活它们的逻辑)。
2. `docs/plans/2026-06-10-rcnn-f1-gap-review.html` — C1、C13 各两份对抗校验意见逐字;实现细节有疑问以校验全文为准。
3. `docs/experiments/exp_finalizer_pixel_or_vs_per_detection.md` — B1 复用其 raw-hint audit 方法学(line 99 附近的表:`raw_box_hint_rate` / `raw_mask_hint_rate` / strict no-proposal)。
4. `/home/gaosh/projects/solar_zerov2/docs/r1_backbone_domain_ablation.md` — B2 的 probe 矩阵、FLOOR 定义(line 50/149/199)、cache 工作流;另读同目录 `ablation_plan.md`。
5. `.claude/rules/05-runpod-inference.md` + `08-runpod-large-files.md` — 仅当用 pod 时;本地 5090 跑则用 tmux(`feedback_use_tmux_local`),不要 nohup-only 或前台阻塞。

## 硬约束

1. **kill bar 预注册不可移动**:B1 = 「<10–15% 漏检 polygon 转化为 IoU≥0.5 可恢复 proposal → 整条 TTA 放弃」;B2 = 「FLOOR = F1≥0.817 **且** σ_Bw≤0.248(JHB CBD25 clean_gt)」。判定写进报告时引用本 prompt 的预注册原文。
2. **本 handoff 只做 probe,不做集成**:B1 通过 bar 也不开工 1–2 周集成(那是另一个 handoff 的事);B2 任何结果都**不触发 Stage-2 unfreeze**(D2 是独立决策,且有 1a′ 前置条件,见 B2 节)。
3. clean_gt 评估锁不动;所有打分走 locked clean_gt + Tier-1 + 双 merge-mode(全局规则 1/2/3)。
4. **B2 的代码改动写进 `/home/gaosh/projects/solar_zerov2/`**(sibling 子仓,共享 venv,从其根 `source scripts/activate_env.sh`),不进主仓;B1 的 pilot 脚本与结果文档进主仓。
5. 锚点先 grep 再动手(下文 file:line 为 2026-06-10 快照);大二进制不进 git。
6. 工具辨认:`scripts/analysis/` 下同时存在 `poly_conf_sweep.py` 与 `polygon_conf_sweep.py` —— 先确认哪个是当前 harness 引用的活脚本再用,不要想当然。

---

## B1 = C1 · TTA 证伪 pilot 〔~0.5 天,本地 GPU〕

### 已核实事实(2026-06-10 快照)

- TTA 是唯一 sanctioned 的推理时 recall lever(`feedback_recall_recovery_constraints`,与杀掉降阈值+cls / rescan / 跨 imagery / NMS-relax 四条是同一份备忘录);全仓库 grep 证实零 TTA 代码。
- **训练增广包络**(`train.py`):hflip/vflip(:254-276)、rot90(:278-284)、scale jitter **0.8–1.2×**(:311-313)。⇒ 常规 TTA 视图(flip/rot90/0.9×/1.25×)几乎全在已训练等变包络内,输出相关、新 proposal 近零——**这些视图禁止入 pilot**。
- **机制定位**:raw_box_hint_rate 已 97.3–100%(G0816/G0817/G0925,score 0.05,见 exp_finalizer 文档表)⇒ headroom **不在 proposal 存在性**(~3%),在 **proposal IoU 质量 / mask undersizing**。pilot 量的是后者。
- 工具链:`detect_direct.py` + `finalize.py` 均在主仓根目录。

### 任务

- [ ] 选 grid:G0925、G0817 + 2–3 个 Ch2-recall 最低的 clean_gt grid(从 Ch2 结果 CSV 排序取),+PTA0292(先查 tiles 是否在本地,不在就跳过并在报告注明)。
- [ ] 预重采样 tiles 到 **1.5× 与 2.0×**(唯二 out-of-envelope 放大视图;400px 滑窗在放大后图上的重叠处理写清楚——同一物理位置必须仍被完整窗口覆盖)。
- [ ] `detect_direct.py` 原样对每个视图各跑一遍(原始 1.0× 已有结果可复用,口径须同;不同则重跑)。**不做 merge/inverse-transform/重校准级联——pilot 只收集 raw proposals。**
- [ ] 测量(对每个漏检 clean_gt polygon,即 1.0× 基线 @production 工作点未匹配者):
  - (i) 在任意放大视图中获得 conf≥0.3 raw proposal 的比例;
  - (ii) **per-missed-polygon best-proposal IoU**(mask hint @0.3 与 @0.5 两档,坐标变换回原始 CRS 后计算;方法学复用 exp_finalizer raw-hint audit,不量 existence)。
- [ ] 判定:**<10–15% 漏检 polygon 转化为 IoU≥0.5 可恢复 proposal → 整条 TTA 放弃**,写 kill memo 归档 `docs/experiments/`。
- [ ] 若通过 bar:报告只写「pilot 通过,集成另立 handoff」+ 实测转化率/最佳 IoU 分布;**预期声明上限 +0.5–1.5pp in-domain @re-swept 工作点**,不许写「跨域更大」(跨域无 FP 吸收层,该声明已被校验删除)。

### 禁止

- in-envelope 视图(flip/rot90/0.9–1.25×)——浪费 GPU 且稀释结论。
- 任何集成工程(merge 策略、NMS 跨视图、poly_conf redo、solar_cls 重校准)。
- 把 (i) 的存在性比例当成主指标报——主指标是 (ii) 的 IoU≥0.5 可恢复率。

---

## B2 = C13 · solar_zerov2 probe 1b + 1a′ 〔~1 GPU-day,本地 5090 或 <$20 pod〕

### 已核实事实

- **Phase 0 decoder 是 random-init**:`solar_zerov2/checkpoints/phase_0/phase_0_stdout.log:9` 原文 `[decoder] instantiating HF Mask2Former (random init for pixel + transformer decoders)`——而 config 意图是 pretrained-M2F。这是 Phase 0 的**第六个 confound**,1a′ 即为把它单独定量。
- Phase 0 用的是 `configs/models/m2f_dinov3_sat_l.yaml`(SAT backbone, FROZEN, cached features `/dev/shm/chip512_features`);**web 权重 config `m2f_dinov3_l.yaml` 已存在**——probe 1b 零新建模代码。
- **feature cache 是 backbone-specific**:1b 换 web-DINOv3-L 必须重提特征 cache(~6h 估计含这一步);1a′ 复用 SAT cache(~3h)。
- `infer.py:249` 输出路径 `<grid>/per_detection/predictions_metric.gpkg` —— **只有 per_detection,没有 pixel-or 路径**。
- FLOOR(`docs/r1_backbone_domain_ablation.md:50,149,199`):V3-C raw per-det = F1 0.817 / σ_Bw 0.248;**any frozen probe clears FLOOR(双条件同时满足)→ bet revived**。
- 文献三重预测 frozen-SAT 失败 / 指向 web+unfreeze:PANGAEA(frozen GFM≈U-Net)、GEO-Bench-2(<10m GSD 下 web 预训练 > SAT)、DINO Soars(unfreeze 值至 +11.9 mIoU)。Rejected #11 只覆盖了 frozen-SAT 这一个最差 cell。

### 任务(按序,顺序有依赖)

- [ ] **Step 0(免费,烧 GPU 前)**:复核 decoder random-init 对 Phase 0 anchor 有效性的影响——Phase 0 的 0.764 是「frozen SAT + random decoder」的成绩,所有 probe 对比基线要带此注记;在 zerov2 的 r1 文档加 correction 注记。
- [ ] **Step 1(kill 判定前置条件)**:给 `infer.py` 补 **pixel-or merge 路径**(instance mask rasterize-OR 后矢量化,镜像主仓 `finalize.py` 语义)。**没有这条路径之前不许下任何 fail 判定**——主仓教训:同 raw 换 merge mode 可差 6pp F1(`feedback_merge_mode_first_class`)。revive 分支可 per-det vs per-det(与 0.817 FLOOR 同口径)。
- [ ] **Step 2 = probe 1b 先行**:frozen **web**-DINOv3-L(`m2f_dinov3_l.yaml`),native GSD,重提特征 cache 后训练(流程照抄 Phase 0,~6h 含 cache)。
- [ ] **Step 3 = probe 1a′**:**SAT backbone + pretrained Mask2Former decoder 权重**(修 decoder 加载 bug 本身就是交付物),复用 SAT cache,~3h。把 decoder-init 效应单独定量。
- [ ] **Step 4 打分**:两 probe 均在 JHB CBD25 clean_gt 上走 `area_aggregate_eval.py` + poly_conf sweep(活脚本先辨认,见硬约束 6),**双 merge-mode**;与 FLOOR 和 Phase 0(带 random-decoder 注记)三方对照。
- [ ] probe 1c(SAT@13.4cm)**严格条件于** 1b 不结论或显示 pretrain 域非主因;实价是「较大窗口 re-cache」不是小 flag——默认不跑,要跑先在报告里论证。

### 判定表(预注册)

| 结果 | 判定 |
|---|---|
| 任一 frozen probe 过 FLOOR(F1≥0.817 且 σ_Bw≤0.248) | **bet revived**(frozen 形态)——报告写明哪个 cell 救活的 |
| 双双 <0.78 **且** 1a′ 显示 decoder-init 非主因 | frozen 路线判 kill;Stage-2 unfreeze 成为 make-or-break——**但 Stage-2 是 D2 的事,本 handoff 到此为止** |
| 双双 <0.78 但 1a′ 显示 decoder-init 是主因 | **不许下 kill 判定**——修复 decoder 后的 cell 才是有效证据 |

- 预期改述(校验后口径):交付物 = **gate 决策本身**;即便 revive,in-domain 上行低个位数封顶(A2 GT capped);战略价值在 **cross-domain σ_Bw**(Rein/PEFT 路线)——Mask R-CNN 系所有候选都不覆盖这一项。

### 禁止

- 跳过 Step 1 直接下 fail 判定(per-det-only 打分判 kill 无效)。
- 「双双 <0.78」就动 Stage-2 unfreeze 代码——1a′ 没排除 decoder 因素前禁止,且 unfreeze 实施(5 处代码:backbone param group / scoped no_grad / grad ckpt / lr_backbone 1e-5 + layer-wise decay + EMA)属于 D2,不在本 handoff。
- 引用 RSPrompter +4 mask AP 类比(已被校验删除)。

---

## 报告要求(收尾时一并交付)

1. **Verdict-first**:第一行就是两个判定(B1 kill/pass + 转化率实测;B2 revive/kill/blocked-on-decoder + 两 probe 双 mode 成绩 vs FLOOR 表)。
2. B1 结果文档进主仓 `docs/experiments/`(kill 或 pass 都写,负结果归档是项目惯例);B2 结果回写 `solar_zerov2/docs/r1_backbone_domain_ablation.md`(含 Phase 0 random-decoder correction 注记)。
3. 回写主计划:`docs/plans/2026-06-10-rcnn-f1-gap-review.md` §4 Tier B checkbox + §5 时间线表(B1/B2 判定落点);若 B1 kill,同步在 §3 否决台账追加条目(格式照 C4/C8/C10:前提事实 + 残值)。
4. 主仓与 zerov2 各自独立 commit;commit co-author 用 canonical 形式 `Claude <noreply@anthropic.com>`。
5. GPU 长任务一律 tmux(本地)或 nohup + 日志(pod,规则 05);pod 大文件遵守规则 08(S3 不 SCP)。
