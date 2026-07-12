# 仓库结构 + CLAUDE.md 体系优化方案(2026-07-10)

来源:8-agent 审计 workflow(5 仓库探索 + 1 web 最佳实践调研 + 2 批判 agent,
sonnet)。本文档记录审计结论、已执行的 CLAUDE.md 体系改造、以及**未执行**的
结构优化建议(按风险/收益排序)。

## 审计核心结论

1. **常驻指令超预算 2 倍以上**:CLAUDE.md(100 行)+ 无 frontmatter 的
   rules 04–08(357 行)= 每会话固定 457 行,远超 Anthropic 官方 <200 行的
   遵守度上限。而 rules 01–03 早已采用 `paths:` frontmatter 按需加载——
   同一机制未应用到 04–08。
2. **RunPod 内容三处重复**:rules 04/05/08 与 `runpod-ops` skill 的
   reference/*.md 大面积重叠,且 skill 侧更新(含 Blackwell/OOM 指引),
   rules 侧反而是旧拷贝。04/08 还是全中文,违反 `agent_docs_english_do_style`
   约定。
3. **CLAUDE.md 本体基本健康**:全部路径可解析、与 architecture.md 无矛盾。
   三处软错误:ct_census_* 简写与真实文件名 3/6 不符;model_registry.yaml
   并无 "current best" 字段(声明过强);"in-repo tiles/results 无真实数据"
   与磁盘现状矛盾(tiles/ 4.1G + results/ 12G,见 `repo_cleanup_2026-07-10`
   memory)。
4. **三大 agent footgun**(结构性,本次不修,须知晓):
   - `detect_and_evaluate.py:229-260` 手工维护一份与 `core/postproc.py`
     平行的 postproc 逻辑("语义对齐"靠人肉);修 filter bug 改一处漏一处。
   - 跨城重叠 grid ID:凡从 ID 模式推 region 的代码都会静默写错城市。
   - tiles/results 12G+ 真实数据在仓内,唯一防线是 Claude-Code 专属
     PreToolUse hook;`.githooks/` 存在但 `core.hooksPath` 未接线,
     裸 shell `git add -A` 无守卫。

## A. CLAUDE.md 体系改造(本次已执行)

| 动作 | 文件 | 说明 |
|---|---|---|
| 重写 | `CLAUDE.md` | 命令区提前;修 ct_census_* 真实文件名;registry 声明改准确;补 tiles/results 已知缺口注记;新增 Routing 区(RunPod→skill、SSH 粘贴→.env 更新契约) |
| 移入 `.claude/retired-rules/` | `.claude/rules/04-runpod-ssh.md` | 内容已在 skill `reference/ssh-and-env.md`(逐条核对);SSH 粘贴自动更新契约上移至 CLAUDE.md Routing 区(保持每会话可见) |
| 移入 `.claude/retired-rules/` | `.claude/rules/08-runpod-large-files.md` | 内容已在 skill `reference/file-transfer.md`(S3/runpodctl/scp 禁令/pod 上现切训练集,逐条核对);代码注释引用已改指 skill(`scripts/training/build_training_pool.py`、`pipeline/hn_ops.py`) |
| 加 `paths:` | `.claude/rules/05-runpod-inference.md` | 内容不动(部分细节比 skill 新);仅在触碰 detect_direct/finalize/ct_census_*/runpod_* 时加载 |
| 加 `paths:` | `.claude/rules/07-annotation-semantics.md` | 内容不动;仅在触碰标注/导出/训练相关文件时加载 |
| 精简重写 | `.claude/rules/06-multi-city.md` | 121→~55 行英文 policy-only;ADR-0002 叙事、日期事件史交还 `docs/adr/0002` + evolution log;保持常驻(跨城正确性是全局横切) |

**效果**:常驻指令 457 → ~160 行(CLAUDE.md ~105 + rule 06 ~55);RunPod/
标注语义在相关会话仍保证加载(paths 触发 + skill 触发双通道)。

**未做**(有意):settings.json 中文 hook 提示语未动(对话语言=中文,合规);
12 个无 frontmatter 的 skills 未统一格式(另列快赢 B7)。

**重要注记**:`CLAUDE.md` 与 `.claude/` 整体在 `.gitignore`(48/50 行,
"local-only, not part of public repo" 是既有治理决定)——本次改动**没有 git
快照**,`.claude/retired-rules/` 是唯一回滚路径。建议考虑给 CLAUDE.md/.claude
建仓外备份通道(类似 annotations 的 Dropbox hook)。

## B. 结构快赢(建议,一个 session 内可完成,未执行)

1. **归档漏网的 C3A-phase0 一次性脚本**:`scripts/analysis/build_c3a_phase0_audit.py`、
   `compute_c3a_phase0_gate.py`、`scripts/training/run_c3a_phase0_scan.py`、
   `sample_c3a_phase0_chips.py`、`scripts/analysis/_li_kml_calib_overlap_check.py`
   ——与 9353be8 归档口径相同的"已结束战役、无活引用"模式,该批被漏掉。
   `git mv` 进 `scripts/archive/c3a_phase0/` + 补 archive README 条目。
2. **修 on-record 的 rule-06 违规**:`scripts/annotations/sam_fn_review.py:55`
   本地重实现 `get_metric_crs()`(自带注释承认应调 core)。单点替换。
3. **消歧近重名脚本**:`poly_conf_sweep.py` vs `polygon_conf_sweep.py`、
   `posthoc_merge_and_spatial_eval.py` vs `posthoc_merge_areaf1.py`、
   `gemini_fp_review_multiscale.py`(archive README 已认定 two_stage 为生产版,
   multiscale 未标记)。头注声明 canonical 或归档被取代者。
4. **`scripts/training/` 加一页 README**:4 个版本化 HN exporter
   (`export_v4_hn` / `export_v4_1_hn`(现行) / `build_v4_3_hn` / `export_targeted_hn`)
   共存无索引;照 `scripts/archive/README.md` 的 campaign-map 模式补一份。
   顺带处理 `scripts/training/jhb_phaseA/`(已失败战役的残留测试,split state)。
5. **接线 git hooks**:跑 `scripts/install_git_hooks.sh`(pre-push =
   validate_registry + pytest)。当前 `core.hooksPath` 为空,PreToolUse hook
   只护 Claude-Code 会话。
6. **根目录未跟踪杂物清扫**:`ct_census_report.html`(2.1M)、`.geobench2_*`
   (508K)、`start-*.{sh,bat}`×8、`CASE_STUDY.*` → `docs/handoffs/` 或仓外。
   全部未跟踪,零风险,降 `ls` 噪声与误 `git add -A` 面。
7. **12 个 skills 补 YAML frontmatter**(name/description),对齐
   `runpod-ops`/`improve-codebase-architecture` 的规范格式,提高触发可靠性。
8. **归档 `commands/supervise-jhb382.md`**:JHB 382 战役已结
   (`jhb_grid_scheme_jnb_canonical`),与 9353be8 同口径。

## C. 中期改造(值得单独排期)

1. **postproc 双实现收敛(footgun #1)**:方向 = `detect_and_evaluate.py`
   的 filter/tier 逻辑改为 import `core.postproc`,以 parity 测试
   (`tests/postproc/test_parity_old.py` 模式)锁语义后切换。注意 dae 是
   frozen reference(`evaluate_predictions.py` 直接 import),只换内脏、
   不动签名。做之前先补两链 D5 tier 语义的 parity 用例。
2. **EPSG 硬编码清扫**:archive 外 24 个文件 grep 命中 `EPSG:3273[45]`
   字面量(含 `ct_census_merge.py` 等活跃脚本),违反 rule 06。批量换
   `get_metric_crs()`,每文件跑其对应测试/dry-run。
3. **`core/training/jhb_phaseA_*` 显式弃用**:训练路线已否
   (`jhb_phaseA_failed`),但 `train.py --jhb-phaseA-spec` 仍活接。加
   DeprecationWarning(不删,测试仍引用)。
4. **results/ 12G 迁移治理**:与 `repo_cleanup_2026-07-10` memory 的
   遗留项对齐——代码直读 `BASE_DIR/results`,迁移需先改 `get_results_root`
   默认根再搬数据,单独出 plan。

## D. 明确不做(与批判 agent 结论一致)

- **不拆 `detect_and_evaluate.py`(2119 行)**:frozen reference 契约在身,
  拆分风险 > 尺寸收益。
- **不把 `scripts/analysis/` 一次性脚本抽公共框架**:archive-on-conclusion
  生命周期模式已经工作得很好,预抽象是反模式。
- **不重排 `core/` 子包边界**:lazy-import 模式已工作,56 个 import 站点
  的搬迁是纯 churn。
- **不上 GitHub CI**:solo + agent 工作流下,PreToolUse hook + 接线后的
  pre-push 已覆盖;有协作者再议。
