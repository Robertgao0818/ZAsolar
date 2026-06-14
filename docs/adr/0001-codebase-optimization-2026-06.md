# ADR-0001: 代码库优化(architecture deepening)追踪

- **Status**: In progress — 主线 11 步完成 6,side items 1/4,1 项语义裁决 PENDING
- **Created**: 2026-06-12 · living tracker,完成项打勾并附验证证据
- **Source review**: [`docs/plans/2026-06-12-architecture-review.html`](../plans/2026-06-12-architecture-review.html)(多 agent review,15 个对抗校验存活候选 + 11 步方案;1 项候选被否决)
- **落地记录(证据明细)**: [`docs/handoffs/2026-06-12-architecture-opt-landing.md`](../handoffs/2026-06-12-architecture-opt-landing.md)

## Context

review 定位的结构问题:F1 主裁判内核埋在 2257-LOC 五合一 module 且零测试;Tier-1 公式 / chip 裁切 / merge-HN / label_source 派生各有 3-5 份副本;声明式 builder monkeypatch 自己标 DEPRECATED 的脚本;merge-mode first-class lever 只覆盖一条推理链;3 处 raw-YAML back-channel 绕过 registry;2 个活 defect(mosaic 静默丢 HN、sync_from_runpod 丢 L-namespace)。

**全程铁律**:评估协议锁定(`docs/evaluation_protocol.md`),每步必须 byte-equivalence gate;绝不中途改数值口径。

## Decisions(语义裁决记录)

| # | 决策 | 状态 | 记录 |
|---|---|---|---|
| D1 | **merge-mode 住 CLI,不住 canonical postproc JSON**:`v4_canonical.json` 删除 `merge_mode`,direct 链调用方必须显式 `--merge-mode`;CLI-vs-JSON 冲突 raise;`v4_poly_diag.json` 自带字段是 diagnostic 设计例外 | ✅ Accepted(2026-06-12,用户裁决选 per-detection) | eval_protocol §3 |
| D2 | **build_training_pool 的 label_source 派生分叉是 by-design**(训练 loader fail-fast raise vs 池构建器 fail-closed,46/112 值矩阵格不同),不统一 | ✅ Accepted(2026-06-12) | `test_build_training_pool_fork_is_divergent_by_design` 锁定 |
| D3 | **提取一律 move + import shim,绝不留第二份实现** | ✅ Accepted(执行惯例) | steps 1/3/8/9 均如此执行 |
| D4 | legacy 链 config.json 的 `merge_mode` 字段为 provenance-only,经 `_CACHE_IGNORE_KEYS` 排除缓存比对(保历史结果不被强制重跑) | ✅ Accepted(2026-06-12) | step 6 |
| D5 | postproc tier 语义:**fall-through vs first-match 统一口径或 op_mode 双语义并存** | ⏳ **PENDING** — block 步骤 11 的 C2 | 见 review C2 |
| D6 | chunked tile glob 收紧为 `{grid}_*_*_geo.tif`(rule 06 规范布局,替代 hn_ops 旧 `*.tif`) | ✅ Accepted(2026-06-12,step 9 有意收紧) | handoff "Notable findings" |

## 主线 11 步

- [x] **1. iou_matching → `core/eval_matching.py`**(候选 #1)— shim 留在 detect_and_evaluate.py;synthetic 快照 byte-identical(14 场景);18 新测试;import 不再触发 matplotlib/set_grid_context 副作用 *(2026-06-12)*
- [ ] **2. 5 个 analysis caller 改 import `core.eval_matching`** — 依赖步骤 1 ✅ 已解锁。实际名单:`compute_ch2_recall` / `tta_probe_baseline` / `validate_checkpoint` / `installation_sym_eval` / `repostprocess`(review 写的 `evaluate_predictions.py` 不存在)
- [x] **3. Tier-1 统计内核 → `core/area_metrics.py`**(候选 #6)— 6 场景快照 byte-identical + 真实 run(jhb_phaseA_vexcel 3 grid)端到端一致;18 新测试;area_aggregate_eval 保留 re-export 兼容 *(2026-06-12)*
- [ ] **4. 两份 Tier-1 公式副本路由到 `core.area_metrics`** — `per_grid_dispersion_audit` 重构为函数;`poly_conf_sweep._agg` 委托 summarize(bootstrap 设可选防拖慢 sweep)。依赖步骤 3 ✅ 已解锁
- [x] **5. `ModelRunConfig.deprecated` + 封 3 处 raw-YAML back-channel**(候选 #5)— 含 `solar_cls` 两脚本;前后 deprecated 标记映射 byte-identical(29 runs);11 registry 测试 *(2026-06-12)*
- [x] **6. merge-mode provenance 最小修复**(候选 #9)— legacy 写 `per_detection_geoai`(cache-safe,D4);finalize 冲突 raise(双向测试);overnight.sh 注释修正;**附加**:D1 裁决落地,顺带解掉 validate_checkpoint per-det 腿的第二个活冲突 *(2026-06-12)*
- [ ] **7. `core/run_provenance.py` 统一 config.json 两方言**(候选 #3)— read 先兼容旧 nested/新 flat 再切写端;`script_sha256` 移出 cache-key(修 spurious 缓存失效);退役 `infer_*` 字符串启发式。依赖步骤 5、6 ✅ 已解锁
- [x] **8. positive-source loader → `core/training/positive_sources.py`**(候选 #4)— dataset_builder monkeypatch 已删;`review_root` 显式参数;build_unified_reviewall 720→534 行薄壳;v2 dry-run manifest byte-identical(指纹 `d01e1bf1`,68 grid / 5903 标注);分叉处置见 D2 *(2026-06-12)*
- [x] **9. `core/chip_extraction.py`**(候选 #7)— 收编 4 份 crop + 4 个 find_tile;**修 mosaic 静默丢 HN bug**(回归测试固化);export_v4_hn 补 region=;真实 CT G1632 chip md5 一致;11 新测试 *(2026-06-12)*
- [ ] **10. 4 份 merge-HN 副本 → `hn_ops.merge_hn_into_coco`**(候选 #8)— 注意 hn_ops 实际在 `pipeline/hn_ops.py`。依赖步骤 9 ✅ 已解锁;迁移前确认 docs/workflows.md 引用的老 CLI 无活跃 reproduce 依赖
- [ ] **11. 收尾三件** — region alias 窄手术(#12,保短别名输出合约);`resolve_gt_spec` 去私名公开(#13);postproc 过滤链统一(#2,**被 D5 裁决 block**)。依赖步骤 1、3 ✅ 已解锁

## Side items(未排期,独立随手可做)

- [ ] **#11(部分)`sync_from_runpod.sh` 修复** — ⚠️ **活 defect,优先级最高**:`grep '^G[0-9]+'` 把 CT census 在产的 L-namespace grid(L1842–L1954)静默丢弃 + `.env` CWD 相对加载;与 Dropbox sync 三克隆合并可拆开做,L-namespace 修复应尽早单独落
- [x] **#10 `building_filter.py` 归档** + 删 detect_and_evaluate 双重死代码块(依赖文件不存在的软门 + 已注释禁用的使用点)— 368 行副本归档至 `~/projects/_archive/building_filter_legacy_2026-06-12/`(附 README,git 历史 `9f55b1c` 亦可恢复);删除 `BUILDINGS_GPKG` 常量、`.exists()` 软门预加载块、tile 循环内 `建筑掩膜已禁用` 注释残留;全仓 grep 零残留引用(`building_filter|BUILDINGS_GPKG|buildings.gpkg`),268 测试通过;`docs/architecture.md` 模块表 + ROADMAP 同步 *(2026-06-12)*
- [ ] **#14 `TrainRunConfig` dataclass**(train.py ~50 flag 收口 + 互斥校验 WARN→raise)— 下次训练改动时顺带做
- [ ] **#15 `CANONICAL_DETECT_ARGS` shell 片段**(7 个字面量参数 ×3 脚本)— 下次新建 batch 脚本时做

## Rejected(对抗校验否决,勿重提)

- `annotation_loader.resolve_gt_path` 塌缝 — friction 不足,否决于 review 阶段

## Consequences

- 新 core 模块 4 个:`eval_matching` / `area_metrics` / `chip_extraction` / `training/positive_sources`,全部带单测(评估内核首次有 CPU-only test surface);`docs/architecture.md` 已同步
- 全仓测试 252 passed(2026-06-12 组合树);主仓净 -555 行
- 残余验证缺口:step 8 的 JHB 集成 dry-run 需 pod 上 `vexcel_2024` tiles(CT 路径已全量 gate)
- 提交纪律:各步文件互斥,可按步拆 commit;`docs/architecture.md` 模块表变更须与对应文件移动同 commit(rule 03-doc-sync)
