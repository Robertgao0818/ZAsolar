# JHB 全量 FP-cut sweep — 暂停/续跑工作记录

**日期：** 2026-05-31（晚间暂停，因 Gemini 账号池将耗尽）
**状态：** ⏸️ 暂停，**完全可 resume**。9 个 batch 中 3 个已完成，从 batch_03 续跑即可。
**相关：** `docs/handoffs/2026-05-31-jhb-full-fp-cut-agent-prompt.md`（任务说明）、
`docs/handoffs/2026-05-31-jhb-two-stage-fp-review-production.md`（pipeline 验证）、
memory `project_gemini_scorer_concurrency`（routing-salt / 并发）。

---

## 一句话现状

用 Gemini 两阶段 FP-review 清洗 JHB Vexcel 全量 362 grid 推理库存。**Pilot（24 grid）+ 前 3 个
sweep batch（120 grid）已完成并过闸门**；因为 3 个 Gemini 账号撞限额/被暂停到只剩 1 个，主动暂停，
晚上账号恢复后一行命令续跑（自动跳过已完成、从 batch_03 续）。

## 进度表

| 单元 | grid | HI cand | LO cand | 丢弃 | 保留 | 闸门 | 状态 |
|---|---|---|---|---|---|---|---|
| **pilot** (JNB0002–0025) | 24 | 2,568 | 173 | 361 | 2,380 | PASS | ✅ APPLIED |
| **batch_00** (JNB0026–0065) | 40 | 9,034 | 607 | 925 | 8,716 | PASS | ✅ APPLIED |
| **batch_01** (JNB0066–…) | 40 | 9,976 | 559 | 864 | 9,671 | PASS | ✅ APPLIED |
| **batch_02** | 40 | 6,821 | 475 | 834 | 6,462 | PASS | ✅ APPLIED |
| batch_03 | 40 | 3,371 | 368 | — | — | — | ⏸️ stage1 223/3371（resume 续） |
| batch_04 | 40 | 2,019 | 217 | — | — | — | ⬜ 未开始 |
| batch_05 | 40 | 1,877 | 201 | — | — | — | ⬜ |
| batch_06 | 40 | 2,936 | 226 | — | — | — | ⬜ |
| batch_07 | 40 | 3,385 | 259 | — | — | — | ⬜ |
| batch_08 | 18 | 2,226 | 167 | — | — | — | ⬜ |

**累计已完成（pilot+00+01+02，144 grid）：丢弃 2,984 / 保留 27,229。** 闸门全 PASS、abstain 0、0 review。
**剩余：** batch_03–08（218 grid，~15.8k HI + ~1.4k LO + stage2 子集 ≈ **~18–19k Gemini 调用**）。
粗估最终库存：47,465 → **~43,000**（按已完成 batch ~9–10% HI drop / ~35% LO drop 外推）。

## 账号 / 并发情况（暂停原因）

- Gemini 池 = sub2api 多账号，每账号 ~10 slot。原 3 账号 = 30 slot。
- sweep 期间陆续撞 5h 滑动窗口限额；用户在网关暂停了撞限额的账号。暂停前只剩 **1 个账号**没撞，
  继续跑只会把最后一个也烧光，故暂停等窗口恢复。
- **配额自动处理已验证有效**：v3 driver 在 abstain 带 quota 特征时自动 `sleep 1h → --resume`，
  全程自动绕过了 4 次限额（16:50 / 18:08 / 19:15 / 21:25 UTC），无需人工。
- workers 历史：30（3 账号）→ 暂停 1 账号后改 20（2 账号）。**续跑前按当时可用账号数 ×10 设 WORKERS**。

## 🌙 晚上回来怎么续跑（账号恢复后）

```bash
cd /home/gaosh/projects/ZAsolar
ROOT=data/analysis/gemini_review_calib/prod_jhb

# 1) 确认没有残留进程（应为空）
ps -eo pid,cmd | grep -E "[g]emini_fp_review|[s]weep_driver_v3" || echo clean

# 2) 按当前可用账号数设并发：WORKERS = 账号数 × 10（3 账号→30，2 账号→20）
#    编辑 $ROOT/sweep_driver_v3.sh 顶部的 WORKERS=  （当前=20；QPS=8 不动）
grep -n '^WORKERS=' "$ROOT/sweep_driver_v3.sh"

# 3) 重启 —— 自动 resume：跳过 pilot/batch_00/01/02（已 APPLIED），
#    batch_03 用 --resume 续 stage1（保留已跑的 223 条），一路跑到 batch_08
nohup bash "$ROOT/sweep_driver_v3.sh" "$ROOT/sweep_grids_remaining.txt" >> "$ROOT/sweep_v3.log" 2>&1 &

# 4) 看进度
tail -f "$ROOT/sweep_v3.log"        # 关注 applied / [quota] 行
cat "$ROOT/sweep_rollup.tsv"        # 每个 batch 完成写一行
```

v3 的安全特性（已验证）：每步 skip 已完成且干净的输出；stage1/LO/stage2 撞限额自动 sleep 1h + `--resume`
（只补没跑成的 candidate，不重花配额，最多等 8h）；fail-closed 闸门必须 exit 0 才 apply。

## ✅ 全部 batch 跑完后要做的（task #10：合并 + backdating 交接）

1. 合并所有 `*_filtered.gpkg`（pilot + batch_00..08）成一份干净 JHB 库存单层 gpkg。
   各 batch 输出在 `$ROOT/{pilot_jnb0002_0025,batch_0N}/filtered/JNB*_filtered.gpkg`（EPSG:32735）。
2. 放到 solar_backdating 锚点清单构建器期望的位置，写交接文档说明这是 FP-cut 交付物 + provenance
   （run dir、conf 分带、各 batch drop 数、闸门状态）。
3. 出最终 rollup：清洗前后库存对比（47,465 → 47,465 − Σdrop）、review_queue 汇总、soak 汇总。

## 关键文件 / 路径

- 推理源（不要动）：`results/johannesburg/unified_reviewall_A_perdet_sam_maskbox_vexcel_2024_full382_sam_maskbox/JNB*/predictions_metric_merge01_c0925.gpkg`（362 grid）
- tiles（已全部本地化 23G）：`~/zasolar_data/tiles/johannesburg/vexcel_2024/JNB*`
- sweep 工作区（gitignored）：`data/analysis/gemini_review_calib/prod_jhb/`
  - `sweep_driver_v3.sh`（quota-autonomous 驱动）、`sweep_grids_remaining.txt`（338 grid 列表）
  - `sweep_rollup.tsv`、`sweep_v3.log`
  - `render_one_chipset.sh`、`jsonl_health.py`（driver 依赖的 helper，已从 /tmp 复制到此工作区持久保存，
    driver 已改为引用 `$ROOT/` 路径，不再依赖 /tmp）
- 已 commit：`apply_two_stage_decisions.py`（--stage1-as-drops, ada0a0d）、其 7 个测试、两份 handoff 的 routing-salt 修正（23d4009）

## ⚠️ 未提交的代码改动

- `scripts/analysis/gemini_fp_review_multiscale.py` 加了 **`--resume`**（保留 usable、只重跑 abstain/缺失，
  无重复行；已单测验证）。**尚未提交**，在工作树里。续跑依赖它——不要 `git checkout` 掉。
  全部跑完后连同最终结果一起提交。
- driver / helper 脚本都在 gitignored 的 `data/analysis/.../prod_jhb/`，不进 git（运行产物区）。

## 设计备忘（避免踩坑）

- **routing-salt 必须 `--routing-salt-mode target`**（flash 在 auto 模式不加 nonce → 全压一个账号）。
  driver 已固定 target。详见 memory `project_gemini_scorer_concurrency`。
- batch_02 的 stage2 曾因限额留下 2 个 review 残留；处理方式 = 删该 batch 的 `two_stage_hi*.jsonl`
  让 stage2 干净重跑（stage1 保留）。重启后已自动跑完。若再遇类似，同样处理。
- 渲染器单线程，已用 `xargs -P 12` 并行渲染（36 chip set ~1h 内跑完，已全部完成，续跑不再渲染）。
