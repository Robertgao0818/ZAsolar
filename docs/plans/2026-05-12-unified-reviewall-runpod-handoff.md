# Handoff: unified_reviewall_20260511 RunPod 训练

**写于**: 2026-05-12 NZST
**目的**: 另开一个 Claude Code 窗口执行 pod 端训练。这份是自洽 handoff，新窗口不需要回看前一窗口对话。
**承接 plan**: `/home/gaosh/.claude/plans/review-aerial-2023-jhb-opengeoai-v3c-parallel-biscuit.md`（详细设计）
**承接日报**: `docs/progress_log/week_2026-05-06/2026-05-11.md`（实现 + commits + codex review）

---

## 1. 实验目标（一句话）

CT Batch003/004/002b/EarlySAM2 (68 grids) + JHB Vexcel 2024 (20 train + 5 val) 用 **per-instance mask_trusted gate**（reviewed_prediction/sam_added_true_fn → mask BCE 跳过；human_manual_* / sam_added_browser → 全权重）破 V3-C halo cycle，**A=V3-C-warm vs G=opengeoai-fresh 双路 A/B**，主战场 JHB Vexcel val-5。

Pass criteria（JHB Vexcel val-5）：
- Ch3 area_F1 ≥ 0.78（V3-C raw 基线）
- Ch2 recall@IoU≥0.5 ≥ 0.55
- bulk_ratio ∈ [0.85, 1.15]
- A vs G 差距 < 2pp → 选 G（实验语义更干净）

CT independent_26 仅作 A 路退化检测，不参与 A vs G 对比（G 路在 CT 上天然吃亏）。

---

## 2. 当前状态（截至 2026-05-12 evening NZST）

### 2.1 代码状态

`origin/main` 已 push 到 `22a2b9c2`，工作树干净。本周 19+2=21 个新 commit（详见 `docs/progress_log/week_2026-05-06/2026-05-11.md` §"工作完成"）。关键 commits：

| Topic | Commit | 文件 |
|---|---|---|
| boundary-mask per-instance gate + box-loss bucketing | `6f744f31` | core/training/boundary_aware_mask.py |
| export mask_trusted COCO field | `79814722` | export_coco_dataset.py |
| train pipeline (mask_trusted + freeze + diff LR + box-log + eval + early stop) | `b98687e1` | train.py |
| build_unified_reviewall.py builder | `d377257c` | scripts/training/build_unified_reviewall.py |
| training_sets.yaml unified_reviewall_20260511 entry | `eb6facf3` | configs/datasets/training_sets.yaml |
| **fix** balance_chips signature | `2cc1b652` | scripts/training/build_unified_reviewall.py |
| **fix** strict mask_trusted_for + omit on missing label_source | `08ad6ee8` | export_coco_dataset.py + train.py |
| **fix** transform aux resize post-transform | `f785fea5` | core/training/boundary_aware_mask.py + train.py |
| **fix** PhaseA UnboundLocalError + missing hook | `1bdf49a3` | train.py |
| **fix** CT sam_fn_review → untrusted + raise on unknown | `22a2b9c2` | scripts/training/build_unified_reviewall.py |

5 个 codex review fix 都已修 + py_compile 过。

### 2.2 数据状态

- **本地**: `~/zasolar_data/tiles/cape_town/aerial_2025/` (4.3 GB, 121 grids JPEG95) + `~/zasolar_data/tiles/johannesburg/vexcel_2024/` (1.5 GB, 25 grids JPEG95)
- **Dropbox 本地缓存**: `/mnt/c/Users/gaosh/Dropbox/zasolar_data/tiles/{cape_town/aerial_2025,johannesburg/vexcel_2024}/` 已 rsync 完，共 5.8 GB
- **Dropbox 云端**: Windows 客户端 background 在推。**新窗口启动前先 `rclone size dropbox:zasolar_data/tiles` 验证已到 ~5.8 GB**（之前查时 ~5% 上去了）
- **Pod**: **不在**。新窗口需要等用户起 pod 提供 `ssh root@... -p ...` 串
- **Annotations**: 在本地 `data/annotations/` 下（CT `Capetown/`，JHB `Joburg/clean_gt` + `results/johannesburg/v3c_vexcel_2024_ch1_sample/<grid>/review/<grid>_{reviewed,sam_added}.gpkg`）

### 2.3 Builder dry-run 验证（fix 后）

```
trusted=3384  untrusted=5691  ratio=1.68  (≪ 4 OK)
CT 68 grids  JHB 20 train + 5 val
```

Default val JHB grids: `G0772 G0816 G0817 G0888 G0925`（致密+中等+稀疏+outlier 失败模式覆盖）。

---

## 3. Pod 端执行步骤

### 3.1 SSH 接入 + 复核 /workspace

用户给 ssh 串后（如 `ssh root@213.173.x.y -p 12345`），按规则 `04-runpod-ssh.md` 自动更新 `.env`：

```bash
# 用户给 ssh 串 → 提取 host/port → 写 .env
sed -i "s|^RUNPOD_SSH_HOST=.*|RUNPOD_SSH_HOST=root@<NEW_IP>|" .env
sed -i "s|^RUNPOD_SSH_PORT=.*|RUNPOD_SSH_PORT=<NEW_PORT>|" .env
ssh-keygen -R "[<NEW_IP>]:<NEW_PORT>" 2>/dev/null
ssh-keyscan -p <NEW_PORT> -t ed25519 <NEW_IP> >> ~/.ssh/known_hosts 2>/dev/null
```

复核 /workspace（**优先复用**，按规则 `08-runpod-large-files.md`）：

```bash
ssh "$(grep ^RUNPOD_SSH_HOST= .env | cut -d= -f2)" -p "$(grep ^RUNPOD_SSH_PORT= .env | cut -d= -f2)" \
  "echo '--- nvidia-smi ---' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; \
   echo '--- /workspace tree ---' && ls /workspace/tiles/ 2>/dev/null; \
   echo '--- CT aerial_2025 grids ---' && ls /workspace/tiles/cape_town/aerial_2025/ 2>/dev/null | wc -l; \
   echo '--- JHB vexcel_2024 grids ---' && ls /workspace/tiles/johannesburg/vexcel_2024/ 2>/dev/null | wc -l; \
   echo '--- /workspace disk usage ---' && du -sh /workspace/* 2>/dev/null | sort -h"
```

### 3.2 拉底图（按 /workspace 实际状态走）

**情况 A**: /workspace 上 CT aerial_2025 + JHB vexcel_2024 都齐了（121 + 25 grids）→ 跳过此步。

**情况 B**: 缺一部分 → 走 Dropbox（**前提**：本地确认 `rclone size dropbox:zasolar_data/tiles` 已到 ~5.8 GB）：

```bash
# 把本地 rclone.conf 推到 pod
HOST="$(grep ^RUNPOD_SSH_HOST= .env | cut -d= -f2)"
PORT="$(grep ^RUNPOD_SSH_PORT= .env | cut -d= -f2)"
scp -P "$PORT" ~/.config/rclone/rclone.conf "$HOST":~/.config/rclone/
# 装 rclone（如缺）
ssh "$HOST" -p "$PORT" "which rclone || (curl -fsSL https://rclone.org/install.sh | bash)"
# Pull
ssh "$HOST" -p "$PORT" "mkdir -p /workspace/tiles && \
  rclone copy dropbox:zasolar_data/tiles /workspace/tiles \
    --transfers 10 --checkers 20 --progress"
```

Dropbox CDN 下行速度 ~50 MB/s 量级，5.8 GB ≈ **2 min**。

### 3.3 同步代码 + annotations 到 pod

代码走 `git pull`（pod 上已有 repo）；annotations 小（<50 MB）走 scp：

```bash
# 代码（pod 上）
ssh "$HOST" -p "$PORT" "cd /workspace/ZAsolar && git fetch && git checkout main && git pull"
# Annotations（本地 → pod）
HOST="$(grep ^RUNPOD_SSH_HOST= .env | cut -d= -f2)"
PORT="$(grep ^RUNPOD_SSH_PORT= .env | cut -d= -f2)"
tar czf /tmp/anns.tar.gz data/annotations/ results/johannesburg/v3c_vexcel_2024_ch1_sample/*/review/ data/annotations_channel2_clean/
scp -P "$PORT" /tmp/anns.tar.gz "$HOST":/workspace/ZAsolar/
ssh "$HOST" -p "$PORT" "cd /workspace/ZAsolar && tar xzf anns.tar.gz && rm anns.tar.gz"
```

如果 pod 上 ZAsolar repo 还不在，先 git clone（也行 scp 整个目录）。

### 3.4 Pod 上跑 builder（端到端切 chip）

```bash
ssh "$HOST" -p "$PORT" << 'EOF'
cd /workspace/ZAsolar
source scripts/activate_env.sh   # 或 source .venv/bin/activate
export SOLAR_TILES_ROOT=/workspace/tiles  # builder 通过此 env 找底图
python scripts/training/build_unified_reviewall.py \
  --output-dir /workspace/coco/unified_reviewall_20260511 \
  --val-jhb-grids G0772 G0816 G0817 G0888 G0925 \
  --chip-size 400 --overlap 64 --neg-ratio 0.15 --seed 42
EOF
```

预期输出：
- `[ASSERT] train pool: trusted=3384 untrusted=5691 ratio=1.68`
- chip 写到 `/workspace/coco/unified_reviewall_20260511/{train,val}/<chip>.tif`
- `train.json` / `val.json` / `manifest.json` 在 output_dir 根

如果 ratio > 4 → 检查是否 grid 列表错乱（R 类比预期多）。

### 3.5 跑训练（A 路 + G 路并行）

按 plan 双路：A=V3-C warm，G=opengeoai fresh。**5090 双卡的话并行；单卡先 A 再 G**。

**A 路（V3-C warm-start，Stage1=3, total max=15）**：

```bash
ssh "$HOST" -p "$PORT" << 'EOF'
cd /workspace/ZAsolar
source scripts/activate_env.sh
mkdir -p /workspace/tiles_shm
# 拷热数据到 /dev/shm 提速（per rule 05-runpod-inference）
mkdir -p /dev/shm/coco && cp -r /workspace/coco/unified_reviewall_20260511 /dev/shm/coco/
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
  --coco-dir /dev/shm/coco/unified_reviewall_20260511 \
  --pretrained checkpoints/exp003_C_targeted_hn/best_model.pth \
  --freeze-mask-head --stage1-epochs 3 \
  --per-source-mask-weight --per-instance-mask-trusted \
  --diff-lr-backbone-mult 0.1 --diff-lr-rpn-box-mult 1.0 --diff-lr-mask-mult 1.0 \
  --log-per-source-box-reg-loss \
  --eval-schedule "2:10:2,11:15:3" \
  --early-stop-metrics f1_85,ap50 --early-stop-min-delta 0.005 --early-stop-patience 3 \
  --best-ckpt-bulk-range 0.85,1.15 \
  --epochs 15 \
  --output-dir /workspace/checkpoints/exp_unified_reviewall_A \
  > /workspace/logs/A_$(date +%Y%m%d_%H%M).log 2>&1 &
echo "A path PID: $!"
EOF
```

**G 路（opengeoai fresh，Stage1=5, total max=25）**：

```bash
ssh "$HOST" -p "$PORT" << 'EOF'
cd /workspace/ZAsolar
source scripts/activate_env.sh
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py \
  --coco-dir /dev/shm/coco/unified_reviewall_20260511 \
  --pretrained ~/zasolar_data/models/giswqs_solar_panel_detection.pth \
  --freeze-mask-head --stage1-epochs 5 \
  --per-source-mask-weight --per-instance-mask-trusted \
  --diff-lr-backbone-mult 0.1 --diff-lr-rpn-box-mult 1.0 --diff-lr-mask-mult 1.0 \
  --log-per-source-box-reg-loss \
  --eval-schedule "2:10:2,11:25:3" \
  --early-stop-metrics f1_85,ap50 --early-stop-min-delta 0.005 --early-stop-patience 3 \
  --best-ckpt-bulk-range 0.85,1.15 \
  --epochs 25 \
  --output-dir /workspace/checkpoints/exp_unified_reviewall_G \
  > /workspace/logs/G_$(date +%Y%m%d_%H%M).log 2>&1 &
echo "G path PID: $!"
EOF
```

预计 GPU 时长（5090 单卡，36 min/epoch）：
- A 路 max 15 epoch ≈ ~9h（早停下 6-7h）
- G 路 max 25 epoch ≈ ~15h（早停下 10-12h）
- 双卡并行 ~15h；单卡串行 ~24h

---

## 4. Training-time 监控

`training_history.json` 每 epoch 末刷写，关注：

```bash
# 主指标 + early-stop 状态
jq '.epochs[-3:] | .[] | {ep: .epoch, f1_85: .f1_85, ap50: .ap50, lr: .lr, es: .early_stop_state}' \
  /workspace/checkpoints/exp_unified_reviewall_A/training_history.json

# 分 source box reg loss 间隙
jq '.epochs[-3:] | .[] | {ep: .epoch, box_H: .box_reg_loss_H, box_R: .box_reg_loss_R, gap_pct: .box_reg_loss_R_vs_H_gap}' \
  /workspace/checkpoints/exp_unified_reviewall_A/training_history.json
```

预警信号：
- f1_85 / ap50 任一连续 3 个 eval 不刷 best → patience 用完会自动停
- box_reg_loss gap > 25% 持续 3 eval epoch → 写 `next_round_signal.json` `box_trusted_recommended: true`（仅记录，不动 LR）
- bulk_ratio 出 [0.5, 2.0] → 当前 ckpt 不入 best 池子，eval 时手动复核

---

## 5. 训练后验证

### 5.1 Best ckpt 选择（手动 post-hoc）

```bash
# 列出所有 eval 点的 bulk_ratio + area_F1 + Ch2 recall
jq '.eval_points[] | {ep: .epoch, area_f1: .grid_area_F1, ch2_r: .ch2_recall_iou05, bulk: .bulk_ratio}' \
  /workspace/checkpoints/exp_unified_reviewall_A/training_history.json
# 选 bulk_ratio ∈ [0.85, 1.15] 内 area_F1 最高的
```

### 5.2 跑两个 benchmark

```bash
# JHB Vexcel val-5（主战场）
python scripts/analysis/run_benchmark.py \
  --checkpoint /workspace/checkpoints/exp_unified_reviewall_A/best_model.pth \
  --suite jhb_vexcel_val5 \
  --postproc-config configs/postproc/v4_canonical.json

# CT independent_26（仅 A 路退化检测）
python scripts/analysis/run_benchmark.py \
  --checkpoint /workspace/checkpoints/exp_unified_reviewall_A/best_model.pth \
  --suite cape_town_independent_26 \
  --postproc-config configs/postproc/v4_canonical.json

# Tier 1 metric suite
python scripts/analysis/area_aggregate_eval.py \
  --pred-results results/.../exp_unified_reviewall_A_vexcel \
  --gt data/annotations_channel2_clean/ \
  --metrics tier1
```

### 5.3 A vs G 决策

仅看 JHB Vexcel val-5 上：
- A 与 G 差距 < 2pp → **选 G**（opengeoai-fresh，实验语义更干净）
- A 显著领先（≥2pp） → 保留 A，V3-C 非 mask 先验确实有价值
- G 显著领先 → opengeoai 路线胜出，下一轮直接从 G 起步
- A 路若 CT independent_26 退化（F1 < 0.78），无论 JHB 怎么样视为 A 路 fail，G 默认胜出

---

## 6. 容易踩的坑（codex review 已修，但新窗口要警觉）

1. **balance_chips 签名是 `(images, annotations, provenance, seed, neg_ratio)`** — 不是 `(images, neg_ratio, seed=...)`。已修。新代码动 builder 时注意。
2. **mask_pixel_weights / ignore_masks 必须经过 `install_transform_aux_resize(model)`** — torchvision 默认把 400 chip resize 到 800 但不管 custom dict 字段；pre-forward hook stash 是错的。已修走 transform wrap。如果手动加新 spatial 字段进 target dict，必须确保 transform wrap 也 resize 它。
3. **mask_trusted_for 现在 strict raise**：未知 label_source 直接 raise，强制显式映射。新增 label_source 类型时记得加进 `_MASK_TRUSTED` dict（`export_coco_dataset.py:52`）+ `_LABEL_SOURCE_TO_MASK_TRUSTED`（`train.py:66`）+ `_LABEL_SOURCE_TO_BOUNDARY_W`（`train.py:50`）。三处保持一致。
4. **CT `sam_fn_marker` 和 `sam_fn_review` 都是非交互式 batch SAM cut → untrusted**（不是 trusted），已修。任何新的 CT source value 出现都会 raise，需要显式分类。
5. **--per-source-mask-weight + --per-instance-mask-trusted 都开时**：boundary_aware_mask 既走 per-instance gate 又走 per-pixel weight；这是设计的组合，不是冲突。但 ignore band（--ignore-band）本轮 **不开**（独立 ablate，per 2026-05-09 plan 第 5 条）。
6. **OR-stop early stop**：任一 metric patience 用完即停（防 area_F1 涨 + Ch2 recall 掉的 collapse 模式）。这是 plan 的设计语义，不要改成 AND。
7. **G 路 max=25 epoch 可能仍是下限**：opengeoai 起点远 + trusted 池 3,533 小数据，如果 G 路 epoch 25 跑完 `patience_counter < 3` 仍在涨，记 followup —— 下轮若 G 胜出，重跑 max=35 看天花板。本轮承认可能是下限，不动该上限防 GPU 时间失控。

---

## 7. 内存上下文（已存）

新窗口可以直接读这些 memory entries 了解背景：
- `feedback_tier1_metric_system.md` — Ch3 model selection 用 area_aggregate_eval Tier 1 全套
- `feedback_pixel_union_with_fragmented_gt.md` — SAM-supp 碎 GT 上 area_R 奖励 roof-swallow
- `feedback_eval_gt_lock_clean.md` — JHB CBD 25 grid 评估 GT 锁 clean_gt（**禁止漂移到 Li GT / micro T1**）
- `feedback_merge_mode_first_class.md` — pixel-or vs per-detection 是 first-class lever
- `feedback_volume_loss_avoid.md` — 不要建议把 BCE 换成 Dice/IoU
- `feedback_bulk_ratio_perverse_incentive.md` — bulk = r/p Goodhart effect
- `project_jhb_phaseA_failed.md` — 上一轮 boundary-aware retrain 三项 pass criteria 全 fail
- `project_train20_val5_hn_failed.md` — train20_val5_hn V3-C+HN warm-start 也失败
- `project_pseudo_label_degradation_terms.md` — Gerstgrasser 2024 accumulation principle（untrusted ≤ 4 × trusted 的依据）

---

## 8. 一句话验证 handoff 接力没漏

新窗口开干前跑一遍：

```bash
cd ~/projects/ZAsolar
git log --oneline -3   # 应该看到 22a2b9c2 / 1bdf49a3 / dd80dbed
source scripts/activate_env.sh
python -m py_compile train.py core/training/boundary_aware_mask.py \
  export_coco_dataset.py scripts/training/build_unified_reviewall.py
python scripts/training/build_unified_reviewall.py \
  --output-dir /tmp/dryrun --val-jhb-grids G0772 G0816 G0817 G0888 G0925 --dry-run \
  2>&1 | grep "ASSERT" 
# 应该看到: [ASSERT] train pool: trusted=3384  untrusted=5691  ratio=1.68
rclone size dropbox:zasolar_data/tiles
# 5.8 GB 完成才能让 pod rclone copy
```

三个都过 → 可以开 pod。

---

## 9. 失败回退路径

如果训练全 fail（A 路 + G 路都不达 pass criteria）：
1. 不要无脑续训。前两轮（train20_val5_hn 续训 + JHB Phase A boundary-aware retrain）都是 V3-C-derived GT 反哺 mask head 的死路。
2. 优先检查：(a) trusted 池实际 supervision 有效率（`mean_box_reg_loss_H` vs `mean_box_reg_loss_R` 差距，gap 太小说明 mask_trusted gate 没真生效）；(b) bulk_ratio 走势是否撞 [0.85, 1.15] guardrail。
3. 下一轮扩 trusted 池：要么把 CT Batch 003/004 SAM-cut FN 升级到交互式重切（`sam_added_true_fn` → `sam_added_browser`，可加 ~250 trusted），要么主动新增 H 类标注（QGIS+GeoSAM 或 browser SAM）。
4. 极端 fallback：V3-C raw 锁为 production，等积累更多 H 类后再下一轮训练。
