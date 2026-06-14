<!--
Progress handoff for Tier B execution (B1 TTA pilot + B2 zerov2 probes).
Written 2026-06-10 ~23:30 after local prep completed; BLOCKED on pod GPU
availability (pod o2ccwdm1thuzxz host has no free GPUs).
Parent prompt: docs/handoffs/2026-06-10-f1-gap-tierB-agent-prompt.md — read it
FIRST; this doc only records delta progress, it does not restate the rules.
Usage: open a fresh session, read parent prompt + this doc, resume at "下一步".
-->

# Tier B 执行进度 handoff (2026-06-10 晚)

## 一句话状态

本地零 GPU 准备**全部完成**（B1 漏检集合锚定 + 三个 pilot 脚本 + B2 全部代码改动与
Step 0 文档注记，含本地验证）；**阻塞点 = pod 无 GPU**：`runpodctl pod start
o2ccwdm1thuzxz` 两次失败 `There are not enough free GPUs on the host machine`。
所有剩余工作都在 pod 侧（B1 三视图推理 ~1.5h + B2 两个 probe ~8h，≈$7 @ $0.69/hr）。

## 预注册 kill bar（原文不可移动，引用自 parent prompt）

- **B1**: 「<10–15% 漏检 polygon 转化为 IoU≥0.5 可恢复 proposal → 整条 TTA 放弃」
- **B2**: FLOOR = F1≥0.817 **且** σ_Bw≤0.248（JHB CBD25 clean_gt，best sweep point）;
  判定表三行（revive / kill / blocked-on-decoder）见 parent prompt B2 节。
- 双 probe 任何 fail 判定前必须双 merge-mode 打分（pixel-or 路径已实现，见下）。

## 已完成（本地，全部零 GPU）

### B1 — TTA 证伪 pilot 准备

| 项 | 结果 |
|---|---|
| Grid 选定 | **G0925, G0817**（指定）+ **G0816 (0.342), G0924 (0.511), G0889 (0.521)**（Ch2 per-grid recall@0.5 升序前三，源 CSV: `results/analysis/jhb_cbd25_v3c_raw_vs_sam_local_eval_20260511/ch2_sam_maskbox_v4_agg/`） |
| PTA0292 | **跳过**——本地 `~/zasolar_data/tiles/` 只有 cape_town/durban/johannesburg，无 pretoria；按 parent prompt 在报告注明即可（pod /workspace 如有 tiles 也不补：无 clean_gt，口径不一致） |
| Pilot 模型 | `checkpoints/exp_unified_reviewall_A/best_model.pth`（JHB production 模型） |
| 1.0× 基线 | unified_A **per-det+SAM @ c=0.925**（production 工作点），预测来自 `results/analysis/jhb_cbd25_3model_20260514/unified_A_perdet_sam_maskbox/<G>/predictions_metric.gpkg`（本地已有） |
| 匹配口径 | Ch2 同语义：`iou_matching(merge_preds=True)` @ IoU 0.5（installation profile） |
| **漏检集合（kill bar 分母）** | G0925: 61/79 · G0817: 32/90 · G0816: 66/111 · G0924: 64/188 · G0889: 45/142 → **合计 268 个漏检 polygon**。落盘 `results/analysis/tta_scale_probe/baseline/{<G>_missed_gt.gpkg, baseline_summary.csv}` |

新脚本（主仓，未 commit）：

1. `scripts/analysis/tta_probe_baseline.py` — 已跑完（上表）。
2. `scripts/analysis/tta_probe_resample_tiles.py` — chunked tiles → 1.5×/2.0× 双线性上采样，
   geotransform 同步缩放，输出镜像 `<out-root>/<region>/<layer>/<grid>/` 供
   `SOLAR_TILES_ROOT` fast-path。JPEG95 输出。
3. `scripts/analysis/tta_probe_audit.py` — 复用 `raw_hint_audit._raw_sets_from_artifact`
   （exp_finalizer 方法学），per-missed-polygon best-proposal IoU（mask 二值化 0.3/0.5 两档
   × score floor 0.3 主 / 0.05 诊断），输出 `per_polygon.csv` + `summary.csv`，
   `conv_iou05_mag` 即 kill-bar 统计量；已用本地 G1238 raw artifact 烟测通过
   （self-IoU=1.0 sanity）。视图命名约定 `x10/x15/x20`，目录结构
   `<view_root>/<view>/<grid>/raw_detections.pkl`。

**预注册的 overlap 决定**（写进 resample 脚本 docstring）：放大视图 detect_direct 用
`--overlap 0.5`（1.0× 维持 0.25 与 3model 口径一致）。理由：2.0× 时 400px 窗口只覆盖
200 native px，目标中位 ~70–80 native px，stride 必须 ≤ chip−target 才能保证完整窗口
覆盖；overlap 0.5 → 物理 stride 100/133 px（2.0×/1.5×），满足。chunk 边界的部分窗口
与 1.0× 一致，报告里注明即可。

### B2 — zerov2 probes 准备（代码全部落盘，未 commit）

`/home/gaosh/projects/solar_zerov2/`（工作树，基于 HEAD 3d9df27）：

1. **Step 0 done** — `docs/r1_backbone_domain_ablation.md` 加 §0.1 CORRECTION
   （Phase 0 decoder random-init，confound C6 入 §2 表，run 表加 1a′ 行）。
2. **Step 1 done（代码）** — `infer.py`：
   - 新增 `pixel_or_polygons()`（镜像主仓 finalize.py pixel-or 语义：OR 全部 raw
     pre-NMS footprint，连通域 = 一个 polygon，confidence = 贡献检测最大 score，
     与 `core/postproc.py PaintedPolygon` 合同一致）；每 grid 额外输出
     `<grid>/pixel_or/predictions_metric.gpkg`。单元测试通过。
   - 新增 `--config`（默认 SAT yaml）：经 `resolve_backbone_spec()` 选 backbone
     model_id + preprocess stats（web yaml stats=null → timm 默认）。
     infer_summary.json 记录 backbone/stats/merge_modes。
3. **1b 解锁** — `train.py::_load_config` 现在解析 `based_on` / `inherit_from`
   （`m2f_dinov3_l.yaml` 是 delta config，原代码会 KeyError）。本地验证：web cfg
   解析出 model_id=vit_large_patch16_dinov3 / queries=300 / epochs=30 / holdout 5 grid，
   SAT cfg 不变。
4. **1a′ 实现** — `train.py` 新增 `--pretrained-decoder` flag +
   `_load_pretrained_decoder()`：从 cfg `decoder.model_id`
   (facebook/mask2former-swin-large-coco-instance) 严格 name+shape 匹配拷贝，
   显式跳过 backbone encoder / 手术替换层（input_projections, lateral）/
   queries(300≠200) / class_predictor(1+1≠80+1)。**本地实测**（已下载 HF 权重到
   `solar_zerov2/.cache/huggingface`）：311 tensors / **19.5M params 载入**，
   skipped encoder=453 surgery=12 shape=6 absent=0；0-tensor 载入会 hard-fail。
   stats 进 model_info → train_config_snapshot.json。
5. `scripts/probes/r0_b8_real_feed_smoke.py::_build_model` 加 `backbone_model_id`
   参数（向后兼容默认 SAT），infer/train 共用。

### 其他已定事实

- **活 sweep 脚本判定**（parent prompt 硬约束 6）：B2 打分用
  `scripts/analysis/poly_conf_sweep.py`（registered-run + JHB clean_gt，Phase 0 同口径，
  r1 doc §5 配方）；`polygon_conf_sweep.py` 是 CT census 链（lock_operating_point 引用），
  不用于本任务。报告里写明这条辨认。
- 本地 GPU = RTX 4070 Laptop 8GB 且本地无 JHB Vexcel tiles
  （`~/zasolar_data/tiles/johannesburg/vexcel_2024/` 为空）→ **B1/B2 GPU 工作只能上 pod**。
- SAT/webL 特征 cache 都在 /dev/shm（已随 pod 重启丢失）→ 1a′ 也要重 cache（~0.5h），
  r1 doc 的 "复用 SAT cache ~3h" 时间估计相应 +0.5h。
- /dev/shm 28GB 放不下两份特征 cache（各 ~14.5GB）+ chip512 manifest → **顺序执行并删除**：
  cache webL → train 1b → rm webL cache → cache SAT → train 1a′ → rm cache → infer。

## 阻塞点与下一步（pod 侧 runbook）

### 0. 拿到 GPU

- 先重试 `bash scripts/runpod_pod.sh start`（宿主机 GPU 可能释放）。
- 不行则建新 pod 挂同一 network volume（/workspace 数据保留）：
  ```bash
  runpodctl create pod --name zasolar-tierb \
    --gpuType "NVIDIA GeForce RTX 5090" --secureCloud \
    --imageName runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 \
    --networkVolumeId <RUNPOD_S3_VOLUME_ID from .env> \
    --volumePath /workspace --containerDiskSize 40 --startSSH \
    --ports 22/tcp
  ```
  新 pod 后更新 `.env` 的 `RUNPOD_POD_ID`，SSH host/port 由 `runpod_pod.sh start` 自动写。
  ⚠️ 双 pod 规则见 `.claude/skills/runpod-ops/reference/ssh-and-env.md`。
- 新 pod 按 `docs/runbook/runpod_session_setup.md` §2–4.9 重建 venv
  （/root/venvs + cu128 torch；smoke test 必跑）。

### 1. Pod 现状探查（决定要传什么）

```bash
ssh ... "ls /workspace/tiles/johannesburg/vexcel_2024/ | head; \
  ls /workspace/coco/; ls /workspace/outputs/ 2>/dev/null; \
  find /workspace -maxdepth 4 -name 'best_model.pth' -path '*unified*' 2>/dev/null; \
  find /workspace -maxdepth 6 -name 'raw_detections.pkl' -path '*unified*' 2>/dev/null | head; \
  ls /workspace/solar_zerov2 2>/dev/null || find /workspace -maxdepth 2 -name train.py"
```
- 期望 `/workspace/tiles/johannesburg/vexcel_2024/G0925` 等 25 grid 在（2026-05-14 跑过）。
- 期望 `/workspace/coco/unified_reviewall_ct_only_chip512_20260514` 在（Phase 0 训练集）。
- unified_A checkpoint 与 5-14 的 unified_A raw_detections.pkl 若在 → B1 的 1.0× 视图直接复用
  （口径核对 config.json: chip400/overlap0.25/score0.05/mask0.3）；不在则 1.0× 重跑（快）。
- 缺 checkpoint 则 S3 上传（~330MB，规则 08 禁 scp）。

### 2. 代码上 pod

主仓 tar（含三个新 tta_probe 脚本 + `results/analysis/tta_scale_probe/baseline/`
的 missed gpkg ~几百 KB，一并打包）+ zerov2 tar（改动的 train.py/infer.py/smoke/docs），
走 S3 推（SOP §3 模式）。zerov2 在 pod 的路径以探查结果为准。

### 3. B1 执行（~1.5h GPU）

```bash
# per grid in G0925 G0817 G0816 G0924 G0889:
# (a) 重采样（CPU，输出到 /dev/shm 或 container disk）
python scripts/analysis/tta_probe_resample_tiles.py \
  --tiles-dir /workspace/tiles/johannesburg/vexcel_2024 \
  --out-root /dev/shm/tta_x15 --factor 1.5 --grids <G...>
python ... --out-root /dev/shm/tta_x20 --factor 2.0 --grids <G...>
# (b) 三视图 detect_direct（unified_A checkpoint；1.0× 若复用旧 raw 则跳过）
#     x10: 默认参数（overlap 0.25）；x15/x20: SOLAR_TILES_ROOT 指向重采样根 + --overlap 0.5
SOLAR_TILES_ROOT=/dev/shm/tta_x15 python detect_direct.py --grid-id <G> \
  --region johannesburg --imagery-layer vexcel_2024 --model-run tta_probe_x15 \
  --model-path <unified_A.pth> --overlap 0.5 \
  --output-dir /workspace/tta_probe/x15/<G>
# (c) audit（pod 侧跑，CSV 拉回本地）
python scripts/analysis/tta_probe_audit.py --view-root /workspace/tta_probe \
  --grids G0925 G0817 G0816 G0924 G0889 \
  --baseline-dir <解包的 baseline 目录> --output-dir /workspace/tta_probe/audit
```
注意 raw_detections.pkl 在放大视图上会大（chips ×4–7），放 /workspace 或 container disk，
不要塞满 /dev/shm（resample 输出按 grid 处理完即删，或 x15/x20 分两轮）。

### 4. B2 执行（~8h GPU，tmux/nohup + 日志）

按顺序（/dev/shm 容量约束）：
```bash
# manifest → shm
cp -r /workspace/coco/unified_reviewall_ct_only_chip512_20260514 /dev/shm/chip512
# --- probe 1b ---
python scripts/training/cache_dinov3_features.py --manifest-dir /dev/shm/chip512 \
  --out-dir /dev/shm/feat_webL --model-id vit_large_patch16_dinov3 --batch 8
python train.py --config configs/models/m2f_dinov3_l.yaml \
  --manifest-dir /dev/shm/chip512 --features-dir /dev/shm/feat_webL \
  --output-dir /workspace/outputs/r1s1_webL_native --batch-size 16 --grad-accum 1
rm -rf /dev/shm/feat_webL
# --- probe 1a' ---
python scripts/training/cache_dinov3_features.py --manifest-dir /dev/shm/chip512 \
  --out-dir /dev/shm/feat_sat --model-id vit_large_patch16_dinov3.sat493m --batch 8
python train.py --config configs/models/m2f_dinov3_sat_l.yaml --pretrained-decoder \
  --manifest-dir /dev/shm/chip512 --features-dir /dev/shm/feat_sat \
  --output-dir /workspace/outputs/r1s1_sat_pretdec --batch-size 16 --grad-accum 1
rm -rf /dev/shm/feat_sat /dev/shm/chip512
# --- infer 两个 checkpoint（tiles 可拷 /dev/shm 提速）---
python infer.py --checkpoint /workspace/outputs/r1s1_webL_native/best.pt \
  --config configs/models/m2f_dinov3_l.yaml \
  --tiles-dir /workspace/tiles/johannesburg/vexcel_2024 \
  --output-dir /workspace/outputs/infer/r1s1_webL_native --region jhb
python infer.py --checkpoint /workspace/outputs/r1s1_sat_pretdec/best.pt \
  --tiles-dir ... --output-dir /workspace/outputs/infer/r1s1_sat_pretdec --region jhb
```
- HF 下载（webL timm 权重 + mask2former-swin-large）pod 上直接拉；1a′ 训练命令需要
  `HF_HOME` 可写。
- 注意 train.py preflight（r1_unlock_check.py）若因环境抱怨可 `--skip-preflight`
  （Phase 0 已过 gate；记录在案）。

### 5. 拉回 + 打分（本地）

- `pack_and_pull_pod_results.sh` 或 S3 拉：两个 run 的 `<G>/{per_detection,pixel_or}/
  predictions_metric.gpkg` + infer_summary + train_log + audit CSV。
- 按 r1 doc §5 配方：symlink staging → regions.yaml 注册 4 个 model_run
  （webL/satdec × perdet/pixor）→ `area_aggregate_eval.py` + `poly_conf_sweep.py`
  （锁 clean_gt）→ 与 FLOOR(0.817/0.248) + Phase 0 (0.764, **random-decoder 注记**) 三方对照。
- 判定表与报告要求见 parent prompt（Verdict-first；B1 进 docs/experiments/；
  B2 回写 r1 doc；主计划 §4/§5 回写；两仓独立 commit，co-author
  `Claude <noreply@anthropic.com>`）。

## 未 commit 状态清单（建议下一 session 开工前先 commit prep）

- **主仓**：`scripts/analysis/tta_probe_{baseline,resample_tiles,audit}.py`（新）+
  `results/analysis/tta_scale_probe/baseline/`（小 gpkg/csv，~百 KB，可入 git 或留本地）+
  本 handoff。注意主仓工作树还有**本任务之前遗留**的改动
  （ROADMAP.md / annotation_manifest.csv / test_region_registry_census_mid_date.py /
  docs/handoffs/2026-06-10-f1-gap-tierB-agent-prompt.md）——不要混进 Tier B commit。
- **zerov2**：`train.py`、`infer.py`、`scripts/probes/r0_b8_real_feed_smoke.py`、
  `docs/r1_backbone_domain_ablation.md`（均改）+ 未跟踪 `docs/research/`（遗留，别动）。
- zerov2 `.cache/huggingface/`（~900MB HF 权重）是 gitignored 缓存，勿入 git。

## Suggested skills

- `/runpod-ops` — pod 生命周期/传输/venv 全部按它来（hard rules 已踩过坑）。
- 起好 pod 后长任务一律 tmux（本地）/ `Bash run_in_background`（pod SSH 会话内，
  `nohup` 不能跨 SSH 断连，见 runpod-ops hard rule 7）。

## 关键文件指针（按需读，不必全读）

- Parent prompt（规则与 kill bar 原文）: `docs/handoffs/2026-06-10-f1-gap-tierB-agent-prompt.md`
- 主计划 Tier B 节 + §3 否决台账: `docs/plans/2026-06-10-rcnn-f1-gap-review.md`
- C1/C13 对抗校验全文: `docs/plans/2026-06-10-rcnn-f1-gap-review.html`
- B1 方法学母本: `docs/experiments/exp_finalizer_pixel_or_vs_per_detection.md` +
  `scripts/analysis/raw_hint_audit.py`
- B2 执行合同: `solar_zerov2/docs/r1_backbone_domain_ablation.md`（含新 §0.1 correction）
- Pod SOP: `docs/runbook/runpod_session_setup.md`
