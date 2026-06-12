# Tier C pod runbook (2026-06-12)

Tier C 非 GPU 工程已于 2026-06-11/12 通过三链并行 workflow 落地(C-1 工具链 / C-2+C-3(b) 训练配方 / C-3(a) Phase-0 harness),全量 `pytest tests/` 193 passed。本文是 pod 上 GPU 步骤的统一入口;细节见各链自己的 handoff。

起 pod 后先走 runpod-ops skill(.env SSH 更新、/workspace read-before-write、tiles → /dev/shm)。

## 1. C-1 retrain(搭 C-2 warmup+EMA 顺风车)

数据集:pod 上现切(rule 08,勿本地切完上传)

```bash
# pipeline.dataset_builder + 新 spec(easy_neg 0.5;targeted-HN 当前 0 chip,
# 因 pool 全部行 training_eligible=false —— 这是预期,见 spec 注释)
configs/pipelines/datasets/unified_reviewall_v3_hn.yaml
```

训练(双 sibling 归因,同数据同 seed,唯一 delta = recipe flags):

```bash
# recipe run
python train.py --coco-dir /workspace/coco/unified_reviewall_v3_hn \
  --pretrained checkpoints/exp003_C_targeted_hn/best_model.pth \
  --output-dir checkpoints/c1_negpool_warmup_ema --seed 42 \
  --freeze-mask-head --per-instance-mask-trusted --per-source-mask-weight \
  --diff-lr-backbone-mult 0.1 \
  --warmup-iters 750 --warmup-start-factor 0.01 --ema --ema-decay 0.999
# legacy sibling(无 --warmup-iters/--ema,其余同)
```

- warmup_iters 实跑前按 Stage-2 总步数的 ~3–5% 重算(750 是 batch32 估值)。
- 产出 raw-best + EMA-best 双 checkpoint;评估两个都跑。

Gate(C-1,计划 line 186):Ch2 recall@0.3 非回归(locked clean_gt)+ Tier-1 全套**双 merge-mode**;σ_Bw 声明只在 solar_cls-attached baseline 之上、held-out 跨域 grid 上量。若与 B1 TTA 同轮:分开报 with/without TTA。

## 2. C-3(b) ignore-band 单 lever retrain(C-1 之后排队)

```bash
python train.py ...同 warm-start... --output-dir checkpoints/c3b_ignore_band --seed 42 \
  --freeze-mask-head --per-instance-mask-trusted --per-source-mask-weight \
  --diff-lr-backbone-mult 0.1 \
  --boundary-ignore-band --boundary-ignore-band-thresholds 400,2500
```

Gate:bulk / σ_Bw / area_F1 vs unified_A on locked JHB CBD25 clean_gt 双 mode;**不得 book 任何 polygon-recall/polygon-F1 收益**。详见 `docs/handoffs/2026-06-11-c2-warmup-ema-c3b-ignore-band.md` 与 `core/training/boundary_ignore_band.py` docstring。

## 3. C-3(a) Phase-0 缺标率测量(决定 C-3(a) 做不做)

完整流程见 `docs/handoffs/2026-06-11-c3a-phase0-runbook.md`。概要:

```bash
# 采样(CPU,但需 JHB vexcel_2024 tiles 在场 —— 本地缺,必须 pod 跑)
python scripts/training/sample_c3a_phase0_chips.py \
  --spec configs/pipelines/datasets/unified_reviewall_v2.yaml \
  --target 180 --seed 42 --out-dir results/analysis/c3a_phase0/<date>_c3a_phase0
# 验收: sample_meta.json::missing_strata_no_local_tiles==[]

# 低 conf 扫描(唯一 GPU 步骤)
python scripts/training/run_c3a_phase0_scan.py --phase detect \
  --run-dir <run-dir> --model-path checkpoints/exp_unified_reviewall_A/best_model.pth \
  --model-run exp_unified_reviewall_A --score-threshold 0.05

# extract(CPU)→ build audit(HTML 标注包)→ gate
python scripts/training/run_c3a_phase0_scan.py --phase extract --run-dir <run-dir> --gt-iof-threshold 0.10
python scripts/analysis/build_c3a_phase0_audit.py --run-dir <run-dir>
# 人工/Gemini 填 audit_label 后:
python scripts/analysis/compute_c3a_phase0_gate.py --audit-csv <run-dir>/audit.csv --threshold 0.05
```

判定:PASS(≥5% chips 受影响)→ C-3(a) 开工;KILL → 杀掉。审计输出即 ignore 语料(confirmed_pv → 转正;lookalike → negative_pool,绝不入 ignore)。

## 4. 明确不做

- **C-4 copy-paste**:硬依赖 C-1 retrain 结果;若 trusted mask 仍 bulk 过冲直接杀,不预写代码。
- Tier B(B1 TTA / B2 zerov2 probe)走另一条线:`docs/handoffs/2026-06-10-f1-gap-tierB-progress.md`。

## 5. 已知坑(本轮 review 修掉的,防回潮)

- `pipeline/hn_ops.py` cropper 已加 CRS reproject(4326 bbox → tile 原生 CRS);Vexcel/3857 图层裁切靠它,别回退。
- `sample_c3a_phase0_chips.py` 的 per-stratum seed 用 `stratum_sub_seed()`(crc32),不要改回 `hash()`(进程加盐,毁复现)。
- EMA shadow 已随 checkpoint 持久化(`state["ema"]`);resume 旧 checkpoint 无该 key 会 WARNING 回退 online 重播种。
- HN chip 进训练包受 `training_eligible` gate;当前 711 行全 false,翻 true 是人工决策(见 `data/negative_pool/README.md` + `hn_breadth_report.py` 诊断)。
