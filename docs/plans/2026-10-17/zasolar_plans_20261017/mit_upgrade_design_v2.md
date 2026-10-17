# MiT-b2 upgrade / alternative-backbone experiment design v2 (DESIGN ONLY; §0 filled 2026-10-17 by CPU read-only check)
Date: 2026-10-17. Mode: exploration (WORKFLOW.md). No training/GPU until 葉 approves.

## 0. Current model (verified read-only on rog, CPU only, 2026-10-17)
- Training code: `~/capetown_eval/ds3m_ft/train2_v2d1.py` (v2d1) and `train2_v2d2.py` (identical recipe, v2d2 data). The latest finished checkpoint is `~/capetown_eval/ds3m_ft/run_v2d1/best.pt` (DONE; best by val IoU). v2c = `run_v2c/best.pt` (copy at `evals/ft_best.pt`). **`run_v2d2` has NOT been trained**: `v2d2_chain.sh.HELD` is on hold.
- Format: plain HF `state_dict` saved with transformers 5.16.1, 380 tensors, 27,350,468 params. Key prefixes: `segformer.stages.{i}.patch_embeddings / blocks.{j}.(layernorm_before, attention.q_proj/k_proj/v_proj/o_proj, attention.sequence_reduction.*, layernorm_after, mlp.fc1/dwconv/fc2) / layer_norm` + `decode_head.*`. This is the new transformers-5 naming. Older hub checkpoints (nvidia/mit-b3) use `segformer.encoder.block...`; loading them may need a key remap (unverified).
- Model: `SegformerConfig(num_labels=3, depths=[3,4,6,3], hidden_sizes=[64,128,320,512], decoder_hidden_size=768)`. Init comes from DeepSolar3M `~/capetown_eval/deepsolar3m/weights/segformer.pt` (remapped, strict=True). The 3rd class (SWH, solar water heater) is initialised by copying the PV classifier row with bias −2. Labels: 0 bg / 1 PV / 2 SWH / 255 ignore.
- Crop: 640 px random crops (`--chip 640`) drawn from 512-px-indexed tiles. Sampling: 70% PV-centred, 5% SWH, 15% hard negative, 10% random.
- Schedule: 30,000 iterations, bs 2 × accum 2 (effective 4), AdamW (lr 6e-5 encoder, ×2 decoder, wd 0.01), 1k linear warmup then poly 0.9, grad clip 1.0, gradient checkpointing. Validation every 2,500 iterations on 300 val tiles.
- Loss: CE + 0.5·Dice, SWH weight 2.0.
- Aug: h/v flip, rot90, brightness ×U(0.85,1.15) + U(−15,15), and 30% "degrade" (downsample to 5/8 resolution, then upsample back). ImageNet mean/std.
- Seed: **no global seed**. Sample RNG is `default_rng(index)`, so the data order is deterministic, but torch init and cuDNN are not seeded. The fair comparison must add explicit seeds (e.g. 0/1/2) to every arm, including the A0 retrain.
- Throughput (run_v2d1 log): 0.385 s/it, 4.1 GB peak → about 3.2 h per 30k run on rog. Final val (internal 300-tile validate(), step 30k): IoU 0.936, P 0.951, R 0.984, bIoU 0.486. These come from the trainer log, not from VERIFICATION.md (different eval protocol).

## 0b. CPU state_dict mapping test (scripts: mit_upgrade_scripts_20261017/map.py, map2.py; random-init HF configs, nothing downloaded)
| Target | Tensors matched / total | Missing | Shape mismatch | Params covered |
|---|---|---|---|---|
| MiT-b3, direct load (strict=False) | 380 / 644 | 264 | 0 | 27.35M / 47.23M (58%) |
| MiT-b3, + duplicate b2 blocks (block j ← j mod depth_b2) | 644 / 644 | 0 | 0 | 47.23M / 47.23M (100%) |
| MiT-b4, direct | 380 / 930 | 550 | 0 | 27.35M / 64.00M |
| MiT-b4, + duplicate blocks | 930 / 930 | 0 | 0 | 64.00M / 64.00M |
| PVTv2-b2 (HF PvtV2, official mlp_ratios [8,8,4,4]), name remap | 329 / 364 | 0 | 35 | 23.54M / 24.85M (95%) |
- Every b3/b4 mismatch is a missing block. There are 0 shape conflicts: all present tensors, including the decode head with 3 classes, load as-is.
- PVTv2-b2 is MiT-b2 with renamed keys. The only shape conflicts are the stage-1/2 MLP layers (35 tensors; PVTv2 uses hidden ratio 8, MiT uses 4). Options: (a) build PVTv2 with mlp_ratio 4 (then it is architecturally identical to MiT-b2, so the arm adds nothing); (b) load the 329 tensors and init the stage-1/2 MLPs from ImageNet PVTv2-b2; (c) widen them via Net2Net (copy the existing units, halve the fc2 weights). PVTv2 is therefore a weak "alternative"; drop A3 or keep it only as (b).
- MSCAN-B and EfficientViT-B2 were not tested: they are different architectures, so only the decode-head idea transfers. Plan for distillation.

## 1. Facts (web-verified, with sources)
| Backbone | Params | FLOPs | Pretrained ckpt | Source |
|---|---|---|---|---|
| MiT-b2 (current) | 24.2M enc; SegFormer-B2 27.5M | 4.0G@224 (enc); 62.4G@512 ADE | HF nvidia/mit-b2 | SegFormer supp. Tab; SegNeXt Tab |
| MiT-b3 | 44.0M enc | 6.9G@224 | HF nvidia/mit-b3 | SegFormer supp. |
| MiT-b4 / b5 | 60.8M / 81.4M | 10.1G / 11.8G | HF nvidia/mit-b4/b5 | SegFormer supp. |
| Segformer++ (2024, arXiv 2405.14467) | same as MiT (no new weights) | B2 up to ~1.9x faster via token merging | uses original MiT weights | arXiv 2405.14467 |
| SegNeXt-B / MSCAN-B | 26.8M enc; 27.6M total | 34.9G@512 ADE | mmseg / Tsinghua Cloud (ADE, Cityscapes) | NeurIPS'22 paper; repo README |
| PVTv2-b2 | 25.4M | 4.0G@224 | GitHub whai362/PVT, timm | arXiv 2106.13797 |
| EfficientViT-B2 (seg) | 15M | 74G MACs@1024x2048 | HF han-cai/efficientvit-seg (ADE, Cityscapes) | mit-han-lab README |

Key point: there is NO official "MiT v2". MiT-b0..b5 remain the latest MiT weights; Segformer++ is an inference-efficiency method on the same weights. "Upgrade" therefore = deeper MiT (b3/b4) or a same-size alternative.
MiT-b2..b5 share widths 64/128/320/512, heads, SR ratios, patch embeds; they differ only in per-stage depth (b2 [3,4,6,3], b3 [3,4,18,3], b4 [3,8,27,3]; SegFormer paper Table 4 — depths cross-check against HF config before use). Decoder (all-MLP) is identical in shape -> fully reusable.

## 2. Weight transfer routes (ranked by feasibility)
1. MiT-b3/b4, partial load + depth expansion: copy patch_embed, norms, decode head, and all blocks of b2 into the matching indices; new blocks (b3 stage-3 blocks 6..17) either (a) from ImageNet nvidia/mit-b3, or (b) duplicated from b2 blocks (repeat/interleave) with drop-path. No pos-embed interpolation needed (MiT has no absolute pos-embed). Strict=False load, log missing/unexpected keys. Feasibility: high.
2. PVTv2-b2: same family (overlapping patch embed, SR attention, Mix-FFN-like DWConv); dims identical to MiT-b2 [64,128,320,512], depths [3,4,6,3]. Partial key remap likely possible for most tensors but naming/shape needs a scripted check — treat as UNVERIFIED; fallback: ImageNet PVTv2 init + distillation. Feasibility: medium.
3. MSCAN-B (SegNeXt) and EfficientViT-B2: conv-attention / linear-attention, no shape match -> ImageNet/ADE init + fine-tune on v2d2, optionally logit/feature distillation from b2 teacher (KD loss on soft masks, T=1..2, alpha 0.5). Feasibility: medium (needs new code path).
4. Pure distillation b2 -> any student: always possible, costliest, last resort.
Segformer++ token merging: zero-training, applicable to current b2 as an inference-speed arm only (check small-panel recall loss).

## 3. Hypothesis
H1: MiT-b3 initialised from DS3M-b2 (route 1) improves object F1@0.5 on the clean test set by >= +0.01 over b2 without lowering small-panel recall.
H2: a same-size alternative (MSCAN-B, PVTv2-b2) matches b2 (|dF1| <= 0.01) with better boundary F or speed.
H0: no candidate beats b2 beyond seed noise -> keep b2.

## 4. Data
- Train: v2d2 (as current b2). Label provenance kept separate: 葉's labels (original SAM hand labels + Mar 3 batch) vs Li's; no Google Earth labels. Ablation per source optional.
- Val: existing 1,768 val tiles (VERIFICATION.md) for model selection only.
- Test (frozen, touched once per arm): Li's later no-KML batch, built by the Cape Town bot — NOT yet available; experiment blocked on it.

## 5. Arms (fair comparison)
A0 MiT-b2 DS3M-ft (retrain with identical recipe, not the old ckpt only) | A1 MiT-b3 route1a | A2 MiT-b3 route1b (duplicated blocks) | A3 PVTv2-b2 | A4 SegNeXt-B (MSCAN-B) | A5 EfficientViT-B2 (optional) | A6 b2+Segformer++ inference (no training).
Same: data split, crop size, epochs/iters, aug, optimizer/LR schedule (LR tuned only on val, small grid shared by all), 3 seeds each, same threshold selection protocol (fixed 0.5 AND val-selected hysteresis, both reported — see VERIFICATION definition ambiguities 1–4).

## 6. Metrics (side by side, every arm, mean ± sd over 3 seeds)
Object P/R/F1 at IoU 0.3/0.5/0.7; small-panel recall by area bins (e.g. <20, 20–50, 50–100, >100 px — confirm bins with current eval); boundary F-score (fixed tolerance, one implementation chosen up front — VERIFICATION notes evalfull vs evalobj bIoU differ, 0.459 vs 0.434 for MiT t0.3); pixel IoU; params, latency/tile, GPU-h. Independent recomputation of all metrics from saved predictions by a separate script.
Baseline reference (verified, VERIFICATION.md): MiT pixel IoU 0.753–0.758 across thresholds; object F1 0.926 (v2d1 0.908) on val. These are val numbers, not test.

## 7. GPU budget (estimate, unverified; calibrate with one b2 epoch timing)
Let T = wall-clock of one current b2 training run on rog. Relative cost ~ FLOPs: b2 1.0T, b3 ~1.7T, PVTv2-b2 ~1.0T, MSCAN-B ~0.6–1.0T, EffViT-B2 ~0.7T.
Measured (§0): T ≈ 3.2 GPU-h on rog (30k it × 0.385 s). So the full study ≈ 21T + 3T ≈ 77 GPU-h; phase 1 (A0, A1, A4 × 1 seed) ≈ 11 GPU-h. b3 cost scaling is still unverified.
Total for 3 seeds: A0 3T, A1 5T, A2 5T, A3 3T, A4 3T, A5 2T ≈ 21T (+ small LR grid ~3T). Recommend phase 1 = A0,A1,A4 single seed (~3.3T), phase 2 = remaining seeds only for arms within 0.01 F1 of best.

## 8. Stop conditions
- Stop an arm after 2 consecutive failed runs (crash/NaN/val F1 < A0−0.03) per WORKFLOW.
- Stop whole study if phase 1 shows no arm > A0 + 1 seed-sd on val.
- Budget cap: 葉 sets max GPU-h; stop at cap.
- Every run logged (config, seed, git hash, ckpt path, metrics JSON).

## 9. Decision informed
Whether the Cape Town census production model stays MiT-b2 or switches to b3 / an alternative; and whether Segformer++ inference is safe for the full-city run.
