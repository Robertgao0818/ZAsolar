# Cape Town PV segmentation backbone comparison: design v3 (DESIGN ONLY; no GPU, no training)
Date 2026-10-17. Supersedes v2 for scope. v1/v2 kept unchanged. §0 facts (current recipe, key mapping) are in v2 and are not repeated here except where needed.
Approved arms (葉, 2026-10-17): A0 MiT-b2 baseline, SegNeXt-B, ConvNeXt-T+UPerNet. MiT-b3 optional. PVTv2-b2 and EfficientViT dropped.

## 1. Arms
| Arm | Model | Params | FLOPs (published) | Init | Source |
|---|---|---|---|---|---|
| A0 | SegFormer MiT-b2 (current) | 27.35M (counted on rog ckpt) | 62.4G @512² | DeepSolar3M SegFormer `~/capetown_eval/deepsolar3m/weights/segformer.pt` (exact recipe of train2_v2d2.py) | SegNeXt paper Tab. (SegFormer-B2) |
| A1 | SegNeXt-B (MSCAN-B + Ham head) | 27.6M | 34.9G @512² | ImageNet-1k MSCAN-B (or ADE20K SegNeXt-B ckpt, head re-init to 3 classes) | SegNeXt NeurIPS'22; github Visual-Attention-Network/SegNeXt |
| A2 | ConvNeXt-T + UPerNet | 60M (backbone 28.6M) | 939G @2048×512 (≈235G @512², linear scaling, unverified) | ImageNet-1k ConvNeXt-T (or ADE20K UPerNet ckpt `upernet_convnext_tiny_1k_512x512.pth`) | ConvNeXt paper Tab. 4 (arXiv 2201.03545); facebookresearch/ConvNeXt semantic_segmentation; mmseg configs/convnext |
| A3 (optional) | MiT-b3 from DS3M-b2 + duplicated blocks | 47.2M | ~79G @512² (SegFormer-B3, from memory, unverified) | b2 → b3, 644/644 tensors map (v2 §0b) | v2 §0b |

Only A0 and A3 can use the DS3M fine-tuned weights directly. A1 and A2 are a different architecture, so the only route for them is a public init plus fine-tuning. **Distillation from the b2 teacher is optional**: an extra arm variant A1-KD/A2-KD with loss = task + 0.5·KL(soft masks, T=2) computed on the b2 v2d2 checkpoint's logits. Run it only if A1/A2 without KD land within 0.01 F1 of A0.

### Remote-sensing pretrained ConvNeXt-T: findings
- SatlasPretrain (allenai, ICCV'23): aerial 0.5–2 m/px RGB models exist, but only Swin-v2-B/T and ResNet50/152, no ConvNeXt. Licence ODC-BY. Source: github allenai/satlaspretrain_models.
- RSP (MillionAID, TGRS'22): ResNet-50, Swin-T, ViTAEv2-S only, no ConvNeXt. MIT licence. Source: github ViTAE-Transformer/RSP.
- SSL4EO-S12: Sentinel-1/2 (10 m), ViT/ResNet only. CC-BY-4.0. Source: HF BiliSakura/SSL4EO-S12-transformers. Wrong resolution and no ConvNeXt.
- MMEarth (ConvNeXt V2 atto/tiny, MP-MAE): the tiny model is 12-channel Sentinel-2 only; the RGB variants are atto-size only. Sentinel resolution, not aerial. Source: github vishalned/MMEarth-train; HF BiliSakura/MMEarth-transformers.
- Conclusion: **no aerial-resolution RGB ConvNeXt-T checkpoint was found**. Use ImageNet-1k (or ADE20K UPerNet) init. If 葉 wants a remote-sensing-pretrained comparator, the closest same-size option is SatlasPretrain Aerial Swin-v2-Tiny (ODC-BY, 0.5–2 m/px, which matches CT aerial ~0.08 m only loosely: the CT tiles are finer than Satlas's range). Not included unless requested.

## 2. Data and label provenance
- Train: v2d2 (`~/data/ZAsolar/geid_temporal/ct_seg_trainset_v2d2_20261017`, 79 train grids, 19,904 chips). 葉's labels (original SAM hand labels + Mar 3 batch) are tracked separately from Li's in the manifest; no Google Earth labels (v2d2 reads only `G*_SAM2_*`).
- Val (model and threshold selection only): the v2d2 val split (14 grids). **It contains V2/V3-C reviewed-prediction labels** (detector predictions reviewed by a human). These may favour models that resemble the generating detector, so val is not used for the final claim.
- Eval (5 grids incl. G2038): reported, but has the same provenance caveat.
- **Clean test (decisive):** Li's later no-KML batch (being built by the Cape Town bot). Used once per final checkpoint.
- **Secondary clean test:** Joburg labels (clean for CT models; never trained on). They test transfer.
- Every metric table carries a provenance column (clean vs reviewed-prediction).

## 3. Common recipe (copied from train2_v2d2.py, differences explicit)
Same for all arms: v2d2 data, the same sampler (70/5/15/10 PV/SWH/HN/random), 640 px crops, flips + rot90 + brightness + 30% degrade, 3 classes (bg/PV/SWH, 255 ignore), CE + 0.5·Dice with SWH weight 2.0, 30k iterations, effective batch 4 (bs 2 × accum 2), 1k warmup + poly 0.9, grad clip 1.0, val every 2.5k on the same 300 val tiles, best-by-val-IoU checkpoint. **Explicit seeds 0, 1, 2** for every arm (torch/numpy/random + sampler offset; cudnn.deterministic off, so the seed is logged rather than exact determinism).
Optimizer per arm:
- A0 / A3: AdamW lr 6e-5, decoder ×2, wd 0.01 (the current recipe).
- A1 SegNeXt-B: AdamW lr 6e-5, wd 0.01 (SegNeXt mmseg ADE config, to be confirmed in the config file before the run), head ×10 per the official config if present.
- A2 ConvNeXt-T+UPerNet: AdamW lr 1e-4, wd 0.05, layer-wise LR decay 0.9, drop-path 0.3–0.4 (ConvNeXt paper App. A.3 sweeps lr {8e-5,1e-4}, LLRD {0.8,0.9}, drop-path {0.3,0.4,0.5}). One mini-sweep on seed 0 only (lr ∈ {8e-5,1e-4} at 10k iterations) counts toward the budget. Fixed thereafter.
- Mixed precision allowed for all arms equally.

## 4. Metrics (side by side, mean ± sd over 3 seeds, per test set)
Object P/R/F1 at IoU 0.3/0.5/0.7; small-panel recall by area bin (reuse the evalobj bins: <4, 4–10, 10–25, >25 m²); boundary F-score (one implementation fixed in advance: evalobj); pixel IoU (PV); SWH IoU; negative FP m²/ha; params; inference latency per 640 tile; GPU-h used. Threshold: fixed 0.5 AND the val-selected hysteresis, both reported. All metrics recomputed from saved probability maps by an independent script (per WORKFLOW).
Baseline reference values (VERIFICATION.md, val, reviewed-label provenance): MiT object F1 0.926 (v2d1 0.908), pixel IoU 0.753–0.758.

## 5. GPU budget (T = 3.2 GPU-h measured for one b2 30k run on rog; others ESTIMATED by FLOPs, UNVERIFIED)
Step-time estimates relative to b2: SegNeXt-B 34.9/62.4 = 0.56× FLOPs → assume 0.6–1.0T (depthwise convs are often slower than their FLOPs suggest) ≈ 1.9–3.2 h. ConvNeXt-T+UPerNet ≈ 235/62.4 = 3.8× FLOPs → assume 2.5–3.8T ≈ 8–12 h (UPerNet's head dominates the cost). MiT-b3 ≈ 1.3–1.7T ≈ 4.2–5.4 h. Optional KD adds a b2 forward pass, ≈ +0.3T per run (unverified).

| Plan | Runs | GPU-h (est.) |
|---|---|---|
| (a) Minimal: A0 b2 × 3 seeds | 3 | 9.6 |
| Phase-1 single seed: A0 + A1 + A2 (seed 0) + ConvNeXt LR mini-sweep (2 × 1/3 run) | 3 + 2 short | 13–18 + 5–8 ≈ 18–26 |
| (b) Main: A0 + A1 + A2 × 3 seeds (+ LR sweep) | 9 | 9.6 + 5.7–9.6 + 24–36 + 5–8 ≈ 44–63 |
| (c) Main + A3 b3 × 3 seeds | 12 | (b) + 12.5–16.3 ≈ 57–79 |
| Optional KD variants (A1-KD, A2-KD, 3 seeds) | 6 | ≈ (A1 + A2 cost) + 6 × 1 ≈ 36–52 extra |
Phase 1 calibrates the step times: re-estimate the budget after the first 500 iterations of each arm (log s/it) before committing to (b).

## 6. Stop conditions
- Per arm: stop after 2 consecutive failed runs (crash, NaN, OOM after one bs/accum fix, or val PV IoU < A0 − 0.05 at 10k iterations).
- Phase gate: proceed to 3 seeds only for arms whose phase-1 val object F1@0.5 ≥ A0 − 0.01.
- Global: stop at 葉's GPU-hour cap; stop the study if no arm is within 0.01 of A0 after phase 1.
- Every run logged (config, seed, git hash, ckpt path, s/it, metrics JSON); results recomputed independently.

## 7. Hypotheses and decision
- H1: SegNeXt-B matches A0 (|ΔF1@0.5| ≤ 0.01 on clean test) at lower or equal latency, so it is a viable faster/alternative production model.
- H2: ConvNeXt-T+UPerNet improves boundary F and small-panel recall (<10 m²) by ≥ 0.02 over A0, at about 2× params.
- H0: no candidate beats A0 beyond seed sd on the clean test, so keep MiT-b2 (v2d2) for the Cape Town census.
- Decision informed: which backbone produces the Cape Town census production masks, and whether DS3M init (domain-specific) beats generic ImageNet/ADE init at equal recipe.
- Blocker: the clean Li no-KML test set must exist before any final comparison. A0 v2d2 training (v2d2_chain.sh.HELD) is the natural first run.
