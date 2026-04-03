# Experiment 003: Hard Negative Training Strategy

**Date**: 2026-03-29
**Status**: Preparing
**Owner**: gaosh

## Motivation

Batch 003 review revealed a 35.7% FP rate (915 FP / 2563 total predictions). The current
training set uses random empty tiles as negatives, but these are mostly "easy" negatives
(e.g., roads, empty land) that don't challenge the model. The hypothesis is that training
with **targeted hard negatives** — chips centered on locations where the model already made
FP errors — will more effectively suppress false positives.

## Experiment Design

Three training variants using the same positive data (all batches through 003):

| Variant | Dataset | Train Images | Annotations | Negative Source |
|---------|---------|-------------|-------------|-----------------|
| A | `coco_v3_no_hn` | 4,740 | 7,720 | None (positive-only) |
| B | `coco_v3_hn` | 9,480 | 7,720 | Random empty tiles, balanced 1:1 |
| C | `coco_v3_targeted_hn` | 5,623 (4,740 + 883 FP) | 7,720 | Reviewed FP locations from batch 003 |

### Variant Details

**A — No Hard Negatives (baseline)**
- Only chips containing at least one annotation
- Tests whether the model can learn from positives alone
- Risk: high FP rate without negative examples

**B — Random Hard Negatives (current default)**
- 1:1 balanced: for every positive chip, one random empty chip
- Negatives come from any tile in annotated grids (87% useful, 13% blank/ocean/flat)
- Represents the status quo training approach

**C — Targeted Hard Negatives (new)**
- Positive-only base + 883 chips centered on batch 003 FP detections
- Every negative is a location where the model **already** produced a false alarm
- FP sources: `results/<GridID>/review/<GridID>_reviewed.gpkg` where `review_status == "delete"`
- **GT overlap filter**: FPs that intersect ground-truth annotations (>5% area overlap) are excluded — these are segmentation boundary errors, not true FPs. Removed 32 of 915 FPs.
- **CRS auto-fix**: Reviewed GPKGs with missing CRS but UTM-like coordinates are assigned EPSG:32734 and reprojected to EPSG:4326
- Positive:negative ratio ~5.4:1 (intentionally NOT balanced 1:1 — targeted negatives are high-quality, diluting with random negatives would reduce signal density)
- Chip extraction: 400×400 centered on FP polygon centroid

## Data Sources

### Positive chips
- **Annotations**: `data/annotations/cleaned/*_SAM2_*.gpkg` (63 grids, all batches)
- **Tiles**: `tiles/<GridID>/` (symlinked from D drive for batch 003)
- **Export**: `export_coco_dataset.py --output-dir <path>`

### Targeted HN chips (Variant C only)
- **FP source**: Batch 003 reviewed predictions (20 grids, 915 FPs)
- **Script**: `scripts/training/export_targeted_hn.py`
- **Provenance**: `coco_v3_targeted_hn/targeted_hn_provenance.csv`

## FP Distribution by Grid

| Grid | FPs | Correct | FP Rate |
|------|-----|---------|---------|
| G1688 | 114 | 281 | 28.9% |
| G1689 | 125 | 93 | 57.3% |
| G1690 | 120 | 94 | 56.1% |
| G1691 | 88 | 114 | 43.6% |
| G1687 | 82 | 197 | 29.4% |
| G1800 | 83 | 133 | 38.4% |
| G1743 | 67 | 153 | 30.4% |
| G1749 | 44 | 109 | 28.8% |
| G1798 | 40 | 75 | 34.8% |
| Others | 152 | 399 | 27.6% |
| **Total** | **915** | **1648** | **35.7%** |

## Evaluation Plan

All three variants will be evaluated on the **same validation set** (2,640 images, 2,948 annotations)
using the `installation` evaluation profile.

### Primary Metrics
- **F1 @ IoU ≥ 0.5** (primary decision metric)
- **Precision** (FP suppression — the main goal of this experiment)
- **Recall** (must not regress significantly)

### Secondary Metrics
- AP50 (COCO standard)
- FP count per grid on batch 003 grids (re-run inference)

### Success Criteria
- Variant C should achieve **higher precision than B** without significant recall loss
- If Variant A outperforms B, it suggests random negatives may be adding noise
- If B outperforms C, the FP locations may not generalize as hard negatives

## Reproduction

```bash
# Variant A: no hard negatives (--no-balance keeps all chips, then filter positives only)
python export_coco_dataset.py --output-dir /mnt/d/ZAsolar/coco_v3_no_hn --no-balance
# Note: scan-then-write ensures only selected chips are written to disk (no orphaned files)

# Variant B: random hard negatives (default 1:1 balance)
python export_coco_dataset.py --output-dir /mnt/d/ZAsolar/coco_v3_hn

# Variant C: targeted hard negatives
python scripts/training/export_targeted_hn.py \
    --base-coco /mnt/d/ZAsolar/coco_v3_no_hn \
    --output-dir /mnt/d/ZAsolar/coco_v3_targeted_hn

# Training (on RunPod)
python train.py --coco-dir <variant_dir> --output-dir checkpoints/<variant>
```

## File Locations

| File | Path |
|------|------|
| Variant A dataset | `/mnt/d/ZAsolar/coco_v3_no_hn/` |
| Variant B dataset | `/mnt/d/ZAsolar/coco_v3_hn/` |
| Variant C dataset | `/mnt/d/ZAsolar/coco_v3_targeted_hn/` |
| Targeted HN script | `scripts/training/export_targeted_hn.py` |
| FP provenance | `coco_v3_targeted_hn/targeted_hn_provenance.csv` |
| This document | `docs/experiments/exp_003_hard_negatives.md` |

## Results

### COCO Validation AP50 (training eval)

| Metric | A (no HN) | B (random HN) | C (targeted HN) |
|--------|-----------|---------------|-----------------|
| AP50 | **0.8058** | 0.7748 | 0.7689 |
| Best Epoch | S2 E19 | S2 E15 | S2 E20 |

Training config: RTX 5090 32GB, bs=32, AMP, /dev/shm, num_workers=8

### Batch 002 Inference (26 grids, GT=2015, post_conf≥0.7)

> **Note**: 早期 V3 batch 002 推理使用了不完整的 GT 数据 (1148 objects)。下表为 2026-03-29 用完整 GT (2015 objects) 重新跑的结果。

| Metric | V2 baseline | A (no HN) | B (random HN) | C (targeted HN) |
|--------|-------------|-----------|---------------|-----------------|
| **Precision** | 35.3% | — | — | **52.1%** |
| **Recall** | **78.2%** | — | — | 74.6% |
| **F1** | 48.6% | — | — | **61.3%** |
| TP | **1575** | — | — | 1503 |
| FP | 2885 | — | — | **1384** |
| FN | **440** | — | — | 512 |

> A/B 未用完整 GT 重跑，早期结果（GT=1148）: A F1=55.4%, B F1=57.4%, C F1=62.4%。趋势一致：C > B > A。

### V3-C vs V2 Delta

| Metric | Delta | 说明 |
|--------|-------|------|
| Precision | **+16.8pp** | 35.3% → 52.1% |
| Recall | -3.6pp | 78.2% → 74.6% |
| F1 | **+12.7pp** | 48.6% → 61.3% |
| FP | **-52%** | 2885 → 1384 |
| TP | -4.6% | 1575 → 1503 |

### Key Findings

1. **V3-C 相比 V2 有显著提升**: F1 +12.7pp，FP 减半，TP 仅少 4.6%。在独立标注的 batch 002 上验证了 targeted HN 训练策略的有效性。

2. **COCO AP50 与实际推理结论相反**: AP50 排序 A > B > C，但实际推理 F1 排序 C > B > A。AP50 是跨阈值面积指标，不代表固定阈值下的 P/R 表现。

3. **Targeted HN 显著提升 Precision**: C 相比 V2 的 FP 减少 52% (2885→1384)，Precision 提升 16.8pp，Recall 仅降 3.6%。验证了假设：用审核 FP 位置作为 hard negative 能有效抑制模型在类似场景下的误检。

4. **Batch 003 评估存在偏差**: V2 在 batch 003 上 F1=85.7% vs V3-C 81.0%，但 batch 003 GT 来源于 V2 推理结果（V2 推理 → 人工审核 → SAM 补标），V2 多边形形状天然匹配 GT → IoU 更高 → recall 虚高。**必须用独立标注数据 (batch 002) 做公平对比**。

5. **Precision 仍是瓶颈**: 即使最好的 V3-C 也只有 52.1% Precision，1384 个 FP。高 confidence FP (0.7-0.99) 说明模型对某些 FP 模式非常自信，需要进一步调查 FP 类型。

### Remaining FP Analysis

FP 样本 (Variant C, batch 002) 保存在 `results/fp_samples_exp003C/`。典型特征：
- 面积 5-76m²，大多为小目标 (<40m²)
- Confidence 极高 (0.72-0.99)
- 需要人工分类确定 FP 类型以指导后续训练策略

### Decision: Batch 004 使用 V3-C 权重

基于 batch 002 公平对比结果，**batch 004 推理使用 V3-C** (`exp003_C_targeted_hn/best_model.pth`)。
