# Experiment: PV vs non-PV Classifier Backbone Ablation

**Date started**: 2026-04-22
**Status**: Planning frozen, implementation pending
**Owner**: gaosh
**Plan**: `/home/gaosh/.claude/plans/codex-efficientnetb0-convnexttiny-found-swirling-muffin.md`

## Motivation

Batch 004 small-FP taxonomy (2026-04-03) found **77% of high-confidence small-target FPs (<30 m², area-reasonable geometry) are solar thermal water heaters**. Geometric post-processing (conf, elongation, area) cannot separate them — thermal geysers look like small PV panels from nadir. A binary PV vs non-PV classifier applied as a post-detection filter on small detections is the planned remedy.

`scripts/classifier/` already has the scaffold (EfficientNet-B0 / ResNet-18, reviewed-decision label map, area-cutoff inference gate), but no backbone comparison has been run. This experiment freezes the candidate set, success criteria, and promotion rule before any training begins, so the decision is reproducible and future `DINOv3` or remote-sensing foundation-model additions attach to a stable evaluation harness.

## Candidate Set

**Mainline (this round)**

| Backbone | Role | Training mode | Notes |
|---|---|---|---|
| `efficientnet_b0` | Default baseline | full FT (frozen warmup → unfreeze) | already supported in `train_cls.py` |
| `resnet18` | Control baseline | full FT | already supported |
| `convnext_tiny` | CNN challenger | full FT | new, via `torchvision.models.convnext_tiny` |
| `dinov2_vits14` | FM challenger | full FT (frozen warmup → unfreeze, layer-wise LR decay 0.75) | new, via `timm` or `torch.hub` |

**Conditional follow-up (not in this round)**

| Backbone | Trigger |
|---|---|
| `dinov2_vitb14` + LoRA r=8–16 | If `dinov2_vits14` yields ≥1pp installation-level precision gain over `efficientnet_b0` with recall drop ≤3pp |
| `dinov2_vitl14` linear probe | Re-evaluate after ViT-B/14 outcome |
| `DINOv3` | After DINOv2 path stabilizes |

**Rationale for capacity scaling**: labeled pool is ~5.1K PV / ~1.9K non-PV small-detection chips (area <30 m²; see `results/analysis/classifier_data_inventory/<run_id>/summary.{json,md}`). Full-FT ViT-S/14 (22M params) sits within safe overfit margin under strong augmentation + region-stratified grid split. ViT-B/14 (86M) requires parameter-efficient fine-tuning at this scale; ViT-L/14 (300M+) should be linear-probe only.

## Success Criteria

**Primary**

- **Installation-level precision gain** on reviewed grids after classifier filtering, via `detect_and_evaluate.py --classifier-filtered-gpkg ... --evaluation-profile installation`
- **Installation-level recall drop ≤ 3pp** (user-locked budget)
- Both criteria must hold on **CT (batch003 + batch004)** and **JHB Sandton** independently

**Secondary**

- Chip-level balanced accuracy on held-out val split
- Class precision / recall (minority class = non-PV / HN)
- Removal accuracy on reviewed `delete` decisions (how often the classifier correctly flags a human-confirmed FP)
- HN subclass breakdown: heater / skylight / shadow / pergola (labels from `results/analysis/small_fp/taxonomy_run/small_fp_taxonomy_labeled.csv`)
- Throughput: chips/sec on RTX 5090

## Promotion Rule

A candidate backbone is promoted to default only if:

1. It beats `efficientnet_b0` on installation-level precision **in both CT and JHB Sandton** (not just one region).
2. Installation-level recall drop vs. pre-filter baseline is **≤ 3pp** in both regions.
3. Runtime penalty is acceptable (<5× chips/sec vs EfficientNet-B0 is acceptable since classifier only touches chips below `area_cutoff_m2 = 30`).

If no candidate meets the rule, `efficientnet_b0` remains default.

## Evaluation Scope

- **Mainline evaluation**: Cape Town batch003 + batch004 (aerial_2025) + Johannesburg Sandton (v4_aerial_2023)
- **Exploratory column (non-binding)**: JHB CBD GEID G1110 — only 1 reviewed grid; results are reported as a domain-shift signal but do **not** enter the promotion rule. Statistical strength is insufficient for mainline decisions until more GEID grids are reviewed.

## Data Protocol

Locked parameters:

- `area_cutoff_m2 = 30` (classifier applied only to detections below this area)
- Crop policy: `400 → 224` (extract 400×400 around detection center, resize to 224 for backbone input)
- Split: **region-stratified whole-grid** 80/20 across {CT_batch003, CT_batch004, JHB_sandton}; no grid appears in both train and val
- Label map: `correct` / `edit` → PV (positive); `delete` → non-PV (negative)
- External HN augmentation: merge `gt_heater_audit/*/audit_labels_phase1.csv` + `small_fp/taxonomy_run/small_fp_taxonomy_labeled.csv` (dedup by grid_id + pred_id / geometric proximity)

Augmentation profile is decided by a sub-ablation in Task 4.5 (`exp_cls_augmentation_ablation.md`); Task 3 and Task 4 use the winning profile.

## Classifier → Detector Integration Contract

Decoupled, not fused. Contract specified in `exp_cls_detector_integration.md`:

1. `classify_predictions.py` reads `predictions_metric.gpkg` → writes `predictions_metric_filtered.gpkg` (parallel artifact, not overwrite).
2. `detect_and_evaluate.py --classifier-filtered-gpkg <path> --evaluation-profile installation` reads the filtered GPKG and runs installation-level evaluation.
3. `config.json` in results records `classifier_model_path`, `classifier_threshold`, `filtered_gpkg` for cache-key provenance.
4. `load_postproc_config` semantics are **not** changed; classifier filtering is an external pre-evaluation step.

## Deliverables

- `results/analysis/classifier_backbones/<run_id>/summary.{csv,md}` — per-backbone per-region metric table + winner
- `configs/classifier/default.json` — promoted default configuration
- `docs/workflows.md` — classifier training + inference section
- `docs/experiments/exp_cls_augmentation_ablation.md` — aug profile decision
- `docs/experiments/exp_cls_dataset_protocol.md` — dataset manifest protocol
- `docs/experiments/exp_cls_detector_integration.md` — integration contract

## Risk Annotations

- **GEID domain shift**: SSIM ≈ 0.21 between JHB CBD GEID and Sandton aerial; classifier trained on aerial may degrade on GEID. Tracked as exploratory, not blocking.
- **Class imbalance 2.6:1**: `train_cls.py` already has a balanced sampler; focal loss is a fallback if ViT minority recall underperforms.
- **Capacity-mode fairness**: Reported metrics will label each backbone's training mode (full_ft / lora / linear_probe) so the comparison is transparent.
- **Dataset builder legacy bug**: `build_cls_dataset.py` originally scanned only flat `results/G*/review/`, missing CT batch004 and JHB. Fix is registry-driven, not glob-additive.
