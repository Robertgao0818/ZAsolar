# Annotation Semantics Rules

## GT authoritative semantics = installation-level

The project's formal ground-truth definition is **installation-level footprint
segmentation** (see `data/annotations/ANNOTATION_SPEC.md`). One GT polygon
represents the footprint of one solar installation on a single roof.

Do not degrade or redefine GT semantics to panel-level unless the user
explicitly requests it with full awareness of the implications.

## Two-Axis Model governs annotation quality

All annotations are classified on two axes (see ANNOTATION_SPEC.md):

- **Axis A (Semantic Conformance)**: A1 (installation-spec compliant) →
  A2 (mostly installation-like) → A3 (weak/fragmentary/noisy)
- **Axis B (Provenance)**: H (human) / R (reviewed prediction) /
  S (SAM-refined) / G (legacy)

Key rule: **T1 (gold evaluation GT) requires A1.** Only annotations where
a human has explicitly confirmed conformance to installation merge/boundary
rules qualify as T1. Do not auto-promote annotations to T1 based solely on
their label source.

## No label source automatically equals gold GT

- **Reviewed predictions (R) and SAM-refined (S)** are model-initiated — high-value
  training supervision but A2 by default, not gold GT.
- **Human-initiated SAM-assisted (H)** annotations are human-initiated but SAM as
  a drawing tool does not guarantee installation-spec conformance — also A2 by default.
- Using SAM as a tool (H) is fundamentally different from SAM-refining a model
  proposal (S). See ANNOTATION_SPEC.md "SAM-as-tool vs SAM-derived" section.
- All new annotations start as T2. Only upgrade to T1 after explicit human review
  against installation merge/boundary rules.

## `label_source` enumeration

Allowed values in `annotation_manifest.csv`:
- `human_manual` — pure freehand, no SAM assistance
- `human_manual_sam_assisted` — human-initiated, SAM as drawing tool (QGIS + GeoSAM)
- `reviewed_prediction` — model prediction accepted as-is after human review
- `sam_refined_review` — model prediction re-segmented with SAM after review
- `legacy_weak_supervision` — early Google Earth / weak-supervision

## Evaluation profile semantics

The `installation` evaluation profile in `detect_and_evaluate.py` performs
**pred-side many-to-one merge matching**: multiple predictions overlapping
one GT installation are unioned before computing IoU.

This is NOT GT-side clustering. Do not write or imply that "evaluation
automatically clusters panel-level GT into installation-level GT."

## COCO export must declare region scope

When creating or modifying COCO dataset exports (`export_coco_dataset.py`),
the region scope must be explicit. Currently the exporter only supports
Cape Town (`data/annotations/Capetown/`). If modifying the exporter, either:
1. Add an explicit `--regions` parameter, or
2. Document the single-region limitation in the script header

Do not silently expand or narrow the annotation discovery scope.

## `quality_tier` vs provenance fields

- `quality_tier` (T1/T2) is the **executable layer** — used by training
  pipelines, evaluation suites, and benchmark configs to select data.
- `label_source` and `semantic_confidence` are the **explanation/provenance
  layer** — they record WHY an annotation has its tier.
- Tier decisions are made by combining provenance + human review, never by
  `label_source` alone.
