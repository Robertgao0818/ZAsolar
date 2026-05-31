# Annotation Specification — Installation Footprint

**Version**: 1.2 (V1.3)
**Effective date**: 2026-04-12
**Label definition**: `installation_footprint`

## Scope

This specification defines the **ground-truth annotation standard** for the solar detection project. All new annotations and all evaluation must conform to this spec. Historical annotations are retroactively classified under this spec with quality tiers.

> **V1.3 workflow note**: The pipeline task definition is "reviewed prediction footprint segmentation." Pipeline output is model predictions reviewed by humans (`batch_finalize_reviews.py` exports `review_status==correct` predictions to `cleaned/`), not installation-merged footprints. However, **this annotation spec is unchanged** — ground-truth polygons still follow installation-level merge and boundary rules below. The `installation` evaluation profile compares reviewed predictions against installation-level GT using pred-side many-to-one merge matching.

## Definition

One annotation polygon represents the **footprint of one solar installation on a single roof**.

- An "installation" is a set of solar panels that form a single, physically connected or near-connected array on one roof surface.
- The polygon boundary traces the **outer envelope of the installation**, not individual panel boundaries.
- Minor gaps between panels within the same installation (e.g., mounting rail spacing) are enclosed within the polygon.

## Rules

### Merge Rule
- Panels on the **same roof** that are physically connected or clearly part of the **same system** (contiguous cluster) → **merge into one polygon**.
- Panels on the **same roof** but belonging to **physically separate systems** (e.g., different roof faces, clear gap > ~1m) → **separate polygons**.

### Boundary Rule
- The polygon should trace the installation footprint as tightly as practical.
- Do NOT trace individual panel outlines — one polygon per installation.
- Do NOT extend the polygon to cover the entire roof or building footprint.
- Small overhangs or shading structures that obscure panel boundaries: use best visual estimate.

### Edge Cases
- **Ground-mounted panels**: Annotate if visible and within the grid extent. Same merge/boundary rules apply.
- **Solar water heaters**: Do NOT annotate. Only photovoltaic installations are in scope.
- **Partially obscured by trees**: Annotate the visible portion. If >50% obscured, skip.
- **Under construction / partially installed**: Annotate what is visibly present.

## Two-Axis Annotation Model

All annotations are classified on two independent axes. This conceptual model provides stable vocabulary for discussing annotation quality and governs tier assignment.

### Axis A: Semantic Conformance

How well does the polygon conform to the installation-level rules above?

| Grade | Name | Meaning |
|-------|------|---------|
| **A1** | installation-spec compliant | Human has explicitly confirmed the polygon satisfies merge and boundary rules. |
| **A2** | mostly installation-like | Polygon is roughly installation-shaped but may be model-shaped, have ambiguous merge decisions, or preserve inter-panel gaps. Not individually verified against the spec. |
| **A3** | weak / fragmentary / noisy | Polygon has known issues: wrong merge, severe geometric offset, covers entire roof, or is clearly panel-level / fragmentary. |

### Axis B: Label Source (Provenance)

How was the polygon created? The critical distinction is between **human-initiated** sources (H) and **model-initiated** sources (R, S).

| Code | Source | Description |
|------|--------|-------------|
| **H** | Human-initiated | Annotator draws the polygon from scratch, possibly using SAM2 as a segmentation tool (point-prompt → polygon). The human decides what to annotate and where; SAM is only a drawing aid. |
| **R** | Reviewed prediction | Model prediction marked as `correct` during human review, exported as-is. Geometry is model-generated; human only decides accept/reject. |
| **S** | SAM-refined review | Model prediction reviewed, then re-segmented via SAM point-prompt. Geometry is SAM-generated, but the annotation was initiated by a model proposal, not by the human. |
| **G** | Legacy weak annotation | Early Google Earth / weak-supervision annotations. Origin and quality uncertain. |

#### SAM-as-tool vs SAM-derived: a key boundary

Using SAM as a drawing tool (H) is fundamentally different from SAM-refining a model proposal (S):

- **H (SAM-assisted human)**: The annotator opens a blank map, identifies a solar installation by eye, clicks a point prompt, and SAM generates the polygon boundary. The human initiates and directs. This is analogous to using a magic wand tool in Photoshop — it's a tool, not a data source.
- **S (SAM-refined review)**: A model prediction already exists. The annotator reviews it, then uses SAM to re-cut a better boundary. The annotation was initiated by the model, not the human. The geometry may still be model-shaped.

Both use SAM, but the provenance chain is different:
- H: human eye → SAM tool → polygon
- S: model prediction → human review → SAM re-cut → polygon

The `label_source` field in the manifest captures this distinction:
- `human_manual` — pure freehand or non-SAM human annotation
- `human_manual_sam_assisted` — human-initiated, SAM as drawing tool (most Batch 002+ annotations)
- `reviewed_prediction` — model prediction accepted as-is after review
- `sam_refined_review` — model prediction re-segmented with SAM after review
- `legacy_weak_supervision` — early Google Earth / weak-supervision

### Typical Combinations

| Code | Description | Typical origin |
|------|-------------|----------------|
| A1-H | Human-drawn and individually verified against installation spec | Future gold GT; currently rare |
| A2-H | SAM2-assisted human annotation, not individually verified against spec | Current Batch 002+ Cape Town annotations; RA annotations via QGIS + GeoSAM |
| A2-R | Reviewed prediction exported as-is | `batch_finalize_reviews.py` output |
| A2-S | Reviewed prediction re-segmented with SAM | Joburg CBD batch1 (V4 review + SAM recut) |
| A3-G | Legacy Google Earth large bounding polygons | G1023, G1134, early weak-supervision |

### Key Principles

1. **Reviewed/SAM-derived labels (R, S) are high-value training supervision**, not automatic gold evaluation GT. They may exhibit:
   - Model-shaped geometry (prediction polygons, not installation envelopes)
   - Ambiguous merge decisions (adjacent panels with inter-panel gaps kept separate)
   - Boundaries that don't conform to installation merge/boundary rules

2. **Human-initiated SAM-assisted labels (H) are also not automatic gold GT.** Using SAM as a tool does not guarantee the annotator followed installation merge/boundary rules. Default classification is A2 (mostly installation-like).

3. **Gold GT (A1) requires explicit human verification** against installation merge/boundary rules, regardless of which tool was used to draw the polygon.

## Quality Tiers

| Tier | Semantic Conformance | Use |
|------|---------------------|-----|
| **T1** | **A1 only** — human has explicitly verified conformance to installation spec. Geometric accuracy sufficient for IoU >= 0.3 matching. | Validation/evaluation set; all evaluation conclusions must reference T1 data. |
| **T2** | **A2 or A3** — not individually verified against installation spec. Includes reviewed predictions, SAM-refined, legacy, and human annotations without explicit spec verification. | Training set (combined with T1). Not suitable as sole evaluation GT. |

### Tier Assignment

- All new annotations start as **T2** regardless of label source.
- Annotations are upgraded to **T1** only after a human has:
  1. Reviewed the polygon against the aerial/satellite imagery
  2. Confirmed it satisfies the merge rule (correct grouping)
  3. Confirmed it satisfies the boundary rule (tight installation envelope)
- The annotation manifest (`annotation_manifest.csv`) tracks each annotation's tier, label source, and semantic conformance.

### Relationship Between Layers

- `quality_tier` (T1/T2) is the **executable layer**: training pipelines, evaluation suites, and benchmark configs select data by tier.
- `label_source` and `semantic_confidence` are the **explanation/provenance layer**: they record why an annotation has its tier.
- Tier is determined by provenance + human review together. **Do not auto-promote** annotations to T1 based solely on label source.

## Historical Correction Policy

Existing annotations are NOT redrawn wholesale. Only fix these **three error types**:

1. **Area too large**: Polygon covers significant non-panel area (e.g., entire roof instead of just the installation).
2. **Wrong merge**: Two physically separate installations incorrectly merged into one polygon.
3. **Severe geometric offset**: Polygon position is shifted such that true IoU with the actual installation < 0.3.

All other imprecisions (slightly loose boundary, minor shape deviations) are accepted under T2 tier.

## Coordinate System

- Source annotations: EPSG:4326 (WGS84)
- Evaluation CRS: determined per region (see `configs/datasets/regions.yaml`)
  - Cape Town: EPSG:32734 (UTM 34S)
  - Johannesburg: EPSG:32735 (UTM 35S)
- All area calculations use the region-appropriate metric CRS.
- Use `core.grid_utils.get_metric_crs(grid_id, region=)` to look up the correct CRS programmatically.

## Onboarding New Annotation Sources

When integrating annotations from a new annotator or source, follow this procedure:

### 1. Determine label source classification

| Workflow | `label_source` | Axis B |
|----------|---------------|--------|
| Annotator draws from blank map, optionally using SAM point-prompt | `human_manual_sam_assisted` | H |
| Annotator draws purely freehand (no SAM) | `human_manual` | H |
| Model predictions reviewed, accepted as-is | `reviewed_prediction` | R |
| Model predictions reviewed, then SAM re-cut | `sam_refined_review` | S |
| Model predictions visually reviewed by Gemini/LLM, accepted as PV without human redraw | `gemini_reviewed_prediction` | R-like weak supervision |

Gemini/LLM review is an automation aid, not a gold annotation source. Treat
`gemini_reviewed_prediction` as T2 weak supervision by default: suitable for
backbone / RPN / box / class learning after confidence and overlap audits, but
not trusted for mask-boundary BCE. To become A1/T1 or mask-trusted H data, the
polygon must be human-reviewed against the installation spec or redrawn through
a human-initiated tool workflow.

### 2. Default provenance assignment

- `semantic_confidence`: **A2** (default for all new sources until individually verified)
- `quality_tier`: **T2** (default for all new annotations)
- Only upgrade to A1/T1 after explicit installation-spec review

### 3. File onboarding steps

1. **Copy** source GPKGs to the appropriate `data/annotations/{CityName}/` directory
2. **Rename** to project convention: `{GridID}_{source_tag}_{YYMMDD}.gpkg`
   - Example: `G1842_RA_SAM_260412.gpkg`
3. **Register** in `configs/datasets/regions.yaml` under the correct region's `grids:` section
4. **Add** manifest entries in `annotation_manifest.csv` with `label_source`, `semantic_confidence`, `quality_tier`
5. **Run** `python scripts/validate_registry.py` to verify cross-references

### Example: RA annotations (QGIS + GeoSAM)

Source: RA handoff GPKGs — human-initiated, SAM-assisted annotation in QGIS.

| Field | Value | Reason |
|-------|-------|--------|
| `label_source` | `human_manual_sam_assisted` | Human identifies installations; SAM generates boundaries |
| `semantic_confidence` | `A2` | Not individually verified against installation merge/boundary rules |
| `quality_tier` | `T2` | A2 → T2 by default |
| Axis code | A2-H | Human-initiated, SAM-assisted, not spec-verified |

These annotations are suitable for training. To use as evaluation GT, each polygon must be individually reviewed against the installation spec and upgraded to A1/T1.

## Category Naming

- New exports and documentation use `solar_installation`.
- Legacy COCO datasets and old checkpoints retain `solar_panel` (category_id=1 is unchanged).
- Both names map to the same model class; the category name is cosmetic.
