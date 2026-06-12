# C-3(a) Phase 0 runbook — unlabeled-real-PV-as-background measurement

> Plan: [`docs/plans/2026-06-10-rcnn-f1-gap-review.md`](../plans/2026-06-10-rcnn-f1-gap-review.md)
> Tier C, lever **C-3(a)** (RPN/box-cls ignore supervision), Phase 0 (lines 205-210).
> **Gate: >= 5 % of sampled chips affected => C-3(a) proceeds; < 5 % => C-3(a) killed.**
> Status: tooling landed 2026-06-11 (this repo, no GPU). Scan + audit + gate run tomorrow on the pod.

## What Phase 0 measures

The production training pool (`unified_reviewall_v2`) supervises every pixel that
is **not** inside an existing GT polygon as background. If real PV installations
sit unlabeled in those background regions, the detector is being taught to
*suppress* real PV. Phase 0 quantifies the fraction of training chips where this
happens, by running the production detector at a low score threshold over the
sampled chips and adjudicating the background-region proposals.

The audit output is itself the future ignore corpus (only the `ignore_candidate`
rows — see disposition routing below).

## Tool chain (4 scripts + 1 shared module)

| Step | Script | GPU? | Output |
|---|---|---|---|
| Shared primitives | `core/training/c3a_phase0.py` | no | (imported) |
| A. Sampler | `scripts/training/sample_c3a_phase0_chips.py` | no | `chip_manifest.csv`, `gt_refs__*.gpkg`, `sample_meta.json` |
| B. Scan runner | `scripts/training/run_c3a_phase0_scan.py` | **detect phase only** | `raw_scans/.../raw_detections.pkl`, `proposals.csv`, `chips/rgb/*.png` |
| C. Audit builder | `scripts/analysis/build_c3a_phase0_audit.py` | no | `audit.csv`, `c3a_phase0_labeler.html` |
| D. Gate calculator | `scripts/analysis/compute_c3a_phase0_gate.py` | no | `gate_result.json` + verdict |

Audit unit = one **training chip window** (400 px, overlap 0.25), identical to the
windows `export_coco_dataset.scan_chips_from_tile` enumerates when the COCO set is
built. A chip is **affected** iff >= 1 of its background-region proposals is
adjudicated `confirmed_pv`.

## Disposition routing (audit decision -> sink)

| audit_label | downstream sink | counts as affected? |
|---|---|---|
| `confirmed_pv` | promote to **positive** (NEVER ignore) | **yes** |
| `lookalike` (skylight / water heater) | `data/negative_pool/` (HN) — **never ignore** | no |
| `ignore_candidate` | unreviewed-margin **ignore** region (per-chip area cap) | no |
| `not_pv_other` | bare roof / shadow / road — neither | no |
| `uncertain` | abstain — does NOT count toward decided or affected | no |

Rationale (plan line 208): confirmed real PV is always promoted to positive
(matches the `sam_added_true_fn` precedent); lookalikes go to the negative pool
because removing HN in a precision-bottlenecked project violates the breadth
constraint; only genuinely unreviewed margins become ignore regions, each with a
per-chip ignore-area cap.

---

## Run order (tomorrow, pod)

Set a run id, e.g. `RUN=2026-06-12_c3a_phase0` and
`RUN_DIR=results/analysis/c3a_phase0/$RUN`.

### 0. Pod prep (see `.claude/skills/runpod-ops` / rules 05, 08)

- Confirm `/workspace` has both strata's tiles. The sampler needs tile **headers**
  (CRS + width/height) for every grid:
  - CT `aerial_2025`: `~/zasolar_data/tiles/cape_town/aerial_2025/<grid>/`
  - JHB `vexcel_2024`: `~/zasolar_data/tiles/johannesburg/vexcel_2024/<grid>/`
  The detect phase needs the **pixels** too — copy hot tiles to `/dev/shm` and
  `export SOLAR_TILES_ROOT=/dev/shm/tiles` (rule 05) for the detect phase only.
- Confirm the checkpoint: `checkpoints/exp_unified_reviewall_A/best_model.pth`
  (model_registry id `exp_unified_reviewall_A`).

### 1. Sample 150-200 chips (CPU — run ON THE POD for full stratification)

```bash
python scripts/training/sample_c3a_phase0_chips.py \
  --spec configs/pipelines/datasets/unified_reviewall_v2.yaml \
  --target 180 --seed 42 \
  --out-dir $RUN_DIR
```

The sampler replays `unified_reviewall_v2`'s positive-source loaders (CPU, no
model) and stratifies by `region:imagery_layer`. Both strata
(`cape_town:aerial_2025`, `johannesburg:vexcel_2024`) must appear in
`sample_meta.json::realized_strata` with `missing_strata_no_local_tiles == []`.

> Local-machine note: this repo's checkout has CT tiles only; running the sampler
> locally yields a **CT-only** sample and flags `johannesburg:vexcel_2024` under
> `missing_strata_no_local_tiles`. Do NOT draw per-stratum gate conclusions from a
> CT-only sample — run the sampler where the JHB Vexcel tiles live.

### 2. Low-conf scan — detect phase (GPU)

```bash
python scripts/training/run_c3a_phase0_scan.py --phase detect \
  --run-dir $RUN_DIR \
  --model-path checkpoints/exp_unified_reviewall_A/best_model.pth \
  --model-run exp_unified_reviewall_A \
  --score-threshold 0.05
```

This is a thin subprocess wrapper around `detect_direct.py` — one call per
distinct `(region, imagery_layer, grid)` in the manifest, with
`--detector-score-threshold 0.05` (the only GPU step). Re-runs skip grids whose
`raw_detections.pkl` already exists (use `--force` to redo). Use `--dry-run` to
print the commands without launching the GPU.

### 3. Low-conf scan — extract phase (CPU)

```bash
python scripts/training/run_c3a_phase0_scan.py --phase extract \
  --run-dir $RUN_DIR --gt-iof-threshold 0.10
```

Reads the artifacts + `gt_refs__*.gpkg`, keeps detections whose
intersection-over-foreground vs every existing GT is below `--gt-iof-threshold`
(= background-region proposals), computes metric area, renders chip RGB PNGs, and
writes `proposals.csv`.

(`--phase both` runs detect then extract in one invocation.)

### 4. Build the audit package (CPU)

```bash
python scripts/analysis/build_c3a_phase0_audit.py --run-dir $RUN_DIR
```

Writes `audit.csv` (schema = `core.training.c3a_phase0.AUDIT_CSV_COLUMNS`) and a
self-contained `c3a_phase0_labeler.html` (chip RGB + green GT boxes + red proposal
box). Pull the HTML to a workstation and open it (Windows path printed by the
script). Keys: `1`=confirmed_pv `2`=lookalike `3`=ignore_candidate (prompts for an
ignore-area cap) `4`=not_pv_other `5`=uncertain `S`=skip `B`=back. Click
**导出 audit.csv** when done and overwrite `$RUN_DIR/audit.csv`.

> Gemini alternative: the same `audit.csv` + chip PNGs can be fed to a Gemini
> reviewer (mirror the FP-review harness in `solar_cls` / project memory
> `project_gemini_fp_review_calibration`). Write the chosen label into the
> `audit_label` column; the gate calculator is agnostic to who labeled.

### 5. Compute the gate (CPU)

```bash
python scripts/analysis/compute_c3a_phase0_gate.py \
  --audit-csv $RUN_DIR/audit.csv \
  --threshold 0.05
```

The denominator is **all sampled chips** (`chip_manifest.csv`, auto-discovered
next to `audit.csv`), so chips that produced zero background proposals correctly
count as 0-affected. Output:

- `affected rate` and the `PASS` / `KILL` / `INSUFFICIENT_DATA` decision
- per-stratum breakdown (CT vs JHB)
- proposal disposition tallies
- `gate_result.json`

`--strict` makes the process exit non-zero on KILL/INSUFFICIENT_DATA (for CI).

---

## Decision

- **PASS** (affected >= 5 %): proceed to C-3(a) proper — RPN/box-cls ignore
  implementation at the two patch points (`RPN.assign_targets_to_anchors` +
  `RoIHeads.assign_targets_to_proposals`, label=-1; plan line 209). The
  `ignore_candidate` rows become the ignore corpus (per-chip area cap honored).
  The `confirmed_pv` rows are promoted to positive; `lookalike` rows go to
  `data/negative_pool/` (a separate, append-only ingestion — do NOT delete or
  rewrite existing pool rows).
- **KILL** (affected < 5 %): C-3(a) is killed; record the negative result. The
  `confirmed_pv` rows are still worth promoting to positive (free recall signal),
  and `lookalike` rows still worth adding to the negative pool — but no ignore
  supervision work.
- **INSUFFICIENT_DATA**: finish labeling more chips and re-run the gate.

## Constraints honored by the tooling

- No GPU except the detect phase (step 2). The sampler, extract, audit builder,
  and gate are CPU-only.
- No region/CRS/EPSG hardcoding — paths resolve via `core.grid_utils` +
  `core.region_registry`; metric area uses `get_metric_crs(grid, region=)`;
  region never inferred from grid id.
- `configs/pipelines/datasets/unified_reviewall_v2.yaml` is read-only (provenance
  of the trained set). The sampler only reads it.
- `data/negative_pool/` is append-only: `lookalike` ingestion adds rows, never
  edits/removes existing ones (that ingestion is a separate downstream step, not
  part of this Phase 0 harness).
- No automatic tier promotion; `label_source` enum unchanged.

## Files

- `core/training/c3a_phase0.py`
- `scripts/training/sample_c3a_phase0_chips.py`
- `scripts/training/run_c3a_phase0_scan.py`
- `scripts/analysis/build_c3a_phase0_audit.py`
- `scripts/analysis/compute_c3a_phase0_gate.py`
- `tests/training/test_c3a_phase0.py`
