# Handoff — JHB two-stage Gemini FP-review → production

**Date:** 2026-05-31
**Branch:** `feat/gemini-two-stage-skylight-review` (commit `c14c823`)
**Scope of this handoff:** take the validated two-stage skylight FP-review into the JHB
production inventory flow. Cape Town is explicitly **out of scope** (see Blockers).
**Related memory:** `project_gemini_two_stage_prelaunch`, `project_gemini_fp_review_calibration`,
`project_gemini_scorer_concurrency`, `project_jhb_production_model_2026-05-14`.

---

## 1. What this is

Gemini replaces the human RA as the detector false-positive suppressor. The reviewer is now a
**two-stage** pass:

- **Stage 1** (`scripts/analysis/gemini_fp_review_multiscale.py`) — sends a **tight 20 m** crop
  (module texture) + a **wide 48 m** crop (roof context) of each detection in one call →
  `{label pv|not_pv, confidence, lookalike_type, reason}`.
- **Stage 2** (`scripts/analysis/gemini_fp_review_two_stage.py`) — re-reviews **only** stage-1
  `not_pv ∧ lookalike_type=skylight` rows with a TP-protective skylight prompt, then merges into a
  production record with explicit fields: `production_action` (keep|drop|review),
  `production_decision_source`, `auto_drop`, `requires_human_review`, and preserved `stage1_*` /
  `stage2_*`.
- **Fail-closed:** stage-2 missing / abstained / unusable ⇒ `production_action=review`,
  `auto_drop=false`, `pv_present=null`. The pipeline never auto-drops on the safety path.

`auto_drop` is the **authoritative deletion flag**. Only rows with `auto_drop=true` are removed from
the inventory; `keep` and `review` rows are retained (review rows additionally go to a human queue).

---

## 2. Prelaunch validation (2026-05-31, verified)

909 real Gemini calls @ `--workers 10 --qps 4`. Numbers re-checked against the eval JSONs on disk
and the fail-closed checker re-run independently.

| Gate criterion | Verdict | Decisive number |
|---|---|---|
| HARD-1 Fail-closed = 0 violations | **PASS** | 4 outputs (141+141+429+286) all 0 violations; negative control → exit 1; pytest green |
| HARD-2 TP-protect (pv_recall) | **PASS (JHB) / FAIL (CT)** | JHB Vexcel two_stage **0.936**; CT 0.852 (−8.4pp, below 0.926 bar) |
| HARD-3 FP-cut (nonpv_recall) | **PASS (JHB baseline)** | JHB Vexcel two_stage **0.851**; CT 0.609 / low-conf 0.888 reported separately, not pooled |
| SOFT-4 Stage-2 overfit | **PASS** | JHB flips 2/141, drop precision 0.854; not systematically restoring FPs |
| SOFT-5 Concurrency soak | **PASS** | usable_rate 1.0000 (909/909), retry 0.0033 all-recovered, p95 15.1s / p99 23.0s, no 503 cluster |

**Decision:** greenlight two-stage as the **skylight-review default for JHB Vexcel conf ≥ 0.95**.
Run 1–2 full batches before any decision to replace the legacy single-stage Flash-cut logic.

---

## 3. Production runbook (the validated review step)

The two-stage **review mechanism** is production-ready. Confirmed env / flags:

- Run from repo root with `.venv/bin/python`; set `PYTHONPATH=/home/gaosh/projects/ZAsolar` and
  `SOLAR_TILES_ROOT=/home/gaosh/zasolar_data/tiles` inline (shell state does not persist).
- Gemini creds: `/home/gaosh/projects/solar_backdating/.env.gemini.local`, model
  `gemini-3-flash-agent`. Concurrency budget = ~30-slot shared account pool; **keep Σ(workers) ≤ 30**
  across simultaneous runs. `--workers 10 --qps 4` validated; do not run `score_target_sequence`
  alongside (it stalls workers).
- Tight crop `--chip-size-m 20`, wide crop `--chip-size-m 48`, `--output-px 768`,
  `--max-targets-per-chip 1`. JHB render CRS default `EPSG:32735` is correct (do NOT override).

**Per-grid (or per-batch) review:**

```bash
# 1. render tight + wide chips from the production candidate manifest (see §4 for how to build it)
PYTHONPATH=$PWD SOLAR_TILES_ROOT=/home/gaosh/zasolar_data/tiles .venv/bin/python \
  scripts/training/build_gemini_detection_review_chips.py \
  --candidate-manifest <prod_candidate_manifest.csv> --output-dir <chips_z20> \
  --chip-size-m 20 --search-radius-m 4 --output-px 768 --max-targets-per-chip 1 --chip-prefix prod_z20
PYTHONPATH=$PWD SOLAR_TILES_ROOT=/home/gaosh/zasolar_data/tiles .venv/bin/python \
  scripts/training/build_gemini_detection_review_chips.py \
  --candidate-manifest <prod_candidate_manifest.csv> --output-dir <chips_z48> \
  --chip-size-m 48 --search-radius-m 5 --output-px 768 --max-targets-per-chip 1 --chip-prefix prod_z48

# 2. stage 1 (multiscale dual-crop)
.venv/bin/python scripts/analysis/gemini_fp_review_multiscale.py \
  --tight-chips-csv <chips_z20>/chip_targets.csv --wide-chips-csv <chips_z48>/chip_targets.csv \
  --output <stage1.jsonl> --summary <stage1_summary.json> \
  --model gemini-3-flash-agent --workers 10 --qps 4
# GUARD: if summary abstain_rate > 0.30, STOP (env/gateway broken) — do not proceed.

# 3. stage 2 router + fail-closed merge → the production decision JSONL
.venv/bin/python scripts/analysis/gemini_fp_review_two_stage.py \
  --stage1-jsonl <stage1.jsonl> \
  --tight-chips-csv <chips_z20>/chip_targets.csv --wide-chips-csv <chips_z48>/chip_targets.csv \
  --output <two_stage.jsonl> --stage2-jsonl <two_stage_stage2.jsonl> --summary <two_stage_summary.json> \
  --model gemini-3-flash-agent --workers 10 --qps 4

# 4. HARD GATE — must pass before applying any drop. Exits non-zero on any violation.
.venv/bin/python scripts/analysis/check_two_stage_failclosed.py <two_stage.jsonl>
```

**Confidence routing (required):** Stage-2 skylight pass is net-positive only at high confidence.
Route by detector confidence:
- **conf ≥ 0.95 → two-stage** (the validated path above).
- **conf < 0.95 → stage-1 only** (multiscale `gemini_fp_review_multiscale.py`, no stage-2). In the
  0.5–0.95 band stage-2 was net-negative (−4.9pp FP-cut for +1 TP); use stage-1's
  `pv_present`/`label` directly there. Stage-1 on that band already cleared both bars
  (nonpv_recall 0.937 / pv_recall 0.944).

---

## 4. Production-wiring — end-to-end (BUILT 2026-05-31)

The full inventory seam is now in place and validated on the 25-grid JHB Vexcel sample. End-to-end:

```
raw predictions_metric.gpkg
  └─(1) build_gemini_review_production_manifest.py  → prod candidate manifest (conf>=0.95)
       └─ build_gemini_detection_review_chips.py    → tight z20 + wide z48 chips
            └─ gemini_fp_review_multiscale.py        → stage1.jsonl   (conf<0.95: stop here, stage-1 only)
                 └─ gemini_fp_review_two_stage.py     → two_stage.jsonl (production decisions)
                      └─ check_two_stage_failclosed.py  (HARD gate, exit 0 required)
                           └─(2) apply_two_stage_decisions.py → <grid>_filtered.gpkg + review_queue.csv
                                └─ detect_and_evaluate.py --classifier-filtered-gpkg <grid>_filtered.gpkg
```

1. **Production candidate manifest** — `scripts/analysis/build_gemini_review_production_manifest.py`
   (DONE). Scans raw `predictions_metric.gpkg` (no RA labels), filters a conf band, emits a fully
   renderable manifest; `pred_id` = positional row index (iloc), `candidate_id = {grid}_pred{idx:06d}`.
   Validated: 335 conf≥0.95 candidates across the 25 JHB grids (of 667 total), 0 tiles missing.
   ```bash
   .venv/bin/python scripts/analysis/build_gemini_review_production_manifest.py \
     --predictions-glob 'results/johannesburg/v3c_vexcel_2024_ch1_sample/G*/predictions_metric.gpkg' \
     --region johannesburg --imagery-layer vexcel_2024 --min-conf 0.95 \
     --out-csv <prod_manifest_conf095.csv>
   # low-conf path: --min-conf 0.5 --max-conf 0.95 into a SEPARATE manifest (stage-1 only)
   ```

2. **Apply-decisions-to-gpkg** — `scripts/analysis/apply_two_stage_decisions.py` (DONE). Joins the
   merged decision JSONL(s) to the predictions gpkg by `(predictions_path, pred_id)`, drops only
   `auto_drop=true`, writes a same-schema/CRS row-subset `<grid>_filtered.gpkg` per grid (directly
   consumable by `detect_and_evaluate.py --classifier-filtered-gpkg`). Fail-closed: keeps `keep` +
   `review` + undecided rows, emits a review queue, aborts on `auto_drop=true ∧ action≠drop`
   integrity violations, resolves cross-file conflicts conservatively (non-drop wins). Validated:
   exact row removal (G0890 70→66 = its 4 auto_drops; 46 total = JSONL auto_drop count).
   ```bash
   .venv/bin/python scripts/analysis/apply_two_stage_decisions.py \
     --decisions <two_stage.jsonl> [<stage1_lowconf.jsonl> ...] --out-dir <filtered_dir>
   # then per grid:
   python detect_and_evaluate.py --grid-id <G> --classifier-filtered-gpkg <filtered_dir>/<G>_filtered.gpkg ...
   ```

3. **Human-review queue** — `apply_two_stage_decisions.py` auto-emits `review_queue.csv`
   (`production_action=review` / `requires_human_review=true` rows, with chip paths). Remaining: wire
   that CSV into a human surface (`build_stage2_flip_audit.py` HTML, or the QGIS/Li review GUI). In
   prelaunch `review`=0, but the fail-closed path *will* fire in production. These rows stay in the
   inventory until a human rules. **(queue produced; review UI still TODO.)**

4. **Per-batch acceptance = re-run the gate (§5).** `check_two_stage_failclosed.py` must gate every
   batch's decision JSONL (exit 0) before `apply_two_stage_decisions.py` runs; the applier also
   re-asserts the drop/action integrity invariant and aborts on violation. Remaining: wrap the
   soak-metric spot-check into the batch runner. **(fail-closed gating wired; soak aggregation still
   manual.)**

---

## 5. Per-batch acceptance gate (ongoing)

Every production batch must clear, before its drops are applied:

- **Fail-closed = 0 violations** — `check_two_stage_failclosed.py <batch>.jsonl` exits 0. (HARD)
- **Soak health** — usable_rate ≥ 0.97 and no 503 / account-exhausted cluster in the
  `error_type` histogram (PATCH1 logs `latency_ms` / `retry_count` / `error_type` per row; aggregate
  them). p95 latency in the validated band (~15 s). (HARD-ish — investigate before trusting drops.)
- **Drop sanity** — drop precision tracked via spot audit (`build_stage2_flip_audit.py`); flips rare
  and principled. Baseline ref: drop precision 0.854, flips ≈ 1.4 %. (SOFT)

---

## 6. Blockers / known issues / follow-ups

- **CT is RED — do not deploy this config to Cape Town.** Cross-imagery-source gap: two_stage
  FP-cut 0.609 / TP-protect 0.852 on CT aerial_2025; neither bar (0.872 / 0.926) cleared. Needs
  CT-specific recalibration before a separate CT gate. (Out of scope for JHB production.)
- **G1688 source-data CRS defect.** `results/G1688/review/G1688_reviewed.gpkg` has `CRS=None` +
  mixed UTM/lon-lat coords; 85/514 CT candidates could not render. Naive `set_crs(32734)` would
  misplace the genuinely-4326 features — needs an upstream fix. (CT-only; blocks completing the CT
  eval, not JHB.)
- **Low-conf stage-2 OFF.** Enforce the §3 confidence routing; do not run stage-2 on conf < 0.95.
- **G0890 over-keep follow-up (ROADMAP).** The one stable −1 FP in JHB conf≥0.95 is a genuine
  skylight embedded in a PV array that stage-2 wrongly restores. Fix as a **geometry/post-proc guard**
  (drop a `stage2_skylight_keep` flip whose footprint is largely covered by other confirmed-PV
  polygons on the same roof), **not** by hardening the stage-2 adjacency prompt (prompt-hardening
  backfires on TP recall). Logged in `ROADMAP.md` → Next Up.
- **Unaudited flips.** CT (31 flips / 80 drops) and the low-conf band (8 flips / 75 drops) were not
  put through `build_stage2_flip_audit.py` (T3 audited JHB conf≥0.95 only). Audit before relying on
  drops from those populations.

---

## 7. Decision rule — replacing the legacy single-stage Flash-cut logic

Keep the old single-stage "Flash directly cuts TP" logic as the **fallback**. Promote two-stage from
"skylight-review default" to **full replacement** only after 1–2 full JHB Vexcel batches confirm, in
production, that: (a) HARD-1/2/3 hold on JHB conf≥0.95 within tolerance, (b) fail-closed = 0
violations on every batch, (c) soak usable_rate / p95 / error histogram match prelaunch, (d) drop
precision holds (~0.854). CT recalibration and the low-conf stage-1-only route must also be landed
before any talk of a global cutover.

---

## 8. File map

| Path | Role | State |
|---|---|---|
| `scripts/analysis/gemini_fp_review_multiscale.py` | Stage-1 dual-crop scorer (+ soak instrumentation) | committed |
| `scripts/analysis/gemini_fp_review_two_stage.py` | Stage-2 router + fail-closed merge | committed |
| `scripts/analysis/check_two_stage_failclosed.py` | Data-level fail-closed gate (CI/preflight) | committed |
| `scripts/analysis/build_stage2_flip_audit.py` | Stage-2 flip/drop audit queue (CSV+HTML) | committed |
| `scripts/analysis/gemini_fp_review.py` | Single-crop scorer (routing-salt + retry backoff) | committed |
| `scripts/analysis/eval_gemini_review_vs_ra.py` | RA-vs-Gemini scorer (**calibration only**, needs RA labels) | pre-existing |
| `scripts/analysis/build_gemini_review_calibration_manifest.py` | RA-labeled calib manifest builder | pre-existing |
| `scripts/analysis/build_gemini_review_production_manifest.py` | **production** candidate manifest from raw predictions (§4.1) | committed |
| `scripts/analysis/apply_two_stage_decisions.py` | apply auto_drop → filtered gpkg + review queue (§4.2) | committed |
| `tests/analysis/test_gemini_fp_review_two_stage.py`, `test_two_stage_failclosed_data.py`, `test_apply_two_stage_decisions.py` | unit + data-guard | committed |
| `data/analysis/gemini_review_calib/**` | prelaunch artifacts (gitignored) | local only |
| **(remaining)** human review-queue UI; soak-metric step wired into batch runner | §4.3 / §4.4 | **TODO** |
