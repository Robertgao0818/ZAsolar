---
title: "ZAsolar — Validation & Robustness Methodology"
date: 2026-06-21
scope: "How the rooftop-solar census and its install-date layer defend against the three things that move a published number: a tuned threshold, an irreproducible pipeline, and a stochastic instrument."
---

# ZAsolar — Validation & Robustness Methodology

## Executive summary

A census figure is only as credible as it is robust to the choices and the noise
that produced it. This document audits the ZAsolar rooftop-solar inventory the way
an applied economist would audit an estimate — by asking, of every headline number,
*what could move it, and by how much.* Three questions, three answers:

| Robustness question | What it guards against | Finding |
|---|---|---|
| **1. Specification robustness** — is the headline an artifact of a tuned threshold? | researcher degrees of freedom / oracle leakage | The operating point is fitted **only on calibration grids disjoint from every reporting suite**, with a ≤1 pp transfer-acceptance gate. Where that gate *fails* (the production per-detection chain), reporting falls back to a **pre-declared, fail-closed** threshold rather than a swept optimum. The failure is recorded, not hidden. |
| **2. Computational replicability** — does the same input reproduce the same number? | silent drift between runs | Not seeded bit-reproduction of model weights, and we do not claim it. What is guaranteed: a single metric kernel, **config-match output reuse**, a **frozen** evaluation ground truth, and a **locked** merge-mode + operating point — so any number is re-derivable from, and traceable to, its source tiles and model run. |
| **3. Instrument reliability** — how much does the one stochastic device move the result? | LLM sampling noise in the install-date layer | The detector and post-processing are deterministic; the **only** stochastic step is the Gemini date-scorer. A test-retest (642 installations, 3 identical-input reruns) finds **individual** install-dates ~81 % reproducible at the production k = 1 setting; the number is confirmed to belong to the production model (re-scoring with `gemini-3-flash` is statistically identical, McNemar *p* ≈ 0.65) and is **not** a loose-decoder artifact (greedy `temperature:0` is already applied and does not reduce it). The actionable lever is a **majority-of-3 vote**, which a K = 6 certification shows lifts reproducibility to **~0.95** (inventory-weighted) at 3× cost. And the errors are near-mean-zero, so they **wash out under aggregation**: the published install-year cohort histogram reproduces to within **~1.7 % per year** (TVD 0.025) regardless. |

The honest one-line version: *individual estimates are noisier than they look, the
aggregate is more robust than the individual estimates, and the one place a rerun
could still move a number — the CT per-detection operating point — is flagged as
fail-closed-pending rather than certified.*

Two semantic anchors hold throughout. (i) The Cape Town evaluation ground truth is
**SAM2 sub-array (Axis-A "A2") annotation, not installation-level gold**, so
per-polygon F1 is demoted to a diagnostic and model selection runs on
area-aggregate dispersion (σ_Bw, RMSE). (ii) The target is an **inventory-level
area/count judgment**, not a per-polygon match.

---

## 1 · Specification robustness — the calibration is not a tuned constant

**The threat.** Sweeping a threshold on a reporting suite and then quoting the
swept number on that same suite is oracle leakage — the estimate flatters itself.
This is not hypothetical here: on the 2026-06-07 evaluation face the model ranking
*flipped* with the sweep caliber — at one fixed threshold v3c wins; at each model's
own best threshold unified_A wins. A threshold tuned on the evaluation set is
therefore not a defensible operating point.

**The control: a leakage-checked lock.** An operating point is fitted **only on a
calibration grid set disjoint from every reporting suite**, then validated for
transfer (`scripts/analysis/lock_operating_point.py`). Three guarantees are
enforced mechanically, not by convention:

- Calibration grids must not intersect any registered reporting suite — the script
  raises `[LEAKAGE]` on intersection (`lock_operating_point.py:283–290`).
- Acceptance = the locked point's `agg_area_F1` is within **1 pp** of the oracle
  sweep maximum **on every** validation suite, else the lock fails
  (`ACCEPT_GAP = 0.01`, gate at `:325–327`).
- Validation suites are read-only — used to *check* transfer, never to *fit*
  (`operating_point_calibration.yaml` rule 4).

Each fit records its calibration roster, ranking rule, per-suite transfer gap, an
`all_suites_pass` boolean, and the run's `git_head`.

**The honest result — the gate has teeth.** It has failed, and the failure is on
the production chain:

| lock | merge-mode | transfer gap | verdict |
|---|---|--:|---|
| `ct_aerial_2025_v3c` | pixel-or | **0.00 pp** | clean lock |
| `ct_aerial_2025_v3c_perdet` | per-detection | **5.22 pp** | **fails acceptance** |

The per-detection lock fails even after the whole calibration chain was re-run in
per-detection mode to remove the merge-mode confound — the gap stays at 5.22 pp.
The diagnosis is structural: on the per-detection chain the ranking rule (σ_Bw +
RMSE) prefers a high threshold while the acceptance metric (agg_F1) peaks lower
(t ≈ 0.925), because per-detection predictions carry no over-paint to trim, so the
two metrics diverge. **Disposition: fail-closed.** Production per-detection
reporting uses a *pre-declared* threshold from the 2026-06-07 trade-off analysis,
never a value swept on the reporting suite. The unified_A × CT lock that would
replace that pre-declared threshold with a certified one is registered with a
leakage-clean calibration set (53 Li-KML grids, ≥23.5 km from all training/reporting
cells) but is **not yet fitted** — its tiles are pending download. That gap is the
section's load-bearing caveat, stated rather than papered over.

**Two levers that an auditor must not conflate.**

1. *Polygon-confidence is calibrated; the pipeline's pixel `post_conf` is fixed.*
   The locked lever is a per-polygon `(confidence, area_m²)` cut applied at
   consumption time on `predictions_metric.gpkg` (`polygon_conf_sweep.py:54–57`),
   area-aware so the dominant false-positive archetype (small, low-confidence
   blobs) can be cut harder than large installations. It is **distinct** from the
   coarse fixed `post_conf_threshold = 0.85` baked into the finalizer
   (`core/postproc.py:1064–1066`). Conflating them mis-attributes the calibration.
   The census deliberately keeps the detector permissive — raising polygon-conf to
   the σ_Bw-optimal 0.97 would cost ~15 pp of count-recall (documented in the
   calibration appendix) — and delegates false-positive removal to the in-domain
   `solar_cls` classifier instead of the confidence knob.

2. *Merge-mode is a first-class, model-dependent lever.* `v4_canonical.json` (the
   cross-experiment comparability anchor) carries **seven** post-processing keys and
   **no `merge_mode` field** — it was stripped under the 2026-06-12
   architecture-optimization landing, and callers must now pass `--merge-mode`
   explicitly; `finalize.py` raises on a CLI-vs-JSON mismatch. This matters because
   the same model under the two modes posts materially different dispersion (on the
   wave-1 face unified_A is σ_Bw 0.270 / RMSE 395 m² per-detection vs 0.361 / 588 m²
   pixel-or), so pixel-or and per-detection are **separate locks with separate
   calibration runs**. Numbers are always reported at a stated merge-mode, never as
   a max-over-modes headline.

---

## 2 · Computational replicability — traceable, not bit-seeded

**What is *not* claimed.** The pipeline does not offer seeded bit-for-bit
reproduction of model outputs. Stating that plainly is part of the contract.

**What *is* guaranteed** is that any reported number can be re-derived from, and
traced back to, its inputs through four locked artifacts:

- **One metric kernel.** The executable definition of "model–ground-truth
  agreement" is a single pure, side-effect-free function, `summarize()` in
  `core/area_metrics.py:84` — extracted on 2026-06-12 as "byte-for-byte the same
  logic" so every caller computes identical statistics. It consumes set-theoretic
  union areas per grid (A = pred, B = GT, A∩B) and emits the Tier-1 suite:
  `agg_area_F1`, `bulk_pred_gt_ratio`, `std_ratio_Bw` (σ_Bw, the B-weighted
  dispersion that is the primary deploy judge), `std_logratio`, `rmse_m2`,
  `thru0_slope`, and `ols_r2`. The **only** seeded component anywhere is the
  bootstrap-CI (`seed=0`) — it governs the confidence intervals, not the model
  outputs. (Note: `cov50` is a CT-census coverage figure from a *separate* script,
  not emitted by this kernel — it is not attributed to the Tier-1 suite.)
- **Config-match output reuse.** Inference reuses a prior result only when its
  `results/<grid>/config.json` matches current code + parameters; `--force` is
  required to override. A number cannot silently come from a stale run with
  different settings.
- **A frozen evaluation ground truth.** Model selection on the JHB CBD face is
  locked to a frozen `clean_gt`; drifting the eval GT to Li GT or micro-T1 is
  barred. The comparison surface does not move under the models being compared.
- **A locked merge-mode + operating point** (§1), plus per-grid provenance fields
  (`imagery_layer_id`, `model_run_id`) that tie every number to specific source
  tiles and a specific inference batch.

So "reproducible" here means **trace-and-rederive under locked configuration**, not
deterministic replay. That is the honest and the useful claim: a second analyst,
given the same config and frozen GT, recomputes the same summary — and can point at
the exact tiles and run behind any cell.

---

## 3 · Instrument reliability — the one stochastic step, measured

The detector (Mask R-CNN), the post-processing, and the area metrics are all
deterministic given fixed weights. The install-date layer adds **one** stochastic
device: a Gemini vision model that reads a time-sequence of aerial chips per
installation and calls, for each date, whether PV is present — yielding a presence
*pattern* and a derived *install-date*. It is the only place in the pipeline where a
rerun on identical inputs could return a different answer. Note this is **not** a
loose-temperature artifact: the scorer already requests greedy `temperature:0`
decoding (`gemini_solar_image_review.py:404`); the residual variance comes from the
model being a *thinking* model whose internal reasoning trace is sampled (no fixed
seed). This section measures how much that moves the result, and what removes it.

**Design — test-retest on identical inputs.** From the 15,859-anchor JHB inventory
we stratified-sampled **244 anchors / 642 installations**, deliberately
*oversampling* the ambiguous strata (non-monotonic, no-recent-anchor,
gemini-failed) where LLM noise is most likely to bite; the headline is then
**re-weighted to the inventory's true stratum mix**. Each installation's chips were
rendered **once** (a fixed 8-date window, 2018–2025) and frozen, then scored
**three times** on those byte-identical images. Reliability is the rep-to-rep
agreement; the input is held constant so this isolates the model's own sampling
variance. The chips and the scoring calls are **freshly generated for this
experiment** — only the anchor list and its production status-stratum are reused
(for stratification and inventory re-weighting); no production install-dates are
recycled. (`solar_backdating/scripts/validation/llm_reliability_{sample,analyze}.py`;
raw + summary under `results/analysis/llm_consistency/reliability/`.)

The headline run scored through the env-default `gemini-3-flash-agent`. Production,
however, produced **almost all** of its dates with the plain `gemini-3-flash` alias
(round types `initial`/`bisection`/`tail`/`walk_back` plus the 2023-census step);
`gemini-3-flash-agent` was used only for the narrow `anchor_recovery` round. To
confirm the number is a property of the *production* instrument and not of the
agent alias, we re-scored a 150-installation stratified slice of the **same frozen
chips** ×3 through `gemini-3-flash` — see "Model parity" below.

**Per-installation reliability — moderate.**

| Stratum | n | P(install-date identical ×3) | P(pattern identical ×3) | inventory weight |
|---|--:|--:|--:|--:|
| done_appears (clean) | 148 | 0.818 | 0.777 | 0.689 |
| ambiguous · no-recent-anchor | 168 | 0.792 | 0.738 | 0.164 |
| ambiguous · non-monotonic | 130 | 0.823 | 0.715 | 0.130 |
| installed-during-census | 62 | 0.839 | 0.758 | 0.011 |
| already-present bound | 53 | 0.830 | 0.755 | 0.004 |
| gemini-failed | 81 | 0.840 | 0.728 | 0.002 |
| **inventory-weighted** | | **0.814** | **0.762** | |
| _sample (unweighted)_ | 642 | 0.818 | 0.745 | |

So **~81 %** of installations get the identical install-date on a rerun and **~76 %**
the identical presence pattern. Reliability is notably *uniform* across strata — the
hard cases are barely worse than the clean ones — which already says the noise is a
property of the marginal date-call, not of a few pathological installations. But the
headline alone is a worst-case, k = 1 reading; three further cuts of the *same* three
reps (no new API; `…/reliability/recut/reliability_recut.json`) reframe how bad it
actually is.

**Model parity — the number is a property of the production model, not the agent
alias.** Re-scoring the 150-installation slice of frozen chips ×3 through the
production `gemini-3-flash` reproduces install-dates at **0.807**, statistically
indistinguishable from the agent alias's **0.820** on the identical targets (paired
McNemar χ² = 0.20, *p* ≈ 0.65; 9 vs 11 discordant of 150). On the dominant clean
stratum the two are likewise close (flash 0.925 vs agent 0.85 on n = 40), and the
inventory-weighted slice figure is, if anything, slightly *higher* under flash
(~0.87). The headline therefore transfers to the model that actually produced the
inventory — it is neither inflated nor deflated by the choice of alias.
(`results/analysis/llm_consistency/reliability/parity/parity_compare.json`.)

**Anatomy of the ~18 % that disagrees — it is wider than a single-step flip, and it
cannot be filtered away by the obvious knobs.** Of the 117 disagreeing installations:

- **41 % (48) are present-vs-absent flips**, not dating errors: on a faint
  installation the model flips between "never visible in 2018–2025" and "appears at
  date X." This is a detection-margin wobble at the threshold of visibility.
- **33 % (39) shift by exactly one imagery step** (~12 months — the headline median):
  57 % of the 69 *date-only* disagreers.
- **26 % (30) jump two or more years**; these multi-step jumps concentrate in the
  small flagged-ambiguous tail.

Coarsening to install-year buys **nothing** here, and it is important to say *why*:
the 8-date window is year-spaced (8 dates falling in 8 distinct years), so
install-year agreement equals exact-date agreement for **every** target *by
construction* — verified, not assumed. Year-level reporting is therefore no more
individually reliable than the exact date. Two diagnostics confirm the noise is
structural rather than gateable: it concentrates in the ~8 % of installs already flagged
non-monotonic/uncertain (rerun-flip rate **0.35 vs 0.17** for clean monotonic
sequences), yet **confidence cannot screen it** — 637 of 642 targets self-report
confidence ≥ 0.9, and the disagreers' mean confidence (0.933) is barely below the
agreers' (0.958). The model stays confident even when it is about to flip, so a
confidence cut is not the lever.

**Lever 1 — vote, don't trust one sample (certified).** Production scores each
installation **once** (k = 1), which is precisely the worst case measured above.
Across the three reps a **well-defined majority (≥ 2 of 3) already exists for
98.4 %** of installations; only **1.6 % (10 of 642)** are irrecoverable three-way
splits. To certify the *denoised* answer's reproducibility — not just "is a majority
defined" — we extended the production-model (`gemini-3-flash`) reps to **K = 6** on
the frozen slice and compared two **independent** majority-of-3 draws across all ten
disjoint 3∣3 splits. Result: a single draw reproduces at **0.910** inventory-weighted
(two independent draws agree), while a **majority-of-3 vote reproduces at 0.951**
(0.980 on the dominant clean stratum) — a **+4.1 pp** lift for **3× the calls**.
A majority-of-k denoiser is therefore not merely plausible but **measured**: it lifts
effective install-date reproducibility into the mid-90s. (A fully grounded
majority-of-5 needs 10 reps; the 6-rep proxy already pins the modal answer in ≥ 4/6
reps for 92 % of installations. `…/reliability/parity/majority_k.json`.)

**Lever 2 — pin the decoder: tested, and it is *not* the fix.** The natural
hypothesis is that the variance is a loose-temperature artifact. It is not. The
native scorer path already sends greedy **`temperature:0`** unconditionally
(`gemini_solar_image_review.py:404`), so both reliability runs were *already* greedy.
And `gemini-3-flash` — a direct, non-agent alias — is **just as variable** as the
agent alias (0.807 vs 0.820 above) despite that greedy request. The residual
rep-to-rep movement is intrinsic to a *thinking* vision-model whose reasoning trace
is sampled on marginal-visibility date calls, not a decoder that was left un-pinned.
This closes off the cheap "config fix" story: the honest levers are the **vote**
(Lever 1) and the **aggregation** (below), not a temperature flag.

**Aggregate reliability — high (the published quantity is robust).** The economic
deliverable is not any single install-date; it is the **install-year cohort
distribution** (how many installations appear each year). Because the per-target
errors are near-mean-zero, they cancel under aggregation. Across the three reruns:

| install-year | rep 1 | rep 2 | rep 3 | max swing |
|---|--:|--:|--:|--:|
| 2022 | 70 | 69 | 67 | 3 |
| 2023 | 95 | 87 | 96 | 9 |
| 2024 | 280 | 291 | 284 | 11 |

The full-distribution agreement is **total-variation distance 0.025** between
reruns, and the **largest** per-year cohort count moves by only **~1.7 %** of the
sample. In economist's terms: the instrument is a noisy ruler at the individual
reading, but the *histogram it produces* — the quantity that enters any downstream
adoption-rate or policy analysis — is reproducible to within a couple of percent.
That is the bound on how much an independent re-estimation of the backdated
inventory could move the published cohort numbers from LLM stochasticity.

**End-to-end reproducibility — the from-scratch rerun (executed 2026-06-23, 3
reruns; results below).** The test-retest above deliberately *froze* its inputs: chips were
rendered once on a single fixed global window and re-scored, so it isolates the
scorer's own sampling variance and — as §4 records — is a **lower** bound on what a
real re-estimation would move. The complementary upper-bound question is the one an
auditor actually asks: *if a second analyst re-ran the entire backdating pipeline
from scratch on the same installations, how often would they recover the same
install interval?* This protocol specifies that run.

**Design — replay the production chain, nothing frozen.** The same stratified
sample drives it (the 244 anchors / 642 installations of §3, reused verbatim from
`sample_anchors.csv`, so the two experiments are directly comparable and the
inventory re-weighting carries over). For each installation the full production
backdating chain is re-executed end to end, 1:1 with how the inventory was produced
— **no input is held fixed**:

- **imagery is re-fetched** from GEHI (not reused), so provider-side capture
  selection / `--allow-nearest` snapping re-runs;
- **the adaptive date-search re-runs** per anchor (`run_adaptive_scan.py`) — each
  run picks its *own* probe vintages round-by-round, rather than the single frozen
  global window §3 used;
- **chips are re-rendered** from the freshly fetched vintages;
- **scoring uses the production model routing** — `gemini-3-flash` for the
  `initial`/`bisection`/`tail`/`walk_back` rounds and the 2023-census step,
  `gemini-3-flash-agent` only for `anchor_recovery` — identical to the run that
  produced the inventory;
- **the install-interval post-processing re-runs** in full
  (`infer_install_dates.py`, including the isolated-dip repair `repair_isolated_dips`)
  — so the comparison is on the *post-processed* deliverable, not a raw score.

**Metric — the install interval, against production.** The unit of agreement is the
**install interval** itself — the `[install_interval_start, install_interval_end]`
bracket that `infer_install_dates.py` emits (a point-dated `done_appears` install is
the degenerate single-step ~1 yr interval; the bounded strata —
`installed-during-census`, `already-present` — carry their open/closed brackets).
Two cuts, mirroring §3:

- **1 run vs production** — a single from-scratch rerun's interval compared, per
  installation, against the **interval the production inventory actually shipped**.
  This is the end-to-end k = 1 reproducibility: the honest "could one independent
  re-run move this installation's date?" number.
- **3 runs — mutual + majority.** Three independent from-scratch runs give the
  three-way agreement rate and a **majority-of-3** denoised interval, the end-to-end
  analogue of §3's certified majority lever.

**What it adds over §3.** §3 bounds *sampling* variance on frozen inputs; this
bounds *total* re-run variance. The difference between the two — end-to-end minus
frozen — is exactly the contribution of imagery re-fetch + adaptive re-search that
§4 flagged as un-measured, so running this converts §3's lower bound into a stated
total. The aggregate check re-runs too: the install-year cohort histogram's rerun
TVD (§3: 0.025 on frozen inputs) is recomputed end to end, since the histogram —
not any single interval — is the published quantity.

**Results (executed 2026-06-23; 3 from-scratch reruns).** The drop from §3 is
large and is the headline finding: **the adaptive re-search — not the scorer's own
sampling — is the dominant source of install-date variance, and it is invisible to
a frozen-input test.** §3 froze each anchor's chips on one global window and only
re-scored them (0.81 k = 1); end to end, every run re-decides *which* vintages to
fetch and score, and a single ambiguous/abstain verdict at a pivotal round derails
the whole bracket. (Verified real, not a harness artifact: the cached chips are
present and readable; the rerun genuinely re-scores frames as ambiguous that
production read cleanly.)

| cut | unit | reference | figure (unweighted / inventory-weighted) |
|---|---|---|---|
| single from-scratch run vs production | install interval (exact start+end) | shipped inventory interval | **0.45–0.47 / 0.57–0.61** (3 reps) |
| **process self-consistency** (pairwise rep↔rep) | install interval | another independent rerun | **0.62–0.66 / 0.74–0.75** |
| 3-way mutual (all 3 reps identical) | install interval | — | 0.53 / 0.66 |
| majority-of-3 vs production | install interval | shipped inventory interval | 0.45 / 0.58 |
| install-year vs production | calendar year (both dated) | shipped inventory year | 0.66 / — |
| **cohort histogram** (published quantity) | install-year distribution | rep-to-rep TVD | **0.037–0.063** (vs prod 0.058–0.071) |

Three readings, in order of importance:

1. **The published economic quantity is robust.** Individual dates jitter, but the
   install-year *distribution* barely moves: rep-to-rep histogram TVD 0.037–0.063
   (cf. §3 frozen 0.025). Errors are substitutional across adjacent years, not
   systematic, so they largely cancel in aggregate. The histogram an economic
   analysis consumes reproduces; a single installation's bracket often does not.
2. **Per-installation reproducibility is modest, and the right denominator is
   rep↔rep, not rep↔production.** Two independent reruns agree on the *exact*
   interval ~0.74 (inv-weighted); agreement against the shipped inventory is lower
   (~0.60) because production was a higher-coverage draw — it dated 434/642 vs the
   reruns' tightly clustered 370/385/371 (~16 % more). **majority-of-3 does not lift
   rep-vs-production** (0.58) because majority denoises toward the *process* centre,
   not toward production; this is the opposite of §3, where production *was* the
   centre and majority climbed to 0.95.
3. **Routing is the churn mechanism.** 46 % of installations (881/1926
   installation-rep pairs) are dated by a *different* pipeline layer than production
   — bidirectionally (e.g. production-`gehi_main`→rerun-undated, and
   production-undated→rerun-`gehi_main`), the signature of genuine search
   instability rather than a one-directional defect. The 2023-census layer is the
   largest single contributor: anchors that come back `no_recent` in a rerun lose
   the clean absent→present exemplar pair the census narrowing needs.

**For the paper:** report the **cohort-histogram TVD** and the **rep↔rep
self-consistency** as the headline reproducibility evidence; carry the exact
per-installation interval (~0.74 inv-wtd rep↔rep, ~0.60 vs the shipped inventory)
as the diagnostic floor; and disclose the ~16 % coverage gap (production dated more
than a fresh rerun) as an honest caveat on the shipped inventory being a favourable
draw. Numbers + per-installation audit trail under
`~/zasolar_data/geid_temporal/llm_endtoend_20260623/analysis_3rep/`
(`endtoend_summary.json`, `derived_cuts.json`, `endtoend_per_installation.csv`).

**Cost (per rerun, measured):** ~38 min wall, ~1,300 Gemini calls (L0 601 + L1 651
+ census 54), 517 anchor GEHI fetches, ~3 GB chips — 3 reruns ≈ 2 h / ~3,900 calls
/ ~9 GB. It supersedes the cheaper "window-faithfulness re-score" half-step noted
in §4 by re-deriving each anchor's window rather than replaying a frozen one.

---

## 4 · What is and is not guaranteed

Stated plainly, so a reader can calibrate trust:

- **Guaranteed.** Leakage-checked operating-point selection (§1); a single,
  extracted, traceable metric kernel and config-match reuse (§2); a measured,
  inventory-weighted bound on LLM-induced aggregate variance (§3).
- **Fail-closed, not certified.** The CT production per-detection operating point
  could not be locked at ≤1 pp transfer, so it ships on a **pre-declared** threshold;
  the unified_A × CT certifying lock is **registered but unfitted** (tiles pending).
  A portfolio reader should not infer the CT operating point is leakage-certified —
  it is fail-closed pending that fit.
- **Documented, not re-derived here.** The ~15 pp count-recall cost of the 0.97
  confidence optimum, and the Platt/temperature-scaling ablation, trace to the
  calibration appendix's own records rather than to a fresh run in this document.
- **Scope of the reliability number.** §3's frozen-input figure measures the JHB
  install-date scorer with a fixed 8-date window; it bounds *sampling* variance on
  frozen inputs, not the additional variance from re-fetching imagery or re-running
  the *adaptive* date-search. The ~81 % figure is the **k = 1** production setting and
  oversamples the ambiguous strata. Several follow-ups flagged earlier have now been
  **executed** (2026-06-23): a model-parity re-score confirmed the number is the
  production model's, not the agent alias's; a greedy-decoding check showed
  `temperature:0` is already applied and does **not** reduce the variance; a
  K = 6 majority-of-3 certification measured the denoiser lift on frozen inputs (→
  ~0.95 inventory-weighted); and — the previously-open one — the **end-to-end
  from-scratch rerun** (§3, "End-to-end reproducibility") now has 3 reruns. Its
  verdict reshapes this caveat: the frozen-input lower bound was *far* below the
  total. End to end the exact-interval reproducibility is ~0.74 inv-weighted between
  two independent reruns (~0.60 vs the shipped inventory, which was a higher-coverage
  draw), with 46 % layer-routing churn — the **adaptive re-search**, not the scorer,
  dominates total variance, and a frozen-input test is structurally blind to it.
  **What survives is the aggregate**: the install-year cohort histogram's rep-to-rep
  TVD is 0.037–0.063, so the published distribution reproduces even though individual
  brackets do not. Report the histogram TVD + rep↔rep self-consistency as the
  reproducibility evidence; treat the per-installation exact interval as a diagnostic
  floor; disclose the ~16 % coverage gap as a caveat on the shipped inventory.

---

## Appendix — Provenance table (claim → source)

| Claim area | Source |
|---|---|
| Leakage-checked lock, ≤1 pp acceptance, `[LEAKAGE]` guard | `scripts/analysis/lock_operating_point.py:283–290, 325–327`; `configs/eval/operating_point_calibration.yaml` |
| Locked CT points (pixel-or PASS 0.00 pp / per-det FAIL 5.22 pp) | `configs/eval/locked_operating_points.json` |
| Polygon-conf sweep vs fixed `post_conf` | `scripts/analysis/polygon_conf_sweep.py:54–57`; `core/postproc.py:1064–1066` |
| v4_canonical = 7 keys, no `merge_mode` (CLI-explicit) | `configs/postproc/v4_canonical.json`; `finalize.py` merge-mode guard |
| Tier-1 metric kernel + emitted keys + seed=0 bootstrap | `core/area_metrics.py:84` (`summarize`), `:70–81` |
| Config-match reuse / frozen clean_gt / provenance fields | `docs/evaluation_protocol.md`; per-grid `results/<grid>/config.json` |
| LLM test-retest design + headline numbers | `solar_backdating/scripts/validation/llm_reliability_sample.py`, `llm_reliability_analyze.py`; `results/analysis/llm_consistency/reliability/{reliability_summary,aggregate_stability}.json` |
| Sample scale (244 anchors / 642 installations / 15,859 population, seed 42) | `~/zasolar_data/geid_temporal/llm_reliability_20260622/sample/sample_manifest.json` (`n_anchors_sampled`, `n_targets`, `population_total`) |
| Majority-of-3 (98.4 %), disagreement geometry, abstain/confidence cuts, year≡date | `solar_backdating/scripts/validation/llm_reliability_recut.py`; `results/analysis/llm_consistency/reliability/recut/reliability_recut.json` |
| Model parity: flash 0.807 vs flash-agent 0.820, McNemar χ²=0.20; production model routing (`cheap_model=gemini-3-flash`, `anchor_recovery`→agent) | `results/analysis/llm_consistency/reliability/parity/{parity_compare,FINDINGS}.{json,md}`; `~/zasolar_data/geid_temporal/jhb_full382_fpcut_scan_2026-06-02/run.log` (`[CFG]` line) |
| Greedy `temperature:0` already applied (not a fixable lever) | `solar_backdating/scripts/validation/gemini_solar_image_review.py:404` |
| Majority-of-3 certification: k=1 0.910 → maj-3 0.951 inv-wtd, K=6, 10 disjoint splits | `solar_backdating`-rendered reps under `~/zasolar_data/geid_temporal/llm_reliability_20260622/parity_flash/`; `results/analysis/llm_consistency/reliability/parity/majority_k.json` |
| Inventory scale (15,859 anchors / 41,393 dated installations) | `~/zasolar_data/geid_temporal/jhb_full382_fpcut_scan_2026-06-02/` |
| End-to-end from-scratch rerun (**executed 2026-06-23, 3 reps**) — full 4-layer chain (L0 `gehi_main` → L1 per-target no-recent → L_census 2023-narrowing → `merge_three_layers.py --with-census` + flatten). Finding: exact-interval rep↔rep ~0.74 inv-wtd (~0.60 vs production), 46 % routing churn, cohort-year TVD 0.037–0.063 (aggregate robust) | reuses production layer scripts unmodified: `solar_backdating/scripts/temporal/{run_adaptive_scan.py, infer_install_dates.py (repair_isolated_dips), run_census2023_scan.py}` + production `merge_three_layers.py`/`flatten_gpkg_to_csv.py`. Harness: `solar_backdating/scripts/validation/llm_endtoend_{build_reference,select_units,expand_l1,splice_merge,analyze}.py` + `run_endtoend_rep.sh`. Reuses §3 `sample_anchors.csv`. Routing seam: production no-recent/census membership reused as the fixed candidate pool; all stochastic steps (GEHI fetch, render, Gemini scoring, infer, merge) re-run; L1 membership re-derived per replica from its own L0; routing flips measured, not absorbed. Results under `~/zasolar_data/geid_temporal/llm_endtoend_20260623/analysis_3rep/` (`endtoend_summary.json`, `derived_cuts.json`) |
