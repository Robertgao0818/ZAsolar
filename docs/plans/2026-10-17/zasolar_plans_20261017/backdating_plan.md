# BACKDATING plan: dating when each PV installation appeared (temporal imagery)

Date: 2026-10-17 (NZDT). Drafted by Claude for 葉 from read-only context (sources in §2). Status: **proposal**; nothing here has been started. Numbers without an "assumption" tag come from the file cited next to them.

Project type: exploratory. Evaluation metrics are **not fixed**, with one exception that 葉 already froze: the SSEG contradiction metric and stop rule for the 3.1-vs-3.5 decision (OWNER_DECISIONS D11, 2026-10-13). §3.1 lists the other candidate metrics and the milestone at which each should be frozen.

---

## Top 5 next steps

1. **Turn the 1,008 SSEG contradictions into a model-attributable error rate (§B.6 aerial adjudication, wk 1–2).** A contradiction means the model still saw "no panel" more than 90 days after the system was officially commissioned. Steps:
   - Freeze the parcel list with a hash.
   - For each parcel, find the aerial frames that can adjudicate it: a capture date A with C+30 d < A ≤ L, from CoCT 2020Jan, 2021Mar, 2022Jan, Vexcel 2024 or CoCT 2025Jan.
   - Have a blinded human read those frames, with 50 non-contradiction controls mixed in and 20% read twice.

   *Why:* 1,008 is a raw count. It mixes model misses, SSEG date or location errors and matching errors. Until it is adjudicated, the project has **no** accuracy-type number for Cape Town dates. *If ignored:* the collaborator appendix and any "stop upgrading" decision rest on an unadjudicated count, which a reviewer will reject.
2. **Make the 3.1-vs-3.5 comparison like-for-like and powered (wk 1–2).**
   - The 1,008 come from the **citywide adaptive-scan 3.1** table. The only 3.5 results are a **10-grid full-timeline (T3) protocol** run on 1,114 anchors, so the two cannot be compared directly.
   - Day 1: count the SSEG-matched parcels inside those 10 grids.
   - If there are fewer than 300 (the §B.7 sample size), rescore only the SSEG-matched anchors (≈5,000 parcels) with 3.5 under the **same adaptive protocol** as the citywide 3.1 run.

   *Why:* D11's stop rule compares two models on the same parcels; protocol differences would be read as model differences. *If ignored:* the stop rule is either unpowered (wide CIs) or confounded.
3. **Freeze the seed inventory for this cycle (Mon decision).**
   - Backdating stays on the delivered 2026-06-16 census: 111,801 Mask R-CNN polygons, citywide table sha256 `c0bf9ed1…`.
   - Dating of the new segmentation census waits for a polygon crosswalk at census M4.

   *Why:* the census plan will change the polygons. Mixing seed versions in one analysis breaks the SSEG joins and the appendix numbers. *If ignored:* parcel-level results become irreproducible.
4. **Draw an independent aerial-year check sample (wk 2): ≈300 anchors × 5 aerial years, stratified by status and array size.** Read each anchor-year as present / absent / unreadable, and compare with the model interval. It is a different instrument from SSEG: true-date aerials at 5–8 cm versus satellite z19.

   *Why:* it is the only route to an accuracy statement that does not depend on SSEG, and it tests whether aerial years can bound the 24.7% "census-bound only" anchors. *If ignored:* the date product keeps the label "coverage description, not accuracy", which the collaborators already said is "not precise enough" (BRIEF-municipal 2026-09-06).
5. **Park the self-hosted student line (r4_v2) for this cycle and record that in OWNER_DECISIONS.** Its prereg was frozen 2026-08-13 and it has not trained since; the run is waiting for a clean commit. The single 8 GB GPU is needed for the census loop. *Why:* an open line nobody runs keeps generating status text and decision debt. *If ignored:* it competes for the GPU with census rounds at the worst time, or silently rots.

---

## 1. Goals and research questions

**Goal.** Give every census installation an install-date interval, plus a statement of how often that interval is wrong, measured against evidence that does not come from the model itself. The economics needs interval-censored dates and adoption panels (snapshot stocks by H3 cell and date, monthly flows) for Cape Town first, then Johannesburg (delivered) and Durban.

Research questions:

- **RQ1. Accuracy of the Cape Town intervals.** What fraction of intervals is contradicted by independent evidence, after adjudication? Two evidence sources: (a) SSEG commission dates as one-sided upper bounds, (b) aerial years. How does that fraction vary by array size, cell, imagery density and period, especially Feb 2022–Feb 2024, when only 25 cm NGI imagery exists?
- **RQ2. Stop rule.** Does upgrading the presence scorer (gemini-3.1-flash-lite → 3.5-flash-lite, possibly 3.8-flash) cut the model-attributable contradiction rate enough to justify a rerun? D11 rule: new-model upper bound ≤10%, gain not significant or <2 points, width and undated share no more than 20% worse.
- **RQ3. Error mechanism.** Why does the full-timeline (T3) protocol produce so many non-monotone ("conflict") roofs (616 of 849 evaluation roofs)? Candidates: unusable frames coded as absent, misregistration, small-array contrast, real removals or replacements. What does that imply for the production adaptive-scan and dip-repair path?
- **RQ4. Hybrid product.** Can aerial years (CoCT 2020–22 at 8 cm, Vexcel 2024 at 6.2 cm, CoCT 2025/2026 at 5 cm), read by humans or by the new census segmenter, turn "census-bound" and "undated" anchors into year-level bounds? Would that be better than more satellite scoring?
- **RQ5. Census-version propagation.** When the census polygons change (segmentation census), how are dates carried over without redoing the citywide scan?
- **RQ6. Extension.** When, and with which imagery, should Durban be dated? eThekwini aerial is 10 cm (2016/18/19/20/22) and GEHI TM has a median of 89.5 dates.

---

## 2. Current state

### 2.1 Pipeline

Sources: `~/projects/solar_backdating/README.md`, `CLAUDE.md`, `docs/replan_v2/TRACKER.md`.

- **Seeds.** Census polygons (Cape Town: the 2026-06-16 Mask R-CNN inventory, 111,801 polygons).
- **Imagery.** GEHistoricalImagery Time Machine is the sole download provider, using the A3 region-batch "dump" path (OWNER_DECISIONS D4–D6). Cape Town has a median of 107 bbox-complete TM z19 dates per anchor (2009–2025) and a median of 9 Wayback dates (TRACKER, satellite H3-7 rollup).
- **Scoring.** Gemini judges each frame present/absent. The adaptive scan uses a census-date cutoff with post-census frames as reference only (ISSUE-26). Dip repair enforces monotone sequences, then intervals are built.
- **Estimator layer.** Changepoint posterior decoder, Turnbull EB prior, fractional deliverable (D19).

### 2.2 Deliverables and their measured properties

**Johannesburg, RUN 3** (`TRACKER.md`, RUN 3 notes):

- 41,393 anchors, bounded 83.2%, delivered 2026-07-18.
- CoJ external audit: gate_nc PASS (1/300 false-present). gate_a fails in one cell: 2015 pre-2019 at 0.891 against a 0.95 bar, so the 2015 layer is down-weighted.
- Presence panels delivered 2026-09-23.

**Cape Town citywide** (`DATA-cape-town-citywide-gemini-descriptive-2026-09-02.md`; TRACKER):

- 111,801 anchors: dated 73,336 (65.6%), bounded 100,927 (90.3%), median interval 37 days.
- Scorer: gemini-3.1-flash-lite primary plus 3.6-high rescue.
- Census-bound only (known only to be before 2025-01-31): 27,591 (24.7%). Left-censored (before the 2019 window): 9,843 (8.8%) (`PLAN-cv-validation-appendix-tasklist-2026-10-13.md` D.7).
- Visit sequences for the 90,348 27-wave anchors: 1,291,477 frames, ≈14 per anchor. Raw and dip-repaired codes differ on 13,157 anchors / 52,212 dates. The Top-52 21,453 rows have no scan state (`DATA-cape-town-presence-sequences-2026-09-23.md`).
- Status: a research preview, **not an accuracy claim**.

**CT-11 human calibration** (`RUN-cape-town-ct11-human-calibration-2026-08-04.md`; BRIEF-municipal):

- Repeat-run state agreement 68%; exact interval recovery 37.5%.
- Release on HOLD.
- A **sealed 500+100 confirmation holdout** has never been opened. It can be used once only.

**10-grid full-timeline reference v2** (1,114 anchors, T3 = candidate frame + census aerial + one post-census reference; `RUN-fulltimeline-reference-v2-10grid-2026-10-05.md`, `RUN-fulltimeline-v2-gemini35-flash-lite-2026-10-07.md`):

| | 3.1 T3 | 3.5 T3 |
|---|---|---|
| eval roofs (849): bounded / conflict / left-cens / undated | 171 / **616** / 6 / 56 | (all 1,114) bounded 119 / conflict 866 / left 1 / undated 128 |
| frame-label agreement 3.1 vs 3.5 | — | 82.1% (41,668 / 50,742); the largest cell is 3.1 present → 3.5 absent (5,224) |
| "unknown" labels | 509 | 2,339 |

- Of 761 production (adaptive) brackets, **539 become conflicts under T3**.
- So scoring every frame exposes non-monotone sequences that the adaptive scan with dip repair hides. Agreement between models is not accuracy (stated in the run notes).

**SSEG comparison** (`DATA-sseg-registration-intake-2026-10-13.md`; `RUN-cv-appendix-figures-v3r1-2026-10-14.md`):

- Register: 17,176 rows; 5,458 commissioned, about 5,150 with coordinates. Commission dates run 2011-11-26 to 2024-07-02.
- Parcel main sample: **5,081**. Of these, 2,755 dated, 1,071 left-censored, 643 mixed, 342 with no model polygon, 241 census-bound only, 29 undated.
- At τ = 90 d: registration lag (model saw the panel before commissioning) 1,566, consistent 570, contradiction **619** among dated parcels.
- **Headline count 1,008** = 619 dated + 173 census-bound + 216 mixed (葉, 2026-10-14).
- For dated parcels, Δ = interval midpoint − C: median −2.83 months, IQR 11.20.
- Monthly-flow cross-correlation: best lag 2, Pearson 0.621, n = 64.
- §B.6 adjudication and the 3.1-vs-3.5 gate are **not done**.
- The uploaded collaborator README still calls 619 the main table. Open items: Q3 (L1952–54 label offset) and Q4 (D2 coordinate rounding).

### 2.3 Imagery for independent checks

Sources: `DATA-ct-vexcel-municipal-recheck-2026-10-13.md`; TRACKER.

- **CoCT aerial.**
  - 2019/20/21/22/25/26 had imagery at all 300 sampled anchors.
  - 2023Jan is NGI 2022 at 25 cm.
  - 2024 covers 161/300.
  - Almost-annual 8 cm layers exist back to 2012.
- **Vexcel Cape Town.** One season, 2024-02-29 … 05-13, ~6.2 cm, covering 98.9% of the 111,801 anchors.
- **Practical year panel:** 2020Jan, 2021Mar, 2022Jan, Vexcel 2024, 2025Jan (census layer) and 2026Jan. The **Feb 2022–Feb 2024 gap**, the load-shedding adoption boom, has no fine aerial.
- **Parcels and footprints.** CoCT parcels (835,018) are already joined (`cv_appendix_20261013/phase0/`). CoCT 2023 building footprints (1,444,950) are available.

### 2.4 Self-hosted scorer line

Source: `docs/dinov3_scorer/TRACKER.md`.

- The fidelity gate-2 FAILED: A_LSAT 0.639 and DINOv2 floor 0.654, against a teacher ceiling of 0.77.
- R4 v1 ended CALIBRATION_FAILED: the quality head scored AUROC 0.57–0.59.
- The r4_v2 (H1) prereg was frozen 2026-08-13. Formal training waits for a clean commit and has not run.
- A second failure would trigger the pre-authorised LoRA round (D2).

### 2.5 Other open items

- In `docs/replan_v2/TRACKER.md`, ISSUE-11 (gold-set adjudication), ISSUE-14/15/16 (fixed-grid pipeline) and ISSUE-25/29 are ready but not run. ISSUE-28 (quota fallback) is awaiting 葉.
- The S2 offline EIG replay (choosing fewer vintages per anchor, zero API quota) was recommended on 2026-09-01 and never dispatched (`docs/briefs/BRIEF-lit-intake-next-plan-2026-09-01.md`).
- **Durban.** No backdating yet. The census itself still needs its FP audit (see `census_plan.md`).

### 2.6 Resources

- One 8 GB GPU, needed by the census loop.
- Free concurrent local Gemini; the citywide scoring contract was 40 in flight at QPS 6.
- Grok Build/Codex for code; Opus acceptance before collaborator uploads (D11 practice).
- Disk: the citywide chips were released locally after Dropbox backup (waves ≈330 GB on Dropbox). Re-scoring needs either a partial restore or a fresh A3 dump.
- Backdating in the next two weeks needs **no GPU** apart from an optional segmenter-reader test (minutes).

---

## 3. Metrics, milestones and exit criteria

### 3.1 Candidate metrics and freeze points

| # | metric | what it measures | set | freeze status |
|---|---|---|---|---|
| B1 | **Model-attributable contradiction rate** (lower and upper bound after §B.6), plus median interval width and dated / undated share as companions | wrong "absent after commissioning" | SSEG main sample | **Frozen** (D11, 2026-10-13; τ = 90 d, controls 50, double-read 20%, bootstrap 2,000 by H3-9, stop-rule thresholds) |
| B2 | Aerial-year agreement: per anchor-year with a readable aerial, does the model state (present if U ≤ A, absent if L ≥ A, "inside interval" otherwise) match the aerial read? Reported as agreement, false-absent and false-present, by area band (<15, 15–40, ≥40 m²) | interval correctness at year points | aerial-year sample (~300 anchors × 5 years) | candidate; **freeze after the 50-anchor pilot (P3a)**, before the remaining ~250 are read |
| B3 | Aerial-bracket overlap: share of anchors whose model interval overlaps the aerial bracket [last aerial absent, first aerial present] | coarse date correctness | same | freeze with B2 |
| B4 | Census-bound and undated reduction: share of such anchors that receive a year-level bound from aerials | value of the hybrid product | same | descriptive; freeze if the hybrid product is adopted (P4) |
| B5 | Conflict share under full scoring, and mechanism breakdown | per-frame label noise | 10-grid T3 | descriptive (RQ3) |
| B6 | Registration lag C − U (median, IQR) | SSEG process, not accuracy | SSEG dated parcels | descriptive (§B.5.3, already defined) |
| B7 | Monthly-flow cross-correlation with SSEG (best lag) | aggregate timing | citywide | descriptive (§B.5.4, already defined) |
| B8 | Repeat stability (same model, same frames) and model-version agreement | reproducibility | 10-grid, CT-11 | descriptive |
| B9 | Calls per anchor and wall-clock | cost | any run | operational |

**When accuracy may be claimed.** Only after B1 (adjudicated) and B2 are both frozen and computed, and only once the sealed 500+100 holdout has been opened under a written prereg. Until then, every output stays "research preview".

### 3.2 Phases

| phase | content | exit criteria | target |
|---|---|---|---|
| **P0. Bookkeeping** (CPU) | freeze the 1,008-parcel list (hash); aerial eligibility per parcel (C+30 < A ≤ L, capture dates per §B.6); SSEG-matched parcels counted inside the 10 grids; seed inventory frozen (D3); reader package spec | counts written in a RUN doc; 葉 decisions B-D1–D4 taken | Tue 10-20 |
| **P1. §B.6 adjudication** (human) | step 1: blinded aerial reads of eligible parcels plus 50 controls, 20% double-read; step 2: blinded 4-image satellite review for the rest | B1 lower and upper bounds for the **citywide 3.1** table; inter-reader agreement reported; residual unresolved count stated | step 1 by Fri 10-23; step 2 by Wed 10-28 |
| **P2. 3.1-vs-3.5 gate** | paired parcels: either the 10-grid T3 pair (if ≥300) or a 3.5 adaptive rescore of the SSEG-matched anchors; new parcel×date contradictions adjudicated once and shared | D11 stop rule evaluated and the verdict written to OWNER_DECISIONS by 葉 | draft Fri 10-30 |
| **P3. Aerial-year check** | P3a: 50-anchor pilot → freeze B2/B3; P3b: the remaining ~250 anchors; P3c: test the census segmenter as an automatic aerial-year reader against the human reads | B2/B3 estimated with CIs by area band; reader decision (human only, or segmenter with human audit) | P3a by Thu 10-29; P3b in weeks 3–4 |
| **P4. Product decision** | (a) satellite + Gemini stays primary, aerial as the check; (b) hybrid: aerial year panel plus satellite within-year refinement; (c) student/self-hosted later | 葉 picks one; collaborator README v3r2 with the 1,008 headline and the adjudicated B1 (Opus acceptance before upload) | wk 3–4 |
| **P5. Extension** | census-version crosswalk (when census M4 lands); Durban backdating after the Durban census FP audit; JHB maintenance | crosswalk spec agreed; Durban start decision | after census M4 |

---

## 4. Experiments ranked by value / effort

| rank | experiment | answers | value | effort | first concrete step |
|---|---|---|---|---|---|
| X1 | **Aerial-eligibility count and frozen contradiction list** | how many of the 1,008 can aerials adjudicate at all | high | low (CPU, ½ d) | From `staging_v3r1/d6_dated_pairs.parquet` plus the census-bound and mixed rows, compute per parcel the aerial layers with C+30 < A ≤ L. Report the count stuck in the Feb 2022–Feb 2024 gap. |
| X2 | **§B.6 step 1: blinded aerial reads** (1,008 eligible subset + 50 controls; 20% double) | B1 bounds | high | low–medium (≈4–6 h of reading, assumption, ~15 s per item) | Build a randomised reader package with parcel outline, aerial chip and date, and no model or SSEG information. |
| X3 | **10-grid SSEG count; if <300, a 3.5 adaptive rescore on the SSEG-matched anchors** | a powered, like-for-like 3.1-vs-3.5 gate | high | medium (chips: Dropbox partial restore or A3 re-dump; ≈14 calls per anchor × ~6–7k anchors ≈ 90k calls, assumption) | Dry run on 100 anchors: chip availability, same scanner code, model string logged as `gemini-3.5-flash-lite`. |
| X4 | **Aerial-year check sample** (300 anchors × 2020Jan / 2021Mar / 2022Jan / Vexcel 2024 / 2025Jan) | B2/B3, hybrid value | high | medium (≈1,500 chips; ≈6 h of reading, assumption) | Stratify by citywide status (dated / census-bound / left-censored / undated) × area band. Seed and prereg written before drawing. |
| X5 | **Census segmenter as aerial-year reader**, validated on the X4 human reads | can aerial years be read automatically at scale (hybrid product, RQ4) | medium–high | low (inference on ≈1,500 chips; needs census M1 model) | Score 0.05 m CoCT 2025 directly. Resample the 8 cm 2020–22 and 6.2 cm Vexcel chips to 0.05 m. Report agreement with human reads by year and size. The JHB 15 cm transfer was poor (F1 0.66), so the 8 cm years must be checked separately. |
| X6 | **T3 conflict diagnosis**: sample 60 of the 616 eval-roof conflicts and classify the cause (unusable frame coded absent, misregistration, small-array contrast, real removal or replacement, model flip) | RQ3; whether 3.5's extra "unknown" labels are a feature | medium | low (≈1–2 h of reading) | Use the existing image pack (`reference_fulltimeline_v2_10grid_20261005/`). No new calls. |
| X7 | **S2 offline EIG replay** (pick fewer vintages per anchor) | cost of any future rescore (X3, Durban) | medium | low (CPU, zero quota) | Replay on the citywide sequences. GO if calls fall ≥20% with containment not falling (already defined in Leg-S S2). |
| X8 | **Census-version crosswalk** (legacy 111,801 ↔ segmentation census: IoU match; dates inherited where matched; new or changed polygons queued for scanning) | RQ5 | medium | low–medium | Spec now; implement when census M4 produces polygons. |
| X9 | **Durban availability catalog** (GEHI info for Durban census anchors; no scoring) | Durban feasibility | low–medium | low | Only after the Durban census FP audit is drawn. |
| X10 | Student r4_v2 / LoRA | reproducibility, no API drift | low now | medium–high, GPU | Park (D5). |
| X11 | Open the sealed 500+100 holdout | a citable accuracy number | high, but one-shot | low | **Not in this window.** Only after B1/B2 are frozen and a prereg is written. |

---

## 5. Data and labelling needs

| need | size | who | hours (assumption) | notes |
|---|---|---|---|---|
| §B.6 step 1 aerial reads | ≤1,008 parcel×date items + 50 controls; 20% double-read | 葉 or an RA (B-D1); the second reader for the 20% must be a different person | 4–6 h + ~1 h | Blind to model, version and SSEG. One verdict per parcel×date, shared by both models (§B.6). |
| §B.6 step 2 satellite reviews | the residual after step 1 (unknown; many will fall in the 2022–24 gap) | same | 2–4 h | 4-image layout fixed by §B.6. |
| Aerial chips | CoCT WMS (2020Jan, 2021Mar, 2022Jan, 2025Jan), Vexcel 2024 per-point | Codex script, CPU and network | — | Store in new dated folders under `~/zasolar_data/geid_temporal/`. Record capture dates (CoCT label month → first day; Vexcel per point). |
| Aerial-year sample reads | ≈300 anchors × 5 years | reader | ≈6 h | 3-state labels (present / absent / unreadable). 20 anchors double-read. |
| 3.5 rescore chips | SSEG-matched anchors (~6–7k, assumption) | Codex: restore from the Dropbox wave tars or re-dump via A3 | disk ~tens of GB (assumption) | Keep within the ≤2-wave hot-set rule; release after scoring. |
| T3 conflict sample | 60 roofs | reader | 1–2 h | Existing images. |
| SSEG | already in place (`~/zasolar_data/sseg/`, sha `40c54ab8…`) | — | — | Read-only. Use `wkt`, never the rounded lat/lon. |

Data hygiene: the citywide table, the SSEG file and the released appendix packages are never edited. New outputs go to new folders such as `cv_appendix_20261013/release_v3r2/` and `sseg_adjudication_v1_<date>/`. Hashes are recorded in RUN docs.

---

## 6. Risks

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| Many contradictions sit in the Feb 2022–Feb 2024 aerial gap | high (assumption; the boom years) | step 1 cannot adjudicate them; step 2 uses the same satellite frames the model saw (correlated errors) | X1 count first; report the unresolved share explicitly as the gap between B1's lower and upper bounds; consider NGI 25 cm as a weak extra layer, labelled as such |
| SSEG coordinates or matching are wrong (two geo sources; 298 commissioned rows unmatched) | medium | contradictions blamed on the model | report by `geo_source`; the aerial read also checks whether the parcel has PV at all; buffer matching as a sensitivity only (frozen) |
| Protocol confound in 3.1 vs 3.5 (adaptive vs T3) | high if unaddressed | stop-rule verdict means nothing | X3: same protocol, same parcels |
| Reader bias or fatigue | medium | biased B1/B2 | blinding; controls; 20% double-read; short sessions |
| Gemini model retirement or drift (3.1-flash-lite) | medium | the citywide table cannot be reproduced or extended | log `modelVersion` (already done); the stop rule decides the future model; keep the student line documented, even while parked |
| Seed inventory changes (census v_seg) | certain later | orphaned dates | D3 freeze for this cycle; X8 crosswalk |
| Over-reading the sealed holdout or reusing it | low | loss of the only clean confirmation set | X11 gated by prereg; never "peek" |
| Segmenter-as-reader fails on 8 cm years (domain shift) | medium–high (JHB 15 cm precedent) | the hybrid product stays manual | validate per year in X5 before any scale-up |
| Human hours (葉) are shared with the census labelling | high | both plans slip | B-D1: assign an RA to the reads; census T-B labelling has priority in wk 1 mornings |

---

## 7. Decisions needed from 葉

1. **Readers for §B.6 and the aerial-year sample.** Who reads (葉 / Li / RA), who is the second reader for the 20%, and the hour budget (proposal: step 1 ≈4–6 h in wk 1; the aerial-year sample ≈6 h in wk 2–3).
2. **3.1-vs-3.5 design.** If the 10 grids hold fewer than 300 matched parcels, approve a 3.5 adaptive rescore restricted to the SSEG-matched anchors (≈90k calls on local Gemini, assumption, plus a chip restore or re-dump). Alternatively run the gate on the underpowered 10-grid T3 pair and accept wide CIs. *Recommendation:* rescore the SSEG-matched anchors.
3. **Seed freeze.** Backdating in this cycle uses only the 2026-06-16 inventory (111,801). Segmentation-census polygons are dated later via a crosswalk.
4. **Contradiction list scope.** Adjudicate all 1,008 (headline, per the 10-14 decision) or start with the 619 dated parcels. *Recommendation:* all 1,008. The 173 census-bound parcels are the most informative.
5. **Student line.** Park r4_v2 and the LoRA option until the census loop converges (GPU), and record this in OWNER_DECISIONS. Or continue.
6. **Aerial-year sample.** Approve 300 anchors × 5 years, stratified, with a prereg; human reads now and segmenter-reader validation later. This stays inside D11/P1-A ("aerial checks at year points as a sampled test, not citywide detection").
7. **Sealed 500+100 holdout.** Keep it sealed this cycle. Name the milestone at which it will be opened (proposal: after B1 and B2 are frozen and P4 is decided).
8. **Collaborator README v3r2.** When to rebuild it with the 1,008 headline (proposal: together with the adjudicated B1, in one Opus-accepted upload, not twice).
9. **S2 EIG replay.** Dispatch it now (zero quota, CPU), so that X3 and Durban are cheaper.
10. **Durban backdating.** Not in this window. At most the GEHI availability catalog after the Durban census FP audit.
11. **Open appendix questions.** Q3: a held-out precision/recall check for the L1952–54 ~1.4 m label offset? Q4: round the public D2 coordinates?

---

## 8. Two-week schedule (Mon 2026-10-19 → Fri 2026-10-30)

Backdating needs no GPU in these two weeks. The census plan has the GPU at night and 葉's labelling in week-1 mornings, so the reading sessions below are placed in afternoons and assume an RA where possible. Every working day ends with `docs/replan_v2/TRACKER.md` updated and a RUN note for any run (Definition of Done).

| day | 葉 / reader | CPU / agents (Codex / Grok) | external calls |
|---|---|---|---|
| **Mon 10-19** | Decisions B-D1–D5 (20 min, same session as census) | X1: freeze the 1,008 list (hash) and compute aerial eligibility. Count SSEG-matched parcels in the 10 grids. RUN doc opened. | — |
| **Tue 10-20** | — | Reader-package builder (blind, randomised, 50 controls, 20% duplicates hidden). Fetch aerial chips for eligible parcel×dates. X3 dry run (100 anchors) if D2 says rescore. | GEHI / A3 dry run |
| **Wed 10-21** | Reading session 1 (~2–3 h) | X6 T3-conflict sample package (60 roofs). X7 S2 replay start (CPU, tmux). | — |
| **Thu 10-22** | Reading session 2 (~2–3 h) plus double-reads | X3 3.5 rescore launch (tmux, resumable, model string asserted). | local Gemini |
| **Fri 10-23** | 30 min review | **P1 step-1 result:** B1 bounds for the citywide 3.1 table, inter-reader agreement, residual list for step 2. RUN note. | rescore continues |
| **Mon 10-26** (NZ Labour Day) | — | — | rescore continues unattended |
| **Tue 10-27** | — | Step-2 package (4-image blind) for the residual. Aerial-year sample drawn (prereg, seed) and chips fetched. | — |
| **Wed 10-28** | Step-2 reads (~2 h); X6 conflict reads (~1 h) | X6 tally by mechanism. | — |
| **Thu 10-29** | Aerial-year pilot reads: 50 anchors (~1 h) | Freeze B2/B3 definitions from the pilot (prereg addendum). 3.5 contradictions computed; new parcel×dates adjudicated with shared verdicts. | — |
| **Fri 10-30** | Review (1 h): stop-rule draft and next steps | Paired 3.1-vs-3.5 B1 with H3-9 bootstrap. Draft OWNER_DECISIONS entry for 葉. README v3r2 draft queued for Opus acceptance. TRACKER updated. | — |

**Two-week deliverables.**

- The first adjudicated, model-attributable contradiction rate for Cape Town dates (bounds).
- A like-for-like 3.1-vs-3.5 gate result, or a precise statement of what is still missing.
- A mechanism breakdown of T3 conflicts.
- A frozen aerial-year metric with a 50-anchor pilot estimate.
- Recorded decisions on seed freeze, the student line and the holdout.
