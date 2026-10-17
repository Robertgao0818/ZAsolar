# ZAsolar forward ROADMAP (census + backdating)

Date: 2026-10-17 (Sat, NZDT). Drafted by Claude for 葉 from read-only sources (§0). Status: **proposal**. Reviewed by Grok Bot 2026-10-17 (corrections marked [Grok Bot ...]; Claude's original kept as ROADMAP_claude_draft.md). **v4** = v3 + Decisions (2026-10-17 17:13 NZDT, 葉); v3 file unchanged. Nothing in this file has been started. No GPU was used to write it.

Mode: **Exploration** (WORKFLOW.md Mode A). Metrics are not locked. 葉 decides models, data versions, labels, GPU budgets and anything that costs money.

Number rule: every number carries a source tag (§0). A number marked **estimate** or **assumption** is not measured. Today's decisions by 葉 are tagged [葉-1017 #n] and override older plans.

## Priorities (2026-10-17 17:04 NZDT, per 葉)

v4 overlay on ROADMAP_v3.md. v1 stays at ROADMAP.md; v2 stays at ROADMAP_v2.md; v3 stays at ROADMAP_v3.md (none modified).

1. **Census model first.** Census work has priority over other tracks.
2. **Backdating and the Durban extras are paused.** No further backdating or Durban extra work until 葉 reopens them.
3. **Continue v2e (built and verified; no training) and the Phase-1 model comparison plan.** Keep the v2e build/verify track and the Phase-1 backbone comparison design moving; do not start training without a further go from 葉.
4. **The r2 sign-off, the touching-label merge decision and all test scoring are ON HOLD.** Inference outputs stay sealed. No scoring until 葉 signs r2 and decides on merging touching labels.
5. **Open / carry-forward:**
   - 葉's pregate cross-grid chip concern (under investigation). Claude on rog was asked; no report yet. May affect r2 / scoring scripts.

## Decisions (2026-10-17 17:13 NZDT, 葉)

Moved out of open items. Also recorded in `~/projects/zasolar_plans_20261017/v2e/DECISIONS_20261017.md`.

1. **G2028 stays in val.** Do not remove it. v2e val = v2d2 val (unchanged), so v2c / v2d / v2e val scores stay comparable. G2028 keeps its SAM3 thin-strip labels.
2. **Keep the v2d ignore edits in G1749 / G1972 / G2029.** Leave those ignore regions as they are in v2e.
3. **Backdating uses 3.5** — confirmed by 葉 (was already [葉-1017b #3]; this closes the open "confirm" item while backdating remains paused).

---

## v2 update (2026-10-17 ~17:30 NZDT, Grok Bot) — 葉's decisions [葉-1017b]
Supersedes the matching "Top decisions" items below (2, 3, 4) and the backdating 3.1-vs-3.5 open item. The text below is v1 unchanged; v1 stays at ROADMAP.md.
1. **Phase-1 GPU cap ≈26 GPU-h approved** (estimate-based cap). Launch details (arm order, seeds within the cap, launch time, stop rules / DECISION_RULE file) are still pending; nothing launched.
2. **SAT DINOv3 ends at 25k steps, no resume.** Closed (cost rule 3.1× > 1.5× MiT). `resume_sat_v2d1.sh` must not be run.
3. **Backdating uses 3.5** (3.1-vs-3.5 decided for 3.5; B0.2/B1.2 gate no longer decides the model choice).
4. **Phase-1 trains on v2e** (replaces "v2d2 as built"): `~/data/ZAsolar/geid_temporal/ct_seg_trainset_v2e_20261017/` = v2c labels + G1189/G1190/G1238 + G1630 seg130 fix + G1972 tightening, without the v2d SAM3 thin-strip replacements in train G1971/G1687/G1240. Val/eval = v2d2 (unchanged; G2028 stays in val with its SAM3 thin-strip labels — decided 2026-10-17 17:13 NZDT, 葉). Design/verification: `~/projects/zasolar_plans_20261017/v2e/`. Train 79 grids / 5,645 polygons / 9,970 tiles / 19,904 chips.
5. **Durban stays at native resolution** (no resampling for Durban training/inference).
Before Phase-1 launch: rebuild the HN coord cache for v2e, write `train2_v2e.py` (copy of train2_v2d2.py with DATA=v2e), a smoke run, and the DECISION_RULE file.


Phase calendar (assumption, set by this roadmap):
- **Phase 0, this week:** Sat 10-17 → Fri 10-23.
- **Phase 1, next 2–4 weeks:** Mon 10-26 (NZ Labour Day) → Fri 11-20 at the latest.
- **Phase 2, expansion:** from late November, gated by Phase 1 exits. No dates are committed.

---

## Top decisions for 葉

Ordered by what blocks the most work. Each item has a recommendation.

1. **Give the GPU go for the approved test inference.** The rules are already frozen: `~/zasolar_data/capetown_manifest_v3_2_20261017/TEST_RULES_frozen_20261017.md` (16:20 NZDT, read-only, md5-pinned; groups A/B/C/D, empty class → ignore, spill clipped, scorer zaeval commit 59911a8). [Grok Bot review] *Remaining:* Claude's code review of 59911a8 against the frozen rules must report no blocker, then the 0.9 GPU-h run (estimate) [葉-1017 #4] can start.
2. **Approve the Phase-1 backbone comparison cap (≈26 GPU-h, estimate)** [葉-1017 #2; MU3 §5]. Also decide whether a second A0 seed (+3.2 GPU-h, estimate from measured T [MU3 §5]) comes out of that cap. *Recommendation:* approve it, and put the seed inside the cap. Without a seed gap, no arm-vs-arm difference can be called real [CP §2.4].
3. **Choose the trainset for the Phase-1 arms.** The options are v2d2 as built [MU3 §2] or the de-duplicated manifest first [葉-1017 #5]. *Recommendation:* v2d2 as built for all arms, judged on the clean Li group A. Then test dedupe and rasterisation on the winning arm only, one change per run (§2.2).
4. **Close SAT DINOv3 at 25k** without more GPU. It fails the cost rule (3.1× MiT) [葉-1017 #3; CP §2.1], and on JHB25 its F1 is within noise of MiT v2c [JA §2, §7]. *Recommendation:* close it and record that in program.md.
5. **Backdating Monday batch** [BP §7]:
   - who reads the §B.6 contradictions (and who is the 20% second reader);
   - the 3.1-vs-3.5 design (rescore the SSEG-matched anchors if the 10 grids hold fewer than 300 matched parcels);
   - seed freeze on the 2026-06-16 inventory (111,801 polygons);
   - parking the r4_v2 student line.

   *Recommendation:* follow BP §7 items 1–5 as written.
6. **Durban carry-overs** (pending since 10-16) [DBN; CP §7 D5]:
   - adopt the v4 candidate, which drops block 030281;
   - look at audit50 before any Gemini hard negative is used;
   - use the 0.05 m resample (hn_v1_r005).

   *Recommendation:* adopt v4. Do audit50 (5 min) before Phase 1.
7. **Small-panel test set T-B** (60–100 new hand-drawn tiles, ≥50 objects <4 m², 6–10 h labelling, assumption [CP §5]). Group A has only about 30 such objects [葉-1017 #4], so its small-panel recall is directional only. *Recommendation:* build T-B in Phase 1 if hypothesis H2 (small-panel and boundary gains) [MU3 §7] is going to drive the backbone choice. Otherwise skip it.
8. **Compute for Phase 2.** One 8 GB GPU [葉-1017 #8] cannot cover all of South Africa. For a sense of scale, one Cape Town pass alone is up to ~8–9 GPU-days (upper-bound assumption [CP §2.8]). *Recommendation:* make no commitment now. Decide after the throughput measurement (E12) in Phase 1. Any cloud GPU spends money, so it is 葉's call [WF rule 8].

---

## 0. Sources and conventions

| tag | file (read-only) |
|---|---|
| [葉-1017 #n] | Facts decided by 葉 on 2026-10-17, item n of `~/projects/zasolar_roadmap_20261017/brief.md` |
| [CP] | `~/projects/zasolar_plans_20261017/census_plan.md` |
| [BP] | `~/projects/zasolar_plans_20261017/backdating_plan.md` |
| [MU3] | `~/projects/zasolar_plans_20261017/mit_upgrade_design_v3.md` |
| [DD] | `~/projects/zasolar_plans_20261017/exp_split_dedupe_v1.md` |
| [RAS] | `~/projects/zasolar_plans_20261017/exp_train_rasterisation_v1.md` |
| [JA] | `~/projects/zasolar_plans_20261017/jhb25_analysis_v1.md` |
| [VER] | `~/projects/zasolar_verification_20261017/VERIFICATION.md` |
| [OQ1] | `~/projects/zasolar_evalfix_20261017/OPEN_QUESTIONS_round1.md` (incl. Grok Bot review at the end) |
| [REV] | `~/projects/zasolar_evalfix_20261017/REVIEW_claude.md` |
| [RES] | `~/projects/zasolar_evalfix_20261017/RESULTS.md` |
| [M3] / [M31] / [M32] | `~/zasolar_data/capetown_manifest_v3_20261017/`, `…v3_1_20261017/`, `…v3_2_20261017/` (pending.md, NAMING_CONVENTION.md, overlap_proof*.json, recount_independent*.txt, tilecount_reconcile.md) |
| [SSEG] | `~/zasolar_data/geid_temporal/ct_sseg_reference_v1_20261017/RUN.md` |
| [JHB25] | `~/capetown_eval/ds3m_ft/joburg25_vexcel/TABLE.md` |
| [DBN] | `~/zasolar_data/durban_overnight_20261016/STATUS.md` |
| [WF] | `~/projects/auto_research_workflow/WORKFLOW.md` |
| [INV] | `~/zasolar_data/training_resources_inventory_20261017.md` |

**Owners** [葉-1017 #9; WF]:

| short name | bot | role in this roadmap |
|---|---|---|
| **主管** | 光伏·项目主管 | design with Claude, reviews, decision packets to 葉 |
| **CT** | 光伏普查·开普敦 | Cape Town execution: manifests, test set, training and inference runs |
| **SAM3** | 光伏普查·SAM3 | Durban, SSEG, backdating |
| **GB** | Grok Build | writes code from a written plan; the owning bot reviews the diff before running it [WF] |
| **葉** | — | decisions, labelling, reading |

**Overrides of older plans:**
- [CP] Top-5 #1 / D1 ("confirm MiT-b2 as THE iteration model") is replaced by the multi-backbone comparison [葉-1017 #2].
- [CP] C10 ("SSEG-matched recall") is retired. SSEG is a precision check only [葉-1017 #6].
- [CP] T-A ("Li's labels as test") is now concrete: the Li no-KML batch (17 grids, 1,260 polygons), with the main metric on group A [葉-1017 #4].
- [CP] bIoU is replaced by boundary F at 0.25/0.5 m, with 1.0 m as a diagnostic [葉-1017 #4; OQ1 Q1].

**Grid IDs:** `^[A-Z]{3}[0-9]{4}$` (CPT, DBN, JNB, PTA, GQB, BFN, ELS, PMB, …). Bare `G####` is retired [葉-1017 #1; M32 NAMING_CONVENTION.md].

---

## 1. Where we stand (17 Oct)

**Census (Cape Town)**
- **Test set:**
  - Li no-KML batch: 17 grids, 1,260 labelled polygons, 233 empty-class polygons [M32 recount; RES].
  - Group A: 8 grids, 150 tiles, 447 polygons, 30 polygons <4 m², 96 of 4–10 m² [OQ1 Q4].
  - All 544 inference tiles are now present; 431 are flagged `scored` [M32 `inference_tiles_v3_2.csv`, counted today].
  - Nearest train/val tile is 22,970 m from the test cells [M32 overlap_proof_v3_2.json].
- **Leakage inside train/val/eval:**
  - 490 shared tiles form 574 cross-split mask pairs [DD].
  - 99.77% of the overlapping pixels are background in both splits, but 72 pairs share about 1,640 m² of PV-on-both pixels [DD].
  - Train∩val share 379 images [M3 overlap_proof.json].
- **Models (MiT-b2 v2d2 training is paused [葉-1017 #2]):**
  - JHB25, with pre-gate, GT 2,068 [葉-1017 #3; JA §1]: MiT v2c F1 0.899, SAT interim 25k 0.888, DINOv3 LVD 0.888, MiT v2d1 0.874 (<4 m² recall 10/26), Mask R-CNN 0.775 [JHB25].
  - At matched config, v2c beats v2d1 by F1 +0.017 [+0.006, +0.031] [JA §7.1].
  - Mask R-CNN's JHB25 row is reference only: the GT is detector-seeded [JA §7.3].
- **Evaluator:**
  - zaeval exists (latest RESULTS.md: 145 passed, 1 skipped); zaeval-2 is committed (59911a8) and under Claude review; new metrics need one independent recompute before reporting [RES; WF].
  - Still open: results are keyed by bare G####, the CPT re-keying is TODO, and seg130 is still kept by the `ct_sam2` loader [REV follow-up; RES].
  - 葉 approved the round-1 changes [葉-1017 #4]: pixel-centre rasterisation for eval, BF 0.25/0.5 m, drop seg130.
- **Legacy census (the baseline to beat):** 111,801 installations, ≈5.0 km² [CP §2.7]. On 99 GT grids: area ratio 1.083, σ_Bw 0.516.

**Census (Durban)**
- v3: 30,604 blocks / 1,889,920 m² [DBN].
- After the rule-D screen: 27,328 blocks / 1,635,102 m². The v4 candidate (drops 030281) is 27,327 blocks / 1,634,973 m² [DBN].
- hn_v1: 4,277 chips. Of these, 3,274 are Gemini negatives and are **unaudited** [DBN].

**Backdating**
- Cape Town citywide: 111,801 anchors, 65.6% dated, 90.3% bounded [BP §2.2]. It is a research preview, not an accuracy claim.
- SSEG headline: 1,008 contradictions [BP §2.2]. Neither the §B.6 adjudication nor the 3.1-vs-3.5 gate is done.
- JHB RUN 3 was delivered (41,393 anchors) [BP §2.2].
- Durban has no backdating yet [BP §2.5].

**Blocked or parked**
- C4 (the probability dump) sits behind an old Joburg chain that is waiting on SAT. Leave it alone [葉-1017 #10]. As a result, zaeval regression class F (an end-to-end check of the full-val 0.926/0.908) stays blocked [RES].
- The brief names tmux `li_infer` and PID 374974. This roadmap assumes nothing about their state and does not touch them. CT should report their status.

---

## 2. CENSUS track

### 2.0 Phase 0: this week (10-17 → 10-23). Freeze the measurement and score the existing models once.

**Goal:** a frozen, signed clean-test protocol and an evaluator (zaeval-2) that implements 葉's approved rules. Also, the first clean-test numbers for MiT v2c and v2d1, and a signed Phase-1 design. No training.

| # | task | owner | inputs | output / done when |
|---|---|---|---|---|
| C0.1 | **DONE 16:20 NZDT** — test-rules freeze doc written before any test score (per the file, no test predictions existed) | 主管 | [OQ1] Q4a–e; `~/zasolar_data/capetown_manifest_v3_2_20261017/pending.md`, `labelled_tiles_canonical_v3_2.csv`, `inference_tiles_v3_2.csv` | `TEST_RULES_frozen_20261017.md` exists in the v3.2 folder, read-only, md5-pinned [Grok Bot review: the draft said it did not exist; it was created at 16:20]. Rules: A = main; B and C go to the extended set after fixes; D is dropped (recall-only if used at all); empty class → ignore; spill clipped. |
| C0.2 | **zaeval-2** (code committed as 59911a8, Claude review of the diff in progress at 16:2x NZDT): pixel-centre GT (`all_touched=False`); BF at 0.25/0.5 m as primary and 0.3/1.0 m as diagnostics; seg130 dropped in new loaders (legacy loader keeps it for reproduction); 254/255 split in the eval masks; results re-keyed to CPT####; `metric_def_version` bumped | GB codes; CT runs the tests; 主管 + Claude review the diff | `~/projects/zasolar_evalfix_20261017/zaeval/`; [OQ1] Q1, Q2, Q5; [REV] open items | The regression suite still passes (latest in RESULTS.md: 145 passed, 1 skipped, 0 failed [RES]). Review finds no blocker. The legacy rows reproduce. Each new metric is recomputed once by an independent script before anyone reports it [WF Mode A.3]. |
| C0.3 | **Test-group column and the next manifest** (name v3.3 is an assumption). Add `test_group` (A/B/C/D). Carry over v3.2's 443 counts and the fixed CPT1688 path. Add the dedupe tile lists (C0.4). | CT; GB for the script | `~/zasolar_data/capetown_manifest_v3_2_20261017/` (v3.2 has no test-group column, checked today); [OQ1] Q3b | A new folder with md5s; v3.2 is left untouched; `overlap_proof` regenerated. |
| C0.4 | **Dedupe statistics on CPU only.** Count how many tiles and how much PV area train loses if every tile shared with val/eval is dropped plus a 100 m buffer. | CT; GB | `~/projects/zasolar_plans_20261017/scripts/split_overlap_check_20261017.json`; [DD] | Counts in a RUN note. If the train PV loss is >10%, report to 葉 and use the "drop only PV-sharing tiles" variant instead [DD stop rule]. |
| C0.5 | **Approved test inference:** MiT v2c and v2d1 on the 544 labelled-area tiles, saving probabilities as uint8. Score on the frozen scored set **only after C0.1 is signed**. Thresholds come from val (h0.9/0.5); no tuning on test. | CT | `~/capetown_eval/ds3m_ft/run_v2c/best.pt`, `run_v2d1/best.pt`; `inference_tiles_v3_2.csv`; tiles at `~/zasolar_data/tiles/cape_town/li_testset_native005_20261017/tiles` | Probability maps saved. Group-A table with P/R/F1 at IoU 0.3/0.5/0.7, recall by size bin with n and CI, BF, pixel IoU and area ratio, independently recomputed and labelled `zaeval-2`. Mask R-CNN is scored on CPU from the existing vectors and flagged "selection-biased": 16 of the 17 grids were its `ct_li_heldout_16` model-selection suite; per the frozen rules §6 they were NOT used for threshold calibration [Grok Bot correction]. |
| C0.6 | **Group B/C repair prep** (no labels change in place): (a) per-grid shift copies for CPT1952/1953/1954, then a rerun of the alignment check; (b) the CPT1896 white-roof supplement as a separate file; (c) the CPT1844 FN check; (d) the CPT1899 thin-strip check | CT; 葉 rules on each | [M31] pending.md #1–5, #7; [OQ1] Q4b | Pass rule after the shift: median offset ≤0.2 m and ≤0.33 of tiles offset >0.5 m [OQ1 Q4b]. If a grid fails, it counts in object metrics only, not in BF. |
| C0.7 | **Phase-1 design packet:** arms A0/A1/A2 (A3 optional), seeds, data version, GPU cap, stop rules | 主管 + Claude → 葉 | [MU3]; `~/projects/zasolar_plans_20261017/mit_upgrade_scripts_20261017/` | 葉 signs the cap and the data version. A `DECISION_RULE`-style file is written before any Phase-1 run [WF Mode B.2 used as good practice]. |
| C0.8 | **Durban carry-overs:** 葉 looks at the audit50 sheet; v4 adopt or reject; r005 yes or no | SAM3 prepares; 葉 decides | `~/zasolar_data/durban_overnight_20261016/hn_v1/audit50/audit50_sheet.jpg`, `…/out/durban_census_inventory_stitch025_holefill_v4_candidate_drop030281.gpkg` | Decisions recorded in program.md. Gemini HN stays out of training until audit50 passes [CP §7 D5]. |
| C0.9 | **SAT closure note:** record the closure if 葉 agrees (Top decision 4). No GPU. | 主管 | `~/capetown_eval/ds3m_ft/SAT_DECISION_RULE.md`; [JA §6–7] | One line in program.md and results.tsv: closed at 25k, cost rule failed. |

**Exit criteria (Phase 0):**
- The test rules are signed and md5-pinned **before** the C0.5 scores are opened.
- zaeval-2 passes regression and its new metrics have one independent recompute.
- The C0.5 group-A table exists for MiT v2c and v2d1, labelled with data version and `metric_def_version`.
- The next manifest (with test_group and dedupe lists) exists in a new folder with md5s.
- 葉 has signed or rejected the Phase-1 cap, the trainset choice and the SAT closure.

**GPU (Phase 0):** 0.9 GPU-h, approved [葉-1017 #4]. This is an **estimate** derived from 2.64 s/tile measured with scoring included, so it is an upper bound [OQ1 Q6 + Grok review]. Nothing else uses the GPU.

**Decisions for 葉 (Phase 0):**
- GPU go for C0.5 once the 59911a8 review is clean (rules already frozen).
- Who does the CPT1896 supplement [OQ1 Q4].
- Accept or reject each B/C repair.
- Phase-1 cap and trainset (Top 2–3).
- SAT closure (Top 4).
- Durban carry-overs (Top 6).

### 2.1 Phase 1: next 2–4 weeks (10-26 → ≤11-20). Pick the backbone, then fix the data one change at a time.

**Goal:** on the clean group A, decide whether any of SegNeXt-B or ConvNeXt-T+UPerNet (or MiT-b3) beats the MiT-b2 baseline beyond seed noise [MU3 §7 H0–H2]. Then measure, on the winner only, the effect of tile dedupe and of training-side rasterisation. Also measure throughput and post-processing gains before any city run.

**Wave 1a: backbone comparison, single seed (needs the Phase-1 cap)**

| # | task | owner | inputs | output / done when |
|---|---|---|---|---|
| C1.1 | A0 MiT-b2 baseline on v2d2 (an exact copy of `train2_v2d2.py`), seed 0 | CT | `~/data/ZAsolar/geid_temporal/ct_seg_trainset_v2d2_20261017`; `v2d2_chain.sh.HELD`; `~/capetown_eval/deepsolar3m/weights/segformer.pt` | best.pt, log, config dump, s/it; logged in results.tsv |
| C1.2 | A0 seed 1 (the noise floor; doubles as the second A0 seed of [MU3] plan b) | CT | same | seed-to-seed gap on val and group A, which defines the noise band |
| C1.3 | A1 SegNeXt-B (public init), seed 0. Log s/it after 500 iterations and re-estimate the budget [MU3 §5]. | GB codes the arm; CT runs it | [MU3] §1, §3 | same outputs as C1.1 |
| C1.4 | A2 ConvNeXt-T+UPerNet: LR mini-sweep {8e-5, 1e-4} at 10k iterations, then seed 0 | GB; CT | [MU3] §3 | same |
| C1.5 | Score every arm on val and group A (544-tile inference, probabilities saved) | CT | C0.5 pipeline | side-by-side table [MU3 §4], each number independently recomputed |
| C1.6 | Apply the phase gate [MU3 §6]: an arm goes to 3 seeds only if its F1@0.5 ≥ A0 − 0.01; stop the study if no arm comes within 0.01 | 主管 + Claude → 葉 | C1.5 table | 葉's verdict: H0, H1 or H2, and whether to fund [MU3] plan b (more seeds) |

**Wave 1b: data changes on the winning arm (each needs separate approval [葉-1017 #4])**

| # | task | owner | inputs | output / done when |
|---|---|---|---|---|
| C1.7 | Dedupe D1: train minus the shared tiles plus a 100 m buffer. D0 is the winner's existing run, so no extra D0 run is needed if the data is the same. | CT | C0.3/C0.4 manifest; [DD] | ΔF1 compared with the noise floor. Decision: does the next trainset dedupe, and do old val numbers need an "optimistic" note? [DD] |
| C1.8 | Rasterisation R1 (pixel-centre), then R2 (ignore band ±2 px) only if R1 passes [RAS stop rules] | CT; GB for mask rebuild | [RAS] | area-ratio and BF change; decision for the v2e labels |
| C1.9 | v2c→v2d1 regression diagnosis (the 44 Gemini-unconfirmed SAM3 polygons → 255, one change) [CP E3]. v2d2 inherits v2d1's label changes (the v3.2 label paths point to `ct_labels_corrected_v2d1`). | CT | [CP] §2.1, E3; [JA] §6 | attributed or declared noise. This matters because JHB25 shows v2d1 <4 m² recall at 10/26 vs v2c 20/26 [JHB25]. |

**Wave 1c: inference-only and CPU work (can run beside 1a/1b)**

| # | task | owner | inputs | output / done when |
|---|---|---|---|---|
| C1.10 | E4 overlapping-window logit averaging plus flip TTA on val | CT; GB | [CP] E4 | split rate and C1 compared with the current stitch |
| C1.11 | E5 rectangle regularisation and footprint-constrained merging (CPU). Footprints may merge and flag, never delete. | GB; CT | [CP] E5; CoCT 2023 footprints (1,444,950) | numbers for BF, area bias, split rate and glue rate. 葉 chooses the area definition and the installation-unit rule [CP D7–D8]. |
| C1.12 | E12 throughput: pure inference on 1,000 tiles (fp16, window size, batch) for the leading arm | CT | [CP] E12 | measured s/tile. This replaces the ~8–9 GPU-day assumption [CP §2.8] used in Phase-2 planning. |
| C1.13 | Group B/C extended set scored (after the C0.6 fixes pass) | CT | C0.6 outputs | extended-set table, reported separately from group A [OQ1 Q4a] |
| C1.14 | (optional, Top 7) T-B small-panel tiles: 60–100 tiles, no pre-fill, ≥50 objects <4 m², 20 double-labelled | 葉 / RA; CT builds the candidate list | [CP] §5 | T-B frozen with a hash and a written protocol |
| C1.15 | Durban FP-screen gold audit: about 400 blocks stratified by Gemini verdict, read blind | SAM3 builds; 葉 / RA reads (2–3 h, assumption [CP §5]) | `~/zasolar_data/durban_ruleD_city_20261012/out/` | an estimated FP-screen precision with a CI, giving a Durban correction factor [CP M5] |
| C1.16 | SSEG as a precision check on census polygons (not recall) | SAM3 | `~/zasolar_data/geid_temporal/ct_sseg_reference_v1_20261017/out/` | Descriptive only. Only 2–4% of labelled parcels have SSEG points, and none sit in the Li test grids [SSEG; 葉-1017 #6]. |

**Exit criteria (Phase 1):**
- The [MU3] §6 gate has been applied, and 葉 has a written verdict on the backbone (H0, H1 or H2), with differences stated against the measured seed gap.
- The dedupe and rasterisation decisions are taken, or explicitly deferred.
- The v2d1 regression is attributed, or declared noise.
- Throughput is measured.
- Post-processing numbers exist, and 葉 has chosen the area definition and the unit rule.
- The Durban gold audit has been read.
- TRACKER and DECISION docs are updated in the same turn as each result [global DoD].

**GPU (Phase 1). Every number is an estimate; none is approved yet.**

| item | GPU-h | basis |
|---|---|---|
| Wave 1a: A0 + A1 + A2 seed 0 + LR mini-sweep | 18–26 | [MU3] §5 (T = 3.2 GPU-h measured; other arms scaled by FLOPs, unverified) |
| C1.2 A0 second seed | 3.2 | T [MU3] §5 |
| C1.5 inference per MiT-class arm on 544 tiles | ≈0.4–0.45 each | half of the 0.8–0.9 for two MiT models [OQ1 Grok review]. Other arms: re-estimate from their measured s/it. |
| C1.7 dedupe D1 (+ eval) | ≈3.2 + ~1 | [DD] |
| C1.8 rasterisation R1, R2 (+ eval) | ≈2 × (3.2 + 0.5) | [RAS] |
| C1.9 v2d1-flag255 | ≈3.2 | one T-length run [MU3] §5. [CP] suggested a shorter 10–15k-iteration round (≈1.1–1.6 h). |
| C1.10 E4 overlap + TTA on val | 1–2 | [CP] E4 |
| C1.12 E12 throughput (1,000 tiles) | ≈0.7 | 1,000 × 2.64 s [OQ1 Q6 measured rate, includes scoring] |
| **Total if everything runs** | **≈38–48** | sum of the rows above (C1.5 counted for 3 arms). This is above the ≈26 cap, so the waves must be sequenced and approved separately. |

At one GPU job at a time [CP §2.8], a 26 GPU-h wave is roughly 3 overnight runs. That is an assumption: about 8–10 h per night.

**Decisions for 葉 (Phase 1):**
- The backbone verdict and whether to fund [MU3] plan b (44–63 GPU-h, estimate) or c (57–79, estimate).
- Whether MiT-b3 (A3) is run at all.
- Approval of each of C1.7, C1.8 and C1.9.
- Area definition and installation-unit rule [CP D7–D8].
- Whether T-B is built.
- Whether Durban HN enters training (after audit50).
- Whether the KD variants [MU3 §1] are worth 36–52 extra GPU-h (estimate). [MU3] says to run them only if A1 or A2 land within 0.01 F1 of A0.

### 2.2 Phase 2: expansion beyond Cape Town and Durban to the rest of South Africa (gated)

**Goal:** one model plus a per-city adaptation recipe that produces a census with an uncertainty statement for each city, starting with Cape Town production.

| # | task | owner | inputs | gate / done when |
|---|---|---|---|---|
| C2.1 | **Cape Town city inference** (census M4): resumable tmux run; pre-gate re-validated at 0.05 m; census v_seg with CIs | CT | Phase-1 winner; the C1.12 throughput number | Area ratio and σ_Bw on the clean test match or beat the legacy census (1.083 / 0.516) [CP M4]. FP-screen precision comes from a stratified audit, not from Gemini. |
| C2.2 | **Polygon crosswalk** from the legacy 111,801 to v_seg (see §4) | SAM3 + CT; GB | v_seg; the legacy census sha in [BP] | crosswalk spec signed and implemented |
| C2.3 | **Durban decision:** keep the Mask R-CNN v4 with a correction factor, or re-run with the CT model after a transfer test on Li's DBN labels | SAM3 | C1.15 audit; Li DBN labels `dropbox:RA_Solar/Li/DBN` (not copied locally [INV]) | 葉 decides on the transfer-test evidence [CP D12] |
| C2.4 | **Johannesburg:** JHB25 already shows usable transfer on Vexcel (MiT v2c F1 0.899 with pre-gate [JHB25]). CoJ 15 cm aerial does not transfer (F1 0.662 [CP §2.6]). Candidate route: fine-tune or verify on 葉's 25 JHB Vexcel grids (3,835 reviewed polygons [CP §2.6]). | CT | `~/zasolar_data/dropbox_label_copies_20261017/`; the JNB grid | 葉 decides JHB in or out [CP D11]. JHB25 GT is detector-seeded, so a clean JHB check is needed before any JHB claim [JA §7.3]. |
| C2.5 | **Other cities** (PTA, GQB, BFN, ELS, PMB and new codes): (a) register the city codes [M32 NAMING_CONVENTION.md]; (b) copy Li's cross-domain labels for BFN/DBN/ELS/GQB/PMB/PTA (part of ≈174 gpkg, ≈58 MB, on Dropbox, not copied [INV]); (c) restore the `vexcel_eval_60` cross-domain eval tiles (4.35 GB, not local [INV]); (d) run a zero-shot transfer table per city | 主管 designs; CT executes; GB scripts | [INV]; [M32] NAMING_CONVENTION.md | Per city: transfer F1 on clean labels, then a choice of zero-shot, fine-tune or skip. Imagery source and licence per city are **unknown** (no source file states them) and must be inventoried first. |
| C2.6 | **Compute plan for national scale** | 主管 → 葉 | C1.12 throughput | 葉 decides on local only or rented GPU (money [WF rule 8]). No national tile count exists in any source, so none is given here. |

**Exit criteria (Phase 2, per city):** a clean-label transfer table; a written adaptation decision; a census file under the ADR-0002 IDs, with a CI and a stated area definition.

**GPU (Phase 2, estimates):**
- Cape Town city pass: up to ~8–9 GPU-days for MiT-b2 (an upper-bound **assumption** [CP §2.8]), to be replaced by the C1.12 measurement.
- More seeds for the backbone: 44–63 (plan b) or 57–79 (plan c) GPU-h [MU3 §5].
- Other cities: **not estimable** until the throughput figure and the tile counts exist.

**Decisions for 葉 (Phase 2):** JHB scope; the Durban strategy; which cities and in what order; compute spending; when to enter Confirmation mode (lock the metrics) [WF Mode B].

---

## 3. BACKDATING track (separate; no GPU in Phases 0–1 [BP §2.6])

### 3.0 Phase 0: this week (10-17 → 10-23). Bookkeeping and the first adjudication reads.

**Goal:** a frozen contradiction list and seed inventory, with the reader package ready. Step 1 of the §B.6 aerial reads should start.

| # | task | owner | inputs | output / done when |
|---|---|---|---|---|
| B0.1 | Freeze the 1,008-parcel contradiction list with a hash. Compute aerial eligibility per parcel (C+30 d < A ≤ L), and count how many fall in the Feb 2022–Feb 2024 gap (X1). | SAM3; GB | `staging_v3r1/d6_dated_pairs.parquet` plus the census-bound and mixed rows (paths as in [BP] X1); `~/zasolar_data/sseg/` (sha `40c54ab8…`) | counts in a new RUN doc [BP P0] |
| B0.2 | Count the SSEG-matched parcels inside the 10 T3 grids (this decides the 3.1-vs-3.5 design) | SAM3 | `reference_fulltimeline_v2_10grid_20261005/` | a number. If it is <300, a rescore is needed [BP §7 D2]. |
| B0.3 | Seed freeze: backdating this cycle uses only the 2026-06-16 inventory (111,801, citywide sha `c0bf9ed1…`) | 葉 decides; SAM3 records it | [BP] Top-5 #3 | OWNER_DECISIONS entry |
| B0.4 | Reader-package builder: blind, randomised, 50 controls, 20% hidden duplicates; aerial chips fetched for eligible parcel×dates | GB; SAM3 | CoCT WMS 2020Jan/2021Mar/2022Jan/2025Jan; Vexcel 2024 [BP §2.3] | package in a new dated folder under `~/zasolar_data/geid_temporal/` |
| B0.5 | §B.6 step-1 reads begin | 葉 or RA (≈4–6 h + ~1 h double-read, assumption [BP §5]) | B0.4 | first verdicts recorded [BP P1 target Fri 10-23] |
| B0.6 | Dispatch X7 (the S2 offline EIG replay: CPU only, zero API quota) | SAM3; GB | citywide sequences [BP X7] | running in tmux; GO if calls fall ≥20% without loss of containment [BP X7] |

**Exit criteria (Phase 0):** list and seed are hashed; the eligibility and 10-grid counts are in a RUN doc; 葉 has decided B-D1–D5 [BP §7]; step-1 reads have started.

**GPU:** 0.

**Decisions for 葉:**
- Readers and second reader [BP D1].
- The 3.1-vs-3.5 design [BP D2].
- Seed freeze [BP D3].
- Scope: all 1,008 or the 619 dated parcels first [BP D4; recommendation: all 1,008].
- Park r4_v2 [BP D5].
- Confirm that the SSEG precision-only rule [葉-1017 #6] covers the census and leaves the frozen D11 date-contradiction metric B1 in place (§9 D3).

### 3.1 Phase 1: next 2–4 weeks. Adjudicated accuracy and the model gate.

**Goal:** the first model-attributable contradiction rate (lower and upper bounds) for Cape Town dates; a like-for-like 3.1-vs-3.5 verdict under the D11 rule; a frozen aerial-year metric.

| # | task | owner | inputs | output / done when |
|---|---|---|---|---|
| B1.1 | §B.6 step 1 completed, then step 2 (blind 4-image satellite review for the residual) | 葉 / RA; SAM3 | B0.4 | B1 bounds for the citywide 3.1 table, with inter-reader agreement [BP P1] |
| B1.2 | 3.1-vs-3.5 gate under the same adaptive protocol: if needed, a 3.5 rescore of the SSEG-matched anchors (≈90k calls, assumption [BP X3]) with chips restored from Dropbox waves or re-dumped through A3 | SAM3 | the waves (≈330 GB on Dropbox [INV]); local Gemini | D11 stop rule evaluated; 葉 writes the verdict [BP P2] |
| B1.3 | X6: classify the causes of 60 T3 conflicts (≈1–2 h of reading, assumption [BP X6]) | SAM3; reader | `reference_fulltimeline_v2_10grid_20261005/` | mechanism breakdown (RQ3) |
| B1.4 | Aerial-year check: P3a pilot on 50 anchors → freeze B2/B3; then P3b on the remaining ~250 (≈6 h of reading, assumption) | SAM3; reader | stratified sample with a written prereg [BP X4] | B2/B3 by area band, with CIs |
| B1.5 | P4 product decision (satellite-primary vs hybrid aerial); README v3r2 with the 1,008 headline plus the adjudicated B1, uploaded once after Opus acceptance | 葉; SAM3 | B1.1–B1.4 | one upload, not two [BP D8] |

**Exit criteria (Phase 1):** B1 adjudicated with bounds; the D11 verdict recorded; B2/B3 frozen and estimated; the P4 product choice made; TRACKER updated.

**GPU:** 0. The only GPU use in [BP] is the optional segmenter-as-reader test (X5, "minutes"), and it waits for the Phase-1 census winner (§4).

**Decisions for 葉:** the D11 verdict; the product choice (P4); the README timing; the aerial-year sample size (300 × 5, [BP D6]).

### 3.2 Phase 2: crosswalk, Durban, other cities

| # | task | owner | gate |
|---|---|---|---|
| B2.1 | Carry dates to the new census through the polygon crosswalk (§4); queue only new or changed polygons for scanning | SAM3 | census C2.1 done |
| B2.2 | Durban: GEHI availability catalogue first (no scoring) [BP X9]. Imagery: eThekwini aerial at 10 cm (2016/18/19/20/22) and GEHI TM with a median of 89.5 dates [BP RQ6] | SAM3 | Durban census FP audit (C1.15) done; 葉 starts it |
| B2.3 | JHB maintenance. The 2015 layer is down-weighted (gate_a 0.891 vs 0.95) [BP §2.2] | SAM3 | — |
| B2.4 | Other cities: GEHI TM is the sole provider [BP §2.1]; each city needs a census seed first | SAM3 | that city's census exists |
| B2.5 | Open the sealed 500+100 CT-11 holdout, once, under a written prereg | 葉 | B1 and B2 frozen and P4 decided [BP D7] |

**GPU:** segmenter-as-reader at scale is optional and unestimated. It needs a per-year validation first, because the 15 cm transfer precedent was poor (F1 0.66 [BP X5]).

---

## 4. Where census and backdating connect

1. **Seed freeze.** Backdating in this cycle dates only the legacy 111,801 Mask R-CNN polygons [BP Top-5 #3]. No new census polygon is dated before the crosswalk exists. *If ignored:* the SSEG joins and the appendix numbers mix seed versions and cannot be reproduced.
2. **Polygon crosswalk (census C2.2 → backdating B2.1).** Match legacy and new polygons by IoU. Dates are inherited where polygons match; new, split or merged polygons are queued for scanning [BP X8; CP M4]. The census installation-unit rule (C1.11) must be fixed first, because it decides what "one polygon" means. The spec can be written in Phase 1; the implementation follows census M4.
3. **Aerial-year checks.** The census segmenter is a candidate automatic reader of the CoCT 2020–22 (8 cm), Vexcel 2024 (6.2 cm) and CoCT 2025 (5 cm) years [BP X5, §2.3]. It can only be validated against the B1.4 human reads, and only after the census Phase-1 winner exists.
4. **SSEG.** In the census it is a precision check only [葉-1017 #6]. In backdating it is the D11 date-contradiction instrument [BP §3.1 B1]. Both tracks should use the same matching code and the `wkt` coordinates, never the rounded lat/lon [SSEG; BP §5].
5. **Shared scarce resources.** The census track owns the single GPU; backdating needs none in Phases 0–1. Both tracks compete for 葉's hours: the census labelling (B/C fixes, T-B) and the backdating reads (§B.6, aerial-year). [BP §6] proposes an RA for the reads.

---

## 5. Dependencies (critical path)

```
C0.1 test rules signed ──► C0.5 open test scores ──► C1.5 score arms on group A ──► C1.6 backbone verdict
C0.2 zaeval-2 ──────────┘                                                                 │
C0.7 Phase-1 cap + trainset signed ──► C1.1–C1.4 runs (one GPU job at a time) ──────────────┘
C1.6 winner ──► C1.7 dedupe / C1.8 rasterisation / C1.9 v2d1 diagnosis (each separately approved)
C1.6 + C1.11 unit rule + C1.12 throughput ──► C2.1 CT city run ──► C2.2 crosswalk ──► B2.1 date carry-over
C0.8 audit50 ──► Durban HN in training (optional) ; C1.15 Durban audit ──► C2.3 Durban decision ──► B2.2
B0.1/B0.2 counts + B-D1–D5 ──► B0.4 reader package ──► B1.1 B1 bounds ──► B1.5 P4 + README
B0.2 (<300?) ──► B1.2 3.5 rescore (chips restore) ──► D11 verdict
C1.6 winner ──► BP X5 segmenter-as-reader test (validated on B1.4 reads)
```

External blockers:
- C4 stays behind the Joburg/SAT chain; leave it [葉-1017 #10].
- The GPU must be free of `li_infer` / PID 374974 before C0.5. CT checks this without killing anything.
- Li DBN labels and the cross-domain tiles are only on Dropbox [INV].

---

## 6. GPU-hour ledger (one 8 GB GPU, the bottleneck [葉-1017 #8])

| phase | item | GPU-h | status |
|---|---|---|---|
| 0 | MiT v2c + v2d1 test inference, 544 tiles | 0.9 (estimate) | **approved**, waits for C0.1 [葉-1017 #4] |
| 1 | Backbone wave 1a | 18–26 (estimate) [MU3] | cap ≈26 **not approved** [葉-1017 #2] |
| 1 | A0 second seed | 3.2 (estimate) | not approved |
| 1 | Data-change runs C1.7–C1.9 | ≈3.2–4.2 each (estimate) [DD; RAS; MU3] | each needs separate approval |
| 1 | E4 + E12 | ≈1.7–2.7 (estimate) [CP; OQ1] | not approved |
| 2 | Backbone more seeds | 44–63 / 57–79 (estimate) [MU3] | conditional on the C1.6 gate |
| 2 | Cape Town city pass | up to ~8–9 GPU-days (upper-bound assumption) [CP §2.8] | replace with the C1.12 measurement |
| any | SAT resume or scoring | 0 recommended | close at 25k |
| any | Backdating | 0 [BP §2.6] | — |

---

## 7. Risks

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| Test scores seen before the rules are frozen, so grouping gets tuned to the scores | medium | the clean test stops being clean | C0.1 is signed before C0.5 is opened; C0.5 saves probabilities so any rescoring is CPU-only [OQ1 Q6] |
| Group A is too small for small panels (~30 objects <4 m²; CI ≈ ±0.18, estimate) | certain | H2 (small-panel and boundary gains) cannot be decided on A | treat it as directional [葉-1017 #4]; T-B (Top 7); JHB25 as a secondary check, keeping in mind its detector-seeded GT [JA §7.3] |
| Arm differences fall within seed or selection noise | high | wrong backbone choice | C1.2 noise floor; MU3 §6 gate; treat point differences ≤0.02 as possible selection noise [JA §7.2] |
| v2d2 inherits the unexplained v2d1 regression | medium | absolute Phase-1 numbers are lower than v2c; the comparison is still fair across arms | C1.9 diagnosis; report v2c as an external reference row |
| Leakage inflates val, which picks best checkpoints and thresholds | medium | slightly optimistic val numbers; the test is unaffected (22,970 m away [M32]) | decide on group A; C1.7 measures the effect |
| Phase-1 FLOP-based cost estimates are wrong (ConvNeXt+UPerNet 8–12 h, unverified [MU3 §5]) | medium | the cap is overrun | re-estimate after 500 iterations; stop at the cap [MU3 §6] |
| Unaudited Gemini hard negatives contain real PV (audit50 shows candidates [DBN]) | high | recall falls silently | audit50 first; unconfirmed chips → 255 [CP §6] |
| Offset grids still have residual shift after one translation (dy IQR is wide [OQ1 Q4b]) | medium | B grids drag BF down | pass rule from C0.6; object metrics only on failure |
| 葉's hours are split between census labelling and backdating reads | high | both tracks slip | RA for the reads [BP D1]; skip T-B unless H2 matters |
| Many contradictions fall in the Feb 2022–Feb 2024 aerial gap | high (assumption [BP §6]) | B1 bounds stay wide | X1 count first; report the unresolved share explicitly |
| National scale outruns the single GPU | high | Phase 2 stalls | measure throughput in C1.12; 葉 decides on compute (money) |
| Imagery for other SA cities is unknown | unknown | Phase 2 cannot be planned | inventory the imagery first (C2.5); no plan numbers until it exists |
| Bot-state drift (numbers living in chat or agent context) | medium | irreproducible decisions | files are the state: results.tsv, STATUS.md, handoff/ [WF]; update the docs in the same turn |

---

## 8. Stop doing

Already decided by 葉 (record each in program.md if it is not there yet):
- Treating MiT-b2 as the only model [葉-1017 #2].
- Training MiT-b2 on v2d2 outside the Phase-1 design (it is paused [葉-1017 #2]).
- Resuming SAT from 25k, or spending more GPU on it [葉-1017 #3].
- Touching C4 or the old Joburg chain [葉-1017 #10].
- Using SSEG as a census recall measure [葉-1017 #6] ([CP] C10 is retired).
- Writing bare `G####` in any new output [葉-1017 #1].
- Using G1630 seg130 in new loaders [葉-1017 #4; OQ1 Q2].
- Changing training-side rasterisation or dedupe without a separate approval [葉-1017 #4].

Recommended (葉 to confirm):
- Using bIoU for any decision, because it swings from 0.16 to 0.90 on the same predictions [CP §2.4]. Use BF instead.
- Using v3.1's `n_label_tiles` (the wrong "471" definition) or the "517" figure [M31 tilecount_reconcile.md].
- Computing precision on partially labelled grids (CPT1843, 1846, 1895); those grids give recall only, if anything [OQ1 Q4c].
- Comparing Mask R-CNN with the segmenters on the Li grids, the 5 eval grids or JHB25 as if those were fair tests (detector-seeded, or used for Mask R-CNN model selection) [OQ1 Q6; JA §7.3].
- Settling model or data questions with single-seed val differences.
- Reporting `zaeval-1`/`-2` numbers before an independent recompute [WF Mode A.3].
- Running the r4_v2 student line this cycle [BP D5].
- Opening the sealed 500+100 holdout before B1 and B2 are frozen [BP D7].
- Using JHB aerial_2023 labels for new training sets [INV].

---

## 9. Discrepancies to resolve (found while writing this roadmap)

- **D1. Scored test tiles, 443 vs 431.** The brief says "scored on 443 canonical tiles" [葉-1017 #4]. v3.2 clips the 12 spill tiles, so `inference_tiles_v3_2.csv` flags 431 tiles as `scored` and `labelled_tiles_canonical_v3_2.csv` marks 12 as `clipped_spill` [M32, counted today]. The 12 spill tiles sit in CPT1841/1788/1894/1787/1840/1955, none of them test grids [M31 tilecount_reconcile.md]. Group A is 150 tiles either way [OQ1 Q4]. **Resolved [Grok Bot review]:** the frozen rules §3 score "443 canonical tiles ∩ test cells", i.e. spill tiles clipped to the test cells; in-grid that is the 431 tiles plus the in-cell parts of the 12 spill tiles. No decision needed.
- **D2. JHB25 GT count.** It is 2,068 in clean_gt, of which 2,065 are scored. An earlier figure of 2,168 was wrong [JA §1]. Use 2,068.
- **D3. Scope of "SSEG precision-only".** [葉-1017 #6] cites census evidence ([SSEG]: 2–4% of labelled parcels, register ends Jul 2024). The backdating B1 metric uses SSEG commission dates as one-sided upper bounds and was frozen by D11 [BP §3.1]. *Assumed here:* fact 6 applies to the census only. 葉 should confirm.
- **D4. Resolved [Grok Bot review]:** `TEST_RULES_frozen_20261017.md` now exists (v3.2 folder, 16:20 NZDT).
