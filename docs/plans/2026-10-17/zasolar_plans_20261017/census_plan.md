# CENSUS plan: present-day rooftop PV segmentation and area census (Cape Town, Durban, maybe JHB)

Date: 2026-10-17 (NZDT). Drafted by Claude for 葉 from read-only context (sources listed in §2). Status: **proposal**. Nothing in this plan has been started. MiT-b2 training on v2d2 stays **on hold** until 葉 approves. Every "assumption" is marked as such; every number without that tag comes from a file cited next to it.

Project type: exploratory. Evaluation metrics are **not fixed**. §3.1 lists candidate metrics and the milestone at which each one should be frozen.

---

## Top 5 next steps

1. **Monday-morning decision batch with 葉 (≈30 min, §7 D1–D8).** The items that block everything else are: confirm MiT-b2 as THE iteration model; allow one diagnostic training night (round R0); choose whether v2d2 is trained as built or split into two runs; decide what happens to the 44 Gemini-unconfirmed SAM3 polygons. *Why:* every GPU night spent before these answers either repeats today's "is v2d1 worse?" argument or bundles several changes into one run. *If ignored:* the single 8 GB GPU sits idle or produces results nobody can attribute.
2. **Build and freeze a clean test set T before the next training result arrives (Mon–Thu, wk 1).** It has two parts. T-A is Li's independent Cape Town labels (different annotator, never seen by the segmenters). T-B is about 60–100 tiles hand-drawn from imagery with no model pre-fill, stratified to hold at least 50 objects under 4 m² plus the main confusers. *Why:* the current val set has only **12** objects under 4 m² (95% CI on recall ±0.2–0.3), and the 5 eval grids have labels derived from Mask R-CNN, so neither can decide anything about small panels or about Mask R-CNN. *If ignored:* every later round is decided by noise or by "looks more like Mask R-CNN".
3. **Measure the noise floor and attribute the v2c→v2d1 drop (R0, one overnight, ≈5 GPU-hours, needs approval).** Run (a) a second seed of the reference recipe and (b) v2d1 with the 44 flagged polygons set to 255, one change only. The 2×2 rescoring already showed the drop is **not** a label-version scoring artifact (`~/capetown_eval/ds3m_ft/rescore_v2d1/TABLE_2x2.md`). *Why:* without a seed-to-seed gap, no metric difference can be called real. *If ignored:* data-version decisions (v2d1, v2d2, HN) stay arguments, not measurements.
4. **Fix the evaluator before trusting boundary and small-object numbers (CPU, 1–2 days, Codex/Grok).** Boundary IoU swings from 0.16 to 0.90 on the same predictions depending only on how the 255 band is handled (TABLE_2x2, "subset bIoU"). Replace it with a boundary F-score at fixed ground tolerances computed on raw polygons. Bin objects by GT polygon area, not by connected-component area. Add paired tests (per-object McNemar, per-grid bootstrap). *Why:* boundaries are one of the three main problems, but today's boundary metric cannot measure them. *If ignored:* a boundary "improvement" could be pure 255 bookkeeping.
5. **Run the inference-only fixes for fragmentation and boundaries (wk 1–2, GPU inference + CPU).** These are overlapping-window logit averaging with flip TTA, then rectangle regularisation plus gap/orientation merging inside CoCT building footprints. *Why:* fragmentation and ragged boundaries are largely tiling and post-processing problems, and fixing them costs no retraining and no labels. *If ignored:* the iteration loop spends 葉's labelling hours on errors that post-processing could remove for free.

---

## 1. Goals and research questions

**Goal.** Produce a defensible present-day rooftop-PV **area and installation inventory** for Cape Town (CoCT aerial 2025, native 0.05 m), then Durban (Vexcel 2026, ~0.075 m), and maybe Johannesburg. "Defensible" means it comes with an uncertainty statement that a reviewer can reproduce. The census also supplies the polygons that the backdating line dates. Changing it therefore changes the backdating seeds (see `backdating_plan.md`).

**Near-term goal (葉's priority).** Choose **one** base model and **one** iterative-training loop. Then use that loop to drive down false positives, fragmentation and boundary error.

Research questions:

- **RQ1. Model.** Is MiT-b2 (SegFormer, DeepSolar-3M init) good enough to be the single iteration model? Do DINOv3 variants add anything beyond a second opinion?
- **RQ2. False positives.** Which confusers dominate (corrugated roofs and skylights, road markings, solar water heaters, pools)? How much does each lever remove: hard negatives, active learning, ensemble disagreement, Gemini/human screens? FP errors that all models share ("correlated") need different fixes than FP errors from one model only ("uncorrelated").
- **RQ3. Fragmentation and the installation unit.** How much fragmentation comes from tiling, how much from inter-row gaps, and how much from the model? Which merge rule turns segments into "installations" (葉 chose "array group" on 2026-10-02)? Is the installation count robust to that rule?
- **RQ4. Boundaries and area.** How large is the area bias, how much of it comes from definition (panel area vs array outline), and how much from real boundary error?
- **RQ5. Small panels.** What is the true recall below 4 m² once it is measured on at least 50 objects?
- **RQ6. Transfer.** Does the Cape Town model carry over to Durban (0.075 m Vexcel) and Johannesburg (Vexcel 2024 ~7 cm, CoJ aerial 15 cm)? How much local labelling does each city need?
- **RQ7. Census-level uncertainty.** How do the per-object errors aggregate into grid-level area error (σ_Bw, RMSE, total area ratio), the numbers the economics actually uses?

---

## 2. Current state

### 2.1 Models on Cape Town val (14 grids, 1,034 GT objects; object-level evaluator `evalobj.py`)

Source: `~/census_overnight_reports/claude_review_20261017_model/brief.md` (table copied from `~/capetown_eval/ds3m_ft/evalobj/TABLE.md` and `evalobj_v2d1/TABLE_v2d1.md`). Each row uses that model's F1-optimal config, chosen on val. All runs are single seed.

| model (trainset) | obj R | obj P | F1 | R <4 m² | R 4–10 m² (n=228) | FP m²/ha | pxIoU | bIoU* | SWH→PV objs |
|---|---|---|---|---|---|---|---|---|---|
| DS3M zero-shot | 0.957 | 0.435 | 0.598 | 10/12 | 212 | 1.007 | 0.479 | 0.109 | 25/28 |
| MiT-b2 (v2c, 30k it) | 0.963 | 0.892 | **0.926** | 11/12 | 207 | 0.076 | 0.758 | 0.422 | 7/28 |
| DINOv3 ViT-B LVD (v2c, 8k it) | 0.969 | 0.887 | **0.926** | 10/12 | 212 | 0.080 | 0.762 | 0.419 | 6/28 |
| MiT-b2 + SWH boost (v2c) | 0.966 | 0.834 | 0.895 | 10/12 | 214 | 0.128 | 0.751 | 0.413 | 7/28 |
| MiT-b2 (v2d1, 30k it) | 0.956 | 0.864 | 0.908 | 8/12 | 209 | 0.101 | 0.761 | 0.394 | 6/28 |
| Mask R-CNN (production unified_A) | 0.948 | 0.936 | 0.942 | 2/12 | 212 | 0.044 | 0.729 | 0.485 | 1/28 |

\*bIoU is not trustworthy as currently computed (§2.4).

What this says:

- MiT-b2 and DINOv3 LVD are **tied** on v2c. Every gap is inside noise.
- Mask R-CNN has the best precision and is the clear loser below 4 m² (2/12). Its val numbers are also optimistic, because it probably saw these grids in training (`OVERNIGHT_SUMMARY.md`, caveats).
- **v2d1 is genuinely lower than v2c on precision (0.864 vs 0.892) and small objects (8/12 vs 11/12).** Re-scoring both models against both label versions gives the same ranking: mitv2c scores 0.926 on both v2c and v2d1 labels, and mitv2d1 scores 0.908 on both (`rescore_v2d1/TABLE_2x2.md`). The drop therefore comes from training, not from the scorer. The hard-negative coordinate cache was checked and is aligned for both runs (`c5/c5_summary.json`: 0 tile mismatches, 0 points on PV), so that cause is ruled out.
- The causes still open are (i) the 44 SAM3 replacement polygons Gemini never confirmed, some skylight-like (`labelfix/README_v2d.md` item 2); (ii) the new ignore slivers along array edges; (iii) seed noise, which has never been measured.

**SAT DINOv3 (ViT-L/16, satellite-pretrained, freeze=6) on v2d1.**

- 葉 paused it at step ~25.6k of 30k; a resume checkpoint exists at step 25,000 (`v2d1_chain.log`, `sat_interim_25k/`).
- Cost is **1.19 s/it vs MiT-b2 0.385 s/it (3.1×)**. Peak memory is 5.9 GB.
- Internal val IoU (300 tiles) plateaued around 0.95 by step 7.5k; it was 0.945 at 25k. That internal metric is not comparable to `evalobj`.
- A switch rule was written before the result: `~/capetown_eval/ds3m_ft/SAT_DECISION_RULE.md`. SAT must gain ≥1.5 F1 points over mitv2d1, must not lose 4–10 m² recall, and must cost ≤1.5× MiT per round. The cost criterion already looks failed (≈3×), unless the F1 gain is large.

**Iteration cost.** MiT-b2 30k iterations ≈ 3.3 h (v2d1 run 22:23→01:44, `v2d1_chain.log`). MiT's internal val was already 0.939 at 2.5k and 0.946 at 7.5k (`SAT_DECISION_RULE.md`), so a 10–15k-iteration round (≈1.1–1.6 h) is probably enough. **Training compute is cheap; the bottlenecks are labelling and evaluation.**

### 2.2 Label and trainset versions

All under `~/data/ZAsolar/geid_temporal/`. Every version is a fresh copy, and md5 checks confirm that older versions are unchanged.

| version | what changed | train grids / chips | status |
|---|---|---|---|
| v2c (`ct_seg_trainset_v2c_20261015`) | classes 0/1/2/255. 186 confirmed collectors as class 2. HN remapped to 0. ±0.3 m ignore band on Class B edges. | 76 / 19,097 | reference |
| v2d (`…v2d_20261016`) | G1630 corrupt seg130 dropped (≈6,400 m² of false PV in **val**). 60 thin_strip arrays replaced by SAM3 polygons, 44 of them never confirmed by Gemini. Edge slivers set to 255. | 76 / 19,097 | superseded by v2d1 |
| v2d1 (`…v2d1_20261016`) | G1971 00047 reverted to the human label. G1972 src113 tightened by SAM3 (240.9→172.1 m²). | 76 / 19,097 | trained (MiT, SAT partial) |
| v2d2 (`…v2d2_20261017`) | +G1189/G1190/G1238 (葉's earliest hand-SAM grids, 475 polygons, alignment confirmed by 葉 on 10-17). The chain also switches to a cleaned HN cache (`hn_clean`: 118,143 of 145,998 HN points kept, 27,855 dropped as not background). | 79 / 19,904 | built, **training held** (`v2d2_chain.sh.HELD`) |

**葉's hand labels** (`~/zasolar_data/capetown_ye_manual_labels_original/README.md`):

- Core hand-SAM batch: 43 grids / 2,982 polygons (≈51,645 m²). Of these, 33 grids are in train (2,286 polygons), 6 in val (201) and 1 in eval (G1349).
- batch004: 12 grids / 143 polygons, detector predictions reviewed by 葉. 葉 ruled on 10-17 that these count as trusted.
- Older Google Earth hand labels: not used (葉, 10-17).

**Splits.**

- Val: 14 grids, 1,768 tiles.
- Eval: 5 Class B grids (G1349 G1686 G1692 G1800 G2038). Their labels started from Mask R-CNN predictions, so they are biased toward Mask R-CNN.
- Five more listed held-out grids (G1620 G1733 G1785 G1842 G2905) are not built into eval/. Whether they have labels must be checked (`ct_seg_trainset_v2c_20261015/README.md`).

**SWH (solar water heaters) are weakly labelled.**

- There are 186 confirmed collectors, about 0.07% of labelled pixels.
- Heaters were recorded only where they had been detector FPs, so many real heaters sit in "background" (`OVERNIGHT_SUMMARY.md`, SWH section).
- Gemini 3.5-flash-lite collector calibration on native imagery failed: P = 0.65 (`ct_seg_v2_native_STATUS.md`). An older gemini-3.8-flash calibration reached P 0.97 but only R 0.37.

### 2.3 Error inventory (the three main problems)

- **False positives.**
  - On Durban DS3M zero-shot "extra" area: 46% corrugated roofing or skylight strips, 31% other (road markings, zebra crossings, pools, ground stacks, shadows), 21% real PV that the census missed, 2% SWH. These are Gemini-only labels (`OVERNIGHT_SUMMARY.md` §A).
  - In the Cape Town val QA (100 tiles), 13 of 32 FP components were true model FPs. Of the rest, 12 looked like missing labels and 7 were heaters (§B of the same file).
  - Most FP area is a **correlated** confusion that every model makes. Fusing models will not remove it; it needs hard negatives or an object-level judge (`claude_review_20261016/claude_answer.md` §1).
- **Fragmentation.**
  - Window tiling cuts large arrays, and the old polygon stitch (IoU>0.1 merge) fails to rejoin small overlaps.
  - Panel-row segmentation leaves inter-row gaps. Zero-shot DS3M on 114 large Durban blocks: 82% split into pieces, 83% holed, pooled area −16%. A 1 m closing still left 67% split (`claude_review_20261016/brief.md` §b).
  - For the fine-tuned models, no fragmentation metric exists yet.
- **Boundaries.**
  - Fine-tuned models reach bIoU ≈0.42, against Mask R-CNN ≈0.49 on val, but bIoU is unreliable (§2.4).
  - SAM3 text-only prompting is much worse than the fine-tuned segmenters as a census engine: area precision 0.43–0.61, IoU 0.41–0.54 across variants (`ct_seg_sam3_fullcover_v1_20261016/metrics_mean.csv`). SAM3 is useful as a labelling aid only.

### 2.4 Evaluation instruments and known flaws

- **`evalobj.py`** (object-level, 14 val grids) is the current best instrument: object P/R/F1 with Wilson CIs, size-bin recall, FP area per ha, pixel IoU, boundary IoU, SWH→PV.
- **Small-object sample is too small.** The <4 m² bin has n = **12** in evalobj, against 42 in the older pixel evaluator, which split arrays into fragments and shrank them by the ignore band (`ds3m_ft/claude_review.md` issues 3–4).
- **Boundary IoU depends on 255 handling.** The same predictions score subset bIoU 0.162 (v2c labels as-is), 0.484 (v2d1 as-is) or 0.898 (v2d1 excluding 255) (`TABLE_2x2.md`). The ±0.3 m ignore band is 6 px wide, while the boundary width is 3 px (`claude_review.md` issue 3).
- **Eval grids are circular for Mask R-CNN**, whose labels seeded them (`OVERNIGHT_SUMMARY.md` caveats).
- **Image tiles are shared across splits** (train∩val 379 tiles). The labelled pixels do not overlap, but one building can be half in train and half in val. Held-out numbers are therefore slightly optimistic (`claude_review.md` issue 8).
- **No run has been repeated.** Seed noise is unknown. A typical rule of thumb is 0.5–1 F1 point (assumption).

### 2.5 Durban

Sources: `~/zasolar_data/durban_overnight_20261016/STATUS.md`, `~/census_overnight_reports/durban_20261016.md`.

- **Census state.**
  - Census v3 (Mask R-CNN, stitch025, hole fill): 30,604 blocks / 1,889,920 m².
  - The SAM3 + Gemini rule-D screen then left 27,328 blocks / 1,635,102 m². It removed 3,276 FP candidates (254,819 m²) and left 703 blocks (36,472 m²) for review.
  - Gemini kept a block that the human label calls FP (030281). A v4 candidate that drops it exists: 27,327 blocks / 1,634,973 m².
- **Hard-negative chips.**
  - hn_v1: 4,277 chips (3,274 Gemini negatives, 3 human negatives, 1,000 PV controls).
  - hn_v1_r005: the same chips resampled to 0.05 m. Chips with heavy padding fell from 92% to 65%.
  - The 3,274 Gemini negatives are **unaudited**. In a 50-chip sheet several look like real PV (#9, #16, #34, #41) (`hn_v1/audit50/`).
- **Open 葉 decisions:** adopt v4 (drop 030281); audit50 before using Gemini HN; use the r005 resample.
- **Labels.** Li's Durban labels exist on Dropbox (`RA_Solar/Li/DBN`) and are not copied locally (`training_resources_inventory_20261017.md`).

### 2.6 Johannesburg (maybe)

- **CoJ aerial 2023 (15 cm), 5 grids, resampled to 0.05 m:** MiT-b2 (v2c) F1 0.662 (objR 0.53, P 0.88); DINOv3 LVD 0.673; Mask R-CNN v4 0.779. Without the resample, the segmenters collapse (F1 0.18–0.28) (`ds3m_ft/joburg5/TABLE.md`). The Cape Town segmenters do **not** transfer out of the box.
- **A JHB CBD 25-grid Vexcel evaluation chain is running.** Downloading and tiling are done and the Mask R-CNN json is written; the segmenter runs are pending (`joburg25_vexcel/chain.log`).
- **Labels available:**
  - 葉's JHB Vexcel 2024: 25 grids, 3,835 reviewed polygons (`ZAsolar/docs/briefs/BRIEF-census-train-resources-2026-10-02.md`).
  - 葉's JHB aerial_2023 labels (56 gpkg): copied from Dropbox, but the recipe is archived and they are not for new training sets (`training_resources_inventory_20261017.md`).
  - Li's JHB CBD supplement: 25 gpkg.

### 2.7 The delivered (legacy) Cape Town census: the baseline to beat

Source: `ZAsolar/docs/briefs/BRIEF-census-detector-speed-2026-09-23.md`.

- Delivered 2026-06-16: Mask R-CNN unified_A, `v4_canonical` post-processing, solar_cls filter. 2,083 grids, **111,801 installations, ≈5.0 km²**.
- On 99 GT grids: total area ratio 1.083, slope through the origin 1.019, aggregate F1 0.833, grid-level dispersion σ_Bw 0.516.
- These 111,801 polygons are the seeds of the Cape Town backdating.
- SSEG side-signal: of 5,081 commissioned SSEG parcels in the main sample, **342 (6.7%) have no census polygon** (`solar_backdating/docs/replan_v2/RUN-cv-appendix-figures-v3r1-2026-10-14.md` §2). This is a rough external recall signal for the legacy census, since coordinate errors, ground mounts and removed systems also land there.

### 2.8 Compute and people

- **GPU and hosts.** One 8 GB GPU on rog (RTX 4070 Laptop); MiT-b2 training uses 5.9 GB. Only one GPU job can run at a time (all chains use a `waitgpu` guard).
- **City-scale inference is unmeasured.** The only timing is evalobj on 632 eval tiles: ≈30 min, ≈3 s per 100 m tile, including scoring (`evalobj_v2d1/run.log`). The Cape Town census area is roughly 250k such tiles (assumption, from 4.29M 400-px production chips at ~8 cm). That implies up to ~8–9 GPU-days for one MiT-b2 pass (upper-bound assumption). SAT ViT-L would cost ≈3× more.
- **Other resources:** free concurrent local Gemini; Grok Build/Codex for code.
- **Roles.** 葉 decides models, data versions and label deletion. 葉's labelling time is the scarcest resource.

---

## 3. Metrics, milestones and exit criteria

### 3.1 Candidate metrics, and when each should be frozen

Rule: a metric is frozen (written down with its threshold or decision rule) **before** the first result it will judge is seen. Until a metric is frozen it is only reported, never used to accept or reject anything.

| # | candidate metric | measures | instrument / set | freeze at | notes |
|---|---|---|---|---|---|
| C1 | Object F1 (plus P, R) at the val-chosen threshold, paired bootstrap over tiles | overall detection | T-B | **M0 exit** (target Fri 10-23) | Proposed **primary** for round accept/reject. |
| C2 | Recall <4 m² and 4–10 m², bins by GT polygon area, paired per object | small panels | T-B (≥50 objects <4 m²) plus val 4–10 m² (n=228) | M0 exit | Guardrail. The <4 m² bin needs T-B; val n=12 cannot decide. |
| C3 | FP objects per ha and FP m² per ha on random tiles, broken down by confuser type | false positives, unbiased rate | T-B random subset (~40 tiles) | M0 exit | Guardrail. Type labels come from Gemini 3.8-flash plus a human check. |
| C4 | FP pixels per chip on held-out Durban HN chips | correlated confusers | hn_v1_r005 val split (695), after the audit50 filter | M0 exit, only if Durban HN enter training | Guardrail. |
| C5 | Share of SWH objects predicted as PV (object level) | heater/PV confusion | val (28 SWH) plus T-B heaters | M0 exit | Guardrail. Pixel SWH IoU is dropped (labels incomplete). |
| C6 | Boundary F-score at d = 0.25 m and 0.5 m on raw polygons (no 255 band) | boundary quality | T-B boundary subset (~100 arrays) | **M2** | Replaces bIoU. Needs the evaluator fix (E1) and a boundary-careful subset. |
| C7 | Split rate (GT arrays hit by ≥2 predicted pieces), glue rate (one predicted unit covering ≥2 GT installations), pieces per hit array | fragmentation and over-merging | val plus T-B, plus Durban XL+L 114 blocks | **M2** | Depends on the installation-unit rule (D8). |
| C8 | Area ratio (pred/true) per grid and in total; σ_Bw and RMSE of grid area ratios | census-level area error, the economics number | T-A (Li grids) plus T-B | **M2** (after D7 area definition) | The legacy census referee (σ_Bw 0.516 on 99 grids) gives a direct comparison. |
| C9 | Installation-count sensitivity to the merge gap {0.25, 0.5, 1, 2} m inside building footprints | robustness of the unit | city sample | M2 | If the count moves by more than ~10%, the unit definition becomes a stated paper risk. |
| C10 | SSEG-matched recall: share of commissioned SSEG parcels with a census polygon (6.7% missing in the legacy census) | external recall proxy | SSEG 5,081 parcels | M4 | Descriptive only (coordinate errors, ground mounts). Shares its matching with the backdating plan. |
| C11 | Cross-city object F1 and area ratio | transfer | Durban (Li DBN), JHB25 Vexcel | M5 | Report only until a city-specific decision is needed. |
| C12 | Human ceiling: F1 and boundary F between two labellers on the same 20 tiles | label ambiguity | T-B double-labelled subset | M0 exit (report) | Defines the "stop when model ≈ humans" rule. |

**Decision rule template (to freeze at M0 exit).** Accept a round only if C1 improves by more than the noise floor and no guardrail (C2–C5) regresses by more than its own noise floor. Noise floor = max(seed-to-seed gap from R0, half-width of the paired-bootstrap 95% CI on T-B).

### 3.2 Phased milestones

| milestone | content | exit criteria | target |
|---|---|---|---|
| **M0. Measurement foundation** | evaluator fixes (E1); T-A and T-B built and frozen; R0 noise floor; v2d1 attribution | T-B frozen with a hash and a written label protocol, holding ≥50 objects <4 m², ≥40 random tiles and ≥5 tiles per main confuser type. Seed gap known for C1–C3. v2d1 drop attributed (44 flagged polygons vs slivers vs noise) or declared noise. C1–C5 frozen in a DECISION doc signed by 葉. | Fri 10-23 |
| **M1. Base model and recipe locked** | MiT-b2 confirmed (or SAT switched by the pre-written rule); iteration count fixed from logs (plateau); trainset baseline chosen (v2c / v2d1 / v2d1-with-flags-255) | 葉 signs: model, init, iterations, batch mix, data version, HN mix. The SAT decision is recorded. | Fri 10-23 to Tue 10-27 |
| **M2. Post-processing and unit definition** | E4 overlap logit averaging plus TTA; E5 regularisation and footprint-constrained merging; area definition (D7) and unit rule (D8) | Split rate on large arrays falls by more than the noise floor with no C1 loss and no glue-rate rise above an agreed cap. Area definition chosen. C6–C9 frozen. | Fri 10-30 |
| **M3. Iterative loop** | rounds R1…Rn, one change each (HN-clean, v2d2 grids, Durban HN, active-learning batches, SWH audit) | Stop at the first of: two consecutive rounds below the noise floor with area bias moving <2%; model–human agreement on T-B ≈ the human ceiling (C12); 5–6 rounds done. | from wk 2, ~4–6 weeks (assumption) |
| **M4. Cape Town city inference and QA** | throughput measured; pregate (if any) re-validated at 0.05 m; full-city run in tmux, resumable; census v_seg with CIs | C8 on T-A beats or matches the legacy census (σ_Bw 0.516, area ratio 1.083). FP-screen precision is estimated by a stratified audit, not taken from Gemini. A census-version crosswalk to the 111,801 legacy polygons is written (feeds backdating). | after M3 |
| **M5. Durban, then JHB (gated)** | Durban: FP-screen gold audit on v3/v4, then a CT-model transfer test on Li DBN labels; JHB only if JHB25 Vexcel shows usable transfer or 葉 funds JHB fine-tuning | Durban 1,635,102 m² gets a correction factor and CI. A re-run decision is taken on evidence. JHB in or out by 葉. | parallel CPU work from wk 1; GPU after M4 |

---

## 4. Experiments ranked by value / effort

Effort counts 葉's hours and GPU hours separately, because those are the two scarce resources. Rows E1–E10 are the realistic two-week scope.

| rank | experiment | fixes which problem | value | effort | first concrete step |
|---|---|---|---|---|---|
| E1 | **Evaluator fixes**: 255 excluded consistently from objects, FP area and boundaries; bins by GT polygon area; boundary F at 0.25/0.5 m; per-object McNemar; per-grid bootstrap; worker seeds made reproducible (`getpid` in the seed, `claude_review.md` issue 10) | measurement | high | low (Codex, CPU, 1–2 d) | Re-score the saved `probs_mitv2c`/`probs_mitv2d1` (`rescore_v2d1/`). The tables must reproduce `TABLE_2x2.md` F1 exactly before new columns are trusted. |
| E2 | **R0 noise floor**: reference recipe with a second seed | measurement | high | low (≈1.6–3.3 GPU-h) | Choose the iteration count from `run_v2c/log.jsonl` and `run_v2d1/log.jsonl` (first val check ≥99% of final), then rerun with a new seed. |
| E3 | **v2d1 attribution**: v2d1 with the 44 FLAG polygons → 255 (one change), plus a paired per-object table and FP typing of v2d1-only FPs (`c4_fp/evalobj_dumpprob.py`) | FP, label quality | high | low (1 run plus CPU) | Build v2d3_flag255 in a new folder. If it recovers v2c precision, the flagged SAM3 polygons were teaching skylights as PV. |
| E4 | **Overlapping-window logit averaging** (50% overlap, cosine weights, one global probability raster per grid before polygonising) plus flip TTA | fragmentation at tile edges, edge FPs | high | low (inference only, ~1–2 GPU-h on val) | Compare stride = window vs window/2 on val and on the Durban XL+L 114 blocks: split rate, coverage, C1. |
| E5 | **Rectangle regularisation plus footprint-constrained merging**: closing r ∈ {0.5, 1, 1.5} m; replace by the minimum rotated rectangle when its IoU with the mask is ≥0.85; merge pieces within gap d and orientation <5° inside one CoCT 2023 building footprint (1,444,950 footprints, `ODP_SPLIT_12/MapServer/11`) | boundaries, holes, row gaps, area bias, unit definition | high | low–medium (CPU, 2–3 d) | Sweep on val. Report C6 proxies, area bias, split rate and glue rate. Footprints are used only to *merge* and to *flag* off-roof objects for review, never to delete (葉 rejected footprint pre-filtering on 2026-10-01 because new builds are missing). |
| E6 | **Hard-negative rounds**, one per round: (a) HN-clean cache (the HN half of v2d2); (b) Durban hn_v1_r005 at 10–15% of each batch | correlated FPs (corrugated, skylights, road markings) | high | medium (needs audit50; 1 run each) | Before (b), drop chips where the current model confidently predicts a ≥2 m² rectangle unless a human confirms them. 21% of DS3M's Durban "extra" area was real PV. Apply the same 0.075→0.05 m resample and blur to Cape Town positives, so the model does not learn "blurry = background". |
| E7 | **Ensemble and disagreement triage**: MiT + DINO LVD probability average; object-level precision by agreement class (both / one model / Mask R-CNN only) | review load, uncorrelated FPs | medium–high | low (CPU on saved probabilities) | If the both-agree class has P ≥0.97 and the single-model class P <0.6, only disagreements go to Gemini or humans. |
| E8 | **Active-learning round 1**: mine ~100–150 unlabelled Cape Town tiles from five buckets (model disagreement, uncertain 0.3–0.7, small <4 m², confident but Gemini-flagged, 10% random; no bucket >30%); 葉 corrects pre-filled masks (training only, never T) | FP, small panels, label gaps | high | medium (≈4–6 h of 葉 per round, assumption; 1 GPU-h mining) | Mining inference over the census area minus the val/T grids. Warm-start, ~⅓ schedule, new tiles oversampled 2×. |
| E9 | **Inference upscale 1.5×** (single run on val) | small panels | medium | very low | If <4 m² recall on T-B rises with no FP cost, resolution is a bottleneck. That unlocks E13. |
| E10 | **SAT DINOv3 closure** under `SAT_DECISION_RULE.md` | model choice | low–medium | low (evalobj on `sat_interim_25k` ≈1 GPU-h, or resume 4.4k it ≈1.5 h) | Score against mitv2d1 with the same evaluator and label hash. Expected outcome: stay on MiT (3.1× cost). |
| E11 | **SWH audit and object-level SWH**: gemini-3.8-flash proposes heaters (P 0.97, R 0.37 on an earlier calibration), a human confirms, then grids are marked "fully audited" | SWH→PV confusion | medium | medium (human confirmation) | Start with the val + T-B grids, so that C5 is measured on complete labels. |
| E12 | **City-scale throughput and pregate check** | feasibility of M4 | high (for M4) | low | Time pure inference on 1,000 tiles for MiT-b2 (fp16, window size, batch). Test whether the DINO 0.45 pregate (validated on 8 cm Mask R-CNN chips) keeps all T-A/T-B PV at 0.05 m. |
| E13 | High-resolution head plus boundary/Lovász loss | boundaries, small panels | medium | high | Only if E9 or E5 show a resolution bottleneck. |
| E14 | Gated SAM box+point refinement on pieces where E5 regularisation fails (accept only if IoU with the seg mask ≥0.7 and area ratio 0.9–1.3) | irregular boundaries | medium | medium | After E5. |
| E15 | Cross-city: finish the JHB25 Vexcel chain; Durban transfer test on Li DBN labels; JHB fine-tune with 葉's 25 Vexcel grids | transfer | medium | medium | Gated by M4 and 葉 (D11, D12). |
| E16 | SAM adapters / CRF | boundaries | low | medium–high | Deferred. Regularisation should cover most of it. |

Notes on the ranking:

- **E1–E5 and E7 need no labels and almost no GPU**, so they can all finish in week 1. They answer "how much of fragmentation and boundaries is post-processing?" before 葉 spends labelling hours.
- **E6 is the main lever for correlated FPs** (46% of Durban extra area was roof/skylight). Fusion and post-processing will not move that share.
- **E8 is where 葉's time goes.** Run it only after M0, so that each round can be judged on T.

---

## 5. Data and labelling needs

| need | size | who | hours (assumption) | when | notes |
|---|---|---|---|---|---|
| **T-A: Li's Cape Town labels** as an independent test | Li's CT grids (L namespace; 18 gpkg in `~/zasolar_data/dropbox_label_copies_20261017/Gao_annotations_Capetown_Li/`; the appendix plan cites 74 grids / ~3,600 panels in total) | Codex: copy, alignment check, protocol check; 葉 rules on usability | 2–4 h CPU, 0.5 h 葉 | wk 1 | Different annotator, never seen by the segmenters. Some grids were used to calibrate Mask R-CNN thresholds, so they are biased for Mask R-CNN comparisons only. A known ~1.4 m offset exists on L1952–54 (backdating appendix Q3). Check whether Li labels outlines or panels and how heaters are handled. |
| **T-B: new hand-drawn tiles**, no pre-fill | 60–100 tiles: ~40 random from grids in neither train nor val; ~40–60 stratified (small panels, skylight, corrugated, SWH, pool, road markings, edge-crossing large arrays); ≥50 objects <4 m² | 葉 or an RA (D6) | 6–10 h | wk 1 (Mon–Thu) | Draw from the 5 unbuilt held-out grids if they are unlabelled, else from new random grids. Freeze with a hash and a written protocol (outline includes row gaps; SWH as its own class; a stated 255 rule). |
| Double labelling (human ceiling) | 20 T-B tiles | a second labeller | 1.5–2 h | wk 1 | Gives C12. |
| Boundary-careful subset | ~100 arrays inside T-B | same labeller, extra care on edges | 1–2 h | wk 1–2 | For C6. |
| 44 flagged SAM3 polygons | 44 | 葉 | ≈1 h, or set to 255 without review | Mon 10-19 | Label-change decision (D4). New version folder only. |
| Durban audit50 | 50 chips | 葉 | 5 min | Mon 10-19 | Gates E6b. |
| Durban FP-screen gold audit | ~400 blocks stratified by Gemini verdict and score | 葉 or RA, blind to the Gemini verdict | 2–3 h | wk 2 | Turns the Durban total into an estimate with a CI. |
| Active-learning corrections | ~100–150 tiles per round | 葉 (pre-filled) | 4–6 h per round | from wk 2 | Training data only. |
| SWH completion | val + T-B grids first | gemini-3.8-flash proposals, human confirms | 1–2 h | wk 2 | Native flash-lite calibration failed (P 0.65), so Gemini alone is not enough. |
| Li Durban labels (`RA_Solar/Li/DBN`) | ~10 grids (assumption) | copy only | 0.5 h | wk 2 | For the M5 transfer test. |

Data hygiene, applied to every item: originals are never edited; each change produces a new versioned folder with a README and md5 check (as v2d/v2d1/v2d2 already do); 葉 approves any label deletion.

---

## 6. Risks

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| Decisions taken on noise (single seeds, 12 small objects) | high (already happened with v2c/v2d1) | wrong data or model choice that persists across rounds | R0 noise floor; T-B ≥50 small objects; paired tests; one change per round |
| Test set contaminated or biased (pre-fill, Mask R-CNN-derived labels, threshold tuning on T) | medium | over-optimistic census claims | T-B drawn without pre-fill; T used only for go/no-go; val for thresholds; the 5 eval grids demoted to "biased dev" |
| Gemini-labelled hard negatives contain real PV | high (audit50 shows candidates) | the model learns to ignore real PV, so recall falls silently | audit50 first; model-confidence filter; human confirmation of rectangles ≥2 m²; unconfirmed chips → 255 |
| 葉's labelling time is the bottleneck | high | the loop stalls | post-processing first (E4/E5); cap rounds at 100–150 tiles; consider an RA for T-B and the audits (D6, D9) |
| City inference takes days on one 8 GB GPU | medium–high | M4 slips; GPU contention with training | E12 measurement first; fp16; pregate only after re-validation; tmux, resumable, one GPU job at a time |
| Area definition unresolved (panel vs outline differ by 10–15%) | medium | the census area cannot be compared with Durban or the legacy census | D7 at M2; report both; the label protocol written into T |
| Unit (merge) rule glues neighbours or splits rows | medium | wrong installation counts for economics and wrong backdating seeds | C7 glue rate as a guardrail; C9 sensitivity; footprint constraint |
| Census polygons change under backdating | certain once M4 runs | backdating dates become orphaned or need redoing | crosswalk at M4 (IoU match, dates inherited where matched); this cycle's backdating uses the 2026-06-16 inventory (see backdating plan) |
| Transfer failure to JHB and Durban | high for JHB (F1 0.66 at 15 cm) | city-specific labelling needed | gate M5; use existing JHB Vexcel labels before new labelling |
| Over-merging with footprints where footprints are missing or wrong (new builds) | medium | missed merges or wrong merges | footprints only merge and flag, never delete; the fallback rule (gap + orientation) applies without a footprint |

---

## 7. Decisions needed from 葉

1. **Base model.** Confirm MiT-b2 (DS3M init) as THE iteration model and DINOv3 ViT-B LVD as inference-only second opinion (for E7 and E8 mining). Settle SAT DINOv3 strictly by `SAT_DECISION_RULE.md`: either score the 25k interim checkpoint (≈1 GPU-h) or resume to 30k (≈1.5 h), or close it unscored. *Recommendation:* score the interim checkpoint once, then close.
2. **R0 diagnostic night.** Approve (a) a second seed of the reference recipe and (b) v2d1 with the 44 flagged → 255, both at the iteration count read from the logs. Or keep all training on hold. *Recommendation:* approve. About 5 GPU-hours, no new labels.
3. **v2d2.** Train as built (3 new grids + HN-clean cache in one run), or split into "v2d2 grids only" and "HN-clean only", each one change. *Recommendation:* split. Two runs of ≈1.6 h fit in one night.
4. **The 44 Gemini-unconfirmed SAM3 polygons.** Review them (≈1 h), or set them to 255 in a new version until reviewed. G1971 00047 and 00250 look skylight-like.
5. **Durban carry-overs** (pending since 10-16): (a) adopt v4 (drop 030281); (b) look at `hn_v1/audit50/audit50_sheet.jpg` before any Gemini HN is used (≤5 bad → use as background, else 255); (c) use hn_v1_r005 (0.05 m) for training.
6. **Test set.** Approve T-A (Li's CT grids) plus T-B (60–100 new tiles, no pre-fill). Decide who labels T-B and the double-labelled 20, and from which grids. Rule that the 5 eval grids are "biased dev" and never decide anything.
7. **Area definition.** Array outline (row gaps included; matches the labels and Durban) as primary, with panel area as secondary. Or the reverse. *Recommendation:* outline as primary.
8. **Installation unit.** Confirm that the "array group" rule from 2026-10-02 is implemented as gap d plus orientation plus same building footprint. Set the glue-rate cap that M2 must not exceed.
9. **Labelling budget.** How many 葉-hours per active-learning round (proposal: 4–6 h, 100–150 tiles)? May gemini-3.8-flash pre-flag FP types and heaters for 葉 to confirm?
10. **Metric freeze.** Agree that C1–C5 are frozen at M0 exit (Fri 10-23) and C6–C9 at M2, each before the results it judges.
11. **JHB scope.** Is JHB in this cycle? *Recommendation:* no; decide after the JHB25 Vexcel numbers and M4.
12. **Durban strategy.** Keep the Mask R-CNN v3/v4 census with an FP-screen gold audit and correction factor, or re-run Durban with the Cape Town segmenter after M4? *Recommendation:* audit now; decide the re-run on the M5 transfer test.
13. **Docs home.** Where the census TRACKER and DECISION docs live. *Proposal:* `ZAsolar/docs/experiments/2026-10-ct-seg-census/` with TRACKER.md and DECISION-*.md. Today the state is spread over `~/capetown_eval/` and `~/census_overnight_reports/`.

---

## 8. Two-week schedule (Mon 2026-10-19 → Fri 2026-10-30)

GPU rule: one GPU job at a time; training runs overnight in tmux with resumable scripts. Daytime GPU is for short inference only. Every working day ends with the TRACKER updated (Definition of Done).

| day | 葉 (decisions / labelling) | CPU / agents (Codex / Grok) | GPU (night unless noted) |
|---|---|---|---|
| **Mon 10-19** | Decision batch D1–D8 (30 min). audit50 (5 min). Review the 44 flagged polygons or rule them →255. | E1 evaluator fixes start. Read val curves → round iteration count. T-B candidate tile list (random plus stratified). T-A copy and alignment check. | Day: E10 SAT interim evalobj (≈1 h). Night: **R0a** seed 2 + **R0b** v2d1-flag255 (if D2 approved). |
| **Tue 10-20** | T-B labelling batch 1 (~30 tiles, ~3 h) | Score R0 with the fixed evaluator → noise floor. Paired per-object table and FP typing (E3). E4 overlap-inference script. | Night: E4 overlap + flip TTA on val; DINO LVD probability dump for E7. |
| **Wed 10-21** | T-B batch 2 (~30 tiles); double-label 10 | E4 analysis (split rate, C1). E5 regularisation/merge sweep (CPU). E7 agreement-class precision. | Night: **v2d2-grids-only** run (if D3 approved). |
| **Thu 10-22** | T-B batch 3 plus double-label 10; boundary subset | T-B frozen (hash, protocol). Draft metric memo (C1–C5 thresholds = noise floor). Durban hn_r005 confidence filter (E6b prep). | Night: **HN-clean-only** run. |
| **Fri 10-23** | **M0 exit review** (1 h): sign the metric DECISION and the recipe (M1) | All runs scored on T-B. Write DECISION doc and TRACKER. | Night/weekend: next single-change run chosen by M0 results (e.g., Durban HN if audit50 passes). |
| **Mon 10-26** (NZ Labour Day) | none | none | Night: active-learning mining inference over unlabelled Cape Town tiles (E8). |
| **Tue 10-27** | — | Select AL batch 1 (100–150 tiles, bucket caps). E12 throughput timing plan. E5 final config. | Day: E12 throughput timing (1,000 tiles), E9 upscale test. |
| **Wed 10-28** | AL batch 1 corrections (~3 h). Durban FP-audit sample review starts (blind). | Durban 400-block stratified sample built. SWH proposals with gemini-3.8-flash on val + T-B grids (E11). | — |
| **Thu 10-29** | AL batch 1 corrections (rest) | Build round-1 trainset (new folder). | Night: **R1** warm-start ⅓ schedule, new tiles 2×. |
| **Fri 10-30** | **M2 review** (1 h): area definition, unit rule, C6–C9 frozen; R1 accept/reject | Score R1 on T-B. Write the post-processing DECISION. Plan the next 2 weeks (rounds R2–R3, M4 preparation). | — |

**Two-week deliverables.**

- A frozen test set T with protocol and hash.
- The noise floor.
- An answer to "was v2d1 worse, and why".
- A signed recipe and metric DECISION.
- Post-processing numbers for fragmentation and boundaries.
- Active-learning round 1 trained and judged.
- The Durban FP-audit sample drawn, plus a throughput estimate for the Cape Town city run.
