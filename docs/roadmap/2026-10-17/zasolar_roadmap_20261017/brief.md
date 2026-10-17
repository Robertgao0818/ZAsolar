# Brief: forward ROADMAP for the ZAsolar project (2026-10-17 NZDT)

Write `~/projects/zasolar_roadmap_20261017/ROADMAP.md` (English, Markdown). This is the ONLY file you may create or edit. Everything else is read-only. No GPU, no training, no downloads, do not touch tmux `li_infer` or PID 374974, do not run inference.

## Sources to read (read-only)
- ~/projects/zasolar_plans_20261017/census_plan.md, backdating_plan.md, mit_upgrade_design_v3.md
- ~/projects/zasolar_verification_20261017/VERIFICATION.md
- ~/projects/zasolar_evalfix_20261017/OPEN_QUESTIONS_round1.md, REVIEW_claude.md (and RESULTS.md if useful)
- ~/zasolar_data/capetown_manifest_v3_20261017/, capetown_manifest_v3_1_20261017/, capetown_manifest_v3_2_20261017/ (pending.md, NAMING_CONVENTION.md, overlap_proof*.json, recount_independent*.txt)
- ~/zasolar_data/geid_temporal/ct_sseg_reference_v1_20261017/RUN.md
- ~/capetown_eval/ds3m_ft/joburg25_vexcel/TABLE.md
- ~/zasolar_data/durban_overnight_20261016/STATUS.md
- ~/projects/auto_research_workflow/WORKFLOW.md
- ~/zasolar_data/training_resources_inventory_20261017.md

## Facts decided today by 葉 (treat as ground truth; they override older plans)
1. Grid IDs follow ADR-0002: CPT1842, JNB####, GQB####, DBN####, etc. (regex ^[A-Z]{3}[0-9]{4}$). Work will expand to all of South Africa.
2. MiT-b2 is NOT the only model (this overrides census_plan "confirm MiT-b2 as THE iteration model"). Candidates: b2 baseline (straight copy of DS3M recipe), SegNeXt-B, ConvNeXt-T(+UPerNet); MiT-b3 optional. Phase 1 cap ≈26 GPU-h, NOT yet approved. MiT-b2 (v2d2) training is paused.
3. JHB25 (with pre-gate, GT 2,068): MiT v2c F1 0.899, SAT DINOv3 interim 25k 0.888, DINOv3 LVD 0.888, MiT v2d1 0.874 (<4 m² recall 10/26), Mask R-CNN 0.775. SAT fails the switch rule (training cost 3.1× MiT) and is paused at 25k.
4. Clean test set = Li's no-KML batch (17 grids, 1,260 polygons). Main metric on group A only (8 aligned, fully labelled grids, 447 polygons; only ~30 panels <4 m², so small-panel recall is directional). ALL of Claude's round-1 recommendations (OPEN_QUESTIONS_round1.md) are approved: boundary F frozen at 0.25/0.5 m (1.0 m diagnostic), drop seg130 in new code, adopt v3_1 then v3.2, test-set grouping rules, eval switches to pixel-centre rasterisation, training-side rasterisation/dedupe changes need separate approval. Test inference approved at 0.9 GPU-h (MiT v2c + v2d1 on the 544 labelled-area tiles, scored on 443 canonical tiles), and waits until the rules freeze.
5. There is train/val/eval tile-sharing leakage; fix in the next manifest (dedupe by tile or buffer). Test set is not affected (nearest train tile ≈23 km).
6. SSEG is useful only as a precision check, not for recall (SSEG parcels are only 2–4% of labelled parcels; register ends Jul 2024; none in Li test grids).
7. Exploration mode: metrics not locked yet; 葉 decides on models, data and labels.
8. The single rog GPU (8 GB) is the bottleneck.
9. Bot roles: 光伏·项目主管 = design with Claude; 光伏普查·开普敦 = Cape Town execution; 光伏普查·SAM3 = Durban, SSEG and backdating. Grok Build writes code (per WORKFLOW.md).
10. C4 (prob dump) is blocked behind an old Joburg chain waiting on SAT; 葉 said leave it alone for now.

## What ROADMAP.md must contain
- Phases ordered: (0) this week, (1) next 2–4 weeks, (2) expansion beyond Cape Town & Durban to the rest of South Africa.
- Census and backdating as SEPARATE tracks, plus a short section on where they connect (seed freeze / polygon crosswalk, aerial-year checks).
- Each phase: goal, concrete tasks, owner bot, inputs (file paths), exit criteria, GPU-hour estimates (always labelled "estimate"), and decisions 葉 must make.
- Dependencies, risks, and an explicit "stop doing" list.
- Every number must come from a source file (cite the file) or be marked "estimate"/"assumption". Do not invent numbers.
- A short "Top decisions for 葉" list at the top.
