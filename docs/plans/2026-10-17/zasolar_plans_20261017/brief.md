# Brief for Claude: two ZAsolar research plans (2026-10-17 NZDT)

Write TWO separate plans, in English, in this folder:
- `census_plan.md` — CENSUS: present-day rooftop PV segmentation + area census, Cape Town and Durban (maybe JHB).
- `backdating_plan.md` — BACKDATING: dating when each PV installation appeared, from temporal imagery.

## Context to read (read-only)
- ~/census_overnight_reports/ (esp. claude_review_20261016/, claude_review_20261017_model/)
- ~/capetown_eval/ds3m_ft/OVERNIGHT_SUMMARY.md, rescore_v2d1/TABLE_2x2.md, SAT_DECISION_RULE.md, claude_review.md
- ~/capetown_eval/labelfix/README_v2d.md, PROGRESS.md
- ~/zasolar_data/durban_overnight_20261016/STATUS.md
- ~/zasolar_data/capetown_ye_manual_labels_original/README.md (葉's own hand SAM labels, 43 grids / 2,982 polygons)
- ~/zasolar_data/training_resources_inventory_20261017.md
- ~/projects/solar_backdating/ (README.md, CLAUDE.md, SHARED_FROM_ZASOLAR.md, docs/replan_v2/, docs/briefs/), ~/projects/ZAsolar/docs/briefs/
- ~/data/ZAsolar/geid_temporal/ (directory names + READMEs only; don't open big files)

## Constraints
- EXPLORATORY project; evaluation metrics are NOT fixed. Propose candidate metrics and the milestone at which each should be fixed. Do not assume fixed metrics.
- Census priority for 葉: choose ONE base model + an iterative-training loop. Current lean MiT-b2, DINOv3 as second opinion. MiT-b2 training on v2d2 is ON HOLD until 葉 approves.
- Main census problems: false positives (corrugated roofs, skylights, road markings, water heaters), fragmentation, boundaries.
- Resources: one 8 GB GPU on rog; free concurrent local Gemini; Grok Build/Codex for code; 葉 decides models, data versions and label deletion.
- Never overwrite originals; new versions go to new folders.

## Each plan must have
1. Goals and research questions  2. Current state (cite file paths and numbers)  3. Phased milestones with exit criteria  4. Experiments ranked by value/effort  5. Data and labeling needs  6. Risks  7. Decisions needed from 葉 (numbered list)  8. Concrete 1–2 week schedule (from Mon 2026-10-19)  9. "Top 5 next steps" section at the very top.
