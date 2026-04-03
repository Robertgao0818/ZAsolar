# Cross-Review

TRIGGER when: user says "/cross-review", "review my changes", "审查", "交叉审查", or after completing a significant code change or experiment run.

## What This Skill Does

Orchestrates a cross-review cycle between an executor (current session) and an independent reviewer (spawned agent).

## Instructions

1. Read `.agents/harness/README.md` for the full protocol
2. Determine scenario from user intent:
   - `code`: code changes to detect_and_evaluate.py, train.py, export_coco_dataset.py, scripts/, configs/
   - `experiment`: training experiment completion, benchmark results interpretation
3. Generate `run_id`: `YYYYMMDD-HHMM-{scenario}-{slug}`
4. Create `active/{run_id}/` directory
5. Write `handoff_r01.md` following `templates/handoff.md`
6. Spawn reviewer agent (coordinator mode, new session):
   - Prompt: "Read `.agents/harness/roles/reviewer-brief.md`, then `.agents/harness/checklists/{scenario}-review.md`, then `.agents/harness/active/{run_id}/handoff_r01.md`. Execute the review and write output to `.agents/harness/active/{run_id}/review_r01.md`."
7. Read review output and report to user
8. If P1 or P2: fix issues → write next handoff → spawn new review round
9. Gate: apply verdict rules from reviewer-brief.md

## Key References

- Protocol: `.agents/harness/README.md`
- Executor brief: `.agents/harness/roles/executor-brief.md`
- Reviewer brief: `.agents/harness/roles/reviewer-brief.md`
- Checklists: `.agents/harness/checklists/`
- Templates: `.agents/harness/templates/`
