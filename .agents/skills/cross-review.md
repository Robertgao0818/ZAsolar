# cross-review Skill

Use this entry when the task should run through the repo's executor/reviewer harness instead of an inline self-review.

## When To Use

- User asks for cross-review, 交叉审查, or an independent second-pass review
- A code change should be reviewed in a fresh session against repo facts
- A training or benchmark run should be reviewed against local artifacts or RunPod evidence

## Protocol Source

Read `.agents/harness/README.md` first. That file is the single source of truth for:

- run layout
- role briefs
- scenario checklists
- handoff/review templates
- gate rules

## Codex Entry Paths

### Acting As Executor

1. Read `.agents/harness/roles/executor-brief.md`
2. Choose `code` or `experiment`
3. Create or reuse `.agents/harness/active/<run_id>/`
4. Write the next `handoff_rNN.md` from `.agents/harness/templates/handoff.md`
5. Ask the reviewer session to read only:
   - `.agents/harness/roles/reviewer-brief.md`
   - `.agents/harness/checklists/<scenario>-review.md`
   - the latest handoff
   - repo files and any handoff-listed RunPod evidence

### Acting As Reviewer

1. Read `.agents/harness/roles/reviewer-brief.md`
2. Read `.agents/harness/checklists/<scenario>-review.md`
3. Read the latest `.agents/harness/active/<run_id>/handoff_rNN.md`
4. Verify claims against repo files and handoff-listed evidence
5. Write `.agents/harness/active/<run_id>/review_rNN.md` from `.agents/harness/templates/review.md`

## Constraints

- Do not create a second protocol outside `.agents/harness/`
- Treat the latest handoff plus repository state as the working context
- If remote evidence is required, use only the paths named in the handoff
- Reviewer-side read-only behavior is a process rule, not a hard permission boundary
