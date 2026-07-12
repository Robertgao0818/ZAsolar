# ESSD Data Descriptor — draft workspace

**Status**: v0 census sections drafted (REBUILT 2026-07-12 — the 2026-07-06 draft
and its `/tmp` handoffs were lost in the WSL→Fedora migration, never committed).
Location `docs/paper/essd/` confirmed by owner; update `docs/architecture.md` in
the same commit that first adds these files (repo doc-sync rules).

**Master reference**: `docs/plans/2026-07-06-storyline-tech-plan-v2.md` (draft v3).

## Scope decisions (as drafted)

1. **Census-only** (Cape Town + Johannesburg detection inventories). The JHB temporal
   panel is NOT included; §7 keeps an extension slot for it as an ESSD living-data
   update. Rationale: census-only depends on no unclosed gate (plan doc §4);
   the P0 chain (displacement audit → Panel v2 → ISSUE-11) gates every temporal claim.
2. **No event-study commitment anywhere.** Load-shedding appears in the introduction
   as motivation only (plan doc §4 risk register, last row).
3. **JHB install-date columns are withheld from the v1 data release** (not just empty):
   the JHB economic table has dates, but no independent accuracy claim is allowed
   before Panel v2 + ISSUE-11 (claim-gate table, plan doc §4). CT date block ships
   empty with `undated_reason=ct_backdating_pending` — as the schema contract says.
   ← OWNER DECISION: confirm withhold-vs-ship-with-caveat for JHB dates.

## Evidence discipline

Every number in the draft carries its caliber inline (analysis unit, IoU/matching,
merge-mode, GT tier) and a source path in an HTML comment. Claims are graded
`[F]` (file/number-backed, path attached) / `[H]` (must verify before submission)
per memory `verdict_evidence_grading`. Cross-caliber comparisons are forbidden —
including comparisons against curated-benchmark literature numbers (Dice 0.998-class);
the GT-ceiling decomposition replaces them.

## Files

- `essd_draft_v0.md` — the manuscript draft (English). Census sections drafted
  2026-07-12; per-number source paths in HTML comments; open items in the draft's
  own PENDING register (P1–P8).
- `research_verification_memo_2026-07-06.md` — **LOST** (was written to `/tmp`
  from workflow `wf_51286802-2e8`, never committed, gone with the machine
  migration). The five pre-submission [H] clusters (global product positioning,
  Overture/GOB lineage, ESSD mechanics, load-shedding/SARS series — Vexcel now
  resolved, see below) need a fresh deep-research pass; tracked as draft P1/P2.

## Known blocking items (tracked in draft as PENDING blocks)

- ~~**Vexcel derived-vector license**~~ **RESOLVED (owner, 2026-07-12)**: the
  imagery was obtained under a Vexcel **academic partnership programme**; citing
  the programme per its terms suffices for publishing derived vectors. The public
  September 2024 EULA analysis (Internal-Use-only Derivatives) does not apply to
  this dataset. Remaining action = draft P3: exact credit line from the agreement
  + keep a copy on file with the DOI deposit. CT (municipal WMS) unaffected.
- ESSD requires the dataset deposited with DOI before/at submission — Zenodo record
  must exist first (draft P2; the memo that verified exact mechanics is lost,
  re-verify).
