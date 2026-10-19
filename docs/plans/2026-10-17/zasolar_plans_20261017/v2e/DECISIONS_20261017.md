# v2e decisions — 2026-10-17

Recorded for the Phase-1 trainset `ct_seg_trainset_v2e_20261017`. Same items also appear in `~/projects/zasolar_roadmap_20261017/ROADMAP_v4.md` under **Decisions (2026-10-17 17:13 NZDT, 葉)**. Existing files were not modified.

Source: 葉, 2026-10-17 17:13 NZDT (Pacific/Auckland). Reply to the three open questions after v2e build+verify: "1 留 2 保留 3 确认".

## Decisions

1. **G2028 stays in val.** Do not remove it from val. v2e val = v2d2 val (unchanged). G2028 keeps its SAM3 thin-strip labels. Rationale (主管 recommendation, accepted by 葉): leave as-is so v2c / v2d / v2e val scores are directly comparable.

2. **Keep the v2d ignore edits in G1749 / G1972 / G2029.** The thin-strip → ignore changes made in v2d for those three grids remain in v2e. Do not revert them.

3. **Backdating uses 3.5** — confirmed. Closes the open confirm item from ROADMAP_v3 Priorities §5. Standing choice while the backdating track is paused (Priorities §2). Was already [葉-1017b #3].

## Still open (not decided here)

- 葉's pregate cross-grid chip concern (does pregate cover chips from neighbouring grids in real detection; size of cross-grid effect). Under investigation; Claude on rog asked. May change r2 / scoring scripts. r2 sign-off, touching-label merge, and all test scoring remain ON HOLD.

## Not started

No training. Before any Phase-1 launch (still needs 葉's go): rebuild HN coord cache for v2e; write `train2_v2e.py`; smoke run; write DECISION_RULE (arm order, seeds, stop rules).
