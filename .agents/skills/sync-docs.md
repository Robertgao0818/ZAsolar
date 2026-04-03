# sync-docs Skill

Synchronize the stable skeleton across CLAUDE.md, AGENTS.md, and README.md.

## When to use

After modifying any of the three entry-point documents, or after changing `docs/architecture.md` or `docs/workflows.md`.

## Steps

1. Read `CLAUDE.md`, `AGENTS.md`, `README.md`.
2. Verify that all three have consistent:
   - **Key References** section (same links, same docs/)
   - **Working Constraints** (same 5 constraints)
   - **Environment** (same 3 lines)
3. If any section is out of sync, update the stale file(s) to match the most recently modified one.
4. Verify the stable skeleton sections are content-consistent (exact wording match for shared sections). Line counts may vary as docs evolve.
5. Report what was changed.

## Constraints

- Do not expand CLAUDE.md or AGENTS.md with content that belongs in docs/.
- README.md may have additional human-facing content (progress table, quick start) that the other two don't need.
