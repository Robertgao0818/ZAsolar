# archive-experiment Skill

Archive experiment results to `docs/experiment-archive/`.

## When to use

After completing a significant experiment run (GPU validation, calibration sweep, cross-validation, etc.) that should be preserved for historical reference.

## Steps

1. Collect the experiment log (metrics tables, acceptance checks, command used, timestamp).
2. Create a new file in `docs/experiment-archive/` with a descriptive name:
   - Format: `<experiment-type>-<version-or-date>.md`
   - Example: `gpu-integration-test-v1.3.md`, `calibration-sweep-2026-03-20.md`
3. Structure the archive file with:
   - Title and date
   - Command(s) used
   - Hardware/runtime info
   - Results tables
   - Acceptance check outcomes
4. If the experiment log was previously inline in ROADMAP.md, remove the detailed log and add a one-line link to the archive file.

## Constraints

- Never delete metrics data that hasn't been archived first.
- Archive files are append-only historical records — do not edit past entries.
