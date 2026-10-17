# ZAsolar tidy-up proposal v2 (fixes CODEX_REVIEW_v1 blockers)
Changes vs v1:
1. **Copy, don't move.** Sources outside the repo (zasolar_roadmap/plans/hn_review/verification_20261017) are not git repos, so git mv is impossible. Plan: new worktree ~/projects/ZAsolar_reorg_wt on new branch reorg-20261017 from origin/durban-citywide-pregate (aa23c2f); `cp -a` (rsync) each source dir WHOLE, structure preserved, into docs/{roadmap,plans,reviews,status}/2026-10-17/<srcname>/. No splitting of code from data (jhb25_analysis_v1/boot.py reads CSVs from cwd; matched/*.py included). Originals stay in place and remain the live source of truth → **no symlinks needed now** (old paths keep working). Verify with md5 manifest of copies.
2. **evalfix**: not moved (pinned worktree 8044d14_wt used by Cape Town bot; worktree .git pointers). Only a docs/eval/ZAEVAL_POINTER.md recording path, branch evalfix, commits 8044d14 / 59911a8(invalid). Later option: `git subtree add` once bots no longer pin it.
3. **Mask R-CNN**: tag + doc only (annotated tag maskrcnn-final-20261017 on aa23c2f, docs/status/MASKRCNN_SUPERSEDED.md). No code moves (detect_direct.py, finalize.py, core/ used by Durban, HN audit pregate, JHB scripts).
4. Main tree's uncommitted ROADMAP.md change and untracked weights/ are untouched; not carried into the branch.
5. Excluded from git: v2e/md5_inputs_*.txt (3.7 MB, path lists), codex_run.log, __pycache__, any tif/npz/gpkg/pth/safetensors/ckpt, >5 MB. Secret scan before commit.
6. Push: only a new branch, no force; requires a working SSH from the main host (Codex sandbox couldn't test).
7. STATUS: docs/status/STATUS_20261017.md (new file; no existing STATUS in docs). Fixed per review (see STATUS_draft_v2.md).
