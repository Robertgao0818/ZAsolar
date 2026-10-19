# ZAsolar tidy-up proposal v1 (2026-10-17 18:15 NZDT) — READ-ONLY, nothing moved
Repo: ~/projects/ZAsolar (origin git@github.com:Robertgao0818/ZAsolar.git; checked-out branch durban-citywide-pregate @ aa23c2f; worktrees ZAsolar_edgefix_wt e9cd82b, ZAsolar_segv1_wt feat/ct-seg-trainset-v1). Existing docs/ has plans/, adr/, runbook/, progress_log/ ... (no roadmap/, eval/, reviews/, status/ yet).

## 1. Move list (proposed; git mv inside a new branch reorg-20261017 created from a separate worktree, NOT the checked-out tree)
| Source | Target | Notes |
|---|---|---|
| ~/projects/zasolar_roadmap_20261017/*.md (+ .md5) | docs/roadmap/2026-10-17/ | claude_run.log -> docs/roadmap/2026-10-17/logs/ |
| ~/projects/zasolar_plans_20261017/*.md, v2e/*.md, v2e/handoff/ | docs/plans/2026-10-17/ | design docs |
| zasolar_plans_20261017/{v2e/build_v2e.py, verify_v2e_independent.py, scripts/, cross_grid_*/*.py, jhb25_analysis_v1/*.py, mit_upgrade_scripts_20261017/} | scripts/analysis/2026-10-17/ | code |
| zasolar_plans_20261017 CSV/JSON results (jhb25 *.csv ~0.3 MB each, per_cell_halo.csv, split_overlap_check json) | docs/plans/2026-10-17/data/ or keep out of git | all <5 MB; v2e/md5_inputs_*.txt 3.7 MB each -> keep out of git; drop __pycache__ |
| ~/projects/zasolar_hn_review_20261017/{CODEX_REVIEW.md,PROMPT.md} | docs/reviews/2026-10-17_hn_review/ | codex_run.log 0.9 MB -> out of git |
| ~/projects/zasolar_verification_20261017/* | docs/status/verification_2026-10-17/ | small |
| ~/projects/zasolar_evalfix_20261017 (git repo, branch evalfix, 57 MB incl. manifests/out json) | keep as its own repo; move whole dir to ~/projects/ZAsolar_evalfix (or `git subtree add --prefix=eval/zaeval` on branch reorg) | Worktrees zasolar_evalfix_8044d14_wt / _59911a8_wt are linked: moving the main dir needs `git worktree repair`. **8044d14_wt is the pinned scorer for the Cape Town bot's score_sealed_v3_r2.sh — do not move.** |
| zasolar_evalfix_*_wt | stay | in use / pinned |
Symlinks planned (after move): each old ~/projects/zasolar_*_20261017 -> new location (dir-level symlink). For the plans dir split across docs/ and scripts/, a symlink farm is needed (dir symlink not possible) — alternative: keep plans dir in place as source of truth and only copy into docs/.

## 2. References to old paths (rg, scripts/md/json)
- ~/capetown_eval/ds3m_ft/hn_crop_audit_v1_20261017/PROMPT.txt (running bot, tmux hnaudit)
- ~/capetown_eval/testset_li/infer_v1_20261017/{RUN.md, li443.labels.json, score_sealed_v1_DISABLED.sh} (+ score_sealed_v3_r2.sh pins zasolar_evalfix_8044d14_wt)
- zasolar_evalfix_20261017 and both worktrees: brief.md, PLAN.md, tasks/run.sh, manifests/*.labels.json, out/*.json (absolute paths embedded in provenance JSON — must not be edited, hashes)
- ROADMAP_v*.md and plan docs cross-reference each other by absolute path.
- No crontab. tmux: only `hnaudit`.
(rg was time-limited; list may be incomplete for ~/zasolar_data.)

## 3. Mask R-CNN archive plan
Inventory: core/models/maskrcnn.py, train.py, detect_direct.py, finalize.py, core/inference/{raw_artifact,tile_dataset}.py, scripts/analysis/run_benchmark.py, scripts/gate_pilot/*, scripts/training/jhb_phaseA/*, tests (presence_pregate, finalize_schema, raw_artifact, mask_supervision_patch). weights/ 5.3 MB, checkpoints* tiny (real weights live elsewhere, see ARCHIVE_LOCATIONS.md). results/ already has README_ARCHIVED.md.
Dependency flags:
- **IN USE**: core.inference.presence_pregate imported live by tmux hnaudit grok job (PID 465938). Pregate shares core/ with Mask R-CNN code.
- **IN USE**: ~/capetown_eval/ds3m_ft/joburg25_vexcel/chain.sh (PID 201732) reads ZAsolar/results/analysis/jhb_cbd25_3model_20260514/unified_A_perdet_sam_maskbox (Mask R-CNN JHB25 outputs); probdump_chain.sh (PID 248289) also running.
- JHB25 eval scripts (evalobj_jhb25*.py, pregate/run_now_v*.sh, joburg5/*) reference ZAsolar/{core,detect_direct,results}.
- Durban v4 census (durban_citywide.py / stitch_census_v1.py / detect_direct.py / pregate) on current branch — needs detect_direct.py + core/ in place.
Proposal: **tag only, no moves now**: `git tag maskrcnn-final-20261017` on aa23c2f; add docs/status/MASKRCNN_SUPERSEDED.md; record env (requirements.lock.txt) + weight locations. Move top-level train.py/detect_direct.py to archive/maskrcnn/ only after chains finish and Durban/JHB scripts are re-pointed. core/ stays (shared).

## 4. Risks
Other agents (codex app-server, grok PID 442453, Cursor shells) have cwd ~/projects/ZAsolar right now; repo working tree must not be switched. Running chains + hnaudit read ZAsolar paths. Provenance JSONs embed absolute paths (symlinks required, no rewriting). Push to GitHub is external.
