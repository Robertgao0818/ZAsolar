#!/bin/bash
cd ~/capetown_eval/ds3m_ft/phase1_eval_pipeline_20261018
codex exec --dangerously-bypass-approvals-and-sandbox -C . "Continue the pipeline in this directory (read LOG.md, TEST_REPORT.md first; never overwrite existing files, suffix new versions e.g. _v2). Same hard rules as PROMPT_task1.txt. Then do: $(cat PROMPT_followup.txt) Also: the eval_orig10 coverage was already checked by the lead (all 126-127 manifest tiles exist; bbox holes are top-row tiles outside the slanted grid footprint) - just confirm in coverage_eval_orig10.csv. Write TEST_REPORT_v2.md with full-run GPU metrics table, runtime, peak GPU memory, and independent recompute check. Log to LOG.md. Stop after two consecutive failures." > codex_task1b.log 2>&1
echo CODEX_EXIT $? >> codex_task1b.log
