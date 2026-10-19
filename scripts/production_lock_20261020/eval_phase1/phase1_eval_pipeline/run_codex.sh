#!/bin/bash
cd ~/capetown_eval/ds3m_ft/phase1_eval_pipeline_20261018
export CUDA_VISIBLE_DEVICES=''
codex exec --dangerously-bypass-approvals-and-sandbox -C . "$(cat PROMPT_task1.txt)" > codex_task1.log 2>&1
echo CODEX_EXIT $? >> codex_task1.log
