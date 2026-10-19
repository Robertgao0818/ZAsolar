#!/bin/bash
cd ~/capetown_eval/jhb_pregate_xfer_v1_20261019
echo "$(date '+%F %T %Z') start" >> run.log
export HF_HUB_OFFLINE=1
python3 score_jhb.py >> score.log 2>&1; echo "$(date '+%F %T %Z') score exit $?" >> run.log
