#!/usr/bin/env python3
"""Single-attempt CPU stages; files hold state, completed stages are resumable."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'rescore_full_v3'


def main():
    os.chdir(ROOT)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONDONTWRITEBYTECODE']='1'
    stages=[('rescore',[sys.executable,'-B','rescore_saved_v3.py','--eval-dir','test_run_full_gpu_v2','--out',str(BASE)]),
            ('independent',[sys.executable,'-B','verify_saved_v3.py','--eval-dir',str(BASE),'--out',str(BASE/'independent_pixel_iou_v3.json')]),
            ('regression',[sys.executable,'-B','verify_results_v3.py','--eval-dir',str(BASE),'--out',str(BASE/'regression_v3.json')])]
    for stage,cmd in stages:
        status=ROOT/f'{stage}_status_v3.json'
        if status.exists():
            previous=json.loads(status.read_text())
            assert previous['returncode']==0, 'Refuse automatic retry of failed stage'
            continue
        start=time.perf_counter()
        with (ROOT/f'{stage}_v3.log').open('x') as f:
            process=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        result={'stage':stage,'command':cmd,'returncode':process.returncode,'runtime_s':time.perf_counter()-start}
        with status.open('x') as f:json.dump(result,f,indent=2)
        with (ROOT/'LOG.md').open('a') as f:
            f.write(f"- v3 CPU {stage}: returncode={process.returncode}, wall={result['runtime_s']:.3f} s; {stage}_v3.log. No automatic retry.\n")
        if process.returncode:sys.exit(process.returncode)


if __name__=='__main__':main()
