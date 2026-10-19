#!/usr/bin/env python3
"""Corrected second attempt; preserve the failed attempt's directory and evidence."""
from pathlib import Path
import run_rescore_v3 as driver

driver.BASE = driver.ROOT / 'rescore_full_attempt2_v3'
# Distinct stage names preserve all first-attempt logs/status, never overwrite.
original_root = driver.ROOT


def main():
    import json, os, subprocess, sys, time
    os.chdir(original_root)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONDONTWRITEBYTECODE']='1'
    base=driver.BASE
    stages=[('rescore_attempt2',[sys.executable,'-B','rescore_saved_v3.py','--eval-dir','test_run_full_gpu_v2','--out',str(base)]),
            ('independent_attempt2',[sys.executable,'-B','verify_saved_v3.py','--eval-dir',str(base),'--out',str(base/'independent_pixel_iou_v3.json')]),
            ('regression_attempt2',[sys.executable,'-B','verify_results_v3.py','--eval-dir',str(base),'--out',str(base/'regression_v3.json')])]
    for name,cmd in stages:
        status=original_root/f'{name}_status_v3.json'
        if status.exists():
            assert json.loads(status.read_text())['returncode']==0, 'STOP: failed stage cannot be retried automatically'
            continue
        start=time.perf_counter()
        with (original_root/f'{name}_v3.log').open('x') as f:
            result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        rec={'stage':name,'command':cmd,'returncode':result.returncode,'runtime_s':time.perf_counter()-start}
        with status.open('x') as f:json.dump(rec,f,indent=2)
        with (original_root/'LOG.md').open('a') as f:f.write(f"- v3 CPU {name}: returncode={result.returncode}, wall={rec['runtime_s']:.3f} s; {name}_v3.log.\n")
        if result.returncode:
            with (original_root/'LOG.md').open('a') as f:f.write('- v3 STOP: second rescore attempt failed; no further execution authorized under the two-failure rule.\n')
            sys.exit(result.returncode)


if __name__=='__main__':main()
