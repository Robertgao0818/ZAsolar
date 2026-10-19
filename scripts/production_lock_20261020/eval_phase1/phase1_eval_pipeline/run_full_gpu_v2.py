#!/usr/bin/env python3
"""One GPU E2E evaluation; CPU render + independent check. Resumable by stage files."""
import os,sys,subprocess,json,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
BASE=ROOT/'test_run_full_gpu_v2'

def main():
    started=time.time(); records=[]; stop=threading.Event()
    def monitor():
        with (ROOT/'gpu_monitor_v2.jsonl').open('x') as f:
            while not stop.is_set():
                r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
                f.write(json.dumps({'time':time.time(),'compute_apps':r.stdout.strip(),'returncode':r.returncode})+'\n');f.flush()
                stop.wait(15)
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    stages=[('evaluation',[sys.executable,'-B','eval_phase1_v2.py','--ckpt','/home/gao/capetown_eval/ds3m_ft/run_v2c/best.pt','--arch','mit_b2','--out',str(BASE),'--split','both','--device','cuda','--precision','bfloat16','--batch-size','1','--threads','4']),
      ('render',[sys.executable,'-B','render_overlays_v2.py','--eval-dir',str(BASE),'--out',str(BASE/'overlays')]),
      ('independent',[sys.executable,'-B','recompute_pixel_iou.py','--eval-dir',str(BASE),'--out',str(BASE/'independent_pixel_iou.json')])]
    try:
        for name,cmd in stages:
            env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['CUDA_VISIBLE_DEVICES']='0' if name=='evaluation' else ''
            t=time.time()
            with (ROOT/f'full_gpu_{name}_v2.log').open('x') as f:
                r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
            rec={'stage':name,'command':cmd,'returncode':r.returncode,'wall_runtime_s':time.time()-t};records.append(rec)
            with (ROOT/f'full_gpu_{name}_status_v2.json').open('x') as f:json.dump(rec,f,indent=2)
            with (ROOT/'LOG.md').open('a') as f:f.write(f"- Full GPU follow-up stage {name}: returncode={r.returncode}, wall time={rec['wall_runtime_s']:.3f} s. See full_gpu_{name}_v2.log. No automatic retries.\n")
            if r.returncode:sys.exit(r.returncode)
    finally:
        stop.set();worker.join(timeout=2)
        with (ROOT/'full_gpu_driver_status_v2.json').open('x') as f:json.dump({'stages':records,'wall_runtime_s':time.time()-started},f,indent=2)

if __name__=='__main__':main()
