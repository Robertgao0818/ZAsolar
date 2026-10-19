#!/usr/bin/env python3
"""Exercise three architecture command routes with a recorder, without loading models."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parent
    fixture=root/'cli_routing_fixture_v3'
    fixture.mkdir(exist_ok=False)
    stub=fixture/'record_python_v3.py'
    with stub.open('x') as f:f.write('''#!/usr/bin/env python3
import json,os,sys
with open(os.environ['CLI_RECORD_V3'],'a') as f:
    f.write(json.dumps({'args':sys.argv[1:],'cuda':os.environ.get('CUDA_VISIBLE_DEVICES')})+'\\n')
''')
    stub.chmod(0o755)
    records=fixture/'calls_v3.jsonl'
    env=os.environ.copy()
    env.update(PYTHON=str(stub),CKPT_ROOT=str(fixture/'runs'),OUT_ROOT=str(fixture/'outputs'),CLI_RECORD_V3=str(records))
    for arch in ('mit_b2','segnext_b','convnext_t'):
        run=fixture/'runs'/f'{arch}_s0';run.mkdir(parents=True,exist_ok=False)
        (run/'best_objf1.pt').touch(exist_ok=False)
        args=['bash',str(root/'pod_bundle_v3/run_phase1_eval_v3.sh'),arch,'0','--device','cpu','--data-root',str(root/'relocation_fixture_v2')]
        subprocess.run(args,env=env,check=True,capture_output=True,text=True)
    calls=[json.loads(line) for line in records.read_text().splitlines()]
    assert len(calls)==9
    for i,arch in enumerate(('mit_b2','segnext_b','convnext_t')):
        evaluate,verify,render=calls[i*3:i*3+3]
        args=evaluate['args']
        assert args[args.index('--arch')+1]==arch
        assert args[args.index('--ckpt')+1]==str(fixture/'runs'/f'{arch}_s0/best_objf1.pt')
        assert args[args.index('--training-log')+1]==str(fixture/'runs'/f'{arch}_s0/log.jsonl')
        assert args[args.index('--out')+1]==str(fixture/'outputs'/f'{arch}_s0')
        assert args[args.index('--data-root')+1]==str(root/'relocation_fixture_v2')
        assert args[args.index('--batch-size')+1]=='1'
        assert args[args.index('--device')+1]=='cpu' and args[args.index('--precision')+1]=='float32'
        assert all(c['cuda']=='' for c in (evaluate,verify,render))
        assert verify['args'][1].endswith('verify_saved_v3.py') and render['args'][1].endswith('render_overlays_v2.py')
    result={'passed':True,'architectures':['mit_b2','segnext_b','convnext_t'],'recorded_calls':len(calls),
            'models_constructed':0,'inference_executed':False,'checkpoint_and_log_and_fresh_output_and_data_root_routing':True}
    with (root/'bundle_cli_checks_v3.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
