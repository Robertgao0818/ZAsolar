# builds evalobj_jhb25_p1cpu.py from an unmodified copy of evalobj_jhb25_par.py; asserts every replacement hits exactly once
s=open('scorer_orig_copy.py').read()
def rep(a,b,n=1):
    global s; assert s.count(a)==n,(a,s.count(a)); s=s.replace(a,b)
s='"""[P1CPU variant 2026-10-19: KIND=p1 loads phase-1 train2_v2e checkpoints (ARCH,SEED env); device via DEV env (default cpu, fp32 on cpu); scoring code unchanged]\n'+s[3:]
rep("""    if KIND in('dino','dinosat'):""","""    if KIND=='p1':
        import importlib.util; _A=os.environ['ARCH']; _S=int(os.environ['SEED']); _E=os.path.expanduser('~/capetown_eval/ds3m_ft'); _TR=os.path.expanduser('~/capetown_eval/joburg_cbd25_phase1_20261018/src/train2_v2e_podpatched.py')
        _sp=importlib.util.spec_from_file_location('tr',_TR); tr=importlib.util.module_from_spec(_sp); _sp.loader.exec_module(tr)
        _RUN=f'{_E}/phase1_runpod_20261017/{_A}_s{_S}'; _cfg=json.load(open(f'{_RUN}/config.json')); import hashlib; assert hashlib.md5(open(_TR,'rb').read()).hexdigest()==_cfg['script_md5']
        _init=f'{_E}/phase1_eval_pipeline_20261018/pod_bundle/sources/weights/mscan_b_20230227-3ab7d230.pth' if _A=='segnext_b' else None
        m=tr.build_model(_A,init=_init,allow_mscan_download=False,pretrained_convnext=False,upernet_ch=int(_cfg['args'].get('upernet_ch',512)),drop_path=float(_cfg['args'].get('drop_path',0.3)))
        sd=tr.extract_state_dict(torch.load(ck,map_location='cpu',weights_only=True))
    elif KIND in('dino','dinosat'):""")
rep("""    sd=torch.load(ck,map_location='cpu')\n""","""    if KIND!='p1': sd=torch.load(ck,map_location='cpu')\n""")
rep("m=m.cuda().eval()","DEV=os.environ.get('DEV','cpu'); torch.set_num_threads(int(os.environ.get('NTHR','16'))); m=m.to(DEV).eval()")
rep(".view(1,3,1,1).cuda()*255",".view(1,3,1,1).to(DEV)*255",2)
rep("WIN=torch.tensor(np.outer(w1,w1)+1e-3).float().cuda()","WIN=torch.tensor(np.outer(w1,w1)+1e-3).float().to(DEV)")
rep("device='cuda'","device=DEV",2)
rep("with torch.autocast('cuda',dtype=torch.bfloat16)","with torch.autocast('cuda',dtype=torch.bfloat16,enabled=DEV=='cuda')")
rep("torch.from_numpy(img).float().cuda()","torch.from_numpy(img).float().to(DEV)")
assert s.count('.cuda()')==1, [l for l in s.splitlines() if '.cuda()' in l]
open('evalobj_jhb25_p1cpu.py','w').write(s)
