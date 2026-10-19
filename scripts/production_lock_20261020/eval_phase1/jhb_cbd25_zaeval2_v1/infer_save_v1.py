"""Joburg CBD25 inference with probability saving (for zaeval-2 rescoring). Model build + sliding window (640/480, hanning,
bf16 autocast on cuda) and the official pregate (pregate/scores.json, 3x3 cells/tile, keep iff >=0.45; PGONLY: only windows touching
kept cells, rejected cells -> prob 0) are copied verbatim from ~/capetown_eval/joburg_cbd25_rescore_v2_20261019/evalobj_jhb25_p1cpu.py.
Saves probs/<NAME>/<grid>/<tile>.npz key 'p' = uint8 rint(prob_PV*255) (same storage convention as CT phase1_followup_valprobs).
usage: infer_save_v1.py NAME KIND CKPT   (KIND mit | p1 with ARCH/SEED env). Resumable: existing npz skipped."""
import os,sys,glob,json,time,numpy as np,torch,torch.nn.functional as F,rasterio
os.environ.setdefault('HF_HUB_OFFLINE','1')
NAME,KIND,ck=sys.argv[1:4]; DEV='cuda'
J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel'); OUT=f'probs/{NAME}'
P=sorted(glob.glob(f'{J}/tiles005/*/*.tif')); assert len(P)==2500
from transformers import SegformerConfig,SegformerForSemanticSegmentation
if KIND=='p1':
    import importlib.util,hashlib; _A=os.environ['ARCH']; _S=int(os.environ['SEED']); _E=os.path.expanduser('~/capetown_eval/ds3m_ft'); _TR=os.path.expanduser('~/capetown_eval/joburg_cbd25_phase1_20261018/src/train2_v2e_podpatched.py')
    _sp=importlib.util.spec_from_file_location('tr',_TR); tr=importlib.util.module_from_spec(_sp); _sp.loader.exec_module(tr)
    _RUN=f'{_E}/phase1_runpod_20261017/{_A}_s{_S}'; _cfg=json.load(open(f'{_RUN}/config.json')); assert hashlib.md5(open(_TR,'rb').read()).hexdigest()==_cfg['script_md5']
    _init=f'{_E}/phase1_eval_pipeline_20261018/pod_bundle/sources/weights/mscan_b_20230227-3ab7d230.pth' if _A=='segnext_b' else None
    m=tr.build_model(_A,init=_init,allow_mscan_download=False,pretrained_convnext=False,upernet_ch=int(_cfg['args'].get('upernet_ch',512)),drop_path=float(_cfg['args'].get('drop_path',0.3)))
    sd=tr.extract_state_dict(torch.load(ck,map_location='cpu',weights_only=True))
else:
    m=SegformerForSemanticSegmentation(SegformerConfig(num_labels=3,depths=[3,4,6,3],hidden_sizes=[64,128,320,512],decoder_hidden_size=768)); sd=torch.load(ck,map_location='cpu')
m.load_state_dict(sd,strict=True); m=m.to(DEV).eval().to(memory_format=torch.channels_last); BS=int(os.environ.get('BS','2'))
MEAN=torch.tensor((.485,.456,.406)).view(1,3,1,1).to(DEV)*255; STD=torch.tensor((.229,.224,.225)).view(1,3,1,1).to(DEV)*255
C=640;S=480;w1=np.hanning(C+2)[1:-1];WIN=torch.tensor(np.outer(w1,w1)+1e-3).float().to(DEV)
def infer(x,km):
    H,W=x.shape[1:]; Pp=torch.zeros(3,H,W,device=DEV);Wt=torch.zeros(H,W,device=DEV)
    ys=sorted(set(list(range(0,H-C+1,S))+[H-C]));xs=sorted(set(list(range(0,W-C+1,S))+[W-C]));pos=[(y,xx) for y in ys for xx in xs]
    pos=[(y,xx) for y,xx in pos if km[y:y+C,xx:xx+C].any()]
    with torch.inference_mode():
        for i in range(0,len(pos),BS):
            bb=pos[i:i+BS]; ch=((torch.stack([x[:,y:y+C,xx:xx+C] for y,xx in bb])-MEAN)/STD).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda',dtype=torch.bfloat16): lo=m(pixel_values=ch).logits
            pr=F.interpolate(lo.float(),size=C,mode='bilinear').softmax(1)
            for k,(y,xx) in enumerate(bb): Pp[:,y:y+C,xx:xx+C]+=pr[k]*WIN; Wt[y:y+C,xx:xx+C]+=WIN
    return (Pp/Wt.clamp(min=1e-6))[1].cpu().numpy()
PG=json.load(open(f'{J}/pregate/scores.json')); PGT=PG['threshold']; assert PGT==0.45
def gatemask(path,H,W):
    sc=PG['scores'][os.path.relpath(path,J)]; km=np.zeros((H,W),bool); e=[0,667,H-667]
    for k,v in enumerate(sc):
        if v>=PGT: i,j=divmod(k,3); km[e[i]:e[i]+667,e[j]:e[j]+667]=True
    return km
t0=time.time(); n=0; nskip=0
for p in P:
    g=os.path.basename(os.path.dirname(p)); f=f'{OUT}/{g}/{os.path.basename(p)[:-4]}.npz'
    if os.path.exists(f): continue
    os.makedirs(os.path.dirname(f),exist_ok=True)
    im=rasterio.open(p).read([1,2,3]); H,W=im.shape[1:]; km=gatemask(p,H,W)
    if km.any(): prob=infer(torch.from_numpy(im).float().to(DEV),km)*km
    else: prob=np.zeros((H,W),np.float32); nskip+=1
    np.savez_compressed(f+'.tmp.npz',p=np.rint(np.clip(prob,0,1)*255).astype(np.uint8)); os.replace(f+'.tmp.npz',f); n+=1
    if n%100==0: print(NAME,n,f'{time.time()-t0:.0f}s',flush=True)
print('DONE',NAME,n,'gated_out_tiles',nskip,f'{time.time()-t0:.0f}s',flush=True)
