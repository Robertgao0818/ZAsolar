"""Phase-1 CT models on JHB CBD25 Vexcel tiles005 (0.05 m). Same sliding window/norm as eval_phase1_v3.infer (chip 640, stride 480, hann, bf16).
usage: infer_jhb25.py ARCH SEED [--limit N]  -> preds/<arch>_s<seed>/<grid>/<tile>.tif (0 bg,1 PV,2 SWH)"""
import os,sys,glob,json,time,importlib.util,numpy as np,rasterio,torch,torch.nn.functional as F
os.environ.setdefault('HF_HUB_OFFLINE','1')
A,S=sys.argv[1],int(sys.argv[2]); LIM=int(sys.argv[sys.argv.index('--limit')+1]) if '--limit' in sys.argv else None
BS=int(os.environ.get('BS','4'))
E=os.path.expanduser('~/capetown_eval/ds3m_ft'); TRAIN=os.path.expanduser('~/capetown_eval/joburg_cbd25_phase1_20261018/src/train2_v2e_podpatched.py')
spec=importlib.util.spec_from_file_location('tr',TRAIN); tr=importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
RUN=f'{E}/phase1_runpod_20261017/{A}_s{S}'; cfg=json.load(open(f'{RUN}/config.json'))
import hashlib; assert hashlib.md5(open(TRAIN,'rb').read()).hexdigest()==cfg['script_md5'],'train src md5 mismatch'
init=f'{E}/phase1_eval_pipeline_20261018/pod_bundle/sources/weights/mscan_b_20230227-3ab7d230.pth' if A=='segnext_b' else None
m=tr.build_model(A,init=init,allow_mscan_download=False,pretrained_convnext=False,upernet_ch=int(cfg['args'].get('upernet_ch',512)),drop_path=float(cfg['args'].get('drop_path',0.3)))
m.load_state_dict(tr.extract_state_dict(torch.load(f'{RUN}/best_objf1.pt',map_location='cpu',weights_only=True)),strict=True); m.cuda().eval()
chip,stride=640,480; hann=np.hanning(chip+2)[1:-1]; win=torch.tensor(np.outer(hann,hann)+1e-3,dtype=torch.float32,device='cuda')
tiles=sorted(glob.glob(f'{E}/joburg25_vexcel/tiles005/*/*.tif'))[:LIM]
O=f'preds/{A}_s{S}'; t0=time.time()
for k,f in enumerate(tiles):
    g=os.path.basename(os.path.dirname(f)); out=f'{O}/{g}/{os.path.basename(f)}'
    if os.path.exists(out): continue
    with rasterio.open(f) as s: x=torch.from_numpy(s.read([1,2,3]).astype(np.float32)); prof=(s.crs,s.transform)
    if x.max()==0: pred=np.zeros(x.shape[1:],np.uint8)
    else:
        h,w=x.shape[1:]; ys=sorted(set(range(0,h-chip+1,stride))|{h-chip}); xs=sorted(set(range(0,w-chip+1,stride))|{w-chip}); pos=[(y,z) for y in ys for z in xs]
        P=torch.zeros(3,h,w,device='cuda'); W=torch.zeros(h,w,device='cuda')
        with torch.inference_mode():
            for i in range(0,len(pos),BS):
                b=pos[i:i+BS]; p=tr.norm(torch.stack([x[:,y:y+chip,z:z+chip] for y,z in b])).cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16): lg=m(pixel_values=p).logits
                p=F.interpolate(lg.float(),size=(chip,chip),mode='bilinear',align_corners=False).softmax(1)
                for j,(y,z) in enumerate(b): P[:,y:y+chip,z:z+chip]+=p[j]*win; W[y:y+chip,z:z+chip]+=win
        pred=(P/W).argmax(0).byte().cpu().numpy()
    os.makedirs(os.path.dirname(out),exist_ok=True)
    with rasterio.open(out,'w',driver='GTiff',height=pred.shape[0],width=pred.shape[1],count=1,dtype='uint8',crs=prof[0],transform=prof[1],compress='deflate') as d: d.write(pred,1)
    if k%100==0: print(json.dumps(dict(arch=A,seed=S,done=k+1,total=len(tiles),s=round(time.time()-t0))),flush=True)
open(f'{O}/DONE','w').write(json.dumps(dict(ckpt=f'{RUN}/best_objf1.pt',tiles=len(tiles),s=time.time()-t0,bs=BS)))
