"""[P1CPU variant 2026-10-19: KIND=p1 loads phase-1 train2_v2e checkpoints (ARCH,SEED env); device via DEV env (default cpu, fp32 on cpu); scoring code unchanged]
[PAR variant 2026-10-17 (copy of evalobj_jhb25_fast.py): ROLE=gpu (default) runs one serialized GPU inference pass, writes per-tile float32 probs to
 /dev/shm, and launches per-grid CPU scorer subprocesses (ROLE=score SG=<grid>, NWORK concurrent) that run the unchanged per-grid scoring code and
 pickle partial accumulators to PART dir (resumable: finished grids are reused); the GPU process then merges partials and runs the unchanged aggregate.]
[FAST variant 2026-10-17: batched windows (BS env), channels_last, inference_mode, prefetch thread; logic otherwise identical to evalobj_jhb25.py]
Joburg variant of evalobj.py (CoJ aerial 2023, labels Joburg/G*_V4_260407.gpkg, EPSG:32735).
usage: evalobj_jhb.py KIND CKPT OUTJSON [--native] [--cfgs h0.9_0.5,t0.5]
--native: feed 0.15 m imagery (tile downsampled to 667 px) and upsample probs; default feeds 0.05 m resampled tiles.
Object-level evaluator (new; evalfull.py untouched).
usage: evalobj.py SPLIT KIND CKPT OUTJSON [--limit N]
KIND: zs | mit | swh | dino | dinosat (SAT mean/std) | mrcnn (CKPT ignored)
GT objects = original label polygons (dedupe + heater removal as in build_ct_seg_trainset_v2),
area from polygon geometry (EPSG:32734), so objects spanning 100 m tiles are whole.
Pixel ignore = v2c mask 255 (per-class band: A 0.15 m / B 0.3 m as built).
Pred objects: 8-connected components, merged across tile edges via global pixel coords.
Configs: plain threshold t, and hysteresis hi/lo (= keep lo-components whose max prob >= hi;
this is identical to an 'object max-prob keep rule' at threshold lo, so one family covers both)."""
import os,sys,re,glob,json,math,numpy as np,pandas as pd,geopandas as gpd,rasterio
from rasterio import features
from scipy import ndimage
os.environ.setdefault('HF_HUB_OFFLINE','1')
split='jhb'; KIND,ck,outj=sys.argv[1:4]; NATIVE='--native' in sys.argv; LIM=int(sys.argv[sys.argv.index('--limit')+1]) if '--limit' in sys.argv else None
R=os.path.expanduser('~/capetown_eval/ds3m_ft'); DATA=os.path.expanduser('~/data/ZAsolar/geid_temporal/ct_seg_trainset_v2c_20261015/data')
sys.path.insert(0,os.path.expanduser('~/projects/ZAsolar_segv1_wt/scripts/training'))
from build_ct_seg_trainset_v1 import dedupe_polygons
from build_ct_seg_trainset_v2 import route_sam_fn_polygons
TEST10={'G1349','G1620','G1686','G1692','G1733','G1785','G1800','G1842','G2038','G2905'}
J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel'); J0=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg5'); T=pd.DataFrame(dict(image=sorted(glob.glob(f"{J}/{os.environ.get('TILEDIR','tiles005')}/*/*.tif")))); T['grid']=T.image.map(lambda p:os.path.basename(os.path.dirname(p))); T['swh_labelled']=False; T['mask']=''
ROLE=os.environ.get('ROLE','gpu'); SG=os.environ.get('SG'); PART=os.environ.get('PART',''); SHM='/dev/shm/jhb_probs_'+os.path.basename(PART.rstrip('/'))
if os.environ.get('TILEPAT'): T=T[T.image.str.contains(os.environ['TILEPAT'])]
GRIDPOLY=gpd.read_file(f'{J0}/grids_32735.gpkg').set_index('gridcell_id').geometry
if LIM: T=T.groupby('grid').head(max(1,LIM//T.grid.nunique()))
PXM=0.05; PXA=PXM*PXM
# ---------------- GT polygons
LAB=os.path.expanduser('~/projects/ZAsolar/data/annotations/Capetown'); G1688=os.path.expanduser('~/zasolar_data/geid_temporal/label_audit_classB_v1_20261014/diag/fixed_G1688_SAM2_260403.gpkg')
SWHG=os.path.expanduser('~/zasolar_data/geid_temporal/water_heater_pass_v1_20261015/solar_water_heaters.gpkg')
swh=gpd.read_file(SWHG); swh_lab=set(swh.loc[swh.origin=='classB_label','ref'].astype(str))
COL=gpd.GeoDataFrame(dict(grid=[],mask_value=[]),geometry=[],crs=32735)
def load_gt(g):
    d=gpd.read_file(f"{os.environ['GTDIR']}/{g}_clean_gt.gpkg" if os.environ.get('GTDIR') else os.path.expanduser(f'~/Dropbox/RA_Solar/Gao/annotations/Joburg/{g}_V4_260407.gpkg')).to_crs(32735); d=d[d.geometry.notna()&~d.geometry.is_empty].copy(); d['geometry']=d.geometry.buffer(0); d=dedupe_polygons(d.reset_index(drop=True))
    o=gpd.GeoDataFrame(dict(cls='B'),index=range(len(d)),geometry=list(d.geometry),crs=32735); o['area']=o.geometry.area; o['oid']=np.arange(1,len(o)+1); return o
def _unused(g):
    fs=sorted(glob.glob(f'{LAB}/{g}_SAM2_*.gpkg')); rows=[]
    for f in fs:
        name=os.path.basename(f); d=gpd.read_file(G1688 if g=='G1688' else f); d=d.set_crs(4326) if d.crs is None else d.to_crs(4326); d=d.reset_index(drop=True)
        d=d[~np.array([f'{name}#{i}' in swh_lab for i in range(len(d))])]
        d=d[d.geometry.notna()&~d.geometry.is_empty].copy(); d['geometry']=d.geometry.buffer(0); d=dedupe_polygons(d.reset_index(drop=True))
        b,a=route_sam_fn_polygons(d)
        for part,cl in ((b,'B'),(a,'A')):
            if part is not None and len(part): p=part.to_crs(32734); rows+= [dict(cls=cl,geometry=q) for q in p.geometry]
    o=gpd.GeoDataFrame(rows,geometry='geometry',crs=32734) if rows else gpd.GeoDataFrame(dict(cls=[]),geometry=[],crs=32734)
    o['area']=o.geometry.area; o['oid']=np.arange(1,len(o)+1); return o
# ---------------- model
torch=None
if KIND!='mrcnn' and ROLE!='score':
    import torch,torch.nn as nn,torch.nn.functional as F
    from transformers import SegformerConfig,SegformerForSemanticSegmentation
    if KIND=='p1':
        import importlib.util; _A=os.environ['ARCH']; _S=int(os.environ['SEED']); _E=os.path.expanduser('~/capetown_eval/ds3m_ft'); _TR=os.path.expanduser('~/capetown_eval/joburg_cbd25_phase1_20261018/src/train2_v2e_podpatched.py')
        _sp=importlib.util.spec_from_file_location('tr',_TR); tr=importlib.util.module_from_spec(_sp); _sp.loader.exec_module(tr)
        _RUN=f'{_E}/phase1_runpod_20261017/{_A}_s{_S}'; _cfg=json.load(open(f'{_RUN}/config.json')); import hashlib; assert hashlib.md5(open(_TR,'rb').read()).hexdigest()==_cfg['script_md5']
        _init=f'{_E}/phase1_eval_pipeline_20261018/pod_bundle/sources/weights/mscan_b_20230227-3ab7d230.pth' if _A=='segnext_b' else None
        m=tr.build_model(_A,init=_init,allow_mscan_download=False,pretrained_convnext=False,upernet_ch=int(_cfg['args'].get('upernet_ch',512)),drop_path=float(_cfg['args'].get('drop_path',0.3)))
        sd=tr.extract_state_dict(torch.load(ck,map_location='cpu',weights_only=True))
    elif KIND in('dino','dinosat'):
        _s=open(f"{R}/{'train3_dinov3_sat.py' if KIND=='dinosat' else 'train3_dinov3.py'}").read(); exec(_s[_s.index('import timm'):_s.index('model=DinoSeg().cuda()')]); m=DinoSeg()
    else: m=SegformerForSemanticSegmentation(SegformerConfig(num_labels=2 if KIND=='zs' else 3,depths=[3,4,6,3],hidden_sizes=[64,128,320,512],decoder_hidden_size=768))
    if KIND!='p1': sd=torch.load(ck,map_location='cpu')
    if KIND=='zs': src=open(os.path.expanduser('~/capetown_eval/deepsolar3m/infer.py')).read(); exec(src[src.index('def remap'):src.index('cfg=SegformerConfig')]); sd=remap(sd)
    m.load_state_dict(sd,strict=True); DEV=os.environ.get('DEV','cpu'); torch.set_num_threads(int(os.environ.get('NTHR','16'))); m=m.to(DEV).eval().to(memory_format=torch.channels_last); BS=int(os.environ.get('BS','2'))
    mean,std=((.430,.411,.296),(.213,.156,.143)) if KIND=='dinosat' else ((.485,.456,.406),(.229,.224,.225))
    MEAN=torch.tensor(mean).view(1,3,1,1).to(DEV)*255; STD=torch.tensor(std).view(1,3,1,1).to(DEV)*255
    C=640;S=480;w1=np.hanning(C+2)[1:-1];WIN=torch.tensor(np.outer(w1,w1)+1e-3).float().to(DEV)
def infer(x,km=None):
    if NATIVE: x=F.interpolate(x[None],size=(667,667),mode='area')[0]
    nc=2 if KIND=='zs' else 3; H,W=x.shape[1:]; P=torch.zeros(nc,H,W,device=DEV);Wt=torch.zeros(H,W,device=DEV)
    ys=sorted(set(list(range(0,H-C+1,S))+[H-C]));xs=sorted(set(list(range(0,W-C+1,S))+[W-C]));pos=[(y,xx) for y in ys for xx in xs]
    if km is not None: pos=[(y,xx) for y,xx in pos if km[y:y+C,xx:xx+C].any()]
    with torch.inference_mode():
        for i in range(0,len(pos),BS):
            bb=pos[i:i+BS]; ch=((torch.stack([x[:,y:y+C,xx:xx+C] for y,xx in bb])-MEAN)/STD).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=DEV=='cuda'): lo=m(pixel_values=ch).logits
            pr=F.interpolate(lo.float(),size=C,mode='bilinear').softmax(1)
            for k,(y,xx) in enumerate(bb): P[:,y:y+C,xx:xx+C]+=pr[k]*WIN; Wt[y:y+C,xx:xx+C]+=WIN
    P=P/Wt.clamp(min=1e-6)
    if NATIVE: P=F.interpolate(P[None],size=(2000,2000),mode='bilinear')[0]
    return P[1].cpu().numpy(),(P.argmax(0)==2).cpu().numpy() if nc==3 else None
CFG=[('t',t,t) for t in (0.3,0.4,0.5,0.6,0.7)]+[('h',hi,lo) for hi,lo in ((0.6,0.3),(0.7,0.3),(0.7,0.4),(0.8,0.4),(0.8,0.5),(0.9,0.5),(0.9,0.6))]
if KIND=='mrcnn': CFG=[('t',0.5,0.5)]
# PRE-GATE (2026-10-17): every config is also run with the presence pre-gate (pregate/scores.json; 3x3 cells/tile, keep cell iff max fold score>=0.45; rejected cells -> prob 0). 'pg_*' = headline.
PG=json.load(open(f'{J}/pregate/scores.json')); PGT=PG['threshold']
PGONLY=os.environ.get('PGONLY')=='1'   # infer only windows touching kept cells; tiles with none are skipped (prob=0); only pg_* configs reported
CFG=[c+(g,) for g in ((1,) if PGONLY else (1,0)) for c in CFG]
cname=lambda c:('pg_' if c[3] else '')+(f't{c[1]}' if c[0]=='t' else f'h{c[1]}_{c[2]}')
def gatemask(path,H,W):
    sc=PG['scores'][os.path.relpath(path,J)]; km=np.zeros((H,W),bool); e=[0,667,H-667]
    for k,v in enumerate(sc):
        if v>=PGT: i,j=divmod(k,3); km[e[i]:e[i]+667,e[j]:e[j]+667]=True
    return km
NSKIP=[0]
MR=os.path.expanduser('~/Dropbox/RA_Solar/Gao/ct_census/results/unifiedA_census_perdet'); mrc={}
S8=np.ones((3,3),bool)
# accumulators
gt_cov={c:{} for c in map(cname,CFG)}   # (grid,oid)-> covered px
gt_px={}; gt_meta={}
predobj={c:[] for c in map(cname,CFG)}  # rows: key,npx,nz,ninv,nswh,maxp
edges={c:{} for c in map(cname,CFG)}    # (grid,gx,gy)->key
pix={c:dict(tp=0,fp=0,fn=0,bi=0,bu=0) for c in map(cname,CFG)}
swhcov={c:{} for c in map(cname,CFG)}; swh_px={}; valid_ha=0.0; neg_ha=0.0
def ero(x,d): return ndimage.binary_erosion(x,iterations=d,border_value=1)
from concurrent.futures import ThreadPoolExecutor   # one worker: read + gate + GPU infer of upcoming tiles, overlapped with CPU scoring in main thread
POOL=ThreadPoolExecutor(1); FUT={}; _L=[p for _,TG_ in T.groupby('grid') for p in TG_.image]; _I={p:k for k,p in enumerate(_L)}
def rd(p):
    with rasterio.open(p) as f: img=f.read([1,2,3]); tr=f.transform; crs=f.crs; bnds=f.bounds
    pr=None
    if KIND!='mrcnn' and ROLE!='score':
        H,W=img.shape[1:]; km_=gatemask(p,H,W)
        if PGONLY and not km_.any(): pr=np.zeros((H,W),np.float32); NSKIP[0]+=1
        else: pr,_=infer(torch.from_numpy(img).float().to(DEV),km_ if PGONLY else None)
    return (img,tr,crs,bnds),pr
NEXT=lambda p:_L[_I[p]+1:_I[p]+4]
import pickle,subprocess,time as _t
T_FULL=T
def _pp(g): return f'{PART}/{g}.pkl'
if ROLE=='score':
    T=T[T.grid==SG]
elif KIND!='mrcnn':
    assert PART and PGONLY, 'par runner is for PGONLY gated runs'
    os.makedirs(PART,exist_ok=True); NW=int(os.environ.get('NWORK','4')); procs={}
    def reap(block):
        while procs and (block or len(procs)>=NW):
            for g,p in list(procs.items()):
                if p.poll() is not None:
                    assert p.returncode==0 and os.path.exists(_pp(g)),f'scorer {g} rc {p.returncode}'; del procs[g]; os.system(f'rm -rf {SHM}/{g}'); print('scored',g,flush=True)
            if procs and (block or len(procs)>=NW): _t.sleep(2)
            else: break
    for n_,(gname,TG) in enumerate(T.groupby('grid')):
        if os.path.exists(_pp(gname)): print('reuse',gname,flush=True); continue
        os.makedirs(f'{SHM}/{gname}',exist_ok=True); t0=_t.time()
        for p_ in TG.image: FUT.setdefault(p_,POOL.submit(rd,p_))
        for p_ in TG.image:
            _,pr=FUT.pop(p_).result(); np.save(f'{SHM}/{gname}/{os.path.basename(p_)}.npy',pr)
        print('infer',n_,gname,len(TG),round(_t.time()-t0,1),'s',flush=True)
        reap(False)
        procs[gname]=subprocess.Popen([sys.executable]+sys.argv,env=dict(os.environ,ROLE='score',SG=gname),stdout=open(f'{PART}/{gname}.log','w'),stderr=subprocess.STDOUT)
    reap(True)
    for gname in sorted(T.grid.unique()):
        P=pickle.load(open(_pp(gname),'rb')); NSKIP[0]+=P['NSKIP']; valid_ha+=P['valid_ha']; neg_ha+=P['neg_ha']; gt_px.update(P['gt_px']); gt_meta.update(P['gt_meta']); swh_px.update(P['swh_px'])
        for c in map(cname,CFG):
            off=len(predobj[c]); predobj[c]+=[[row[0]+off]+list(row[1:]) for row in P['predobj'][c]]
            for kk,v in P['edges'][c].items(): edges[c][kk]=v+off
            gt_cov[c].update(P['gt_cov'][c]); swhcov[c].update(P['swhcov'][c])
            for kk in pix[c]: pix[c][kk]+=P['pix'][c][kk]
    T=T.iloc[0:0]
if ROLE=='score':
    _L=list(T.image); _I={p:k for k,p in enumerate(_L)}
    for p_ in _L[:3]: FUT[p_]=POOL.submit(rd,p_)
for n_,(gname,TG) in enumerate(T.groupby('grid')):
    gt=load_gt(gname); swl=bool(TG.swh_labelled.iloc[0])
    for o in gt.itertuples(): gt_meta[(gname,o.oid)]=(o.area,o.cls)
    cg=COL[(COL.grid==gname)&(COL.mask_value==2)].reset_index(drop=True)
    X0=None
    for _,r in TG.iterrows():
        (img,tr,crs,bnds),PRE=FUT.pop(r.image).result()
        for p_ in NEXT(r.image): FUT.setdefault(p_,POOL.submit(rd,p_))
        H,W=img.shape[1:]
        pvm=features.rasterize([(q,1) for q in gt.cx[bnds.left:bnds.right,bnds.bottom:bnds.top].geometry],(H,W),transform=tr).astype(bool) if len(gt) else np.zeros((H,W),bool)
        valid=features.rasterize([(GRIDPOLY[gname],1)],(H,W),transform=tr).astype(bool)&img.any(0)
        st=ndimage.generate_binary_structure(2,1); band=ndimage.binary_dilation(pvm,st,iterations=6)&~ndimage.binary_erosion(pvm,st,iterations=6)
        mk=np.zeros((H,W),np.uint8); mk[pvm]=1; mk[band]=255; mk[~valid]=255
        npv=int(pvm.sum())
        gs=gt.cx[bnds.left:bnds.right,bnds.bottom:bnds.top]
        idr=features.rasterize([(q,int(i)) for q,i in zip(gs.geometry,gs.oid)],(H,W),transform=tr,all_touched=True,dtype='int32') if len(gs) else np.zeros((H,W),'int32')
        for i,c in zip(*np.unique(idr[idr>0],return_counts=True)): gt_px[(gname,i)]=gt_px.get((gname,i),0)+c
        cs=cg.cx[bnds.left:bnds.right,bnds.bottom:bnds.top] if swl and len(cg) else cg.iloc[0:0]
        sr=features.rasterize([(q,i+1) for q,i in zip(cs.geometry,cs.index)],(H,W),transform=tr,all_touched=True,dtype='int32') if len(cs) else None
        if sr is not None:
            for i,c in zip(*np.unique(sr[sr>0],return_counts=True)): swh_px[(gname,i)]=swh_px.get((gname,i),0)+c
        v=mk!=255; g=mk==1; gfull=idr>0; inv=(mk==255)&~ndimage.binary_dilation(gfull,iterations=8)
        valid_ha+=v.sum()*PXA/1e4
        if npv==0: neg_ha+=v.sum()*PXA/1e4
        gx0=int(round((tr.c-bnds.left)/PXM)); gx0=int(round(tr.c/PXM)); gy0=int(round(-tr.f/PXM))
        zone=ndimage.binary_dilation(gfull|g,iterations=8)
        if KIND=='mrcnn':
            cpt=gname
            if cpt not in mrc:
                f_=(f'{ck}/{gname}/predictions_metric.gpkg' if ck.startswith('/') else os.path.expanduser(f'~/projects/ZAsolar/results/johannesburg/{ck}/{gname}/predictions_metric.gpkg')); mrc[cpt]=gpd.read_file(f_).to_crs(crs) if os.path.exists(f_) else None
            gg=mrc[cpt]; pm=features.rasterize([(q,1) for q in gg.geometry if q is not None],(H,W),transform=tr).astype(np.float32) if gg is not None and len(gg) else np.zeros((H,W),np.float32)
            prob=pm
        else:
            import torch
            prob=np.load(f'{SHM}/{gname}/{os.path.basename(r.image)}.npy')
        km=gatemask(r.image,H,W); probg=prob*km; prob0=prob
        for c in CFG:
            cn=cname(c); prob=probg if c[3] else prob0
            if c[0]=='t': pred=prob>=c[1]; lab,nl=ndimage.label(pred,S8)
            else:
                lab,nl=ndimage.label(prob>=c[2],S8)
                if nl:
                    mx=ndimage.maximum(prob,lab,np.arange(1,nl+1)); keep=np.r_[False,mx>=c[1]]; pred=keep[lab]; lab,nl=ndimage.label(pred,S8)
                else: pred=lab>0
            P_=pix[cn]; P_['tp']+=int((pred&g&v).sum()); P_['fp']+=int((pred&~g&v).sum()); P_['fn']+=int((~pred&g&v).sum())
            reg=v|gfull; bp=pred&~ero(pred,10); bg=gfull&~ero(gfull,10); P_['bi']+=int((bp&bg&reg).sum()); P_['bu']+=int(((bp|bg)&reg).sum())
            if len(gs):
                ids,cnt=np.unique(idr[pred&(idr>0)],return_counts=True); d=gt_cov[cn]
                for i,cc in zip(ids,cnt): d[(gname,i)]=d.get((gname,i),0)+cc
            if sr is not None:
                ids,cnt=np.unique(sr[pred&(sr>0)],return_counts=True); d=swhcov[cn]
                for i,cc in zip(ids,cnt): d[(gname,i)]=d.get((gname,i),0)+cc
            if nl:
                idx=np.arange(1,nl+1); npx=ndimage.sum(np.ones_like(lab,np.int32),lab,idx); nz=ndimage.sum(zone,lab,idx); ni=ndimage.sum(inv,lab,idx); ns=ndimage.sum(mk==2,lab,idx); mp=ndimage.maximum(prob,lab,idx)
                base=len(predobj[cn])
                for k in range(nl): predobj[cn].append([base+k,npx[k],nz[k],ni[k],ns[k],mp[k]])
                E=edges[cn]
                for sl,(yy,xx) in ((lab[0,:],(0,None)),(lab[-1,:],(H-1,None)),(lab[:,0],(None,0)),(lab[:,-1],(None,W-1))):
                    for p_ in np.nonzero(sl)[0]:
                        y=yy if yy is not None else p_; x=xx if xx is not None else p_
                        E[(gname,gx0+x,gy0+y)]=base+sl[p_]-1
    print(n_,gname,len(TG),flush=True)
if ROLE=='score':
    pickle.dump(dict(NSKIP=0,valid_ha=valid_ha,neg_ha=neg_ha,gt_px=gt_px,gt_meta=gt_meta,swh_px=swh_px,predobj=predobj,edges=edges,gt_cov=gt_cov,swhcov=swhcov,pix=pix),open(_pp(SG)+'.tmp','wb')); os.replace(_pp(SG)+'.tmp',_pp(SG)); sys.exit(0)
T=T_FULL
# ---------------- aggregate
def wilson(k,n,z=1.96):
    if n==0: return (None,None)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return (round(c-h,3),round(c+h,3))
BINS=[(0,4,'lt4'),(4,10,'4_10'),(10,25,'10_25'),(25,1e9,'gt25')]
res={}
for c in map(cname,CFG):
    # union-find across tile edges
    par=list(range(len(predobj[c])))
    def fd(a):
        while par[a]!=a: par[a]=par[par[a]]; a=par[a]
        return a
    E=edges[c]
    for (gname,x,y),k in E.items():
        for dx in(-1,0,1):
            for dy in(-1,0,1):
                j=E.get((gname,x+dx,y+dy))
                if j is not None and j!=k: par[fd(j)]=fd(k)
    po=pd.DataFrame(predobj[c],columns=['k','npx','nz','ninv','nswh','mp']); po['root']=[fd(k) for k in po.k]
    po=po.groupby('root').agg(npx=('npx','sum'),nz=('nz','sum'),ninv=('ninv','sum'),nswh=('nswh','sum'),mp=('mp','max'))
    po['area']=po.npx*PXA; po=po[(po.ninv/po.npx<0.5)&(po.area>=0.25)]
    po['tp']=po.nz/po.npx>=0.2; po['swh']=~po.tp&(po.nswh/po.npx>=0.5)
    r={}
    keys=list(gt_meta); ar=np.array([gt_meta[k][0] for k in keys]); cov=np.array([gt_cov[c].get(k,0)/max(gt_px.get(k,0),1) for k in keys]); seen=np.array([gt_px.get(k,0)>0 for k in keys])
    ar,cov=ar[seen],cov[seen]; hit=cov>=0.5
    for lo,hi,nm in BINS:
        s=(ar>=lo)&(ar<hi); k_,n=int((hit&s).sum()),int(s.sum()); r[f'rec_{nm}']=round(k_/max(n,1),3); r[f'ci_{nm}']=wilson(k_,n); r[f'hit_{nm}']=f'{k_}/{n}'
        sp=(po.area>=lo)&(po.area<hi); r[f'prec_{nm}']=round(po.tp[sp].sum()/max(sp.sum(),1),3); r[f'fp_ha_{nm}']=round(int((~po.tp&sp).sum())/valid_ha,3)
    R_=hit.mean(); P__=po.tp.mean() if len(po) else 0; r.update(obj_rec=round(R_,3),obj_rec_ci=wilson(int(hit.sum()),len(hit)),obj_prec=round(P__,3),obj_prec_ci=wilson(int(po.tp.sum()),len(po)),obj_f1=round(2*P__*R_/max(P__+R_,1e-9),3),
        n_gt=int(len(hit)),n_pred=int(len(po)),fp_per_ha=round(int((~po.tp).sum())/valid_ha,3))
    p=pix[c]; r.update(px_iou=round(p['tp']/max(p['tp']+p['fp']+p['fn'],1),3),px_prec=round(p['tp']/max(p['tp']+p['fp'],1),3),px_rec=round(p['tp']/max(p['tp']+p['fn'],1),3),biou=round(p['bi']/max(p['bu'],1),3))
    sk=list(swh_px); sh=sum(swhcov[c].get(k,0)/swh_px[k]>=0.5 for k in sk); r.update(swh_as_pv=round(sh/max(len(sk),1),3),swh_as_pv_hits=f'{sh}/{len(sk)}',swh_fp_objs=int(po.swh.sum()))
    res[c]=r
json.dump(dict(pgonly=PGONLY,tiles_skipped=NSKIP[0],split=split,kind=KIND,ckpt=ck,tiles=len(T),grids=sorted(T.grid.unique()),valid_ha=valid_ha,res=res),open(outj,'w'),indent=1,default=str)
print('pregate cell keep',PGT); print(pd.DataFrame(res).T[['obj_rec','obj_prec','obj_f1','rec_lt4','ci_lt4','hit_lt4','prec_lt4','fp_ha_lt4','px_iou','biou','swh_as_pv']].to_string())
