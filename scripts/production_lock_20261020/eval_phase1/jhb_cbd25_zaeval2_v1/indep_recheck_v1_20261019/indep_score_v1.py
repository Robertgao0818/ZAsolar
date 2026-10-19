"""Independent re-score of Joburg CBD25 saved probs (pregate 0.45 cell gate already baked in the probs).
Own implementation of zaeval-2@8044d14 objects.py rules (does NOT import zaeval):
 pred = hysteresis h0.9_0.5 (8-conn), components 8-conn per tile, merged across tile edges (global px 8-adjacency),
 drop merged comps with area<0.25 m2 or 0 valid px; GT = clean_gt buffer(0)+dedupe, oid 1..n, area = polygon area;
 mask per G6/score_z2: 1 = pixel-centre polygon, 255 = 6px cross band (dilate&~erode, border_value 0), 255 = outside grid poly or nodata;
 exclude-pixels: valid = mask!=255; gt_id zeroed off valid; GT in n_gt iff valid px>0. IoU on valid px; greedy one-to-one desc IoU.
Also: integer-threshold binarize on uint8 (>=230 / >=128) vs float /255 compare; gate-consistency per tile; FN/FP diagnostics.
usage: indep_score_v1.py GRID  -> parts/GRID.pkl"""
import os,sys,glob,json,pickle,numpy as np,geopandas as gpd,rasterio
from rasterio import features
from scipy import ndimage
sys.path.insert(0,os.path.expanduser('~/projects/ZAsolar_segv1_wt/scripts/training'))
from build_ct_seg_trainset_v1 import dedupe_polygons
G=sys.argv[1]; SRC=os.path.expanduser('~/capetown_eval/jhb_cbd25_zaeval2_v1_20261019')
J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel'); J0=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg5')
MODELS=['convnext_t_s0','mit_b2_s0','v2c']; ALL=MODELS+['avg3']
out=f'parts/{G}.pkl'; os.makedirs('parts',exist_ok=True); assert not os.path.exists(out)
GP=gpd.read_file(f'{J0}/grids_32735.gpkg').set_index('gridcell_id').geometry[G]
PG=json.load(open(f'{J}/pregate/scores.json'))
S4=ndimage.generate_binary_structure(2,1); S8=np.ones((3,3),bool)
d=gpd.read_file(f'{J}/clean_gt/{G}_clean_gt.gpkg').to_crs(32735); d=d[d.geometry.notna()&~d.geometry.is_empty].copy(); d['geometry']=d.geometry.buffer(0); d=dedupe_polygons(d.reset_index(drop=True))
geoms=list(d.geometry); area=np.array([q.area for q in geoms]); NG=len(geoms); oids=np.arange(1,NG+1)
gt_px=np.zeros(NG+1,np.int64); gt_valid=np.zeros(NG+1,np.int64); gt_cov={m:np.zeros(NG+1,np.int64) for m in ALL}
gt_px_at=np.zeros(NG+1,np.int64)  # all_touched raster px (G6 style n_gt)
def hyst(p,hi,lo):
    lab,nl=ndimage.label(p>=lo,S8)
    if nl==0: return np.zeros(p.shape,bool)
    return np.r_[False,ndimage.maximum(p,lab,np.arange(1,nl+1))>=hi][lab]
parts={m:[] for m in ALL}; edges={m:{} for m in ALL}; diag=dict(int_vs_float_px_diff=0,borderline_comps={m:0 for m in MODELS},gate_violation_px={m:0 for m in MODELS},zero_tiles={m:0 for m in MODELS},gate_zero_tiles=0,ntiles=0,shapes=set())
P=sorted(glob.glob(f'{J}/tiles005/{G}/*.tif'))
for p in P:
    with rasterio.open(p) as s: img=s.read([1,2,3]); tr=s.transform; b=s.bounds
    H,W=img.shape[1:]; t=os.path.basename(p)[:-4]; diag['ntiles']+=1; diag['shapes'].add((H,W))
    sel=[i for i,q in enumerate(geoms) if q.intersects(__import__('shapely').geometry.box(*b))]
    ids=features.rasterize([(geoms[i],i+1) for i in sel],(H,W),transform=tr,dtype='int32') if sel else np.zeros((H,W),np.int32)
    if sel:
        ida=features.rasterize([(geoms[i],i+1) for i in sel],(H,W),transform=tr,dtype='int32',all_touched=True); gt_px_at+=np.bincount(ida.ravel(),minlength=NG+1)
    pv=ids>0
    valid_t=features.rasterize([(GP,1)],(H,W),transform=tr).astype(bool)&img.any(0)
    band=ndimage.binary_dilation(pv,S4,iterations=6)&~ndimage.binary_erosion(pv,S4,iterations=6)
    valid=valid_t&~band   # == (mask!=255)
    gid=np.where(valid,ids,0)
    gt_px+=np.bincount(ids.ravel(),minlength=NG+1); gt_valid+=np.bincount(gid.ravel(),minlength=NG+1)
    # gate
    sc=PG['scores'][os.path.relpath(p,J)]; km=np.zeros((H,W),bool); e=[0,667,H-667]
    for k,v in enumerate(sc):
        if v>=0.45: i,j=divmod(k,3); km[e[i]:e[i]+667,e[j]:e[j]+667]=True
    if not km.any(): diag['gate_zero_tiles']+=1
    u8={m:np.load(f'{SRC}/probs/{m}/{G}/{t}.npz')['p'] for m in MODELS}
    preds={}
    for m in MODELS:
        a=u8[m]; assert a.dtype==np.uint8 and a.shape==(H,W)
        diag['gate_violation_px'][m]+=int((a[~km]>0).sum()); diag['zero_tiles'][m]+=int(not a.any())
        pi=hyst(a,230,128); pf=hyst(a.astype(np.float32)/255.,0.9,0.5); diag['int_vs_float_px_diff']+=int((pi!=pf).sum())
        lab,nl=ndimage.label(a>=128,S8)
        if nl: mx=ndimage.maximum(a,lab,np.arange(1,nl+1)); diag['borderline_comps'][m]+=int(((mx==229)|(mx==230)).sum())
        preds[m]=pi
    avg=np.mean([u8[m].astype(np.float32)/255. for m in MODELS],0).astype(np.float32); preds['avg3']=hyst(avg,0.9,0.5)
    gx0=int(round(tr.c/0.05)); gy0=int(round(-tr.f/0.05))
    ys=np.r_[np.zeros(W,int),np.full(W,H-1),np.arange(1,H-1),np.arange(1,H-1)]; xs=np.r_[np.arange(W),np.arange(W),np.zeros(H-2,int),np.full(H-2,W-1)]
    for m in ALL:
        pm=preds[m]; gt_cov[m]+=np.bincount(ids[pm],minlength=NG+1)
        lab,nl=ndimage.label(pm,S8)
        if nl==0: continue
        lr=lab.ravel(); npx=np.bincount(lr,minlength=nl+1); nva=np.bincount(lr[valid.ravel()],minlength=nl+1)
        bo=valid.ravel()&(lr>0)&(gid.ravel()>0); inter=[dict() for _ in range(nl+1)]
        if bo.any():
            pr,cnt=np.unique(np.stack([lr[bo],gid.ravel()[bo]]),axis=1,return_counts=True)
            for (c,o),n in zip(pr.T,cnt): inter[c][int(o)]=int(n)
        base=len(parts[m])
        for c in range(1,nl+1): parts[m].append([int(npx[c]),int(nva[c]),inter[c]])
        bl=lab[ys,xs]; f=bl>0
        for y,x,c in zip(ys[f],xs[f],bl[f]): edges[m][(gx0+int(x),gy0+int(y))]=base+int(c)-1
pickle.dump(dict(grid=G,NG=NG,area=area,gt_px=gt_px,gt_valid=gt_valid,gt_px_at=gt_px_at,gt_cov=gt_cov,parts=parts,edges=edges,diag=diag),open(out+'.tmp','wb')); os.replace(out+'.tmp',out); print('DONE',G,flush=True)
