"""[jhb_union_v1: copy of ct_val_group_diag_v1/score_jhb_v1.py; models conv, b2 s0, avg2, union=OR of own h0.9_0.5 masks, maxp=hyst(max), avg2 at hi 0.82/0.85/0.88; + area px within grid&nodata]
Joburg CBD25 (2500 tiles, saved uint8 probs incl. official 0.45 cell pregate) -> parts/jhb_GRID.pkl. Labels as G6/score_z2_v1
(clean_gt buffer0+dedupe; mask 1 pixel-centre, 255 6px band + outside grid/nodata; exclude-pixels). Models convnext_t_s0,
mit_b2_s0, v2c, avg3 (3-model mean, as score_z2), avg2 (conv+b2 s0, like-for-like with CT prob-avg). usage: score_jhb_v1.py GRID"""
import os,sys,glob,pickle,numpy as np,geopandas as gpd,rasterio,shapely
from rasterio import features
from scipy import ndimage
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from core_v1 import *
sys.path.insert(0,os.path.expanduser('~/projects/ZAsolar_segv1_wt/scripts/training'))
from build_ct_seg_trainset_v1 import dedupe_polygons
G=sys.argv[1]; out=f'parts/jhb_{G}.pkl'; os.makedirs('parts',exist_ok=True); assert not os.path.exists(out)
SRC=os.path.expanduser('~/capetown_eval/jhb_cbd25_zaeval2_v1_20261019')
J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel'); J0=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg5')
GP=gpd.read_file(f'{J0}/grids_32735.gpkg').set_index('gridcell_id').geometry[G]; S4=ndimage.generate_binary_structure(2,1)
d=gpd.read_file(f'{J}/clean_gt/{G}_clean_gt.gpkg').to_crs(32735); d=d[d.geometry.notna()&~d.geometry.is_empty].copy(); d['geometry']=d.geometry.buffer(0); d=dedupe_polygons(d.reset_index(drop=True))
gl=list(d.geometry); NG=len(gl); geoms={i+1:q for i,q in enumerate(gl)}; area={i+1:float(q.area) for i,q in enumerate(gl)}
M3=['convnext_t_s0','mit_b2_s0']; MODELS=M3+['avg2','union','maxp','avg2_h082','avg2_h085','avg2_h088']; tiles=[]
for p in sorted(glob.glob(f'{J}/tiles005/{G}/*.tif')):
    with rasterio.open(p) as s: img=s.read([1,2,3]); tr=s.transform; b=s.bounds
    H,W=img.shape[1:]; t=os.path.basename(p)[:-4]
    sel=[i for i,q in enumerate(gl) if q.intersects(shapely.geometry.box(*b))]
    ids=features.rasterize([(gl[i],i+1) for i in sel],(H,W),transform=tr,dtype='int32') if sel else np.zeros((H,W),np.int32)
    pv=ids>0; vt=features.rasterize([(GP,1)],(H,W),transform=tr).astype(bool)&img.any(0)
    band=ndimage.binary_dilation(pv,S4,iterations=6)&~ndimage.binary_erosion(pv,S4,iterations=6)
    valid=vt&~band; gid=np.where(valid,ids,0).astype(np.int64)
    u={m:np.load(f'{SRC}/probs/{m}/{G}/{t}.npz')['p'].astype(np.float32)/255. for m in M3}
    masks={m:hyst(u[m],0.9,0.5) for m in M3}; a2=(u['convnext_t_s0']+u['mit_b2_s0'])/2; masks['avg2']=hyst(a2,0.9,0.5)
    masks['union']=masks['convnext_t_s0']|masks['mit_b2_s0']; masks['maxp']=hyst(np.maximum(u['convnext_t_s0'],u['mit_b2_s0']),0.9,0.5)
    for h in (0.82,0.85,0.88): masks[f'avg2_h{int(round(h*100)):03d}']=hyst(a2,h,0.5)
    del a2,u
    gx0=int(round(tr.c/0.05)); gy0=int(round(-tr.f/0.05))
    T=dict(tile=t,scope='all',gt_valid=np.bincount(gid[valid].ravel(),minlength=NG+1),gt_px=np.bincount(ids.ravel(),minlength=NG+1),models={})
    for m in MODELS: T['models'][m]=tile_parts(masks[m],valid,gid,gx0,gy0); T[f'cov_{m}']=np.bincount(ids[masks[m]],minlength=NG+1)
    T['gt_area_px']=int((pv&vt).sum()); T['pred_area_px']={m:int((masks[m]&vt).sum()) for m in MODELS}; del masks
    tiles.append(T)
pickle.dump(dict(city='JHB',grid=G,NG=NG,geoms=geoms,area=area,crs='EPSG:32735',tiles=tiles,models=MODELS),open(out+'.tmp','wb')); os.replace(out+'.tmp',out); print('DONE',G,len(tiles),flush=True)
