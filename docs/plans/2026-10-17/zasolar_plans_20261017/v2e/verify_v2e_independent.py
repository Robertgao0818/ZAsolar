# independent check of v2e (written by bot, not Grok Build). CPU only, read-only.
import os,hashlib,glob,json,numpy as np,pandas as pd,geopandas as gpd,rasterio
from rasterio.features import rasterize
B='/home/gao/data/ZAsolar/geid_temporal/'; E=B+'ct_seg_trainset_v2e_20261017/data'; C=B+'ct_seg_trainset_v2c_20261015/data'; D=B+'ct_seg_trainset_v2d2_20261017/data'
md=lambda f:hashlib.md5(open(f,'rb').read()).hexdigest()
R={}
M=pd.read_csv(E+'/manifest.csv'); R['grids']=M.groupby('split').grid.nunique().to_dict(); R['tiles']=M.split.value_counts().to_dict()
R['pv_px']=M.groupby('split').n_pv.sum().to_dict()
R['chips']=pd.read_csv(E+'/chips_512px_index.csv').split.value_counts().to_dict()
REV={'G1971','G1687','G1240'}; bad=[]; n=0
for f in glob.glob(E+'/*/*/*_mask.tif'):
    sp,g,fn=f.split('/')[-3:]; ref=(C if g in REV else D)+f'/{sp}/{g}/{fn}'
    n+=1
    if not os.path.exists(ref) or md(f)!=md(ref): bad.append(f)
R['masks_checked']=n; R['mask_mismatch_vs_expected_source']=len(bad)
R['reverted_split']={g:M[M.grid==g].split.unique().tolist() for g in REV|{'G2028'}}
# manifest n_pv recompute on 300 random rows
s=M.sample(300,random_state=0); err=0
for r in s.itertuples():
    m=rasterio.open(r.mask).read(1); err+=int((m==1).sum()!=r.n_pv)
R['n_pv_recount_errors_of_300']=err
# polygons per train grid from v2e labels copy (non-empty geoms)
L='/home/gao/zasolar_data/geid_temporal/ct_labels_corrected_v2e_20261017/Capetown'
G=pd.read_csv(E+'/grid_manifest_v2e.csv'); tr=G[G.split=='train']; tot=0
for r in tr.itertuples():
    if isinstance(r.label_file,str) and os.path.exists(r.label_file):
        d=gpd.read_file(r.label_file); tot+=int((d.geometry.notna()&~d.geometry.is_empty).sum())
R['train_grids_in_grid_manifest']=len(tr); R['train_polygons_recount']=tot; R['grid_manifest_polygon_sum']=int(tr.polygon_count.sum())
R['ids_ok']=bool(G.grid_id.str.fullmatch(r'CPT[0-9]{4}').all() and G.grid_id.is_unique)
R['reverted_labels_eq_original']={g:md(glob.glob(f'{L}/{g}_SAM2_*.gpkg')[0])==md(glob.glob(f'/home/gao/projects/ZAsolar/data/annotations/Capetown/{g}_SAM2_*.gpkg')[0]) for g in REV}
# rasterisation spot-check: v2e labels vs v2e mask PV on G1971/G1687 tiles with PV
sc=[]
for g in ['G1971','G1687','G1972']:
    d=gpd.read_file(glob.glob(f'{L}/{g}_SAM2_*.gpkg')[0]); d=(d.set_crs(4326) if d.crs is None else d).to_crs(32734); d=d[d.geometry.notna()&~d.geometry.is_empty]
    for r in M[(M.grid==g)&(M.n_pv>0)].sample(3,random_state=1).itertuples():
        src=rasterio.open(r.mask); m=src.read(1)
        p=rasterize([(x.buffer(0),1) for x in d.geometry],out_shape=m.shape,transform=src.transform)
        core=rasterize([(x.buffer(-0.3),1) for x in d.geometry if not x.buffer(-0.3).is_empty] or [(d.geometry.iloc[0],0)],out_shape=m.shape,transform=src.transform)
        sc.append(dict(grid=g,tile=os.path.basename(r.mask),mask_pv_inside_poly=round(float(p[m==1].mean()),4),core_px_marked_pv_or_ignore=round(float(np.isin(m[core==1],[1,255]).mean()),4) if core.any() else None))
R['raster_spotcheck']=sc
bad5=0
for line in open(E.rsplit('/',1)[0]+'/MD5SUMS'):
    h,f=line.split(None,1); f=f.strip(); f=f if f.startswith('/') else E.rsplit('/',1)[0]+'/'+f
    bad5+= md(f)!=h
R['MD5SUMS_bad']=bad5
print(json.dumps(R,indent=1,default=str)); json.dump(R,open('verify_v2e_independent.json','w'),indent=1,default=str)
