"""Vectorized counts only (same rules as analyze.py: 8-conn comps >=0.5 m2; approx TP = >=50% of comp inside clean_gt; GT hit = >=50% of GT poly covered)."""
import os,glob,numpy as np,pandas as pd,geopandas as gpd
from rasterio.merge import merge; from rasterio import features; from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor
J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel'); RUNS=['convnext_t_s0','mit_b2_s0','mit_b2_s1','mit_b2_s2','segnext_b_s0']
def one(g):
    out=[]; ref,T=merge(sorted(glob.glob(f'preds/convnext_t_s0/{g}/*.tif'))); H,W=ref.shape[1:]
    gt=gpd.read_file(f'{J}/clean_gt/{g}_clean_gt.gpkg').to_crs(32735); gt=gt[gt.geometry.notna()&~gt.geometry.is_empty]
    gtr=features.rasterize([(x,i+1) for i,x in enumerate(gt.geometry)],out_shape=(H,W),transform=T,dtype='int32') if len(gt) else np.zeros((H,W),np.int32)
    ga=np.bincount(gtr.ravel(),minlength=len(gt)+1)
    for r in RUNS:
        pa=ref if r=='convnext_t_s0' else merge(sorted(glob.glob(f'preds/{r}/{g}/*.tif')))[0]; pv=pa[0]==1
        lab,n=ndimage.label(pv,np.ones((3,3))); sz=np.bincount(lab.ravel()); ing=np.bincount(lab.ravel(),weights=(gtr>0).ravel(),minlength=n+1)
        k=np.arange(len(sz)); keep=(k>0)&(sz>=200); cov=np.bincount(gtr.ravel(),weights=pv.ravel(),minlength=len(gt)+1)
        out.append(dict(model=r,grid=g,n_obj=int(keep.sum()),area_m2=round(float(sz[keep].sum()*0.0025),1),n_gt=len(gt),gt_area_m2=round(float(gt.area.sum()),1),
          approx_tp=int((keep&(ing>=0.5*sz)).sum()),approx_fp_clear=int((keep&(ing<0.1*sz)).sum()),approx_gt_hit=int(((ga[1:]>0)&(cov[1:]>=0.5*ga[1:])).sum())))
    print(g,flush=True); return out
grids=sorted(os.listdir(f'{J}/tiles005'))
with ProcessPoolExecutor(4) as ex: rows=[x for o in ex.map(one,grids) for x in o]
df=pd.DataFrame(rows); df.to_csv('per_grid_counts_fast.csv',index=False)
s=df.groupby('model').agg(n_obj=('n_obj','sum'),area_m2=('area_m2','sum'),tp=('approx_tp','sum'),fp_clear=('approx_fp_clear','sum'),gt_hit=('approx_gt_hit','sum'),n_gt=('n_gt','sum'),gt_area=('gt_area_m2','sum'))
s['P']=(s.tp/s.n_obj).round(3); s['R']=(s.gt_hit/s.n_gt).round(3); s['F1']=(2*s.P*s.R/(s.P+s.R)).round(3); s.to_csv('per_model_summary_fast.csv'); print(s.to_string()); print('CNTDONE')
