"""Per-model, per-grid counts + approximate GT overlap (clean_gt, NOT the official zaeval scorer) + polygons + renders."""
import os,sys,glob,json,numpy as np,pandas as pd,geopandas as gpd,rasterio
from rasterio.merge import merge; from rasterio import features; from scipy import ndimage
from PIL import Image
J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel'); MIN_PX=int(0.5/0.0025)
RUNS=[r for r in ['convnext_t_s0','mit_b2_s0','mit_b2_s1','mit_b2_s2','segnext_b_s0'] if os.path.exists(f'preds/{r}/DONE')]
grids=sorted(os.listdir(f'{J}/tiles005')); rows=[]; os.chdir('analysis'); os.makedirs('polygons',exist_ok=True); os.makedirs('renders',exist_ok=True)
def mos(files):
    a,t=merge(files); return a,t
fps={r:[] for r in RUNS}
for g in grids:
    img,T=mos(sorted(glob.glob(f'{J}/tiles005/{g}/*.tif'))); H,W=img.shape[1:]
    gt=gpd.read_file(f'{J}/clean_gt/{g}_clean_gt.gpkg').to_crs(32735); gt=gt[gt.geometry.notna()&~gt.geometry.is_empty]
    gtr=features.rasterize([(geom,i+1) for i,geom in enumerate(gt.geometry)],out_shape=(H,W),transform=T,dtype='int32') if len(gt) else np.zeros((H,W),np.int32)
    gtarea=np.bincount(gtr.ravel(),minlength=len(gt)+1)
    small=img[:,::8,::8].transpose(1,2,0); panels=[small]
    gov=small.copy(); gov[ndimage.binary_dilation(gtr[::8,::8]>0)^(gtr[::8,::8]>0)]=[0,255,255]; panels[0]=gov
    for r in RUNS:
        pa,_=mos(sorted(glob.glob(f'../preds/{r}/{g}/*.tif'))); pv=pa[0]==1
        lab,n=ndimage.label(pv,np.ones((3,3)))
        sz=np.bincount(lab.ravel()); keep=np.where(sz>=MIN_PX)[0]; keep=keep[keep>0]
        ingt=np.bincount(lab.ravel(),weights=(gtr>0).ravel(),minlength=n+1)
        tp=int(sum(ingt[k]/sz[k]>=0.5 for k in keep)); npred=len(keep)
        cov=np.bincount(gtr.ravel(),weights=pv.ravel(),minlength=len(gt)+1); hit=int(sum(cov[i]/gtarea[i]>=0.5 for i in range(1,len(gt)+1) if gtarea[i]>0))
        area=float(sz[keep].sum()*0.0025)
        rows.append(dict(model=r,grid=g,n_obj=npred,area_m2=round(area,1),n_gt=len(gt),gt_area_m2=round(float(gt.area.sum()),1),approx_tp=tp,approx_gt_hit=hit,n_swh_px=int((pa[0]==2).sum())))
        for k in keep:
            if ingt[k]/sz[k]<0.1: sl=ndimage.find_objects((lab==k).astype(np.uint8))[0] if False else None; fps[r].append((g,int(k),int(sz[k])))
        # polygons
        km=np.isin(lab,keep)
        shp=[dict(geometry=__import__('shapely.geometry',fromlist=['shape']).shape(s),grid=g) for s,v in features.shapes(km.astype(np.uint8),mask=km,transform=T) if v==1]
        if shp: gpd.GeoDataFrame(shp,crs=32735).assign(area_m2=lambda d:d.area).to_file(f'polygons/{r}.gpkg',layer=r,mode='a' if os.path.exists(f'polygons/{r}.gpkg') else 'w')
        ov=small.copy(); s=km[::8,::8]; ov[s]=(0.4*ov[s]+0.6*np.array([255,0,255])).astype(np.uint8); e=ndimage.binary_dilation(gtr[::8,::8]>0)^(gtr[::8,::8]>0); ov[e]=[0,255,255]; panels.append(ov)
        # FP crops (largest unmatched comps)
        bad=sorted([k for k in keep if ingt[k]/sz[k]<0.1],key=lambda k:-sz[k])[:6]
        objs=ndimage.find_objects(lab)
        for k in bad:
            sl=objs[k-1]; cy=(sl[0].start+sl[0].stop)//2; cx=(sl[1].start+sl[1].stop)//2; h=max(200,(sl[0].stop-sl[0].start)//2+60,(sl[1].stop-sl[1].start)//2+60)
            y0,x0=max(0,cy-h),max(0,cx-h); c=img[:,y0:cy+h,x0:cx+h].transpose(1,2,0).copy(); m=(lab[y0:cy+h,x0:cx+h]==k); edge=ndimage.binary_dilation(m,iterations=3)^m; c[edge]=[255,0,255]
            os.makedirs(f'renders/fp_crops/{r}',exist_ok=True); Image.fromarray(c).resize((400,400)).save(f'renders/fp_crops/{r}/{g}_{k}_{int(sz[k]*0.0025)}m2.jpg',quality=85)
        del pa,lab,pv,km
    hh=min(p.shape[0] for p in panels); ww=min(p.shape[1] for p in panels)
    Image.fromarray(np.concatenate([p[:hh,:ww] for p in panels],1)).save(f'renders/{g}_sidebyside.jpg',quality=85); print(g,flush=True)
df=pd.DataFrame(rows); df.to_csv('per_grid_counts.csv',index=False)
s=df.groupby('model').agg(n_obj=('n_obj','sum'),area_m2=('area_m2','sum'),approx_tp=('approx_tp','sum'),approx_gt_hit=('approx_gt_hit','sum'),n_gt=('n_gt','sum'),gt_area=('gt_area_m2','sum'))
s['approx_P']=(s.approx_tp/s.n_obj).round(3); s['approx_R']=(s.approx_gt_hit/s.n_gt).round(3); s['approx_FP']=s.n_obj-s.approx_tp
s.to_csv('per_model_summary.csv'); print(s.to_string())
