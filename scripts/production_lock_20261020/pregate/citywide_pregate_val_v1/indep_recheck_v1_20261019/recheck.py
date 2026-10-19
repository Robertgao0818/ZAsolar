import json,glob,os,re,numpy as np,pandas as pd,rasterio
from rasterio.warp import transform_bounds
from statsmodels.stats.proportion import proportion_confint as ci
B='/home/gao/capetown_eval/citywide_pregate_val_v1_20261019/'
T=pd.read_csv(B+'tile_scores_v1.csv'); O=pd.read_csv(B+'objects_v1.csv'); O=O[O.area>=0.5]
C=pd.read_csv(B+'chip_scores_v1.csv'); C['m']=C[['s0','s1','s2','s3']].max(1)
s2=C.groupby('image').m.max(); d=(T.set_index('image').s-s2).abs().max(); print('tile score recompute maxdiff',d, 'chips/tile',C.groupby('image').size().unique())
# own neighbour max
S={(e,n):s for e,n,s in zip(T.e,T.n,T.s)}
nb=np.array([max([S.get((e+i,n+j),0) for i in(-1,0,1) for j in(-1,0,1) if i or j]) for e,n in zip(T.e,T.n)])
print('nb maxdiff',np.abs(nb-T.nb).max())
# leakage
sel=json.load(open('/home/gao/zasolar_data/gate_pilot_2026-09/learning_curve_2026-09-28/selection.json'))
tr=set(int(g[3:]) for g in sel['train_30']); vg=set(int(g[1:]) for g in T.grid)
print('grid-id overlap train30 vs val/eval:',sorted(tr&vg)); print('holdout overlap',sorted(set(int(g[3:]) for g in sel['holdout'])&vg))
box=[]
for g in sel['train_30']:
  for f in glob.glob(f'/home/gao/zasolar_data/tiles/cape_town/aerial_2025/G{g[3:]}/*.tif'):
    r=rasterio.open(f); box.append(transform_bounds(r.crs,T_crs:= 'EPSG:32734',*r.bounds))
box=np.array(box); print('train tif boxes',len(box))
ov=[]
for e,n in zip(T.e,T.n):
  x0,y0=e*100,n*100
  ov.append(((box[:,0]<x0+100)&(box[:,2]>x0)&(box[:,1]<y0+100)&(box[:,3]>y0)).any())
T['leak']=ov; print('val tiles spatially overlapping training tifs:',int(T.leak.sum()),'of',len(T))
K=dict(zip(T.image,T.index)); oi=O.image.map(K)
def w(k,n):
  if n==0: return 'n/a'
  lo,hi=ci(k,n,method='wilson'); return f'{k/n:.4f}[{lo:.3f},{hi:.3f}]n={n}'
rows=[]
for thr in [0.001,0.01,0.1,0.45,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.98,0.99]:
  for rule in ['tile','tile|nb','tile|nb|old']:
    keep=(T.s>=thr).values.copy()
    if 'nb' in rule: keep|=nb>=max(thr,0.45)
    if 'old' in rule: keep|=T.old.values
    ok=keep[oi.values]; a=O.area.values
    rows.append(dict(thr=thr,rule=rule,keep=round(keep.mean(),4),keep_notold=round(keep[~T.old.values].mean(),4),
      obj=w(ok.sum(),len(ok)),lt4=w(ok[a<4].sum(),(a<4).sum()),b4_10=w(ok[(a>=4)&(a<10)].sum(),((a>=4)&(a<10)).sum()),ge10=w(ok[a>=10].sum(),(a>=10).sum())))
R=pd.DataFrame(rows); R.to_csv('recheck_sweep_v1.csv',index=False); print(R.to_string())
# raw object max score at its tile
o=T.s.values[oi.values]; a=O.area.values
for nm,m in [('all',a>0),('4-10',(a>=4)&(a<10)),('<4',a<4)]:
  print(nm,'tile-score quantiles of PV objects',np.quantile(o[m],[0,0.01,0.02,0.05,0.1]).round(4))
