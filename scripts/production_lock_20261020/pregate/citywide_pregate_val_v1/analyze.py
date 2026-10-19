import os,re,numpy as np,pandas as pd,rasterio,pyogrio,json
from scipy import ndimage
from statsmodels.stats.proportion import proportion_confint as ci
M=pd.read_csv('/home/gao/data/ZAsolar/geid_temporal/ct_seg_trainset_v2d2_20261017/data/manifest.csv')
M=M[M.split.isin(['val','eval'])].drop_duplicates('image')
C=pd.read_csv('chip_scores_v1.csv'); C['s']=C[['s0','s1','s2','s3']].max(1)
T=C.groupby('image').agg(s=('s','max'),grid=('grid','first'),split=('split','first')).reset_index()
T=T.merge(M[['image','mask','n_pv']],on='image')
def en(p): m=re.search(r'E(\d+)_N(\d+)',p); return int(m[1])//100,int(m[2])//100
T['e'],T['n']=zip(*T.image.map(en))
# objects
if os.path.exists('objects_v1.csv'): O=pd.read_csv('objects_v1.csv')
else:
  rows=[]
  for _,r in T[T.n_pv>0].iterrows():
    a=rasterio.open(r['mask']).read(1)==1
    lab,k=ndimage.label(a,structure=np.ones((3,3)))
    for i,c in enumerate(ndimage.sum(a,lab,range(1,k+1))): rows.append((r.image,i,c*0.0025))
  O=pd.DataFrame(rows,columns=['image','obj','area']); O.to_csv('objects_v1.csv',index=False)
O=O[O.area>=0.5]  # drop specks <0.5 m2
# old layer tiles
g=pyogrio.read_dataframe('/home/gao/RA_solar/backdating/capetown/citywide_v1/deliverable_2026-09-02/ct_citywide_install_dated_20260902.gpkg',columns=[])
old=set()
for b in g.geometry.bounds.itertuples():
  for e in range(int(b.minx//100),int(b.maxx//100)+1):
    for n in range(int(b.miny//100),int(b.maxy//100)+1): old.add((e,n))
T['old']=[(e,n) in old for e,n in zip(T.e,T.n)]
S=dict(zip(zip(T.e,T.n),T.s))
def nbmax(e,n): return max([S.get((e+i,n+j),0) for i in(-1,0,1) for j in(-1,0,1) if i or j])
T['nb']=[nbmax(e,n) for e,n in zip(T.e,T.n)]
def w(k,n):
  if n==0: return 'n/a'
  lo,hi=ci(k,n,method='wilson'); return f'{k/n:.4f} [{lo:.3f},{hi:.3f}] (n={n})'
res=[];tdict=T.set_index('image')
for thr in [0.45,0.2,0.1,0.05,0.02,0.01,0.005,0.003,0.002,0.001]:
  for rule in ['tile','tile|nb','tile|old','tile|nb|old']:
    keep=T.s>=thr
    if 'nb' in rule: keep|=T.nb>=max(thr,0.45)
    if 'old' in rule: keep|=T.old
    K=dict(zip(T.image,keep)); ok=O.image.map(K)
    pv=T.n_pv>0
    def rr(m): return w(int(ok[m].sum()),int(m.sum()))
    res.append(dict(thr=thr,rule=rule,keep_all=f'{keep.mean():.3f}',keep_val=f'{keep[T.split=="val"].mean():.3f}',keep_eval=f'{keep[T.split=="eval"].mean():.3f}',
      keep_given_old=f'{keep[T.old].mean():.3f}',keep_given_notold=f'{keep[~T.old].mean():.3f}',
      tile_pv_recall=w(int(keep[pv].sum()),int(pv.sum())),obj_all=rr(O.area>0),obj_lt4=rr(O.area<4),obj_4_10=rr((O.area>=4)&(O.area<10)),obj_ge10=rr(O.area>=10),
      area_recall=f'{O.area[ok].sum()/O.area.sum():.4f}',
      obj_val=rr(O.image.map(tdict.split)=='val'),obj_eval=rr(O.image.map(tdict.split)=='eval')))
R=pd.DataFrame(res); R.to_csv('pregate_val_results_v1.csv',index=False)
print(R[['thr','rule','keep_all','keep_given_old','keep_given_notold','tile_pv_recall','obj_all','obj_4_10','obj_lt4','area_recall']].to_string())
print('tiles',len(T),'old frac',T.old.mean(),'pv tiles',(T.n_pv>0).sum(),'objects',len(O),'old tiles citywide',len(old))
json.dump(dict(n_old_tiles=len(old),val_old_frac=float(T.old.mean())),open('meta_v1.json','w'))
T.to_csv('tile_scores_v1.csv',index=False)
