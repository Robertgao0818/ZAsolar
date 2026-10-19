# Read-only compare: Mask R-CNN v3c (reviewed) vs phase-1 models, JHB CBD25. Writes only into compare_web/.
import geopandas as gpd, pandas as pd, numpy as np, glob, os, json, random, subprocess
import rasterio, rasterio.windows
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
H=os.path.expanduser('~'); B=f'{H}/capetown_eval/joburg_cbd25_phase1_20261018'; O=f'{B}/compare_web'
R=f'{H}/projects/ZAsolar/results/johannesburg/v3c_vexcel_2024_ch1_sample'; T=f'{H}/capetown_eval/ds3m_ft/joburg25_vexcel/tiles005'
MODELS=['convnext_t_s0','mit_b2_s0','mit_b2_s1','mit_b2_s2','segnext_b_s0']
COL={'mrcnn':'#ffff00','convnext_t_s0':'#ff2020','mit_b2_s0':'#00e5ff','mit_b2_s1':'#2f7bff','mit_b2_s2':'#b04dff','segnext_b_s0':'#30ff60'}
mr=[]
for f in sorted(glob.glob(f'{R}/G*/review/G*_reviewed.gpkg')):
    d=gpd.read_file(f).to_crs(32735); d['grid']=os.path.basename(f)[:5]; mr.append(d)
mr=pd.concat(mr,ignore_index=True); mr=mr[mr.geometry.notna()&~mr.geometry.is_empty].copy(); mr['geometry']=mr.geometry.buffer(0)
print('mrcnn source values',mr['source'].value_counts(dropna=False).to_dict())
mr['review_status']=mr.review_status.fillna('nan'); mr['mid']=range(len(mr)); mr['area']=mr.geometry.area
pm={}
for m in MODELS:
    d=gpd.read_file(f'{O}/polygons/{m}.gpkg').to_crs(32735); d=d[d.geometry.notna()&~d.geometry.is_empty].copy(); d['geometry']=d.geometry.buffer(0); d['area']=d.geometry.area; d=d.reset_index(drop=True); pm[m]=d
# match rule: overlap if IoU>=0.1 OR intersection >= 50% of the smaller polygon
def match(a,b):
    j=gpd.sjoin(a[['geometry']].reset_index(names='ia'),b[['geometry']].reset_index(names='ib'),predicate='intersects')
    out=[]
    for ia,ib in zip(j.ia,j.ib):
        ga,gb=a.geometry.iloc[ia],b.geometry.iloc[ib]; i=ga.intersection(gb).area
        if i<=0: continue
        if i/ga.union(gb).area>=0.1 or i/min(ga.area,gb.area)>=0.5: out.append((ia,ib))
    return out
mr=mr.reset_index(drop=True)
hit={m:set() for m in MODELS}; mhit={m:set() for m in MODELS}
for m in MODELS:
    for ia,ib in match(mr,pm[m]): hit[m].add(ia); mhit[m].add(ib)
mr['models']=[','.join(m for m in MODELS if i in hit[m]) for i in range(len(mr))]; mr['n_models']=[sum(i in hit[m] for m in MODELS) for i in range(len(mr))]
# new-only objects: cluster new-model objects not matching MRCNN, union-find across models
no=[]
for m in MODELS:
    d=pm[m].loc[[i for i in range(len(pm[m])) if i not in mhit[m]]].copy(); d['model']=m; no.append(d)
no=gpd.GeoDataFrame(pd.concat(no,ignore_index=True),crs=32735)
par=list(range(len(no)))
def fd(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
for ia,ib in match(no,no):
    if ia!=ib: par[fd(ia)]=fd(ib)
no['cl']=[fd(i) for i in range(len(no))]
cl=no.groupby('cl').agg(models=('model',lambda s:','.join(m for m in MODELS if m in set(s))),n_models=('model','nunique'),grid=('grid','first'),area=('area','max'))
cl['geometry']=no.groupby('cl').geometry.apply(lambda s:s.union_all() if hasattr(s,'union_all') else s.unary_union)
cl=gpd.GeoDataFrame(cl.reset_index(),crs=32735)
keep=mr.review_status.isin(['correct','edit'])
A=mr[keep&(mr.n_models==0)]; C=mr[keep&(mr.n_models>0)]; D=mr[(mr.review_status=='delete')&(mr.n_models>0)]; Dn=mr[(mr.review_status=='delete')&(mr.n_models==0)]
# tables
rows=[]
for nm,d in [('(a) MRCNN kept, no new model',A),('(c) MRCNN kept & >=1 new model',C),('(d) MRCNN deleted & >=1 new model',D),('MRCNN deleted & no new model (agree)',Dn)]:
    rows.append([nm,len(d),round(d.area.sum())]+[int(d.models.str.contains(m).sum()) for m in MODELS])
rows.append(['(b) new-model only (clusters, any model)',len(cl),round(cl.area.sum())]+[int(cl.models.str.contains(m).sum()) for m in MODELS])
rows.append(['(b) of which ConvNeXt-T',int(cl.models.str.contains('convnext').sum()),round(cl[cl.models.str.contains('convnext')].area.sum())]+['']*5)
tab=pd.DataFrame(rows,columns=['category','n','area_m2']+MODELS); tab.to_csv(f'{O}/summary_categories.csv',index=False)
pr=[]
for m in MODELS:
    d=pm[m]; mt=d.index.isin(list(mhit[m]))
    pr.append([m,len(d),round(d.area.sum()),int(mt.sum()),int((~mt).sum()),round(d[~mt].area.sum()),len(set(hit[m])&set(np.where(keep)[0])),int(keep.sum())-len(set(hit[m])&set(np.where(keep)[0])),len(set(hit[m])&set(D.index))])
pt=pd.DataFrame(pr,columns=['model','n_obj','area_m2','match_mrcnn','new_only','new_only_area','mrcnn_kept_hit','mrcnn_kept_missed','mrcnn_deleted_hit']); pt.to_csv(f'{O}/summary_per_model.csv',index=False)
print(tab.to_string()); print(pt.to_string()); print('review status',mr.review_status.value_counts().to_dict())
D.drop(columns='geometry').assign(wkt=D.geometry.to_wkt()).to_csv(f'{O}/category_d_all.csv',index=False)
# crops
for g in []:
    v=''
    if not os.path.exists(v): subprocess.run(['gdalbuildvrt','-q',v]+sorted(glob.glob(f'{T}/{g}/*.tif')),check=True)
RAWD=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel/raw')
from PIL import Image, ImageDraw
from shapely.ops import transform as stf
import pyproj
TO3857=pyproj.Transformer.from_crs(32735,3857,always_xy=True).transform
for g in mr.grid.unique():
    v=f'{O}/vrt_raw/{g}.vrt'
    if not os.path.exists(v): subprocess.run(['gdalbuildvrt','-q','-srcnodata','0',v]+sorted(glob.glob(f'{RAWD}/{g}/*.tif')),check=True)
def stretch(im):
    # mild display-only stretch: 1-99 pct per channel, blended 40% with original
    f=im.astype(np.float32); out=np.empty_like(f)
    for k in range(3):
        lo,hi=np.percentile(f[k][f.sum(0)>0] if (f.sum(0)>0).any() else f[k],[1,99]); out[k]=np.clip((f[k]-lo)/max(hi-lo,1)*255,0,255)
    return np.clip(0.6*f+0.4*out,0,255).astype(np.uint8)
def crop(name,geom,grid):
    p=f'{O}/crops_v2/{name}.jpg'; pc=f'{O}/crops_v2/{name}_clean.jpg'
    if os.path.exists(p) and os.path.exists(pc): return
    c=geom.centroid; r=max(12,np.sqrt(geom.envelope.area)*1.2)
    from shapely.geometry import box; bb=box(c.x-r,c.y-r,c.x+r,c.y+r); bb3=stf(TO3857,bb)
    with rasterio.open(f'{O}/vrt_raw/{grid}.vrt') as s:
        w=rasterio.windows.from_bounds(*bb3.bounds,s.transform).round_offsets().round_lengths()
        im=s.read([1,2,3],window=w,boundless=True,fill_value=0); tr=s.window_transform(w)
    im=stretch(im); h,wd=im.shape[1:]; sc=max(1,int(np.ceil(800/max(h,wd))))
    img=Image.fromarray(np.moveaxis(im,0,-1)).resize((wd*sc,h*sc),Image.NEAREST); img.save(pc,quality=95)
    inv=~tr
    def px(x,y): cc,rr=inv*(x,y); return (cc*sc,rr*sc)
    dr=ImageDraw.Draw(img)
    def draw(gs,col,lw,dash=False):
        for gg in gs:
            gg=stf(TO3857,gg)
            for pp in (gg.geoms if hasattr(gg,'geoms') else [gg]):
                if pp.geom_type!='Polygon': continue
                pts=[px(x,y) for x,y in pp.exterior.coords]
                if dash:
                    for i in range(0,len(pts)-1,2): dr.line(pts[i:i+2],fill=col,width=lw)
                else: dr.line(pts,fill=col,width=lw,joint='curve')
    sub=mr[mr.intersects(bb)]
    for m in MODELS: draw(pm[m][pm[m].intersects(bb)].geometry,COL[m],1)
    draw(sub[sub.review_status!='delete'].geometry,COL['mrcnn'],2); draw(sub[sub.review_status=='delete'].geometry.map(lambda g:g.segmentize(0.4)),COL['mrcnn'],2,True)
    img.save(p,quality=95)
random.seed(42); items=[]
def add(cat,df,n,kind):
    idx=list(df.index); 
    if n and len(idx)>n: idx=random.sample(idx,n)
    for i in idx:
        r=df.loc[i]; nm=f'{cat}_{kind}{i}'; crop(nm,r.geometry,r.grid)
        items.append(dict(id=nm,cat=cat,grid=r.grid,area=round(float(r.area),1),models=r.models,n_models=int(r.n_models),status=r.get('review_status',''),conf=round(float(r.get('confidence',np.nan)),3) if 'confidence' in r else None,img=f'crops_v2/{nm}.jpg',clean=f'crops_v2/{nm}_clean.jpg'))
add('d',D,0,'m'); add('a',A,30,'m'); add('c',C,30,'m')
bc=cl[cl.models.str.contains('convnext')]; add('b_cnx',bc,30,'n'); add('b_other',cl[~cl.models.str.contains('convnext')],30,'n')
json.dump(dict(items=items,colors=COL,table=tab.fillna('').to_dict('records'),per_model=pt.to_dict('records')),open(f'{O}/data.js.json','w'))
open(f'{O}/data.js','w').write('window.DATA='+open(f'{O}/data.js.json').read())
print('crops',len(items),'D',len(D))
