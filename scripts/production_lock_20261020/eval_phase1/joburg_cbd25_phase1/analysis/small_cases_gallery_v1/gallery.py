import glob, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
A='/home/gao/capetown_eval/joburg_cbd25_phase1_20261018/analysis/'
ref=gpd.read_file(A+'recall_with_fn_v1/reference_with_hits.gpkg').to_crs(32735)
ref['oid']=range(len(ref)); s=ref[(ref.area>=4)&(ref.area<10)].copy()
cn,b2=s.hit_convnext_t_s0.astype(bool),s.hit_mit_b2_s0.astype(bool)
cats={'C miss / b2 hit':s[~cn&b2],'both miss':s[~cn&~b2],'C hit / b2 miss':s[cn&~b2],'both hit':s[cn&b2]}
cnt={k:len(v) for k,v in cats.items()}; print('n small',len(s),cnt)
pd.Series(cnt).to_csv('category_counts.csv')
for k,v in cats.items(): v.drop(columns='geometry').to_csv(f"cases_{k.replace(' ','').replace('/','_')}.csv",index=False)
pick=[]
for k,n in [('C miss / b2 hit',6),('both miss',3),('C hit / b2 miss',3)]:
    v=cats[k]; pick+= [(k,r) for r in v.sample(min(n,len(v)),random_state=20261019).itertuples()]
P={m:gpd.read_file(A+f'polygons/{m}.gpkg').to_crs(32735) for m in ['convnext_t_s0','mit_b2_s0']}
tiles=sorted(glob.glob('/home/gao/capetown_eval/ds3m_ft/joburg25_vexcel/tiles005/*/*.tif'))
idx=[]
for t in tiles:
    with rasterio.open(t) as d: idx.append((t,box(*d.bounds),d.crs))
fig,ax=plt.subplots(3,4,figsize=(16,12.8)); ax=ax.ravel()
for i,(k,r) in enumerate(pick):
    c=r.geometry.centroid; H=8; bb=(c.x-H,c.y-H,c.x+H,c.y+H)
    ts=[t for t,b,cr in idx if b.intersects(box(*bb))]
    from rasterio.merge import merge
    im,_=merge(ts,bounds=bb,res=0.05,indexes=[1,2,3]); im=im.transpose(1,2,0)
    a=ax[i]; a.imshow(im,extent=(bb[0],bb[2],bb[1],bb[3]))
    for gg in gpd.GeoSeries([r.geometry]).explode(): a.plot(*gg.exterior.xy,color='cyan',lw=2)
    for m,col,ls in [('convnext_t_s0','yellow','-'),('mit_b2_s0','magenta','--')]:
        q=P[m]; q=q[q.intersects(box(*bb))]
        for gg in q.geometry.intersection(box(*bb)).explode():
            if gg.geom_type=='Polygon': a.plot(*gg.exterior.xy,color=col,lw=1.3,ls=ls)
    a.set_xlim(bb[0],bb[2]); a.set_ylim(bb[1],bb[3]); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"{k}\n#{r.oid} {r.grid} {r.group} {r.geometry.area:.1f} m²",fontsize=9)
for a in ax[len(pick):]: a.axis('off')
fig.suptitle(f"JHB CBD25 small reference panels 4–10 m² (n={len(s)}): cyan=reference, yellow=ConvNeXt-T s0, magenta dashed=MiT-b2 s0. 16 m crops, Vexcel 2024 5 cm\ncounts: "+', '.join(f'{k} {v}' for k,v in cnt.items()),fontsize=10)
plt.tight_layout(); plt.savefig('joburg_small_cases.png',dpi=100)
