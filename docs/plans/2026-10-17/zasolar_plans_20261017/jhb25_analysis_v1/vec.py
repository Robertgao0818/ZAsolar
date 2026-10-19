# Independent VECTOR implementation: pregate recall ceiling (all models) + Mask R-CNN gated P/R from predictions_metric.gpkg
import os,glob,json,numpy as np,pandas as pd,geopandas as gpd,rasterio
from shapely.geometry import box
from shapely.ops import unary_union
J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel'); PG=json.load(open(f'{J}/pregate/scores.json')); thr=PG['threshold']
GRID=gpd.read_file(os.path.expanduser('~/capetown_eval/ds3m_ft/joburg5/grids_32735.gpkg')).set_index('gridcell_id').geometry
MR='/home/gao/projects/ZAsolar/results/analysis/jhb_cbd25_3model_20260514/unified_A_perdet_sam_maskbox'
cells=[];gtrows=[];mr=[]
for g in sorted(GRID.index.intersection([os.path.basename(d) for d in glob.glob(f'{J}/tiles005/G*')])):
  kept=[];allc=0
  for t in sorted(glob.glob(f'{J}/tiles005/{g}/*.tif')):
    with rasterio.open(t) as s: b=s.bounds; res=s.res[0]
    sc=PG['scores'][os.path.relpath(t,J)]; e=[0,667,2000-667]
    for k,v in enumerate(sc):
      i,j=divmod(k,3); y0=b.top-e[i]*res; x0=b.left+e[j]*res; allc+=1
      if v>=thr: kept.append(box(x0,y0-667*res,x0+667*res,y0))
  K=unary_union(kept).intersection(GRID[g]); cells.append(dict(grid=g,kept_frac_cells=len(kept)/allc,kept_area_frac=K.area/GRID[g].area))
  gt=gpd.read_file(f'{J}/clean_gt/{g}_clean_gt.gpkg').to_crs(32735); gt['geometry']=gt.buffer(0)
  for i,q in enumerate(gt.geometry):
    gtrows.append(dict(grid=g,idx=i,area=q.area,in_gate=q.intersection(K).area/max(q.area,1e-9),cx=q.centroid.x,cy=q.centroid.y))
  f=f'{MR}/{g}/predictions_metric.gpkg'
  if os.path.exists(f):
    p=gpd.read_file(f).to_crs(32735); p=p[p.geometry.notna()]; U=unary_union(list(p.geometry.buffer(0))).intersection(K).intersection(GRID[g])
    comps=list(U.geoms) if hasattr(U,'geoms') else ([U] if not U.is_empty else [])
    comps=[c for c in comps if c.area>=0.25]; Z=unary_union(list(gt.geometry)).buffer(0.4)
    for c in comps: mr.append(dict(grid=g,area=c.area,tp=c.intersection(Z).area/c.area>=0.2))
    for i,q in enumerate(gt.geometry): gtrows[-len(gt)+i]['mr_cov']=q.intersection(U).area/max(q.area,1e-9)
  print(g,flush=True)
pd.DataFrame(cells).to_csv('gate_cells.csv',index=False); G=pd.DataFrame(gtrows); G.to_csv('gt_vec.csv',index=False); M=pd.DataFrame(mr); M.to_csv('mrcnn_vec_preds.csv',index=False)
print('GT',len(G),'reachable(>=50% in kept cells)',(G.in_gate>=.5).sum(),'fully lost(0%)',(G.in_gate==0).sum())
for lo,hi in((0,4),(4,10),(10,25),(25,1e9)): s=(G.area>=lo)&(G.area<hi); print(lo,hi,s.sum(),'reach',(G.in_gate[s]>=.5).sum())
if len(M):
  R=(G.mr_cov>=.5).mean(); P=M.tp.mean(); print('MRCNN vec P%.3f R%.3f F1%.3f FP/ha %.3f npred %d'%(P,R,2*P*R/(P+R),(~M.tp).sum()/2500,len(M)))
  for lo,hi in((0,4),(4,10)): s=(G.area>=lo)&(G.area<hi); print(lo,hi,(G.mr_cov[s]>=.5).sum(),'/',s.sum())
