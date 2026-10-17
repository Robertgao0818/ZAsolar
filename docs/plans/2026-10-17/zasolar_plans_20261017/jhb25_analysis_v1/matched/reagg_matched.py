# Independent re-aggregation of saved per-grid partials (pregate/part/*/G*.pkl). Own union-find, own filters, per-grid stats.
import pickle,glob,os,json,numpy as np,pandas as pd
J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel/pregate/part'); PXA=0.0025
MODELS={'mit_v2c':'pg_h0.9_0.5','dinosat_v2d1_interim25k':'pg_h0.9_0.5','mit_v2d1':'pg_h0.9_0.5'}
rows=[];objs=[];preds=[]
for m,cfg in MODELS.items():
  for f in sorted(glob.glob(f'{J}/{m}/G*.pkl')):
    g=os.path.basename(f)[:-4]; P=pickle.load(open(f,'rb'))
    po=np.array([r[1:6] for r in P['predobj'][cfg]],float).reshape(-1,5)  # npx nz ninv nswh mp
    n=len(po); parent=np.arange(n)
    def find(a):
      r=a
      while parent[r]!=r: r=parent[r]
      while parent[a]!=r: parent[a],a=r,parent[a]
      return r
    E=P['edges'][cfg]
    for (gg,x,y),k in E.items():
      for dx in (-1,0,1):
        for dy in (-1,0,1):
          j=E.get((gg,x+dx,y+dy))
          if j is not None and j!=k:
            a,b=find(int(j)),find(int(k))
            if a!=b: parent[a]=b
    root=np.array([find(i) for i in range(n)])
    df=pd.DataFrame(po,columns=['npx','nz','ninv','nswh','mp']); df['root']=root
    c=df.groupby('root').agg(npx=('npx','sum'),nz=('nz','sum'),ninv=('ninv','sum'),mp=('mp','max'))
    c['area']=c.npx*PXA; keep=(c.ninv/c.npx<0.5)&(c.area>=0.25); c=c[keep]; c['tp']=c.nz/c.npx>=0.2; c['zfrac']=c.nz/c.npx
    for r in c.itertuples(): preds.append(dict(model=m,grid=g,area=r.area,tp=r.tp,zfrac=r.zfrac,mp=r.mp))
    for k,(a,cl) in P['gt_meta'].items():
      px=P['gt_px'].get(k,0)
      if px>0: objs.append(dict(model=m,grid=g,oid=int(k[1]),area=a,cv=P['gt_cov'][cfg].get(k,0)/px))
    rows.append(dict(model=m,grid=g,valid_ha=P['valid_ha']))
pd.DataFrame(rows).to_csv('grid_valid.csv',index=False); pd.DataFrame(objs).to_csv('gt_objs.csv',index=False); pd.DataFrame(preds).to_csv('pred_objs.csv',index=False)
O=pd.DataFrame(objs);Pp=pd.DataFrame(preds);V=pd.DataFrame(rows)
for m in MODELS:
  o=O[O.model==m];p=Pp[Pp.model==m];va=V[V.model==m].valid_ha.sum();R=(o.cv>=.5).mean();Pr=p.tp.mean()
  b=lambda lo,hi:f"{int(((o.cv>=.5)&(o.area>=lo)&(o.area<hi)).sum())}/{int(((o.area>=lo)&(o.area<hi)).sum())}"
  print(m,len(o),len(p),'P%.3f R%.3f F1%.3f FP/ha%.3f'%(Pr,R,2*Pr*R/(Pr+R),(~p.tp).sum()/va),'<4',b(0,4),'4-10',b(4,10),'valid_ha%.1f'%va)
