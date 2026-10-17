import numpy as np,pandas as pd
from scipy.stats import binomtest
O=pd.read_csv('gt_objs.csv');P=pd.read_csv('pred_objs.csv');V=pd.read_csv('grid_valid.csv');Gv=pd.read_csv('gt_vec.csv');M=pd.read_csv('mrcnn_vec_preds.csv')
O['hit']=O.cv>=.5; ms=['mit_v2c','dinosat_v2d1_interim25k','mit_v2d1']; grids=sorted(O.grid.unique())
def tab(m):
  o=O[O.model==m].groupby('grid'); p=P[P.model==m].groupby('grid')
  t=pd.DataFrame(dict(nh=o.hit.sum(),ng=o.hit.size(),ntp=p.tp.sum(),npred=p.tp.size())).reindex(grids).fillna(0)
  s=O[(O.model==m)&(O.area>=4)&(O.area<10)].groupby('grid'); t['h410']=s.hit.sum(); t['n410']=s.hit.size(); t=t.fillna(0)
  t['va']=V[V.model==m].set_index('grid').valid_ha; return t
Tm={m:tab(m) for m in ms}
# mrcnn per grid (vector)
g=Gv.groupby('grid'); tm=pd.DataFrame(dict(nh=(Gv.mr_cov>=.5).groupby(Gv.grid).sum(),ng=g.size())); pm=M.groupby('grid'); tm['ntp']=pm.tp.sum(); tm['npred']=pm.tp.size()
s=Gv[(Gv.area>=4)&(Gv.area<10)]; tm['h410']=(s.mr_cov>=.5).groupby(s.grid).sum(); tm['n410']=s.groupby('grid').size(); tm=tm.reindex(grids).fillna(0); tm['va']=Tm['mit_v2c'].va; Tm['mrcnn_vec']=tm
def met(t):
  R=t.nh.sum()/t.ng.sum();Pr=t.ntp.sum()/t.npred.sum(); return np.array([Pr,R,2*Pr*R/(Pr+R),(t.npred-t.ntp).sum()/t.va.sum(),t.h410.sum()/t.n410.sum()])
rng=np.random.default_rng(0);B=10000;idx=rng.integers(0,25,(B,25))
bs={m:np.array([met(t.iloc[i]) for i in idx]) for m,t in Tm.items()}
nm=['P','R','F1','FP/ha','R4-10']
print('== point [95% grid-bootstrap CI]')
for m,t in Tm.items():
  pt=met(t);lo,hi=np.percentile(bs[m],[2.5,97.5],0); print(m,'  '.join(f'{n} {a:.3f} [{l:.3f},{h:.3f}]' for n,a,l,h in zip(nm,pt,lo,hi)))
print('== paired differences A-B [95% CI] two-sided bootstrap p')
for a,b in (('mit_v2c','dinosat_v2d1_interim25k'),('mit_v2c','mit_v2d1'),('dinosat_v2d1_interim25k','mit_v2d1')):
  d=bs[a]-bs[b];pt=met(Tm[a])-met(Tm[b]);lo,hi=np.percentile(d,[2.5,97.5],0);p=2*np.minimum((d<=0).mean(0),(d>=0).mean(0))
  print(a,'-',b,'  '.join(f'{n} {x:+.3f} [{l:+.3f},{h:+.3f}] p={q:.3f}' for n,x,l,h,q in zip(nm,pt,lo,hi,p)))
print('== object-level McNemar on recall (same GT objects)')
W=O.pivot_table(index=['grid','oid'],columns='model',values='hit').astype(bool); W['area']=O[O.model=='mit_v2c'].set_index(['grid','oid']).area
for a,b in (('mit_v2c','dinosat_v2d1_interim25k'),('mit_v2c','mit_v2d1'),('dinosat_v2d1_interim25k','mit_v2d1')):
  for lab,s in (('all',W.index==W.index),('<4',W.area<4),('4-10',(W.area>=4)&(W.area<10))):
    x=W[s];n10=int((x[a]&~x[b]).sum());n01=int((~x[a]&x[b]).sum()); p=binomtest(n10,n10+n01).pvalue if n10+n01 else 1
    print(a,b,lab,'only A',n10,'only B',n01,'p=%.4f'%p)
print('== per grid')
pg=pd.DataFrame({f'{m}_{k}':v for m,t in Tm.items() for k,v in (('R',t.nh/t.ng),('FP',t.npred-t.ntp))}); pg['ngt']=Tm['mit_v2c'].ng; pg=pg.join(pd.read_csv('gate_cells.csv').set_index('grid'))
pg.to_csv('per_grid.csv'); print(pg.round(3).to_string())
for m in Tm: fp=(Tm[m].npred-Tm[m].ntp).sort_values(ascending=False); print(m,'FP total',int(fp.sum()),'top5 grids',dict(fp.head(5).astype(int)),'share top5 %.2f'%(fp.head(5).sum()/fp.sum()))
print('== FP types')
for m in ms:
  f=P[(P.model==m)&~P.tp]; print(m,'n',len(f),'area bins',np.histogram(f.area,[0,1,4,10,25,1e9])[0],'zfrac==0',(f.zfrac==0).sum(),'0<z<0.2(near-GT partial)',((f.zfrac>0)&(f.zfrac<.2)).sum(),'mp>=0.99',(f.mp>=.99).sum())
print('== misses by bin + gate-unreachable')
for m in ms:
  o=O[O.model==m]; print(m,'missed',int((~o.hit).sum()),'bins',np.histogram(o.area[~o.hit],[0,4,10,25,1e9])[0])
print('gate unreachable GT(vec)',int((Gv.in_gate<.5).sum()),'bins',np.histogram(Gv.area[Gv.in_gate<.5],[0,4,10,25,1e9])[0],'per grid',dict(Gv[Gv.in_gate<.5].grid.value_counts()))
