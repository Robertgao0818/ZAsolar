# Independent re-aggregation from per-grid pickles (CPU). Own implementation of matching rules:
# pred obj = 8-conn comps merged across tile edges (within grid); drop if ninv/npx>=0.5 or area<0.25 m2; TP iff zone frac>=0.2
# GT hit iff covered px / gt px >=0.5; bins by polygon area. Bootstrap resamples GRIDS (tile-level not recoverable from pickles).
import pickle,glob,os,json,math,numpy as np
B=os.path.expanduser('~/capetown_eval/joburg_cbd25_rescore_v2_20261019/out'); PXA=0.0025
M={'convnext':'part_p1_convnext_t_s0','b2':'part_p1_mit_b2_s0','v2c':'part_v2c_verify'}
def wil(k,n,z=1.96):
    p=k/n;d=1+z*z/n;c=(p+z*z/2/n)/d;h=z*math.sqrt(p*(1-p)/n+z*z/4/n/n)/d;return round(c-h,3),round(c+h,3)
def grid_stats(P,c,g):
    rows=P['predobj'][c]; n=len(rows); par=list(range(n))
    def f(a):
        while par[a]!=a: par[a]=par[par[a]]; a=par[a]
        return a
    E=P['edges'][c]
    for (gg,x,y),k in E.items():
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                j=E.get((gg,x+dx,y+dy))
                if j is not None and j!=k: par[f(j)]=f(k)
    agg={}
    for r in rows:
        a=agg.setdefault(f(int(r[0])),[0,0,0]); a[0]+=r[1];a[1]+=r[2];a[2]+=r[3]
    tp=fp=0; small_pred=[0,0]
    for npx,nz,ni in agg.values():
        if ni/npx>=0.5 or npx*PXA<0.25: continue
        t=nz/npx>=0.2; tp+=t; fp+=(not t)
    gts=[]
    for k,(ar,_) in P['gt_meta'].items():
        gp=P['gt_px'].get(k,0)
        if gp>0: gts.append((ar,P['gt_cov'][c].get(k,0)/gp>=0.5))
    return dict(tp=tp,fp=fp,gt=gts)
def summ(S):
    tp=sum(s['tp'] for s in S); fp=sum(s['fp'] for s in S); gt=[x for s in S for x in s['gt']]
    hit=sum(h for _,h in gt); P=tp/max(tp+fp,1); R=hit/len(gt); F=2*P*R/max(P+R,1e-9)
    sm=[h for a,h in gt if a<4]; m4=[h for a,h in gt if 4<=a<10]
    return dict(P=P,R=R,F1=F,ngt=len(gt),npred=tp+fp,lt4=(sum(sm),len(sm)),b4_10=(sum(m4),len(m4)))
out={}; per={}
for m,d in M.items():
    fs=sorted(glob.glob(f'{B}/{d}/G*.pkl')); Ps={os.path.basename(x)[:-4]:pickle.load(open(x,'rb')) for x in fs}
    cfgs=list(next(iter(Ps.values()))['predobj'])
    per[m]={c:{g:grid_stats(P,c,g) for g,P in Ps.items()} for c in cfgs}
    out[m]={}
    for c in cfgs:
        s=summ(list(per[m][c].values())); k,n=s['lt4']; k2,n2=s['b4_10']
        out[m][c]=dict(P=round(s['P'],3),R=round(s['R'],3),F1=round(s['F1'],3),n_gt=s['ngt'],n_pred=s['npred'],lt4=f'{k}/{n}',lt4_wilson=wil(k,n),b4_10=f'{k2}/{n2}',b4_10_wilson=wil(k2,n2) if n2 else None)
    out[m]['_grids']=sorted(Ps); out[m]['_gtkeys_hash']=hash(tuple(sorted(Ps[g]['gt_meta'] and tuple(sorted(Ps[g]['gt_meta'].items())) for g in sorted(Ps))))
c='pg_h0.9_0.5'; G=out['v2c']['_grids']; rng=np.random.default_rng(0); bs={}
for a,b in [('b2','convnext'),('v2c','convnext'),('v2c','b2')]:
    obs=summ(list(per[a][c].values()))['F1']-summ(list(per[b][c].values()))['F1']; ds=[]
    for _ in range(2000):
        gs=rng.choice(G,len(G)); ds.append(summ([per[a][c][g] for g in gs])['F1']-summ([per[b][c][g] for g in gs])['F1'])
    bs[f'{a}-{b}']=dict(dF1=round(obs,3),ci95=[round(float(np.percentile(ds,q)),3) for q in (2.5,97.5)])
out['bootstrap_by_grid_F1diff_'+c]=bs
json.dump(out,open(os.path.dirname(os.path.abspath(__file__))+'/recheck_results_v1.json','x'),indent=1,default=str)
for m in M:
    for c in ['pg_h0.9_0.5','pg_t0.5','pg_h0.8_0.4','pg_h0.7_0.3','pg_h0.6_0.3','pg_t0.3']: print(m,c,out[m][c])
print(bs); print({m:out[m]['_gtkeys_hash'] for m in M})
