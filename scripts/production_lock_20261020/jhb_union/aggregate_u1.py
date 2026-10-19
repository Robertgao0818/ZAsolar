"""[jhb_union_v1 copy of ct_val_group_diag_v1/aggregate_v1.py: merge only, + area, comps incl union/maxp; writes results_u1.json]
Aggregate parts/{ct,jhb}_*.pkl. Metrics per model x merge (nomerge = per-tile comps; merge = 8-adjacent cross-tile, zaeval/G7 A1):
 std  : zaeval-2 one-to-one, pred object vs GT polygon, IoU>=0.5 (also 0.3) on valid px, greedy desc IoU
 group: diagnostic; pred TP if IoU(pred, union of all GT polygons it overlaps)>=0.5; GT hit if in such a group with >=half its valid px covered
 unit : census unit 0.25 m. GT units = polygons whose 0.125 m buffers intersect (vector, transitive); pred units = components
        connected after disk dilation r=2.5 px (0.125 m) per tile (+ dilated-border adjacency across tiles in 'merge');
        one-to-one IoU>=0.5 between units (unit px = sum of members' valid px).
CT 'pool' (= holdout + sel300 scored separately, summed) reproduced for std only. Grid bootstrap 10k. Writes results_v1.json (no overwrite)."""
import glob,pickle,json,os,numpy as np
from shapely.strtree import STRtree
assert not os.path.exists('results_u1.json')
class UF:
    def __init__(s,n): s.p=list(range(n))
    def f(s,a):
        while s.p[a]!=a: s.p[a]=s.p[s.p[a]]; a=s.p[a]
        return a
    def u(s,a,b):
        a,b=s.f(a),s.f(b)
        if a!=b: s.p[b]=a
def link_edges(uf,emap):
    for (k,x,y),i in emap.items():
        for dx in(-1,0,1):
            for dy in(-1,0,1):
                j=emap.get((k,x+dx,y+dy))
                if j is not None and j!=i: uf.u(i,j)
def gt_units(geoms,keep):
    ids=[o for o in sorted(geoms) if o in keep]; bufs=[geoms[o].buffer(0.125) for o in ids]; uf=UF(len(ids)); tr=STRtree(bufs)
    for i,b in enumerate(bufs):
        for j in tr.query(b,predicate='intersects'):
            if j!=i: uf.u(i,int(j))
    g={}
    for i,o in enumerate(ids): g.setdefault(uf.f(i),[]).append(o)
    return list(g.values())
def greedy(pairs,thr):
    pairs=sorted([p for p in pairs if p[0]>=thr],key=lambda t:(-t[0],t[1],str(t[2]))); up=set(); ug={}
    for iou,pi,gi in pairs:
        if pi in up or gi in ug: continue
        up.add(pi); ug[gi]=pi
    return up,ug
def binof(a): return 'lt4' if a<4 else ('4_10' if a<10 else 'ge10')
def score(D,m,merge,scope_key):
    """scope_key(tile)->scope label. Returns per-scope dict of metric counts."""
    comps=[];emap={};dmap={};tl={}
    for ti,t in enumerate(D['tiles']):
        P=t['models'][m]
        if P is None: continue
        sk=scope_key(t); base=len(comps)
        for c in P['comps']: comps.append((sk,ti)+c)
        if merge:
            for (x,y),c in P['edges'].items(): emap[(sk,x,y)]=base+c
        for (x,y),dlab in P['dedges'].items(): dmap[(sk,x,y,ti)]=(ti,dlab)
    n=len(comps); uf=UF(n)
    if merge: link_edges(uf,emap)
    # unit grouping: start from std grouping, add same (tile,dlabel), add dilated cross-tile adjacency
    ufd=UF(n)
    for i in range(n): ufd.u(i,uf.f(i))
    first={}
    for i,c in enumerate(comps):
        k=(c[1],c[5]); 
        if k in first: ufd.u(first[k],i)
        else: first[k]=i
    if merge:
        dm={}
        for (sk,x,y,ti),(t2,dl) in dmap.items():
            if (t2,dl) in first: dm[(sk,x,y)]=first[(t2,dl)]
        link_edges(ufd,dm)
    def groups(u):
        g={}
        for i in range(n): g.setdefault(u.f(i),[]).append(i)
        out=[]
        for mem in g.values():
            npx=sum(comps[i][2] for i in mem); nv=sum(comps[i][3] for i in mem); it={}
            for i in mem:
                for o,c in comps[i][4].items(): it[o]=it.get(o,0)+c
            if npx*0.0025>=0.25 and nv>0: out.append((comps[mem[0]][0],nv,it))
        return out
    std=groups(uf); uni=groups(ufd)
    res={}
    for sk in sorted({scope_key(t) for t in D['tiles']}):
        gv=np.zeros(D['NG']+1,np.int64)
        for t in D['tiles']:
            if scope_key(t)==sk: gv+=t['gt_valid']
        inc={o for o in range(1,D['NG']+1) if gv[o]>0}
        S=[p for p in std if p[0]==sk]; U=[p for p in uni if p[0]==sk]
        pairs=[(c/(nv+gv[o]-c),pi,o) for pi,(_,nv,it) in enumerate(S) for o,c in it.items() if o in inc and c>0]
        r=dict(n_gt=len(inc),n_pred=len(S))
        for thr in(0.3,0.5):
            up,ug=greedy(pairs,thr); r[f'std{thr}']=dict(tp=len(ug),fp=len(S)-len(up),fn=len(inc)-len(ug))
            if thr==0.5:
                b={k:[0,0] for k in('lt4','4_10','ge10')}
                for o in inc: k=binof(D['area'][o]); b[k][0]+=1; b[k][1]+=o in ug
                r['std_bins']=b
                predgts={}; gtp={}
                for iou,pi,o in pairs: predgts.setdefault(pi,[]).append(o); gtp.setdefault(o,[]).append((iou,pi))
                cat=dict(no_overlap=0,glued=0,other=0)
                for o in inc:
                    if o in ug: continue
                    if o not in gtp: cat['no_overlap']+=1
                    elif len(predgts[max(gtp[o])[1]])>=2: cat['glued']+=1
                    else: cat['other']+=1
                r['fn_cat']=cat; r['n_pred_glue']=sum(len(v)>=2 for v in predgts.values())
        # group diag
        tp=fp=0; hit=set()
        for _,nv,it in S:
            g=[o for o,c in it.items() if o in inc and c>0]; si=sum(it[o] for o in g); su=nv+sum(gv[o] for o in g)-si
            if g and si/su>=0.5: tp+=1; hit|={o for o in g if it[o]>=0.5*gv[o]}
            else: fp+=1
        r['group']=dict(tp_pred=tp,fp=fp,hit=len(hit&inc),n_gt=len(inc))
        # unit metric
        units=gt_units(D['geoms'],inc); uid={o:k for k,us in enumerate(units) for o in us}; uv=[sum(gv[o] for o in us) for us in units]
        up_=[]
        for pi,(_,nv,it) in enumerate(U):
            acc={}
            for o,c in it.items():
                if o in uid: acc[uid[o]]=acc.get(uid[o],0)+c
            for k,c in acc.items(): up_.append((c/(nv+uv[k]-c),pi,k))
        a,b=greedy(up_,0.5); r['unit']=dict(tp=len(b),fp=len(U)-len(a),fn=len(units)-len(b),n_units=len(units),n_pred_units=len(U))
        res[sk]=r
    return res
def prf(tp,fp,fn): P=tp/max(tp+fp,1); R=tp/max(tp+fn,1); return dict(P=round(P,4),R=round(R,4),F1=round(2*P*R/max(P+R,1e-9),4))
OUT={}; rng=np.random.default_rng(20261019)
for city,pre in (('CT','ct'),('JHB','jhb')):
    files=sorted(glob.glob(f'parts/{pre}_*.pkl')); Ds=[pickle.load(open(f,'rb')) for f in files]; models=Ds[0]['models']
    C=OUT[city]={'grids':len(Ds),'models':{},'labels':{}}
    # label splitting stats (all GT with valid px in grid scope)
    npoly=nnear=0; usz=[]
    for D in Ds:
        gv=sum(t['gt_valid'] for t in D['tiles']); inc={o for o in range(1,D['NG']+1) if gv[o]>0}
        us=gt_units(D['geoms'],inc); usz+= [len(u) for u in us]; npoly+=len(inc); nnear+=sum(len(u) for u in us if len(u)>1)
    usz=np.array(usz); C['labels']=dict(n_gt=npoly,frac_within_0p25m_of_another=round(nnear/npoly,4),n_units=len(usz),poly_per_unit_mean=round(float(usz.mean()),3),
        units_multi=int((usz>1).sum()),poly_per_multi_unit_mean=round(float(usz[usz>1].mean()),2) if (usz>1).any() else None,max_unit=int(usz.max()),
        hist={str(int(k)):int(v) for k,v in zip(*np.unique(np.minimum(usz,10),return_counts=True))})
    per={}
    for m in models:
        for merge in(True,):
            key=f'{m}|{"merge" if merge else "nomerge"}'; rows=[score(D,m,merge,lambda t:'all')['all'] for D in Ds]; per[key]=rows
            agg=lambda f:{k:sum(f(r)[k] for r in rows) for k in f(rows[0])}
            s5=agg(lambda r:r['std0.5']); s3=agg(lambda r:r['std0.3']); u=agg(lambda r:r['unit']); g=agg(lambda r:r['group'])
            bins={k:f"{sum(r['std_bins'][k][1] for r in rows)}/{sum(r['std_bins'][k][0] for r in rows)}" for k in('lt4','4_10','ge10')}
            gp=g['tp_pred']/max(g['tp_pred']+g['fp'],1); gr=g['hit']/g['n_gt']
            C['models'][key]=dict(n_gt=sum(r['n_gt'] for r in rows),std_iou05=dict(**prf(**s5),**s5),std_iou03_F1=prf(**s3)['F1'],std_bins=bins,
                fn_cat=agg(lambda r:r['fn_cat']),n_pred_glue=sum(r['n_pred_glue'] for r in rows),
                group=dict(P=round(gp,4),R=round(gr,4),F1=round(2*gp*gr/(gp+gr),4)),unit=dict(**prf(u['tp'],u['fp'],u['fn']),**u))
            ga=sum(t['gt_area_px'] for D in Ds for t in D['tiles']); pa=sum(t['pred_area_px'][m] for D in Ds for t in D['tiles'])
            C['models'][key]['area_m2']=dict(pred=round(pa*0.0025,1),gt=round(ga*0.0025,1),ratio=round(pa/ga,4))
            if city=='CT' and merge:
                pool=[score(D,m,True,lambda t:t['scope']) for D in Ds]; tot={k:0 for k in('tp','fp','fn')}; ng=0
                for p in pool:
                    for sk,r in p.items():
                        for k in tot: tot[k]+=r['std0.5'][k]
                        ng+=r['n_gt']
                C['models'][key]['pool_std_iou05']=dict(**prf(**tot),**tot,n_gt=ng)
    # bootstrap
    ng=len(Ds); idx=rng.integers(0,ng,(10000,ng)); C['bootstrap']={}
    def arr(key,met): return np.array([[r[met][k] for k in('tp','fp','fn')] for r in per[key]])
    f1=lambda a:2*a[...,0]/np.maximum(2*a[...,0]+a[...,1]+a[...,2],1)
    b2=[m for m in models if m.startswith('mit_b2')][0]
    comps=[(b2,'convnext_t_s0'),('avg2',b2),('union',b2),('maxp',b2),('union','avg2')]
    for met in('std0.5',):
        for merge in('merge',):
            for a,b in comps:
                A=arr(f'{a}|{merge}',met); B=arr(f'{b}|{merge}',met); d=f1(A[idx].sum(1))-f1(B[idx].sum(1))
                C['bootstrap'][f'{met}|{merge}|{a}-{b}']=dict(diff=round(float(f1(A.sum(0))-f1(B.sum(0))),4),ci95=[round(float(np.percentile(d,q)),4) for q in(2.5,97.5)])
json.dump(OUT,open('results_u1.json','w'),indent=1,default=int); print(json.dumps(OUT,indent=1,default=int))
