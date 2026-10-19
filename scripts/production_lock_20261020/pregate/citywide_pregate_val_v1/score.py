import sys,os,time,numpy as np,pandas as pd,torch,torch.nn.functional as F,rasterio
sys.path.insert(0,os.path.expanduser('~/projects/ZAsolar'))
from core.inference.presence_pregate import PresencePregate, default_head_paths
torch.set_num_threads(int(os.environ.get('NT','18')))
M=pd.read_csv('/home/gao/data/ZAsolar/geid_temporal/ct_seg_trainset_v2d2_20261017/data/manifest.csv')
M=M[M.split.isin(['val','eval'])].drop_duplicates('image')
LIM=int(os.environ.get('LIM','0'));
if LIM: M=M.head(LIM)
out='chip_scores_v1.csv'; done=set()
if os.path.exists(out): done=set(pd.read_csv(out).image)
g=PresencePregate(default_head_paths(),'cpu')
S=1250; starts=[0,300,600,850]
f=open(out,'a');
if not done: f.write('grid,split,image,chip,x0,y0,s0,s1,s2,s3\n')
t0=time.time();n=0
for _,r in M.iterrows():
    if r.image in done: continue
    a=torch.from_numpy(rasterio.open(r.image).read()).float()[None]/255.
    a=F.interpolate(a,size=(S,S),mode='bilinear',align_corners=False,antialias=True)[0].clamp(0,1)
    chips=[];xy=[]
    for y in starts:
        for x in starts: chips.append(a[:,y:y+400,x:x+400]); xy.append((x,y))
    sc=g.score(chips)
    for i,((x,y),s) in enumerate(zip(xy,sc)):
        f.write(f'{r.grid},{r.split},{r.image},{i},{x},{y},'+','.join(f'{v:.6g}' for v in s)+'\n')
    f.flush(); n+=1
    if n%20==0: print(n,len(M),f'{(time.time()-t0)/n:.2f}s/tile',flush=True)
print('DONE',n,time.time()-t0,flush=True)
