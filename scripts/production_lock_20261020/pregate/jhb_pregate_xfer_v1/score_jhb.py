"""Joburg transfer of the CT pregate (citywide_pregate_design_v1 §2/§7). Same code path as
~/capetown_eval/citywide_pregate_val_v1_20261019/score.py: 2000px 0.05m tile -> bilinear antialias 1250px,
16 chips 400px at {0,300,600,850}^2, vit30_fold0..3 heads; device cuda. Resumable (appends)."""
import sys,os,glob,time,numpy as np,pandas as pd,torch,torch.nn.functional as F,rasterio
sys.path.insert(0,os.path.expanduser('~/projects/ZAsolar'))
from core.inference.presence_pregate import PresencePregate, default_head_paths
J=os.path.expanduser('~/capetown_eval/ds3m_ft/joburg25_vexcel')
P=sorted(glob.glob(f'{J}/tiles005/*/*.tif')); assert len(P)==2500,len(P)
out='jhb_chip_scores_v1.csv'; done=set()
if os.path.exists(out): done=set(pd.read_csv(out).image)
g=PresencePregate(default_head_paths(),'cuda'); print('heads',default_head_paths(),flush=True)
S=1250; starts=[0,300,600,850]
f=open(out,'a')
if not done: f.write('grid,image,chip,x0,y0,s0,s1,s2,s3\n')
t0=time.time();n=0;tm=0.0
for p in P:
    if p in done: continue
    a=torch.from_numpy(rasterio.open(p).read([1,2,3])).float()[None]/255.
    a=F.interpolate(a,size=(S,S),mode='bilinear',align_corners=False,antialias=True)[0].clamp(0,1)
    chips=[a[:,y:y+400,x:x+400] for y in starts for x in starts]; xy=[(x,y) for y in starts for x in starts]
    torch.cuda.synchronize(); t1=time.time(); sc=g.score(chips); torch.cuda.synchronize(); tm+=time.time()-t1
    gr=os.path.basename(os.path.dirname(p))
    for i,((x,y),s) in enumerate(zip(xy,sc)): f.write(f'{gr},{p},{i},{x},{y},'+','.join(f'{v:.6g}' for v in s)+'\n')
    f.flush(); n+=1
    if n%100==0: print(n,len(P),f'{n/(time.time()-t0):.2f} tiles/s wall, {n/tm:.2f} tiles/s model-only',flush=True)
el=time.time()-t0
print('DONE',n,f'wall {el:.1f}s',f'{n/max(el,1e-9):.3f} tiles/s wall',f'model-only {tm:.1f}s {n/max(tm,1e-9):.3f} tiles/s',flush=True)
open('timing_v1.txt','a').write(f'{time.strftime("%F %T %Z")} n={n} wall_s={el:.1f} tiles_per_s_wall={n/max(el,1e-9):.3f} model_s={tm:.1f} tiles_per_s_model={n/max(tm,1e-9):.3f}\n')
