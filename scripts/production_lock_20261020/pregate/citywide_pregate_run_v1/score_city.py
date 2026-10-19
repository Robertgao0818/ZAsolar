"""CT citywide pregate SCORE ONLY (no keep/drop). Same model/adaptation as citywide_pregate_val_v1_20261019/score.py and Task A:
2000px 0.05m -> bilinear antialias 1250px, 16 chips 400px at {0,300,600,850}^2, vit30_fold0..3, cuda.
Inputs: tiles listed in aerial_2025_native005_v2dl/manifest_tiles_v2.csv (written by the downloader after a tile is complete) whose
file size matches the manifest. Skips tiles already in chip_scores_v1.csv. LIM env = stop after N new tiles (speed test).
Outputs (append): chip_scores_v1.csv (tile,chip,x0,y0,s0..s3), tile_scores_v1.csv (tile,max_score), timing_v1.txt."""
import sys,os,time,numpy as np,pandas as pd,torch,torch.nn.functional as F,rasterio
sys.path.insert(0,os.path.expanduser('~/projects/ZAsolar'))
from core.inference.presence_pregate import PresencePregate, default_head_paths
B='/home/gao/data/ZAsolar/tiles/cape_town/aerial_2025_native005_v2dl'
M=pd.read_csv(f'{B}/manifest_tiles_v2.csv',header=None,names=['tile','size','sha','t'])
M=M.drop_duplicates('tile',keep='last')
co,to='chip_scores_v1.csv','tile_scores_v1.csv'
done=set(pd.read_csv(to).tile) if os.path.exists(to) else set()
todo=[r for r in M.itertuples() if r.tile not in done and os.path.exists(f'{B}/tiles/{r.tile}') and os.path.getsize(f'{B}/tiles/{r.tile}')==r.size]
LIM=int(os.environ.get('LIM','0'))
if LIM: todo=todo[:LIM]
print(time.strftime('%F %T %Z'),'manifest',len(M),'done',len(done),'todo',len(todo),flush=True)
if not todo: sys.exit(0)
g=PresencePregate(default_head_paths(),'cuda'); S=1250; st=[0,300,600,850]
nc=not os.path.exists(co); fc=open(co,'a'); ft=open(to,'a')
if nc: fc.write('tile,chip,x0,y0,s0,s1,s2,s3\n'); ft.write('tile,max_score\n')
t0=time.time(); n=0
for r in todo:
    a=torch.from_numpy(rasterio.open(f'{B}/tiles/{r.tile}').read([1,2,3])).float()[None]/255.
    a=F.interpolate(a,size=(S,S),mode='bilinear',align_corners=False,antialias=True)[0].clamp(0,1)
    xy=[(x,y) for y in st for x in st]; sc=g.score([a[:,y:y+400,x:x+400] for x,y in xy])
    fc.write(''.join(f'{r.tile},{i},{x},{y},'+','.join(f'{v:.6g}' for v in s)+'\n' for i,((x,y),s) in enumerate(zip(xy,sc))))
    fc.flush(); ft.write(f'{r.tile},{float(np.max(sc)):.6g}\n'); ft.flush(); n+=1   # tile row written last = commit marker
    if n%500==0: print(n,len(todo),f'{(time.time()-t0)/n:.3f} s/tile',flush=True)
el=time.time()-t0; print('DONE',n,f'{el/n:.3f} s/tile',flush=True)
open('timing_v1.txt','a').write(f'{time.strftime("%F %T %Z")} LIM={LIM} n={n} wall_s={el:.1f} s_per_tile={el/n:.4f}\n')
