import sys,os,time,pandas as pd,torch,torch.nn.functional as F,rasterio
sys.path.insert(0,os.path.expanduser('~/projects/ZAsolar'))
from core.inference.presence_pregate import PresencePregate, default_head_paths
torch.set_num_threads(18)
T=pd.read_csv('../tile_scores_v1.csv').sample(4,random_state=0)
g=PresencePregate(default_head_paths(),'cpu'); starts=[0,300,600,850]
for im in T.image:
  t=time.time(); a=torch.from_numpy(rasterio.open(im).read()).float()[None]/255.; t1=time.time()
  a=F.interpolate(a,size=(1250,1250),mode='bilinear',align_corners=False,antialias=True)[0].clamp(0,1); t2=time.time()
  ch=[a[:,y:y+400,x:x+400] for y in starts for x in starts]; sc=g.score(ch); t3=time.time()
  print(f'read {t1-t:.2f}s resize {t2-t1:.2f}s score16 {t3-t2:.2f}s')
