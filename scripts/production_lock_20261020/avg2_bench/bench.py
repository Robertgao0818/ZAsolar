"""avg2 production-path timing benchmark, ONE CT val grid (default G1689), rog 4070. Same inference as
phase1_followup_20261018/infer_val_probs.py (640/480 Hann+1e-3, bf16, bs=2, softmax, p_PV uint8) for convnext_t_s0 and mit_b2_s2
(md5-checked), per-model uint8 COG ZSTD15+pred2 256-block save (G16 spec), avg2=(pc+pb)/2, then h0.9_0.5 + A3 merge (xtile_merge,
unmodified) + polygons. Each stage timed separately (cuda synchronized). Outputs under ./out (bench only)."""
import sys,os,time,json,hashlib
from pathlib import Path
PIPE=Path('~/capetown_eval/ds3m_ft/phase1_eval_pipeline_20261018').expanduser(); sys.path.insert(0,str(PIPE))
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pipeline_common_v3 import *  # noqa
os.environ['CUDA_VISIBLE_DEVICES']='0'  # as infer_val_probs.py
sys.path.insert(0,str(Path('~/capetown_eval/ct_xtile_merge_v1_20261019').expanduser()))
from xtile_merge import TileIn, merge_plan, plan_tile_ids, plan_objects, global_origin, polygons
GRID=os.environ.get('GRID','G1689'); OUT=Path('out'); OUT.mkdir(exist_ok=True)
CK={'convnext_t_s0':('convnext_t','a7365650d0f3b99c89d385e80e26d0bc'),'mit_b2_s2':('mit_b2','d74b8273a1e1aa61b5365615f24109e0')}
models={}
for name,(arch,md5) in CK.items():
    ck=Path(f'~/capetown_eval/ds3m_ft/phase1_runpod_20261017/{name}/best_objf1.pt').expanduser(); assert hashlib.md5(ck.read_bytes()).hexdigest()==md5
    cfg=json.loads((ck.parent/'config.json').read_text())['args']
    m=build_model(arch,init=None,allow_mscan_download=False,pretrained_convnext=False,upernet_ch=int(cfg.get('upernet_ch',512)),drop_path=float(cfg.get('drop_path',0.3)))
    m.load_state_dict(training.extract_state_dict(torch.load(ck,map_location='cpu',weights_only=True)),strict=True); models[name]=m.to('cuda').eval()
rows,gm=select_rows('val'); rows,gm,_=validation_scopes(rows); rows=rows[rows.grid==GRID].drop_duplicates('image')
refs=[ref_for(r) for _,r in rows.iterrows()]; print('tiles',len(refs),flush=True)
chip,stride,bs=640,480,2; hann=np.hanning(chip+2)[1:-1]; win=torch.tensor(np.outer(hann,hann)+1e-3,dtype=torch.float32,device='cuda')
@torch.inference_mode()
def infer(model,x,h,w):
    ys=sorted(set(range(0,h-chip+1,stride))|{h-chip}); xs=sorted(set(range(0,w-chip+1,stride))|{w-chip}); pos=[(y,xx) for y in ys for xx in xs]
    prob=torch.zeros(3,h,w,device='cuda'); wt=torch.zeros(h,w,device='cuda')
    for s in range(0,len(pos),bs):
        b=pos[s:s+bs]; pt=norm(torch.stack([x[:,y:y+chip,xx:xx+chip] for y,xx in b])).to('cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16): lg=model(pixel_values=pt).logits
        p=F.interpolate(lg.float(),size=(chip,chip),mode='bilinear',align_corners=False).softmax(1)
        for i,(y,xx) in enumerate(b): prob[:,y:y+chip,xx:xx+chip]+=p[i]*win; wt[y:y+chip,xx:xx+chip]+=win
    return (prob/wt.clamp(min=1e-6))[1].mul(255).round().to(torch.uint8).cpu().numpy()
def save_cog(path,a,ref):
    with rasterio.open(path,'w',driver='COG',width=a.shape[1],height=a.shape[0],count=1,dtype='uint8',crs=ref.crs,transform=ref.transform,COMPRESS='ZSTD',LEVEL=15,PREDICTOR=2,BLOCKSIZE=256,OVERVIEWS='NONE') as d: d.write(a,1)
T={k:0.0 for k in ['read','infer_convnext_t_s0','infer_mit_b2_s2','save_cog_2models','avg2']}; S=lambda: torch.cuda.synchronize()
# warm-up (not timed)
with rasterio.open(refs[0].raster_path) as s: x0=torch.from_numpy(s.read([1,2,3]).astype(np.float32))
for m in models.values(): infer(m,x0,*refs[0].shape)
S(); avg={}; t_all=time.time()
for i,ref in enumerate(refs):
    t=time.time()
    with rasterio.open(ref.raster_path) as s: x=torch.from_numpy(s.read([1,2,3]).astype(np.float32))
    T['read']+=time.time()-t; q={}
    for name,m in models.items():
        S(); t=time.time(); q[name]=infer(m,x,*ref.shape); S(); T[f'infer_{name}']+=time.time()-t
    t=time.time()
    for name in q: save_cog(OUT/f'{name}__{ref.tile_id}.tif',q[name],ref)
    T['save_cog_2models']+=time.time()-t
    t=time.time(); avg[ref.tile_id]=((q['convnext_t_s0'].astype(np.float32)+q['mit_b2_s2'].astype(np.float32))/2/255.).astype(np.float32); T['avg2']+=time.time()-t
    if i%25==0: print(i,{k:round(v/(i+1),3) for k,v in T.items()},flush=True)
gpu_wall=time.time()-t_all
tiles=[TileIn(r.tile_id,*global_origin(r.transform,r.gsd_m),(lambda k=r.tile_id: avg[k])) for r in refs]
t=time.time(); plan=merge_plan(tiles,'h0.9_0.5'); T['a3_plan']=time.time()-t
t=time.time(); parts=[polygons({tt.key:plan_tile_ids(plan,tt)},{tt.key:r.transform},refs[0].crs) for tt,r in zip(tiles,refs)]
P=pd.concat([p for p in parts if len(p)],ignore_index=True); multi=P.oid.duplicated(keep=False)
Dd=pd.concat([P[~multi],P[multi].dissolve(by='oid',as_index=False)],ignore_index=True); T['a3_ids_polygons']=time.time()-t
n=len(refs); res=dict(grid=GRID,n_tiles=n,n_objects=len(Dd),gpu_loop_wall_s=round(gpu_wall,1),stage_total_s={k:round(v,1) for k,v in T.items()},stage_s_per_tile={k:round(v/n,4) for k,v in T.items()},
    cog_bytes_per_tile=round(sum(f.stat().st_size for f in OUT.glob('*.tif'))/n,0),time=time.strftime('%F %T %Z'))
json.dump(res,open('bench_result_v1.json','w'),indent=1); print(json.dumps(res,indent=1),flush=True)
