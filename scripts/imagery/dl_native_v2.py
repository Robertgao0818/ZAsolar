"""dl_native_v2: same output as dl_native.py (EPSG:32734, 0.05 m, 2000x2000 tiles E{x}_N{y}.tif, JPEG q95 YCbCr 512-tiled)
but fetches 2x2 tiles per WMS GetMap (4000x4000 px = server max) and cuts locally. Fewer requests (~4x), thread pool, keep-alive.
Usage: python3 dl_native_v2.py --root DIR [--grids grid_list.csv] [--workers 6] [--only-tiles E1_N2,...] [--dry]
Outputs go to ROOT/tiles, ROOT/vrt, ROOT/manifest_tiles_v2.csv, ROOT/manifest_grids_v2.csv. Skips valid existing tiles. Never overwrites a valid tile.
Manifest contract (fix 2026-10-22): EVERY wanted tile that ends up valid is in manifest_tiles_v2.csv (tile,size,sha256,time),
including tiles already present in ROOT/tiles or an --existing dir. Before this fix, already-present tiles were skipped
without a manifest row, so the CT pregate (which scores manifest tiles only) never saw 12,922 tiles.
manifest_tiles_v2_src.csv (tile,path,source) gives the file actually used; source = downloaded | existing.
Consumers must resolve tile paths via this sidecar (existing tiles may live outside ROOT/tiles), and the
coverage gate scripts/check_coverage.py must pass before merge/delivery."""
import os,io,sys,time,math,csv,hashlib,argparse,subprocess,threading,numpy as np,pandas as pd,requests,rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
from PIL import Image
from concurrent.futures import ThreadPoolExecutor,as_completed
U="https://cityimg.capetown.gov.za/erdas-iws/ogc/wms/GeoSpatial Datasets";L="Aerial Imagery_Aerial Imagery 2025Jan"
R=0.05;T=2000;STEP=100;BUF=50.0;B=2  # block = BxB tiles
ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--grids",default=None)
ap.add_argument("--existing",action="append",default=[],help="extra tiles dir(s) to treat as already present (read-only)")
ap.add_argument("--workers",type=int,default=6);ap.add_argument("--only-tiles",default=None);ap.add_argument("--dry",action="store_true")
a=ap.parse_args();os.makedirs(f"{a.root}/tiles",exist_ok=True);os.makedirs(f"{a.root}/vrt",exist_ok=True)
tf=Transformer.from_crs(4326,32734,always_xy=True);tl=threading.local();lk=threading.Lock()
def tiles_for(r):
    xs,ys=tf.transform([r.minx,r.maxx,r.minx,r.maxx],[r.miny,r.miny,r.maxy,r.maxy])
    x0=math.floor((min(xs)-BUF)/STEP)*STEP;x1=math.ceil((max(xs)+BUF)/STEP)*STEP
    y0=math.floor((min(ys)-BUF)/STEP)*STEP;y1=math.ceil((max(ys)+BUF)/STEP)*STEP
    return [(int(x),int(y)) for x in range(int(x0),int(x1),STEP) for y in range(int(y0),int(y1),STEP)]
def ok(p):
    try:
        with rasterio.open(p) as s: return s.width==T and s.height==T
    except Exception: return False
def find(x,y):
    n=f"E{x}_N{y}.tif";return next((f"{d}/{n}" for d in [f"{a.root}/tiles"]+a.existing if ok(f"{d}/{n}")),None)
def have(x,y): return find(x,y) is not None
MAN=f"{a.root}/manifest_tiles_v2.csv";SRC=f"{a.root}/manifest_tiles_v2_src.csv"
def _read_keys(p):
    try:
        with open(p) as f: return {r[0] for r in csv.reader(f) if r}
    except FileNotFoundError: return set()
def _read_src(p):
    try:
        with open(p) as f: return {r[0]:r[1] for r in csv.reader(f) if len(r)>=2}  # last row per tile wins
    except FileNotFoundError: return {}
def _read_man(p):
    try:
        with open(p) as f: return {r[0]:r[1] for r in csv.reader(f) if len(r)>=2}  # tile -> size (last row wins)
    except FileNotFoundError: return {}
in_man=_read_man(MAN);in_src=_read_src(SRC)
def _rewrite_manifest(n,row):
    """Replace (row) or drop (row=None) tile n's manifest row(s) atomically, keeping one row per tile.
    Called under lk. Needed so legacy readers (header=None, all rows) never see stale/duplicate rows."""
    with open(MAN) as f: rows=[r for r in csv.reader(f) if r and r[0]!=n]
    if row: rows.append(row)
    tmp=MAN+".tmp_rewrite"
    with open(tmp,"w",newline="") as f: csv.writer(f).writerows(rows)
    os.replace(tmp,MAN)
def link_into_root(n,path):
    """Expose a valid tile from an --existing dir at ROOT/tiles/<n> (symlink) so manifest consumers that open
    ROOT/tiles/<tile> can read it. Returns the ROOT path, or None if ROOT/tiles/<n> holds an invalid real file
    (never replaced; the tile is then left OUT of the manifest so the coverage gate reports it). Only a dangling
    symlink (no data) is replaced."""
    dst=f"{a.root}/tiles/{n}"
    if os.path.abspath(path)==os.path.abspath(dst): return dst
    if os.path.islink(dst) and not os.path.exists(dst): os.unlink(dst)
    elif os.path.lexists(dst):
        if os.path.islink(dst) and ok(dst): return dst  # live link to a valid tile already
        print(f"WARN {dst} exists but is invalid; valid copy at {path}; NOT replaced and NOT manifested",flush=True);return None
    os.symlink(os.path.abspath(path),dst);return dst
def record(x,y,path,source):
    """Append tile to the manifest (once) and its resolved path to the sidecar. Thread-safe."""
    n=f"E{x}_N{y}.tif"
    with lk:
        if source=="existing":
            path=link_into_root(n,path)
            if path is None:  # ROOT/tiles/<n> is corrupt: drop any earlier manifest row so nobody reads it
                if n in in_man: _rewrite_manifest(n,None);del in_man[n]
                return
        sz=str(os.path.getsize(path))
        if in_man.get(n)!=sz:  # new tile, or file replaced since its row: exactly one row per tile
            row=[n,sz,hashlib.sha256(open(path,"rb").read()).hexdigest(),time.strftime("%Y-%m-%dT%H:%M:%S%z")]
            if n in in_man: _rewrite_manifest(n,row)
            else:
                with open(MAN,"a") as f: csv.writer(f).writerow(row)
            in_man[n]=sz
        p=os.path.abspath(path)
        if in_src.get(n)!=p:  # new tile or path changed on resume: append; readers take the last row
            with open(SRC,"a") as f: csv.writer(f).writerow([n,p,source])
            in_src[n]=p
def sess():
    if not hasattr(tl,"s"): tl.s=requests.Session()
    return tl.s
def fetch_block(bx,by,need):
    xs=sorted({x for x,_ in need});ys=sorted({y for _,y in need})
    x0,y0=min(xs),min(ys);nx=(max(xs)-x0)//STEP+1;ny=(max(ys)-y0)//STEP+1
    p=dict(service="WMS",version="1.1.1",request="GetMap",layers=L,styles="",srs="EPSG:32734",bbox=f"{x0},{y0},{x0+nx*STEP},{y0+ny*STEP}",width=T*nx,height=T*ny,format="image/jpeg")
    for k in range(5):
        try:
            r=sess().get(U,params=p,timeout=300);r.raise_for_status()
            if "image" not in r.headers.get("Content-Type",""): raise RuntimeError(r.text[:120])
            im=np.array(Image.open(io.BytesIO(r.content)).convert("RGB"));assert im.shape[:2]==(T*ny,T*nx);break
        except Exception as e: err=str(e)[:100];time.sleep(10*(k+1))
    else: return [(x,y,"fail:"+err) for x,y in need]
    out=[]
    for x,y in need:
        i=(x-x0)//STEP;j=(y-y0)//STEP;r0=(ny-1-j)*T;sub=im[r0:r0+T,i*T:i*T+T]
        dst=f"{a.root}/tiles/E{x}_N{y}.tif";tmp=dst+".tmpv2"
        if ok(dst): record(x,y,dst,"existing");out.append((x,y,"exists"));continue
        with rasterio.open(tmp,"w",driver="GTiff",width=T,height=T,count=3,dtype="uint8",crs="EPSG:32734",transform=from_origin(x,y+STEP,R,R),compress="JPEG",jpeg_quality=95,photometric="YCBCR",tiled=True,blockxsize=512,blockysize=512) as o: o.write(sub.transpose(2,0,1))
        os.replace(tmp,dst);record(x,y,dst,"downloaded")
        out.append((x,y,"ok"))
    return out
# collect all needed tiles (dedup across grids), group into BxB blocks on a global grid
grid_tiles={}
if a.only_tiles: want=[tuple(int(v[1:]) for v in t.split("_")) for t in a.only_tiles.split(",")]
else:
    G=pd.read_csv(a.grids or f"{a.root}/grid_list.csv")
    for r in G.itertuples(): grid_tiles[r.grid]=tiles_for(r)
    want=sorted({t for v in grid_tiles.values() for t in v})
loc={t:find(*t) for t in want};need=[t for t in want if loc[t] is None]
present=[t for t in want if loc[t] is not None]
blocks={}
for x,y in need: blocks.setdefault((x//(STEP*B),y//(STEP*B)),[]).append((x,y))
print(f"tiles wanted={len(want)} missing={len(need)} requests={len(blocks)} (v1 would use {len(need)})",flush=True)
unlisted=sum(f"E{x}_N{y}.tif" not in in_man for x,y in present)
print(f"already present={len(present)} (not yet in manifest: {unlisted})",flush=True)
if a.dry: sys.exit()
for x,y in present: record(x,y,loc[(x,y)],"existing" if not loc[(x,y)].startswith(f"{a.root}/tiles/") else "downloaded")
for x,y in need: loc[(x,y)]=None
st={};t0=time.time();done=0
with ThreadPoolExecutor(a.workers) as ex:
    for f in as_completed([ex.submit(fetch_block,bx,by,v) for (bx,by),v in blocks.items()]):
        for x,y,s in f.result(): st[(x,y)]=s
        done+=1
        if done%50==0 or done==len(blocks): print(f"{done}/{len(blocks)} req, {time.time()-t0:.0f}s, fails={sum(v.startswith('fail') for v in st.values())}",flush=True)
for g,ts in grid_tiles.items():
    files=[]
    for x,y in ts:
        files.append(loc.get((x,y)) or find(x,y) or f"{a.root}/tiles/E{x}_N{y}.tif")  # same validity rule as the manifest
    nf=sum(st.get(t,"").startswith("fail") for t in ts)
    if nf==0 and not os.path.exists(f"{a.root}/vrt/{g}.vrt"): subprocess.run(["gdalbuildvrt","-q",f"{a.root}/vrt/{g}.vrt"]+files,check=True)
    with open(f"{a.root}/manifest_grids_v2.csv","a") as f: csv.writer(f).writerow([g,len(ts),nf,"done" if nf==0 else "incomplete",time.strftime("%Y-%m-%dT%H:%M:%S%z")])
