# Re-vectorise ALL 25 grids from preds/*.tif (PV class 1), same rule as analyze.py; writes compare_web/polygons only.
import glob,os,numpy as np,geopandas as gpd,rasterio,sys
from rasterio.merge import merge; from rasterio import features; from scipy import ndimage; from shapely.geometry import shape
B=os.path.expanduser('~/capetown_eval/joburg_cbd25_phase1_20261018'); MIN_PX=int(sys.argv[1])
for m in ['convnext_t_s0','mit_b2_s0','mit_b2_s1','mit_b2_s2','segnext_b_s0']:
    rows=[]
    for gd in sorted(glob.glob(f'{B}/preds/{m}/G*')):
        g=os.path.basename(gd); fs=sorted(glob.glob(f'{gd}/*.tif'))
        if len(fs)!=100: print('WARN',m,g,len(fs))
        a,T=merge([rasterio.open(f) for f in fs]); pv=a[0]==1
        lab,n=ndimage.label(pv,np.ones((3,3))); sz=np.bincount(lab.ravel()); keep=np.where(sz>=MIN_PX)[0]; keep=keep[keep>0]; km=np.isin(lab,keep)
        rows+= [dict(geometry=shape(s),grid=g) for s,v in features.shapes(km.astype(np.uint8),mask=km,transform=T) if v==1]
        print(m,g,len(rows),flush=True)
    d=gpd.GeoDataFrame(rows,crs=32735); d['area_m2']=d.area; d.to_file(f'polygons/{m}.gpkg')
print('DONE')
