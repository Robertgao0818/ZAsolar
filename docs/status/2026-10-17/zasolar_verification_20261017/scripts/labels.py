import geopandas as gpd,glob,os,pandas as pd,fiona
R=os.path.expanduser('~/zasolar_data/capetown_ye_manual_labels_original/')
for d in sorted(glob.glob(R+'*/')):
    fs=sorted(glob.glob(d+'*.gpkg')); n=0;a=0;grids=set();per={}
    for f in fs:
        for L in fiona.listlayers(f):
            g=gpd.read_file(f,layer=L); g=g[g.geometry.notna()]
            k=len(g); ar=g.to_crs(32734).area.sum(); n+=k;a+=ar; gr=os.path.basename(f)[:5]; grids.add(gr); per[gr]=per.get(gr,0)+k
    print(os.path.basename(d[:-1]),'files',len(fs),'grids',len(grids),'polys',n,'area_utm34s',round(a,1))
    for k in ['G1189','G1190','G1238']:
        if k in per: print(' ',k,per[k])
