import csv,geopandas as g,numpy as np,os
R={'train':[],'val':[]}
for r in csv.DictReader(open('grid_manifest_v2e.csv')):
    if r['split'] not in R: continue
    f=os.path.expanduser(r['label_file'])
    try: d=g.read_file(f)
    except Exception as e: print('ERR',f,str(e)[:80]); continue
    if d.crs is not None and d.crs.is_geographic: d=d.to_crs(32734)
    for p in d.geometry:
        if p is None or p.is_empty or p.area>5000: continue
        rr=p.minimum_rotated_rectangle
        if rr.geom_type!='Polygon': continue
        x,y=rr.exterior.coords.xy; R[r['split']].append(max(np.hypot(x[i+1]-x[i],y[i+1]-y[i]) for i in range(2)))
for k,L in R.items():
    L=np.array(L); print(k,len(L),np.percentile(L,[50,90,99,99.9]).round(1),L.max().round(1),(L>30).sum(),(L>50).sum())
