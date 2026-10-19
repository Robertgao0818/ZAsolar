#!/usr/bin/env python3
"""Check authorized eval grid footprints against manifest tile footprints."""
import csv, json, math, re
from pathlib import Path
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Polygon, box

ROOT=Path(__file__).resolve().parent
BASE=Path('/home/gao/zasolar_data/tiles/cape_town/aerial_2025_native005')
DATA=Path('/home/gao/data/ZAsolar/geid_temporal/ct_seg_trainset_v2f_20261017/data')
GRIDS={'G1349','G1686','G1692','G1800','G2038'}

def main():
    m=pd.read_csv(DATA/'manifest.csv'); m=m.loc[(m.split=='eval') & m.grid.isin(GRIDS)]
    gl=pd.read_csv(BASE/'grid_list.csv')
    tf=Transformer.from_crs(4326,32734,always_xy=True)
    details=[]; summary=[]; footprints={'type':'FeatureCollection','features':[]}
    from shapely.geometry import mapping
    for grid in sorted(GRIDS):
        r=gl.loc[gl.grid=='CPT'+grid[1:]].iloc[0]
        # Densify each lon/lat edge; do not substitute its axis-aligned projected bbox.
        corners=[(r.minx,r.miny),(r.maxx,r.miny),(r.maxx,r.maxy),(r.minx,r.maxy)]
        points=[]
        for a,b in zip(corners,corners[1:]+corners[:1]):
            for i in range(101):
                t=i/101; points.append(tf.transform(a[0]*(1-t)+b[0]*t,a[1]*(1-t)+b[1]*t))
        poly=Polygon(points)
        footprints['features'].append({'type':'Feature','properties':{'grid':grid,'crs':'EPSG:32734'},'geometry':mapping(poly)})
        paths={Path(p).stem:Path(p) for p in m.loc[m.grid==grid,'image']}
        x0,y0,x1,y1=poly.bounds
        bbox={f'E{x}_N{y}':box(x,y,x+100,y+100) for x in range(math.floor(x0/100)*100,math.ceil(x1/100)*100,100) for y in range(math.floor(y0/100)*100,math.ceil(y1/100)*100,100)}
        needed={t for t,b in bbox.items() if poly.intersection(b).area>1e-6}
        missing=needed-set(paths)
        absent=[]
        for tile in sorted(set(bbox)|set(paths)):
            path=paths.get(tile,BASE/'tiles'/f'{tile}.tif')
            exists=path.is_file(); size=path.stat().st_size if exists else 0
            if tile in needed and not exists:absent.append(tile)
            xy=re.fullmatch(r'E(\d+)_N(\d+)',tile); x,y=map(int,xy.groups())
            details.append({'grid':grid,'tile':tile,'needed_for_footprint':tile in needed,'in_manifest':tile in paths,
               'file_exists':exists,'file_size_bytes':size,'image':str(path),'intersection_area_m2':poly.intersection(box(x,y,x+100,y+100)).area,
               'bbox_only_outside_footprint':tile in bbox and tile not in needed})
        summary.append({'grid':grid,'manifest_tiles':len(paths),'footprint_needed_tiles':len(needed),'needed_missing_from_manifest':len(missing),
           'needed_missing_on_disk':len(absent),'manifest_files_exist':sum(p.is_file() for p in paths.values()),
           'bbox_tiles':len(bbox),'bbox_only_outside_footprint':len(set(bbox)-needed),
           'bbox_holes_outside_footprint':len((set(bbox)-set(paths))-needed),
           'missing_tiles':';'.join(sorted(missing|set(absent))),
           'missing_file_sizes_bytes':';'.join(f'{t}:{(BASE/"tiles"/(t+".tif")).stat().st_size if (BASE/"tiles"/(t+".tif")).is_file() else 0}' for t in sorted(missing|set(absent)))})
    for filename,rows in [('coverage_eval_orig10.csv',summary),('coverage_eval_orig10_tiles_v2.csv',details)]:
        with (ROOT/filename).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (ROOT/'eval_footprints_v2.geojson').open('x') as f:json.dump(footprints,f)
    print(json.dumps(summary,indent=2))
    assert all(r['needed_missing_from_manifest']==r['needed_missing_on_disk']==0 for r in summary)

if __name__=='__main__':main()
