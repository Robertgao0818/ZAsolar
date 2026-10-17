"""Empirical cross-grid duplicates in the CT census inventories (CPU, read-only). Prints JSON; writes census_dupes.json next to this script."""
import os, json, numpy as np, geopandas as gpd, shapely
H=os.path.expanduser; OUT=os.path.dirname(os.path.abspath(__file__)); R={}
P=gpd.read_file(H('~/projects/ZAsolar/data/task_grid_cpt.gpkg')).to_crs(32734).set_index('gridcell_id').geometry
for name in ['ct_census_inventory_cpt.gpkg','ct_census_inventory_cpt_merged_iou010.gpkg']:
    G=gpd.read_file(H(f'~/Dropbox/RA_Solar/Gao/ct_census/{name}')).to_crs(32734)
    G['geometry']=G.geometry.make_valid()
    r=dict(n=len(G), cols=[c for c in G.columns if c!='geometry'][:30], area_m2=round(G.area.sum()))
    gc='gridcell_id' if 'gridcell_id' in G else None
    if gc and G[gc].astype(str).str.startswith('CPT').all():
        cell=P.reindex(G[gc]).values
        cen=G.geometry.representative_point()
        out=~shapely.within(cen.values, cell)
        r['n_reppoint_outside_own_cell']=int(out.sum()); r['area_outside_own_cell_m2']=round(float((G.area-shapely.intersection(G.geometry.values,cell).area()).sum() if False else sum(g.difference(c).area for g,c in zip(G.geometry,cell))))
        l,rr=G.sindex.query(G.geometry,predicate='intersects'); m=(l<rr)&(G[gc].values[l]!=G[gc].values[rr])
        l,rr=l[m],rr[m]; inter=shapely.intersection(G.geometry.values[l],G.geometry.values[rr]); ia=shapely.area(inter)
        a=G.area.values; iou=ia/(a[l]+a[rr]-ia)
        k=ia>0.01
        r['cross_grid_overlapping_pairs']=int(k.sum()); r['cross_grid_overlap_area_m2']=round(float(ia[k].sum()))
        r['pairs_iou_gt_0.1']=int((iou>0.1).sum()); r['pairs_iou_le_0.1_overlap']=int(((iou<=0.1)&k).sum())
        r['area_in_pairs_iou_le_0.1_m2']=round(float(ia[(iou<=0.1)&k].sum()))
    R[name]=r
json.dump(R,open(f'{OUT}/census_dupes.json','w'),indent=1); print(json.dumps(R,indent=1))
