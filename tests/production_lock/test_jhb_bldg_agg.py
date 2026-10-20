import json, os, sys
import geopandas as gpd, pytest, shapely
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'production_lock_20261020', 'jhb_bldg_agg'))
import jhb_bldg_agg_v1 as J

CRS = 'EPSG:32735'


def _w(p, geoms, **cols):
    gpd.GeoDataFrame(cols, geometry=geoms, crs=CRS).to_file(p, driver='GPKG')


def test_split_unmatched_and_resume(tmp_path):
    b = tmp_path / 'b.gpkg'; o = tmp_path / 'o.gpkg'
    # two buildings 2 m apart (not merged: no 1 m rule), one far building near a block edge (x=1000)
    _w(b, [shapely.box(0, 0, 10, 10), shapely.box(12, 0, 22, 10), shapely.box(995, 0, 1010, 10)])
    objs = [shapely.box(8, 2, 14, 4),        # spans both buildings -> split
            shapely.box(2, 2, 4, 4),          # on building 1
            shapely.box(500, 500, 510, 510),  # no building -> unmatched, kept
            shapely.box(1001, 2, 1004, 4),    # building crosses block edge; object owned by block 1
            shapely.box(990, 2, 997, 4)]      # same building, object owned by block 0
    _w(o, objs, oid=[1, 2, 3, 4, 5])
    d = tmp_path / 'out'
    J.main(['--objects', str(o), '--buildings', str(b), '--dest', str(d), '--workers', '1'])
    S = json.load(open(d / 'summary.json'))
    assert S['objects_in'] == 5 and S['objects_split_2plus_buildings'] == 1
    assert S['unmatched_no_building_n'] == 1 and S['unmatched_no_building_area_m2'] == pytest.approx(100)
    assert S['buildings_with_pv'] == 3  # edge building counted once across blocks
    assert abs(S['area_check_m2']) < 1e-6
    assert S['offroof_residual_n'] == 1 and S['offroof_residual_area_m2'] == pytest.approx(9)  # obj 5: off-roof frac 0.64 >= 0.3
    i = gpd.read_file(d / 'installations.gpkg'); edge = i[i.bld_id == 'B_1002.5_5.0']
    assert len(edge) == 1 and edge.pv_area_m2.iloc[0] == pytest.approx(6 + 5) and edge.n_objects.iloc[0] == 2
    with pytest.raises(SystemExit):  # finished run is never overwritten
        J.main(['--objects', str(o), '--buildings', str(b), '--dest', str(d), '--workers', '1'])
