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
    i = gpd.read_file(d / 'installations.gpkg'); edge = i[i.bld_area_m2.round(3) == 150]
    assert len(edge) == 1 and edge.pv_area_m2.iloc[0] == pytest.approx(6 + 5) and edge.n_objects.iloc[0] == 2
    with pytest.raises(SystemExit):  # finished run is never overwritten
        J.main(['--objects', str(o), '--buildings', str(b), '--dest', str(d), '--workers', '1'])


def test_ids_unique_and_consistent():
    g = [shapely.box(0, 0, 10, 10), shapely.box(0.04, 20, 10.04, 30), shapely.box(0, 0, 10, 10)]
    ids = J.bid_of(g)
    assert ids[0] != ids[1] and ids[0] == ids[2]


def test_nonconvergent_window_fails(tmp_path, monkeypatch):
    b = tmp_path / 'b.gpkg'; o = tmp_path / 'o.gpkg'
    _w(b, [shapely.box(0, 0, 3000, 10)]); _w(o, [shapely.box(5, 2, 8, 4)], oid=[1])
    monkeypatch.setattr(J, 'buildings_for', lambda *a, **k: (gpd.GeoDataFrame(geometry=[], crs=CRS), False))
    with pytest.raises(RuntimeError):
        J.main(['--objects', str(o), '--buildings', str(b), '--dest', str(tmp_path / 'out'), '--workers', '1'])
    assert not (tmp_path / 'out' / 'summary.json').exists()


def test_resume_refuses_changed_inputs_and_existing_outputs(tmp_path):
    b = tmp_path / 'b.gpkg'; o = tmp_path / 'o.gpkg'; d = tmp_path / 'out'
    _w(b, [shapely.box(0, 0, 10, 10)]); _w(o, [shapely.box(2, 2, 4, 4)], oid=[1])
    d.mkdir(); (d / 'pieces.gpkg').write_text('keep')
    with pytest.raises(SystemExit):
        J.main(['--objects', str(o), '--buildings', str(b), '--dest', str(d), '--workers', '1'])
    assert (d / 'pieces.gpkg').read_text() == 'keep'
    d2 = tmp_path / 'out2'; d2.mkdir(); (d2 / 'parts').mkdir()
    import json as js
    J.main(['--objects', str(o), '--buildings', str(b), '--dest', str(d2), '--workers', '1'])
    (d2 / 'summary.json').unlink()  # simulate interrupted run (test-only tmp dir)
    for n in ['pieces.gpkg', 'installations.gpkg', 'installations.csv']: (d2 / n).unlink()
    _w(b, [shapely.box(0, 0, 11, 10)])  # building layer changed in place
    with pytest.raises(SystemExit):
        J.main(['--objects', str(o), '--buildings', str(b), '--dest', str(d2), '--workers', '1'])
