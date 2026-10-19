"""v3 evaluation retains the pinned v2 input/scoring contracts."""
from pipeline_common_v2 import *


def validation_scopes(rows=None):
    """Rebuild the fixed selection BEFORE any spatial sorting or smoke restriction."""
    m, gm = manifests()
    provenance = json.loads((ROOT / 'metadata_v3/selection_provenance_v3.json').read_text())
    order_path = ROOT / 'metadata_v3/load_bundle_order_v3.csv'
    assert digest(order_path) == provenance['order_sha256']
    assert provenance['v2f_manifest_md5'] == MANIFEST_MD5
    order = pd.read_csv(order_path)
    assert not order.duplicated(['grid', 'image_name']).any()
    mapping = {(r.grid, r.image_name): int(r.load_bundle_row) for r in order.itertuples()}
    assert set(mapping) == {(r.grid, Path(r.image).name) for r in m.itertuples()}
    m['load_bundle_row'] = [mapping[(r.grid, Path(r.image).name)] for r in m.itertuples()]
    pool = m.loc[(m.split == 'val') & m.included_default].sort_values('load_bundle_row').copy()
    assert len(pool) == 1703
    selected = pool.sample(300, random_state=1)
    rank = {i: n for n, i in enumerate(selected.load_bundle_row)}
    m['scope_v3'] = np.where(m.split == 'eval', 'eval',
        np.where(m.load_bundle_row.isin(rank), 'val_sel300', 'val_holdout'))
    m['selection_rank'] = m.load_bundle_row.map(rank).astype('Int64')
    allowed = m.loc[m.included_default].copy()
    if rows is not None:
        keys = {(r.grid, Path(r.image).name) for r in rows.itertuples()}
        allowed = allowed.loc[[(r.grid, Path(r.image).name) in keys for r in allowed.itertuples()]]
        assert len(allowed) == len(rows), 'Rows are not an eligible val/eval subset'
    return allowed, gm, provenance
