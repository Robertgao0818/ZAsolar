"""CT citywide avg2 production run, portable for a rented GPU pod (G13a; production_lock_schedule v4/v5;
葉 2026-10-20 11:58 "存太阳能热水器", 12:14 "改成 3，租云 GPU 跑"). Supersedes ct_city_infer_v1.py (kept).

Stage `infer` (GPU): for each tile in the tile-list CSV (column `tile`, e.g. seg_tiles_kept_plus_qa_v1.csv) and each
model (convnext_t_s0, mit_b2_s2) write two uint8 COGs, value = round(255 * softmax prob):
    <out>/<model>/pv/<stem>.tif   class 1 = PV
    <out>/<model>/swh/<stem>.tif  class 2 = SWH (solar water heater)
Inference as avg2_bench/bench.py (640/480 Hann+1e-3 window, bf16 autocast on cuda, bs=2, softmax).
Save: COG, ZSTD level 3, PREDICTOR=2 (lossless), 256 blocks, no overviews. Saving runs in separate writer
processes (--writers) so it does not block the GPU loop; reading is prefetched in a thread.
Resumable: a tile is skipped when all its outputs exist. Outputs are published from a unique temp file via
os.link, so an existing file is never replaced and a crash never leaves a partial file under the final name.

Stage `merge` (CPU): avg2 = (pv_convnext + pv_b2) / 2 / 255 (float, exactly as bench.py) then the frozen
h0.9_0.5 + A3 cross-tile merge (xtile_merge.merge_plan, unmodified). Done per 1 km block (10x10 tiles) with a
1-tile halo; an object is kept by the block containing its representative point, so each object is written once.
Objects touching the outer halo edge are flagged `halo_cut` (may be truncated; expected ~0). Missing tiles
(pregate-dropped) count as background. Output: <out>/merge/<block>.gpkg (+ .json), skipped if present.
SWH is saved only; no SWH merge here. No keep/drop or thresholds beyond h0.9_0.5.
"""
import argparse, json, os, re, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cpu_workers as W  # noqa: E402
CLASS_IDX = {'pv': 1, 'swh': 2}
MODELS = {'convnext_t_s0': ('convnext_t', 'a7365650d0f3b99c89d385e80e26d0bc'),
          'mit_b2_s2': ('mit_b2', 'd74b8273a1e1aa61b5365615f24109e0')}
TRAIN_MD5 = '961e2e8e1a3657d1f72aca27c968f83f'
CHIP, STRIDE, BS = 640, 480, 2
COG_OPTS = dict(driver='COG', COMPRESS='ZSTD', LEVEL=3, PREDICTOR=2, BLOCKSIZE=256, OVERVIEWS='NONE')
TILE_M, BLOCK_M, CFG = 100, 1000, 'h0.9_0.5'
NAME_RE = re.compile(r'^E(\d+)_N(\d+)$')


# ---------------- shared helpers ----------------
def read_tile_list(csv_path):
    import csv
    with open(csv_path, newline='') as f:
        tiles = [r['tile'].strip() for r in csv.DictReader(f) if r['tile'].strip()]
    if len(set(tiles)) != len(tiles):
        raise ValueError('duplicate tiles in list')
    return tiles


def tile_xy(name):
    """'E251300_N6230400.tif' -> (251300, 6230400) = tile lower-left corner in EPSG:32734 metres."""
    m = NAME_RE.match(Path(name).stem)
    if not m:
        raise ValueError(f'bad tile name {name}')
    return int(m.group(1)), int(m.group(2))


def out_path(out, model, cls, tile):
    return Path(out) / model / cls / f'{Path(tile).stem}.tif'


ACTIVE = list(MODELS)  # models this process runs (--model splits the two models over two processes)


def tile_outputs(out, tile):
    return [out_path(out, m, c, tile) for m in ACTIVE for c in CLASS_IDX]


def manifest_name():
    return 'done_manifest.txt' if ACTIVE == list(MODELS) else f'done_manifest_{ACTIVE[0]}.txt'


def to_u8(p):
    import numpy as np, torch
    if isinstance(p, torch.Tensor):
        return p.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    return np.clip(np.rint(np.asarray(p, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)


def publish(path, write_fn, suffix='.tif'):
    """write_fn(tmp_path) writes the file; then os.link publishes it (FileExistsError if target exists)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp' + suffix, dir=path.parent); os.close(fd)
    try:
        write_fn(tmp)
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def save_cog(path, a, crs_wkt, transform_gdal):
    """Runs in writer processes. a: uint8 [H,W]."""
    import rasterio
    from affine import Affine

    def w(tmp):
        with rasterio.open(tmp, 'w', width=a.shape[1], height=a.shape[0], count=1, dtype='uint8',
                           crs=crs_wkt, transform=Affine.from_gdal(*transform_gdal), **COG_OPTS) as d:
            d.write(a, 1)
    try:
        publish(path, w)
    except FileExistsError:
        return 'exists'
    return 'ok'


# ---------------- infer ----------------
def load_models(ckpt_root, device):
    import hashlib, importlib.util, torch
    tr = HERE / 'train2_v2e.py'
    assert hashlib.md5(tr.read_bytes()).hexdigest() == TRAIN_MD5, 'train2_v2e.py md5 mismatch'
    spec = importlib.util.spec_from_file_location('train2_v2e_pinned', tr); training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    models = {}
    for name, (arch, md5) in ((k, MODELS[k]) for k in ACTIVE):
        ck = Path(ckpt_root) / name / 'best_objf1.pt'
        h = hashlib.md5()
        with open(ck, 'rb') as f:
            for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
        assert h.hexdigest() == md5, f'md5 mismatch {ck}'
        cfg = json.loads((ck.parent / 'config.json').read_text())['args']
        if arch == 'mit_b2':  # architecture only; build_model('mit_b2') would load the DS3M init from ~/capetown_eval
            m = training.SegformerForSemanticSegmentation(training.mit_config(training.B2_DEPTHS))
        else:
            m = training.build_model(arch, init=None, allow_mscan_download=False, pretrained_convnext=False,
                                     upernet_ch=int(cfg.get('upernet_ch', 512)), drop_path=float(cfg.get('drop_path', 0.3)))
        m.load_state_dict(training.extract_state_dict(torch.load(ck, map_location='cpu', weights_only=True)), strict=True)
        models[name] = m.to(device).eval()
    return models, training.norm


def sliding_probs(model, x, device, norm):
    """x: float32 tensor [3,H,W] 0..255 -> prob [3,H,W] on device."""
    import numpy as np, torch, torch.nn.functional as F
    h, w = x.shape[1:]
    hann = np.hanning(CHIP + 2)[1:-1]
    win = torch.tensor(np.outer(hann, hann) + 1e-3, dtype=torch.float32, device=device)
    ys = sorted(set(range(0, h - CHIP + 1, STRIDE)) | {h - CHIP}); xs = sorted(set(range(0, w - CHIP + 1, STRIDE)) | {w - CHIP})
    pos = [(y, xx) for y in ys for xx in xs]
    prob = torch.zeros(3, h, w, device=device); wt = torch.zeros(h, w, device=device)
    for s in range(0, len(pos), BS):
        b = pos[s:s + BS]
        pt = norm(torch.stack([x[:, y:y + CHIP, xx:xx + CHIP] for y, xx in b])).to(device)
        if device == 'cuda':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(pixel_values=pt).logits
        else:
            lg = model(pixel_values=pt).logits
        p = F.interpolate(lg.float(), size=(CHIP, CHIP), mode='bilinear', align_corners=False).softmax(1)
        for i, (y, xx) in enumerate(b):
            prob[:, y:y + CHIP, xx:xx + CHIP] += p[i] * win; wt[y:y + CHIP, xx:xx + CHIP] += win
    return prob / wt.clamp(min=1e-6)


def read_tile(path):
    import numpy as np, rasterio
    with rasterio.open(path) as s:
        return s.read([1, 2, 3]).astype(np.float32), s.crs.to_wkt(), s.transform.to_gdal()


def run_infer(tiles, tile_dir, out, prob_fn, writers=4, readers=2, max_pending=32, max_consec_fail=2, log=None):
    """prob_fn(model_name, x_float32_numpy) -> prob [3,H,W] (tensor or array). Returns counts."""
    import concurrent.futures as cf, multiprocessing as mp, threading
    out = Path(out)
    todo = [t for t in tiles if not all(p.exists() for p in tile_outputs(out, t))]
    skipped = len(tiles) - len(todo)
    absent = [t for t in todo if not (Path(tile_dir) / t).is_file()]
    if absent:  # tiles not transferred yet: skip now, a later rerun picks them up
        todo = [t for t in todo if (Path(tile_dir) / t).is_file()]
        if log: log.write(f'ABSENT,{len(absent)},first={absent[0]}\n'); log.flush()
    ex = cf.ProcessPoolExecutor(max_workers=writers, mp_context=mp.get_context('spawn'))
    rd = cf.ThreadPoolExecutor(max_workers=max(1, readers))
    sem = threading.BoundedSemaphore(max_pending); pending = set(); lock = threading.Lock(); errors = []
    def done_cb(f):
        sem.release()
        with lock:
            pending.discard(f)
            if f.exception() is not None: errors.append(repr(f.exception()))
    n = fails = consec = 0; t0 = time.time()
    futs = [rd.submit(read_tile, str(Path(tile_dir) / t)) for t in todo[:2]]
    try:
        for i, t in enumerate(todo):
            if i + 2 < len(todo): futs.append(rd.submit(read_tile, str(Path(tile_dir) / todo[i + 2])))
            try:
                x, crs, tr = futs[i].result(); futs[i] = None
                arrays = {}
                for m in ACTIVE:
                    prob = prob_fn(m, x)
                    for c, k in CLASS_IDX.items():
                        arrays[out_path(out, m, c, t)] = to_u8(prob[k])
                consec = 0
            except Exception as e:  # noqa: BLE001 - logged, run stops after max_consec_fail
                fails += 1; consec += 1
                if log: log.write(f'FAIL,{t},{e!r}\n'); log.flush()
                if consec >= max_consec_fail:
                    raise RuntimeError(f'{consec} consecutive tile failures, last {t}: {e!r}')
                continue
            for p, a in arrays.items():
                if p.exists(): continue
                sem.acquire()
                f = ex.submit(save_cog, p, a, crs, tr)
                with lock: pending.add(f)
                f.add_done_callback(done_cb)
            if errors: raise RuntimeError(f'writer failed: {errors[0]}')
            n += 1
            if log and n % 200 == 0:
                el = time.time() - t0; log.write(f'PROGRESS,{n},{len(todo)},{el / n:.3f} s/tile,{time.strftime("%F %T %Z")}\n'); log.flush()
    finally:
        ex.shutdown(wait=True); rd.shutdown(wait=True)
    if errors: raise RuntimeError(f'writer failed: {errors[0]}')
    return dict(done=n, skipped=skipped, absent=len(absent), failed=fails, wall_s=round(time.time() - t0, 1),
                s_per_tile=round((time.time() - t0) / max(n, 1), 3))


def update_done_manifest(tiles, out):
    """Append tiles of THIS list whose 4 outputs all exist to <out>/done_manifest.txt (once each).
    Returns how many tiles of this list are complete, re-checked on disk (the manifest is a record, not the truth)."""
    mf = Path(out) / manifest_name()
    have = set(mf.read_text().split()) if mf.exists() else set()
    complete = [t for t in tiles if all(p.exists() for p in tile_outputs(out, t))]
    new = [t for t in complete if t not in have]
    if new:
        with open(mf, 'a') as f: f.write(''.join(t + '\n' for t in new))
    return len(complete)


def read_ready(path):
    p = Path(path)
    return {Path(l.strip()).name for l in p.read_text().splitlines() if l.strip()} if p.exists() else set()


def follow_infer(tiles, tile_dir, out, prob_fn, ready_file=None, poll_s=30, writers=4, readers=2, log=None, max_idle_h=6.0):
    """Process tiles as they become ready (listed in ready_file, if given) until all are done."""
    total = len(tiles); idle_since = time.time(); passes = []
    while True:
        ready = read_ready(ready_file) if ready_file else None
        now = [t for t in tiles if ready is None or t in ready]
        r = run_infer(now, tile_dir, out, prob_fn, writers=writers, readers=readers, log=log) if now else dict(done=0)
        ndone = update_done_manifest(tiles, out); passes.append(r)
        if log: log.write(f'PASS,{json.dumps(r)},done_total={ndone}/{total},{time.strftime("%F %T %Z")}\n'); log.flush()
        if ndone >= total: return dict(done_total=ndone, passes=len(passes))
        if r.get('done'): idle_since = time.time()
        elif time.time() - idle_since > max_idle_h * 3600:
            raise RuntimeError(f'no new tiles for {max_idle_h} h; done {ndone}/{total}')
        time.sleep(poll_s)


# ---------------- merge ----------------
def block_of(name):
    x, y = tile_xy(name)
    return x // BLOCK_M * BLOCK_M, y // BLOCK_M * BLOCK_M


def load_avg2(out, tile):
    """float32 avg2 PV in [0,1] exactly as bench.py: (pc + pb) / 2 / 255."""
    import numpy as np, rasterio
    a = [rasterio.open(out_path(out, m, 'pv', tile)).read(1).astype(np.float32) for m in MODELS]
    return ((a[0] + a[1]) / 2 / 255.).astype(np.float32)


def merge_block(out, block, by_xy, gsd=0.05, halo_tiles=1, crs='EPSG:32734'):
    """Returns GeoDataFrame of objects owned by this block, or None if the block has no tiles."""
    import numpy as np, geopandas as gpd, pandas as pd
    from rasterio.transform import from_origin
    from shapely.geometry import box
    sys.path.insert(0, str(HERE.parent / 'ct_xtile_merge'))
    from xtile_merge import TileIn, merge_plan, plan_tile_ids, plan_objects, global_origin, polygons
    bx, by = block; h = halo_tiles * TILE_M
    sel = [(x, y, t) for (x, y), t in by_xy.items() if bx - h <= x < bx + BLOCK_M + h and by - h <= y < by + BLOCK_M + h]
    if not any(bx <= x < bx + BLOCK_M and by <= y < by + BLOCK_M for x, y, _ in sel):
        return None
    trs = {t: from_origin(x, y + TILE_M, gsd, gsd) for x, y, t in sel}
    tiles = [TileIn(t, *global_origin(trs[t], gsd), (lambda t=t: load_avg2(out, t))) for _, _, t in sel]
    plan = merge_plan(tiles, CFG)          # pass 1, memory O(components + labelled border px)
    objs = plan_objects(plan)
    parts = []
    for tt in tiles:                        # pass 2 one tile at a time (bench.py: per-tile polygons, then dissolve)
        ids = plan_tile_ids(plan, tt)
        if ids.any():
            parts.append(polygons({tt.key: ids}, {tt.key: trs[tt.key]}, crs))
    if not parts:
        return gpd.GeoDataFrame(dict(oid=[], n_px=[], max_p=[], n_tiles=[], halo_cut=[], block=[]), geometry=[], crs=crs)
    P = pd.concat(parts, ignore_index=True); multi = P.oid.duplicated(keep=False)
    P = gpd.GeoDataFrame(pd.concat([P[~multi], P[multi].dissolve(by='oid', as_index=False)], ignore_index=True), geometry='geometry', crs=crs)
    core = box(bx, by, bx + BLOCK_M, by + BLOCK_M)
    rp = P.geometry.representative_point()
    P = P[rp.within(core) | (rp.intersects(core.boundary) & (rp.x < bx + BLOCK_M) & (rp.y < by + BLOCK_M))].copy()
    outer = box(bx - h, by - h, bx + BLOCK_M + h, by + BLOCK_M + h).boundary
    edge_tiles = {t for x, y, t in sel if not (bx - h < x and x + TILE_M < bx + BLOCK_M + h and by - h < y and y + TILE_M < by + BLOCK_M + h)}
    P['n_px'] = P.oid.map(lambda o: objs[o]['n_px']); P['max_p'] = P.oid.map(lambda o: objs[o]['max_p'])
    P['n_tiles'] = P.oid.map(lambda o: objs[o]['n_tiles'])
    P['halo_cut'] = P.geometry.distance(outer) < gsd * 1.5
    P['halo_cut'] &= P.oid.map(lambda o: bool(set(objs[o]['tile_keys']) & edge_tiles)) if h else True
    P['block'] = f'E{bx}_N{by}'
    P['oid'] = [f'E{bx}_N{by}_{i}' for i in range(len(P))]
    return P.reset_index(drop=True)


def _merge_one(out, block, by_xy, gsd=0.05):
    mdir = Path(out) / 'merge'; g = mdir / f'E{block[0]}_N{block[1]}.gpkg'
    if g.exists(): return 'skip', 0, 0
    P = merge_block(out, block, by_xy, gsd=gsd)
    if P is None: return 'empty', 0, 0
    publish(g, lambda tmp: P.to_file(tmp, driver='GPKG', layer='objects'), suffix='.gpkg')
    return 'ok', len(P), int(P.halo_cut.sum()) if len(P) else 0


def run_merge(tiles, out, workers=4, gsd=0.05):
    import concurrent.futures as cf, multiprocessing as mp
    missing = [t for t in tiles if not all(out_path(out, m, 'pv', t).exists() for m in MODELS)]
    if missing:
        raise RuntimeError(f'{len(missing)} tiles lack PV outputs (first {missing[0]}); finish infer first')
    by_xy = {tile_xy(t): t for t in tiles}
    blocks = sorted({block_of(t) for t in tiles})
    res = dict(ok=0, skip=0, empty=0, objects=0, halo_cut=0)
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as ex:
        for st, n, hc in ex.map(_merge_one, [out] * len(blocks), blocks, [by_xy] * len(blocks), [gsd] * len(blocks)):
            res[st] += 1; res['objects'] += n; res['halo_cut'] += hc
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='stage', required=True)
    a1 = sp.add_parser('infer'); a2 = sp.add_parser('merge')
    for a in (a1, a2):
        a.add_argument('--tiles-csv', required=True); a.add_argument('--out', required=True); a.add_argument('--limit', type=int, default=0)
    a1.add_argument('--tile-dir', required=True); a1.add_argument('--ckpt-root', required=True)
    a1.add_argument('--device', default='cuda', choices=['cuda', 'cpu']); a1.add_argument('--writers', default='auto', help="COG writer processes; 'auto' = min(cores-2, mem cap / 0.4 GB)"); a1.add_argument('--readers', default='auto', help="tile read/decode threads; 'auto' = min(4, writers)"); a1.add_argument('--mem-cap-gb', type=float, default=6.0)
    a1.add_argument('--follow', action='store_true', help='keep polling until every listed tile is done')
    a1.add_argument('--ready-file', default=None, help='only process tiles named in this file (e.g. READY.txt)')
    a1.add_argument('--poll-s', type=int, default=30)
    a1.add_argument('--model', choices=list(MODELS), default=None, help='run only this model (two processes share the GPU)')
    a2.add_argument('--workers', default='auto'); a2.add_argument('--mem-cap-gb', type=float, default=6.0); a2.add_argument('--allow-unreviewed-merge', action='store_true')
    a = ap.parse_args(argv)
    if hasattr(a, 'writers'):
        a.writers = max(1, W.resolve(a.writers, 0.4, a.mem_cap_gb))
        a.readers = max(1, min(4, a.writers) if str(a.readers).lower() == 'auto' else int(a.readers))
        print(f'workers: writers={a.writers} readers={a.readers}', flush=True)
    tiles = read_tile_list(a.tiles_csv)
    if a.limit: tiles = tiles[:a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    global ACTIVE
    if getattr(a, 'model', None): ACTIVE = [a.model]
    log = open(out / (f'{a.stage}_log_v2.csv' if ACTIVE == list(MODELS) else f'{a.stage}_log_v2_{ACTIVE[0]}.csv'), 'a')
    log.write(f'START,{a.stage},{len(tiles)},{time.strftime("%F %T %Z")},{json.dumps(vars(a))}\n'); log.flush()
    if a.stage == 'infer':
        import torch
        models, norm = load_models(a.ckpt_root, a.device)
        def prob_fn(name, x):
            with torch.inference_mode():
                # move the tile to the device first so chip slicing + normalisation run on the GPU (was CPU-bound);
                # same fp32 ops, same result
                xt = torch.from_numpy(x)
                if a.device == 'cuda': xt = xt.pin_memory().to('cuda', non_blocking=True)
                return sliding_probs(models[name], xt, a.device, norm)
        if a.follow or a.ready_file:
            ready = read_ready(a.ready_file) if a.ready_file else None
            res = follow_infer(tiles, a.tile_dir, out, prob_fn, ready_file=a.ready_file, poll_s=a.poll_s if a.follow else 0,
                               writers=a.writers, readers=a.readers, log=log) if a.follow else run_infer(
                [t for t in tiles if t in ready], a.tile_dir, out, prob_fn, writers=a.writers, readers=a.readers, log=log)
        else:
            res = run_infer(tiles, a.tile_dir, out, prob_fn, writers=a.writers, readers=a.readers, log=log)
        update_done_manifest(tiles, out)
    else:
        if not a.allow_unreviewed_merge:
            raise SystemExit('merge stage is NOT reviewed yet (Codex r3: block-halo ownership can double-count long objects); '
                             'run only infer on the pod. --allow-unreviewed-merge for development only.')
        if a.limit:
            raise SystemExit('merge --limit would write partial blocks into the real output; not allowed')
        res = run_merge(tiles, out, workers=W.resolve(a.workers, 1.0, a.mem_cap_gb))
    log.write(f'END,{a.stage},{json.dumps(res)},{time.strftime("%F %T %Z")}\n'); log.flush()
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
