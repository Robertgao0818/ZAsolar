"""CT citywide inference, production avg2 path (G13a, production_lock_schedule v4/v5).

Per tile and per model (convnext_t_s0, mit_b2_s2) this writes TWO uint8 COGs:
  <out>/<model>/pv/<tile>.tif   = round(255 * softmax[class 1 = PV])
  <out>/<model>/swh/<tile>.tif  = round(255 * softmax[class 2 = SWH, solar water heater])   (v5, 葉 2026-10-20 11:58)
Same inference as avg2_bench/bench.py (640/480 Hann+1e-3 window, bf16 autocast on cuda, bs=2, softmax) and
the same G16 save convention (uint8 COG, ZSTD level 15, predictor 2, 256 blocks, no overviews).
avg2 = (pv_convnext + pv_b2)/2 is recomputed downstream from the saved per-model uint8 maps; --save-avg also
writes <out>/avg2/{pv,swh}/<tile>.tif (uint8 round of the mean of the two uint8 maps).

Resumable: a tile is skipped only when every file it should produce exists; files are written to a unique temp file and
published with os.link, so a crash never leaves a half-written COG under the final name and an existing output
is never replaced, even by a concurrent run.
Score/save only: no thresholding, merging or keep/drop here.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

CLASS_IDX = {'pv': 1, 'swh': 2}
CK = {'convnext_t_s0': ('convnext_t', 'a7365650d0f3b99c89d385e80e26d0bc'),
      'mit_b2_s2': ('mit_b2', 'd74b8273a1e1aa61b5365615f24109e0')}
CK_ROOT = Path('~/capetown_eval/ds3m_ft/phase1_runpod_20261017').expanduser()
PIPE = Path('~/capetown_eval/ds3m_ft/phase1_eval_pipeline_20261018').expanduser()
CHIP, STRIDE, BS = 640, 480, 2


def to_u8(p):
    """float prob tensor/array in [0,1] -> uint8 with round(p*255) (same as bench.py)."""
    import torch
    if isinstance(p, torch.Tensor):
        return p.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    import numpy as np
    return np.clip(np.rint(np.asarray(p, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)


def avg_u8(a, b):
    """uint8 mean of two uint8 maps, round half to even like np.rint (deterministic)."""
    import numpy as np
    return np.rint((a.astype(np.float32) + b.astype(np.float32)) / 2.0).astype(np.uint8)


def out_paths(out, tile, models, save_avg):
    stem = Path(tile).stem
    names = list(models) + (['avg2'] if save_avg else [])
    return {(n, c): Path(out) / n / c / f'{stem}.tif' for n in names for c in CLASS_IDX}


def save_cog(path, a, profile_src):
    """Write a uint8 COG to a unique temp file, then publish with os.link (fails if the target exists),
    so concurrent writers or reruns can never replace an existing output."""
    import tempfile, rasterio
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent); os.close(fd)
    try:
        with rasterio.open(tmp, 'w', driver='COG', width=a.shape[1], height=a.shape[0], count=1, dtype='uint8',
                           crs=profile_src.crs, transform=profile_src.transform, COMPRESS='ZSTD', LEVEL=15,
                           PREDICTOR=2, BLOCKSIZE=256, OVERVIEWS='NONE') as d:
            d.write(a, 1)
        os.link(tmp, path)  # atomic, raises FileExistsError if another writer published first
    finally:
        os.unlink(tmp)  # our own temp name only


def process_tile(tile, models, out, save_avg, prob_fn):
    """Infer one tile and write every missing output. prob_fn(model, x) -> prob [3,H,W] in [0,1].
    Returns the number of files written (0 = tile already complete)."""
    import numpy as np, rasterio
    paths = out_paths(out, tile, models, save_avg)
    if all(p.exists() for p in paths.values()):
        return 0
    with rasterio.open(tile) as src:
        x = src.read([1, 2, 3]).astype(np.float32)
        q = {}
        for name, m in models.items():
            prob = prob_fn(m, x)
            for c, k in CLASS_IDX.items():
                q[(name, c)] = to_u8(prob[k])
        if save_avg:
            for c in CLASS_IDX:
                q[('avg2', c)] = avg_u8(q[('convnext_t_s0', c)], q[('mit_b2_s2', c)])
        n = 0
        for key, p in paths.items():
            if not p.exists():
                save_cog(p, q[key], src); n += 1
    return n


def sliding_probs(model, x, device, norm):
    """x: float32 [3,H,W] 0..255 -> float32 prob [3,H,W] on device (Hann-weighted 640/480)."""
    import numpy as np, torch, torch.nn.functional as F
    h, w = x.shape[1:]
    hann = np.hanning(CHIP + 2)[1:-1]
    win = torch.tensor(np.outer(hann, hann) + 1e-3, dtype=torch.float32, device=device)
    ys = sorted(set(range(0, h - CHIP + 1, STRIDE)) | {h - CHIP})
    xs = sorted(set(range(0, w - CHIP + 1, STRIDE)) | {w - CHIP})
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tiles', required=True, help='text file, one tile path per line (the pregate keep list)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--save-avg', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(argv)
    import numpy as np, rasterio, torch
    sys.path.insert(0, str(PIPE))
    import pipeline_common_v3 as pc  # build_model, training, norm
    models = {}
    for name, (arch, md5) in CK.items():
        ck = CK_ROOT / name / 'best_objf1.pt'
        assert hashlib.md5(ck.read_bytes()).hexdigest() == md5, f'md5 mismatch {ck}'
        cfg = json.loads((ck.parent / 'config.json').read_text())['args']
        m = pc.build_model(arch, init=None, allow_mscan_download=False, pretrained_convnext=False,
                           upernet_ch=int(cfg.get('upernet_ch', 512)), drop_path=float(cfg.get('drop_path', 0.3)))
        m.load_state_dict(pc.training.extract_state_dict(torch.load(ck, map_location='cpu', weights_only=True)), strict=True)
        models[name] = m.to(a.device).eval()
    tiles = [l.strip() for l in open(a.tiles) if l.strip()]
    if a.limit: tiles = tiles[:a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'infer_log_v1.csv', 'a')
    t0 = time.time(); n = skipped = 0

    def prob_fn(m, x):
        with torch.inference_mode():
            return sliding_probs(m, torch.from_numpy(x), a.device, pc.norm)

    for t in tiles:
        ts = time.time()
        if process_tile(t, models, out, a.save_avg, prob_fn) == 0:
            skipped += 1; continue
        n += 1
        log.write(f'{t},{time.time() - ts:.3f},{time.strftime("%F %T %Z")}\n'); log.flush()
    el = time.time() - t0
    print(json.dumps(dict(done=n, skipped=skipped, wall_s=round(el, 1), s_per_tile=round(el / max(n, 1), 3))), flush=True)


if __name__ == '__main__':
    main()
