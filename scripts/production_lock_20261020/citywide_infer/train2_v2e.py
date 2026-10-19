"""Fine-tune 3-class PV/SWH segmentation on the Cape Town v2e trainset.

v2e mask: 0 background (including non-collector hard negatives), 1 PV, 2 SWH
(solar water heaters and pool heaters), 255 ignore.

Sampler (the v2d2 code path; the old 75/15/10 header was stale):
  70% PV-centred, 5% SWH-centred, 15% hard-negative-centred, 10% random valid pixel.
Hard negatives enter only through this sampler. There is no hard-negative loss weight.
If the SWH list is empty, the 5% band falls through to hard negatives when that list
is non-empty, otherwise to random. If the hard-negative list is empty, its 15% band
falls through to random. PV and random-valid lists are required.

Common recipe (design v3, section 3), with two explicit differences from that note:
best checkpoint is chosen by object F1, not pixel IoU, and the data root is v2e.
640 crops, flips + rot90 + brightness, 30% degrade to 0.08 m and back,
CE(ignore_index=255, SWH class weight 2.0) + 0.5 soft-Dice on PV (255 excluded),
class-2 logit masked on grids with swh_labelled False, bf16 autocast, gradient
checkpointing, batch 2 x accum 2, 1k warmup times poly power 0.9, grad clip 1.0,
30k iterations, val every 2.5k on 300 fixed val tiles.

Held-out ids must not appear in train or val. The manifest's eval split is 5 of
those ids and is never used for training or for checkpoint selection.
By default a train row whose aerial image is also in val or eval is dropped,
and a val row whose image is also in eval is dropped. --allow_shared_tiles
restores the old shared-tile behaviour.
"""
import argparse, gc, hashlib, json, os, random, re, subprocess, sys, time, warnings
from types import SimpleNamespace
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.windows import Window
from scipy import ndimage
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

HELD = {'G1349', 'G1620', 'G1686', 'G1692', 'G1733', 'G1785', 'G1800', 'G1842', 'G2038', 'G2905'}
DATA = os.path.expanduser('~/data/ZAsolar/geid_temporal/ct_seg_trainset_v2f_20261017/data')
# Li clean-test list: 507 rows, 442 aerial tiles (tile_id like E300300_N6224600@CPT1952).
# Checked 2026-10-17: zero overlap with train/val tiles. Asserted at startup when the file exists.
LI_TILES = os.path.expanduser('~/zasolar_data/capetown_testset_li_z2_20261017/tiles.csv')
# SegNeXt validate redraws NMF bases. This seed is applied inside fork_rng so the draw is
# fixed for checkpoint selection and the training RNG is unchanged.
VAL_RNG_SEED = 12345
DS3M_WEIGHTS = os.path.expanduser('~/capetown_eval/deepsolar3m/weights/segformer.pt')
INFER_PY = os.path.expanduser('~/capetown_eval/deepsolar3m/infer.py')
# Verified 2026-10-17: HTTP 200, Content-Length 322341934. Official mmseg SegNeXt-B
# ImageNet MSCAN-B init (configs/segnext/segnext_mscan-b_... checkpoint_file).
MSCAN_URL = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_b_20230227-3ab7d230.pth'
MSCAN_NAME = 'mscan_b_20230227-3ab7d230.pth'
ARCHS = ('mit_b2', 'mit_b3', 'segnext_b', 'convnext_t')
# ImageNet mean/std applied to pixels that are still on the 0-255 scale.
MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1) * 255
OBJ_MIN_PX = 100  # 0.25 m^2 at 0.05 m GSD (0.05*0.05*100 = 0.25)
S8 = np.ones((3, 3), bool)
B2_DEPTHS = (3, 4, 6, 3)
B3_DEPTHS = (3, 4, 18, 3)
# ConvNeXt-T stage-wise groups follow mmseg get_stage_id_for_convnext.
# The tiny config is decay_type='stage_wise', num_layers=6 (not the Base layer_wise / 12).
# num_layers_for_scale = num_layers + 2 = 8. scale = 0.9 ** (8 - layer_id - 1).
# Head id is 7, so its scale is 1. Ids 5 and 6 are unused.
CONVNEXT_NUM_LAYERS = 6


def loud(msg):
    bar = '!' * 78
    print(f'\n{bar}\nWARNING: {msg}\n{bar}\n', flush=True)


def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_hash():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=d, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            parent = os.path.dirname(d)
            if parent == d:
                return None
            d = parent
    return None


def ds3m_remap():
    """The remap() defined in deepsolar3m/infer.py, executed so the key rewrite stays identical."""
    src = open(INFER_PY).read()
    ns = {'re': re}
    exec(src[src.index('def remap'):src.index('cfg=SegformerConfig')], ns)
    return ns['remap']


def torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False)


def extract_state_dict(blob):
    if isinstance(blob, dict):
        for key in ('state_dict', 'model', 'model_state'):
            inner = blob.get(key)
            if isinstance(inner, dict) and any(torch.is_tensor(v) for v in inner.values()):
                return inner
        if any(torch.is_tensor(v) for v in blob.values()):
            return blob
    raise ValueError('checkpoint has no tensor state_dict')


def atomic_torch_save(obj, path):
    """Write a checkpoint via a temp file and os.replace so a killed process keeps the previous file."""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def aerial_tile_key(path):
    """Identity of an aerial tile: basename cut at the first '@' or '.tif'.

    Train images are named ``E256700_N6214100.tif``. Li ``tile_id`` values are
    ``E300300_N6224600@CPT1952``. Comparing the full strings would never match.
    """
    base = os.path.basename(str(path))
    cut = len(base)
    for sep in ('@', '.tif', '.TIF', '.tiff', '.TIFF'):
        i = base.find(sep)
        if i >= 0:
            cut = min(cut, i)
    return base[:cut]


def assert_li_clean_test(images):
    """Refuse to start if any train/val aerial tile is in the Li clean-test list.

    No-op when ``LI_TILES`` is absent. The overlap checked on 2026-10-17 was 0.
    """
    if not os.path.isfile(LI_TILES):
        print(json.dumps(dict(event='li_tiles', path=LI_TILES, exists=False)), flush=True)
        return
    li = pd.read_csv(LI_TILES)
    if 'tile_id' not in li.columns:
        raise RuntimeError(f'{LI_TILES} has no tile_id column')
    li_keys = {aerial_tile_key(t) for t in li['tile_id']}
    if any('@' in k or k.lower().endswith('.tif') for k in li_keys):
        raise AssertionError('Li tile_id keys were not cut before @ / .tif')
    ours = {aerial_tile_key(p) for p in images}
    hit = sorted(ours & li_keys)
    if hit:
        raise AssertionError(
            f'{len(hit)} train/val tiles are in the Li clean test {LI_TILES}: {hit[:8]}')
    print(json.dumps(dict(
        event='li_tiles', path=LI_TILES, rows=int(len(li)), tiles=len(li_keys),
        train_val_tiles=len(ours), overlap=0)), flush=True)


def strip_prefix(key):
    changed = True
    while changed:
        changed = False
        for pref in ('module.', 'backbone.', 'model.'):
            if key.startswith(pref):
                key = key[len(pref):]
                changed = True
    return key


# --------------------------------------------------------------------------------------
# SegNeXt-B: MSCAN-B + LightHamHead, pure PyTorch (no mmseg).
# Module names follow mmseg's mscan.py so an ImageNet MSCAN-B state dict loads.
# --------------------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, p=0.):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p == 0. or not self.training:
            return x
        keep = 1. - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class MscanMlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class StemConv(nn.Module):
    """Two stride-2 convs: RGB -> stride 4."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch // 2),
            nn.GELU(),
            nn.Conv2d(out_ch // 2, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        return x.flatten(2).transpose(1, 2), h, w


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=3, stride=2):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.norm(self.proj(x))
        _, _, h, w = x.shape
        return x.flatten(2).transpose(1, 2), h, w


class MSCAAttention(nn.Module):
    def __init__(self, channels, kernel_sizes, paddings):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_sizes[0], padding=paddings[0], groups=channels)
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            for i_kernel, i_pad, suffix in (
                (kernel_size, padding, '1'),
                (kernel_size[::-1], padding[::-1], '2'),
            ):
                self.add_module(
                    f'conv{i}_{suffix}',
                    nn.Conv2d(channels, channels, tuple(i_kernel), padding=tuple(i_pad), groups=channels))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = attn + self.conv0_2(self.conv0_1(attn)) + self.conv1_2(self.conv1_1(attn)) + self.conv2_2(self.conv2_1(attn))
        return self.conv3(attn) * u


class MSCASpatialAttention(nn.Module):
    def __init__(self, channels, kernel_sizes, paddings):
        super().__init__()
        self.proj_1 = nn.Conv2d(channels, channels, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = MSCAAttention(channels, kernel_sizes, paddings)
        self.proj_2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_2(self.spatial_gating_unit(self.activation(self.proj_1(x))))
        return x + shortcut


class MSCABlock(nn.Module):
    def __init__(self, channels, mlp_ratio, drop_path, kernel_sizes, paddings, drop=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attn = MSCASpatialAttention(channels, kernel_sizes, paddings)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(channels)
        self.mlp = MscanMlp(channels, int(channels * mlp_ratio), drop=drop)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(channels))
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(channels))
        self.use_ckpt = False

    def _fwd(self, x, h, w):
        b, n, c = x.shape
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        x = x + self.drop_path(self.layer_scale_1[:, None, None] * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2[:, None, None] * self.mlp(self.norm2(x)))
        return x.reshape(b, c, n).permute(0, 2, 1)

    def forward(self, x, h, w):
        if self.use_ckpt and self.training:
            def fn(t):
                return self._fwd(t, h, w)
            return checkpoint(fn, x, use_reentrant=False)
        return self._fwd(x, h, w)


class MSCAN(nn.Module):
    """MSCAN-B. mlp ratios and kernels are the mmseg SegNeXt-B config (inherited from the tiny base)."""
    def __init__(self, embed_dims=(64, 128, 320, 512), mlp_ratios=(8, 8, 4, 4), depths=(3, 3, 12, 3),
                 drop_path_rate=0.1, drop_rate=0.):
        super().__init__()
        kernels = [5, [1, 7], [1, 11], [1, 21]]
        paddings = [2, [0, 3], [0, 5], [0, 10]]
        dpr = [float(v) for v in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.depths = list(depths)
        self.num_stages = 4
        for i in range(4):
            if i == 0:
                patch = StemConv(3, embed_dims[0])
            else:
                patch = OverlapPatchEmbed(embed_dims[i - 1], embed_dims[i], patch_size=3, stride=2)
            blocks = nn.ModuleList([
                MSCABlock(embed_dims[i], mlp_ratios[i], dpr[cur + j], kernels, paddings, drop_rate)
                for j in range(depths[i])
            ])
            cur += depths[i]
            setattr(self, f'patch_embed{i + 1}', patch)
            setattr(self, f'block{i + 1}', blocks)
            setattr(self, f'norm{i + 1}', nn.LayerNorm(embed_dims[i]))

    def set_ckpt(self, enabled):
        for i in range(4):
            for blk in getattr(self, f'block{i + 1}'):
                blk.use_ckpt = enabled
        if enabled:
            # Non-reentrant checkpoint recomputes the forward in train mode, so every
            # BatchNorm applies its momentum twice per optimiser step. Default 0.1
            # twice is about 0.19 (1 - 0.9**2). momentum 0.05 twice is 1 - 0.95**2 = 0.0975.
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = 0.05

    def forward(self, x):
        outs = []
        b = x.shape[0]
        for i in range(4):
            x, h, w = getattr(self, f'patch_embed{i + 1}')(x)
            for blk in getattr(self, f'block{i + 1}'):
                x = blk(x, h, w)
            x = getattr(self, f'norm{i + 1}')(x)
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


class NMF2D(nn.Module):
    """Hamburger NMF. rand_init=True matches the official LightHam config: bases are redrawn every forward.

    The multiplicative updates run under no_grad. Only the last compute_coef keeps a gradient
    (mmseg local_inference is one-step, not backprop through all 6 iterations). The matmuls
    run in fp32 with autocast disabled.
    """
    def __init__(self, md_s=1, md_r=16, train_steps=6, eval_steps=7, rand_init=True):
        super().__init__()
        self.S = md_s
        self.R = md_r
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.rand_init = rand_init

    def forward(self, x):
        b, c, h, w = x.shape
        with torch.autocast(x.device.type, enabled=False):
            x = x.float().reshape(b * self.S, c // self.S, h * w)
            bases = F.normalize(torch.rand(b * self.S, c // self.S, self.R, device=x.device), dim=1)
            with torch.no_grad():
                coef = F.softmax(torch.bmm(x.transpose(1, 2), bases), dim=-1)  # NMF2D sets inv_t = 1
                for _ in range(self.train_steps if self.training else self.eval_steps):
                    coef = coef * torch.bmm(x.transpose(1, 2), bases) / (coef.bmm(bases.transpose(1, 2).bmm(bases)) + 1e-6)
                    bases = bases * torch.bmm(x, coef) / (bases.bmm(coef.transpose(1, 2).bmm(coef)) + 1e-6)
            coef = coef * torch.bmm(x.transpose(1, 2), bases) / (coef.bmm(bases.transpose(1, 2).bmm(bases)) + 1e-6)
            return torch.bmm(bases, coef.transpose(1, 2)).reshape(b, c, h, w)


class ConvGN(nn.Module):
    def __init__(self, cin, cout, norm=True, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, bias=not norm)
        self.norm = nn.GroupNorm(32, cout) if norm else None
        self.act = nn.ReLU(inplace=False) if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class LightHamHead(nn.Module):
    """Official LightHamHead on backbone stages 2/3/4 (strides 8/16/32), channels 512, MD_R 16, GN-32, dropout 0.1.

    Logits are then bilinear-upsampled x2 to stride 4 so every arm hands the loss a 1/4-resolution map.
    That upsample has no parameters. The hamburger itself still runs at stride 8, as in the published head.
    """
    def __init__(self, in_channels=(128, 320, 512), channels=512, ham_channels=512, num_classes=3, dropout=0.1):
        super().__init__()
        self.squeeze = ConvGN(sum(in_channels), ham_channels, norm=True, act=True)
        self.ham_in = nn.Conv2d(ham_channels, ham_channels, 1, bias=True)
        self.ham = NMF2D()
        self.ham_out = ConvGN(ham_channels, ham_channels, norm=True, act=False)
        self.align = ConvGN(ham_channels, channels, norm=True, act=True)
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, feats):
        # feats: stage1..stage4. Official in_index [1, 2, 3].
        xs = feats[1:]
        target = xs[0].shape[-2:]
        xs = [F.interpolate(t, size=target, mode='bilinear', align_corners=False) if t.shape[-2:] != target else t for t in xs]
        x = self.squeeze(torch.cat(xs, dim=1))
        enjoy = F.relu(self.ham_in(x), inplace=False)
        enjoy = self.ham_out(self.ham(enjoy))
        x = F.relu(x + enjoy, inplace=False)
        return self.conv_seg(self.dropout(self.align(x)))


class SegNeXtB(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MSCAN()
        self.head = LightHamHead()

    def enable_checkpointing(self):
        self.backbone.set_ckpt(True)

    def forward(self, pixel_values):
        feats = self.backbone(pixel_values)
        logits = self.head(feats)
        size = (pixel_values.shape[-2] // 4, pixel_values.shape[-1] // 4)
        if logits.shape[-2:] != size:
            logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=False)
        return SimpleNamespace(logits=logits)


def find_mscan_files():
    """Search likely weight locations plus the plocate index. Does not walk Dropbox, Drive, or imagery trees."""
    found = []
    roots = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.expanduser('~/.cache/huggingface'),
        os.path.expanduser('~/.cache/torch'),
        os.path.expanduser('~/capetown_eval'),
        os.path.expanduser('~/projects'),
        os.path.expanduser('~/Downloads'),
        os.path.expanduser('~/weights'),
        os.path.expanduser('~/models'),
    ]
    skip = {'data', 'zasolar_data', '.git', 'node_modules', '__pycache__', 'tiles', 'raw', 'crops'}
    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip]
            for fn in filenames:
                low = fn.lower()
                if low.startswith('mscan_b') and low.endswith(('.pth', '.pt')):
                    path = os.path.join(dirpath, fn)
                    if path not in seen:
                        seen.add(path)
                        found.append(path)
    try:
        out = subprocess.check_output(
            ['locate', '-b', '-i', '-r', r'mscan_b.*\.(pth|pt)$'], stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            if line and os.path.isfile(line) and line not in seen:
                seen.add(line)
                found.append(line)
    except Exception:
        pass
    return found


def download_mscan(dest):
    import urllib.request
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + '.partial'
    print(f'downloading MSCAN-B weights\n  {MSCAN_URL}\n  -> {dest}', flush=True)
    urllib.request.urlretrieve(MSCAN_URL, tmp)
    os.replace(tmp, dest)
    return dest


def resolve_mscan_path(init, allow_download):
    if init:
        if not os.path.isfile(init):
            raise SystemExit(f'--init is not a file: {init}')
        return init, 'init'
    found = find_mscan_files()
    if found:
        print('MSCAN-B weights found:', found[0], flush=True)
        return found[0], 'search'
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', MSCAN_NAME)
    if os.path.isfile(dest):
        return dest, 'cached-download'
    if allow_download:
        return download_mscan(dest), 'download'
    loud('SegNeXt-B MSCAN backbone is RANDOMLY INITIALISED. No mscan_b*.pth was found '
         f'under the searched locations or the locate index. The public file exists ({MSCAN_URL}, '
         'HTTP 200, 322341934 bytes, checked 2026-10-17) and a training run downloads it unless '
         '--no_mscan_download is set. A random backbone is not the ImageNet recipe.')
    return None, 'random'


def load_mscan_weights(backbone, path):
    raw = extract_state_dict(torch_load(path))
    cleaned = {}
    for k, v in raw.items():
        if torch.is_tensor(v):
            cleaned[strip_prefix(k)] = v
    own = backbone.state_dict()
    keep, shape_skip = {}, []
    for k, v in cleaned.items():
        if k in own and own[k].shape == v.shape:
            keep[k] = v
        elif k in own:
            shape_skip.append(k)
    missing, unexpected = backbone.load_state_dict(keep, strict=False)
    n_own = len(own)
    print(json.dumps(dict(
        event='mscan_load', path=path, loaded=len(keep), backbone_tensors=n_own,
        missing=len(missing), shape_skip=len(shape_skip), unused=len(unexpected))), flush=True)
    if len(keep) < n_own * 0.9:
        loud(f'MSCAN load kept only {len(keep)}/{n_own} backbone tensors from {path}. '
             'The backbone is mostly not the ImageNet checkpoint.')
    return len(keep)


# --------------------------------------------------------------------------------------
# ConvNeXt-T + UPerNet
# --------------------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class UPerNetHead(nn.Module):
    """mmseg UPerHead without the auxiliary FCN head. Output is the stride-4 FPN resolution."""
    def __init__(self, in_channels, channels=512, num_classes=3, pool_scales=(1, 2, 3, 6), dropout=0.1):
        super().__init__()
        self.psp = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(s), ConvBNReLU(in_channels[-1], channels, 1))
                                  for s in pool_scales])
        self.psp_bot = ConvBNReLU(in_channels[-1] + len(pool_scales) * channels, channels, 3, 1)
        self.lateral = nn.ModuleList([ConvBNReLU(c, channels, 1) for c in in_channels[:-1]])
        self.fpn = nn.ModuleList([ConvBNReLU(channels, channels, 3, 1) for _ in in_channels[:-1]])
        self.fuse = ConvBNReLU(channels * len(in_channels), channels, 3, 1)
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        self.cls = nn.Conv2d(channels, num_classes, 1)

    def psp_forward(self, x):
        parts = [x]
        for branch in self.psp:
            y = branch(x)
            parts.append(F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False))
        return self.psp_bot(torch.cat(parts, dim=1))

    def forward(self, feats):
        laterals = [lat(feats[i]) for i, lat in enumerate(self.lateral)]
        laterals.append(self.psp_forward(feats[-1]))
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='bilinear', align_corners=False)
        outs = [self.fpn[i](laterals[i]) for i in range(len(laterals) - 1)]
        outs.append(laterals[-1])
        for i in range(1, len(outs)):
            outs[i] = F.interpolate(outs[i], size=outs[0].shape[-2:], mode='bilinear', align_corners=False)
        return self.cls(self.dropout(self.fuse(torch.cat(outs, dim=1))))


class ConvNeXtUPer(nn.Module):
    def __init__(self, upernet_ch=512, drop_path_rate=0.3, pretrained=True):
        super().__init__()
        import timm
        # fb_in1k is ImageNet-1k. The bare name convnext_tiny defaults to in12k_ft_in1k in timm 1.0.25.
        name = 'convnext_tiny.fb_in1k'
        try:
            self.backbone = timm.create_model(name, pretrained=pretrained, features_only=True, drop_path_rate=drop_path_rate)
            self.weight_source = name if pretrained else name + ' (random)'
        except Exception as e:
            loud(f'ConvNeXt-T ImageNet-1k weights failed to load ({e}). Backbone is RANDOM, not ImageNet-1k.')
            self.backbone = timm.create_model('convnext_tiny', pretrained=False, features_only=True, drop_path_rate=drop_path_rate)
            self.weight_source = 'convnext_tiny random'
        chs = [f['num_chs'] for f in self.backbone.feature_info]
        self.head = UPerNetHead(chs, channels=upernet_ch, num_classes=3)

    def enable_checkpointing(self):
        self.backbone.set_grad_checkpointing(True)

    def forward(self, pixel_values):
        feats = self.backbone(pixel_values)
        logits = self.head(feats)
        size = (pixel_values.shape[-2] // 4, pixel_values.shape[-1] // 4)
        if logits.shape[-2:] != size:
            logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=False)
        return SimpleNamespace(logits=logits)


def convnext_layer_id(name, num_layers=CONVNEXT_NUM_LAYERS):
    """timm names mapped like mmseg get_stage_id_for_convnext (stage_wise, num_layers=6).

    stem and every downsample -> 0
    stages_s blocks -> s + 1   (stage 0..3 -> ids 1..4)
    head -> num_layers + 1 (7), which keeps the base learning rate
    scale = llrd ** (num_layers + 2 - id - 1): 0.478 / 0.531 / 0.590 / 0.656 / 0.729 / 1
    """
    max_id = num_layers + 2  # 8
    if name.startswith('head.'):
        return max_id - 1  # 7
    if not name.startswith('backbone.'):
        raise KeyError(name)
    body = name[len('backbone.'):]
    if body.startswith('stem_') or '.downsample.' in body:
        return 0
    if not body.startswith('stages_'):
        raise KeyError(name)
    stage = int(body.split('_', 1)[1].split('.', 1)[0])
    if stage not in (0, 1, 2, 3):
        raise KeyError(name)
    return stage + 1


# --------------------------------------------------------------------------------------
# MiT-b2 / MiT-b3
# --------------------------------------------------------------------------------------

def mit_config(depths, num_labels=3):
    return SegformerConfig(
        num_labels=num_labels, depths=list(depths), hidden_sizes=[64, 128, 320, 512],
        num_attention_heads=[1, 2, 5, 8], decoder_hidden_size=768, sr_ratios=[8, 4, 2, 1])


def load_ds3m_b2_3class():
    """DS3M SegFormer (2 classes) -> 3 classes. Row 2 of the classifier is the PV row with bias - 2."""
    sd = ds3m_remap()(torch_load(DS3M_WEIGHTS))
    w = sd['decode_head.classifier.weight']
    b = sd['decode_head.classifier.bias']
    sd['decode_head.classifier.weight'] = torch.cat([w, w[1:2]])
    sd['decode_head.classifier.bias'] = torch.cat([b, b[1:2] - 2])
    return sd


def expand_b2_to_b3(sd):
    """Copy a 3-class b2 state dict into b3 depths [3, 4, 18, 3].

    Stages 0, 1 and 3 keep a 1:1 block copy (depths 3, 4 and 3).
    Stage 2 is the only stage that grows (6 -> 18). Block j of b3 receives b2 block
    min(j * 6 // 18, 5), which is the triple repeat
    j = 0..17 -> source 0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5.
    Patch embeddings, stage norms and the decoder are copied unchanged.
    """
    out = {}
    for k, v in sd.items():
        if '.blocks.' not in k:
            out[k] = v
    for stage, (n_src, n_dst) in enumerate(zip(B2_DEPTHS, B3_DEPTHS)):
        for j in range(n_dst):
            src = j if n_src == n_dst else min(j * n_src // n_dst, n_src - 1)
            sp = f'segformer.stages.{stage}.blocks.{src}.'
            dp = f'segformer.stages.{stage}.blocks.{j}.'
            copied = 0
            for k, v in sd.items():
                if k.startswith(sp):
                    out[dp + k[len(sp):]] = v
                    copied += 1
            if copied == 0:
                raise RuntimeError(f'no b2 tensors for {sp}')
    return out


def assert_b3_mapping(model):
    sd = model.state_dict()
    w = lambda j: sd[f'segformer.stages.2.blocks.{j}.mlp.fc1.weight']
    assert torch.equal(w(0), w(1)) and torch.equal(w(0), w(2))
    assert not torch.equal(w(0), w(3))
    assert torch.equal(w(15), w(16)) and torch.equal(w(15), w(17))
    bias = sd['decode_head.classifier.bias']
    assert torch.allclose(bias[2], bias[1] - 2)


def build_mit(arch):
    sd = load_ds3m_b2_3class()
    if arch == 'mit_b2':
        model = SegformerForSemanticSegmentation(mit_config(B2_DEPTHS))
        # v2d2 loads with strict=True. Checked: every key matches, including buffers.
        model.load_state_dict(sd, strict=True)
        bias = model.state_dict()['decode_head.classifier.bias']
        assert torch.allclose(bias[2], bias[1] - 2)
        return model
    if arch == 'mit_b3':
        model = SegformerForSemanticSegmentation(mit_config(B3_DEPTHS))
        sd3 = expand_b2_to_b3(sd)
        missing, unexpected = model.load_state_dict(sd3, strict=False)
        missing = [k for k in missing if not k.endswith('num_batches_tracked')]
        if missing or unexpected:
            raise RuntimeError(f'mit_b3 load missing={missing[:12]} n_miss={len(missing)} unexpected={unexpected[:8]}')
        assert_b3_mapping(model)
        return model
    raise ValueError(arch)


def build_model(arch, init=None, allow_mscan_download=False, upernet_ch=512, drop_path=0.3, pretrained_convnext=True):
    if arch in ('mit_b2', 'mit_b3'):
        model = build_mit(arch)
        model.weight_source = DS3M_WEIGHTS
        return model
    if arch == 'segnext_b':
        model = SegNeXtB()
        path, how = resolve_mscan_path(init, allow_mscan_download)
        if path:
            load_mscan_weights(model.backbone, path)
            model.weight_source = path
        else:
            model.weight_source = 'random MSCAN-B'
        model.mscan_init = how
        return model
    if arch == 'convnext_t':
        return ConvNeXtUPer(upernet_ch=upernet_ch, drop_path_rate=drop_path, pretrained=pretrained_convnext)
    raise ValueError(arch)


def enable_ckpt(model):
    if hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable()
            return 'hf'
        except Exception as e:
            print('no hf grad ckpt', e, flush=True)
    if hasattr(model, 'enable_checkpointing'):
        model.enable_checkpointing()
        return 'torch.utils.checkpoint'
    return 'none'


def make_optimizer(model, arch, lr, wd, dec_mult=2.0, head_mult=10.0, llrd=0.9):
    if arch in ('mit_b2', 'mit_b3'):
        enc, dec = [], []
        for n, p in model.named_parameters():
            (enc if n.startswith('segformer.') else dec).append(p)
        opt = torch.optim.AdamW(
            [{'params': enc, 'lr': lr, 'group': 'encoder'},
             {'params': dec, 'lr': lr * dec_mult, 'group': 'head'}],
            weight_decay=wd)
        return opt
    if arch == 'segnext_b':
        # mmseg custom_keys: pos_block decay_mult 0, norm decay_mult 0, head lr_mult 10.
        # mmengine picks the longest matching key, and on a tie the alphabetically earlier one.
        # 'head' and 'norm' have the same length and 'head' wins, so head GroupNorm keeps wd.
        bb_decay, bb_nodecay, hd = [], [], []
        for n, p in model.named_parameters():
            if n.startswith('head.'):
                hd.append(p)
            elif n.startswith('backbone.'):
                if 'norm' in n or 'pos_block' in n:
                    bb_nodecay.append(p)
                else:
                    bb_decay.append(p)
            else:
                raise RuntimeError(f'segnext param outside backbone/head: {n}')
        if len(bb_decay) + len(bb_nodecay) + len(hd) != sum(1 for _ in model.parameters()):
            raise RuntimeError('segnext param split missed a tensor')
        if not bb_nodecay:
            raise RuntimeError('segnext backbone has no norm/pos_block parameters')
        return torch.optim.AdamW([
            {'params': bb_decay, 'lr': lr, 'weight_decay': wd, 'group': 'backbone'},
            {'params': bb_nodecay, 'lr': lr, 'weight_decay': 0.0, 'group': 'backbone_norm'},
            {'params': hd, 'lr': lr * head_mult, 'weight_decay': wd, 'group': 'head'},
        ])
    if arch == 'convnext_t':
        groups = {}
        for n, p in model.named_parameters():
            lid = convnext_layer_id(n)
            decay = not (p.ndim == 1 or n.endswith('.bias'))
            groups.setdefault((lid, decay), []).append(p)
        param_groups = []
        num_layers = CONVNEXT_NUM_LAYERS + 2  # 8; scale = llrd ** (8 - layer_id - 1)
        head_id = CONVNEXT_NUM_LAYERS + 1  # 7
        for (lid, decay), ps in sorted(groups.items()):
            scale = llrd ** (num_layers - lid - 1)
            param_groups.append({
                'params': ps, 'lr': lr * scale, 'weight_decay': wd if decay else 0.0,
                'layer_id': lid, 'group': 'head' if lid == head_id else 'backbone'})
        return torch.optim.AdamW(param_groups)
    raise ValueError(arch)


def head_lr(opt):
    """Learning rate of the head group. Group 0 is the stem/encoder, which is scaled for ConvNeXt."""
    hits = [g['lr'] for g in opt.param_groups if g.get('group') == 'head']
    if not hits:
        raise RuntimeError('optimizer has no head group')
    return hits[0]


# --------------------------------------------------------------------------------------
# Data, sampler, loss, validation
# --------------------------------------------------------------------------------------

def load_bundle(allow_shared_tiles=False):
    sys.path.insert(0, os.path.expanduser('~/capetown_eval/ds3m_ft/v2f_255_r2_20261017'))
    import hn_clean_v2f as hn_clean_v2e  # v2f load_M: identical function, DATA=v2f
    if os.path.abspath(hn_clean_v2e.DATA) != os.path.abspath(DATA):
        raise RuntimeError(f'load_M DATA {hn_clean_v2e.DATA} != {DATA}')
    M, M2 = hn_clean_v2e.load_M()
    M = M.copy()
    M['valid'] = M.n_bg + M.n_pv + M.n_swh
    manifest = f'{DATA}/manifest.csv'
    assert (M.image.iloc[:len(M2)].values == M2.image.values).all()
    assert len(M) == len(pd.read_csv(manifest))
    data_md5 = md5_file(manifest)
    assert data_md5[:8] == 'f182b9c3', data_md5
    cc = os.path.expanduser(f'~/capetown_eval/ds3m_ft/coord_cache_v2f_clean_{data_md5[:8]}.npz')
    assert os.path.basename(cc) == 'coord_cache_v2f_clean_f182b9c3.npz'
    assert md5_file(cc).startswith('49269ac1'), 'v2f HN cache md5'
    assert os.path.isfile(cc), cc
    splits = {}
    for sp in ('train', 'val', 'eval'):
        splits[sp] = set(M.grid[M.split == sp])
    assert len(splits['train']) == 79 and len(splits['val']) == 14 and len(splits['eval']) == 5
    # HELD must not leak into train or val. The eval split is exactly 5 of the 10 HELD ids
    # (G1349, G1686, G1692, G1800, G2038). Requiring HELD ∩ eval = ∅ would reject this manifest.
    # Those eval grids are still held out of training and out of checkpoint selection.
    assert splits['train'].isdisjoint(HELD), sorted(splits['train'] & HELD)
    assert splits['val'].isdisjoint(HELD), sorted(splits['val'] & HELD)
    assert splits['train'].isdisjoint(splits['val'])
    assert splits['train'].isdisjoint(splits['eval'])
    assert splits['val'].isdisjoint(splits['eval'])
    assert splits['eval'] <= HELD, sorted(splits['eval'] - HELD)
    coord = dict(np.load(cc))
    # Adjacent grids share a 100 m tile, and each grid's mask labels the whole tile.
    # A train row whose image is also in val or eval would train on held-out pixels.
    # A val row whose image is also in eval would score those pixels at checkpoint selection.
    train_pool = M[(M.split == 'train') & (M.valid > 100000)]
    val_pool = M[(M.split == 'val') & (M.valid > 100000)]
    held_imgs = set(M.image[M.split.isin(['val', 'eval'])])
    eval_imgs = set(M.image[M.split == 'eval'])
    train_shared = train_pool.image.isin(held_imgs)
    val_shared = val_pool.image.isin(eval_imgs)
    pv_shared = sum(1 for i in train_pool.loc[train_shared].index if f'{i}_pv' in coord)
    if allow_shared_tiles:
        tr = train_pool.copy()
        val = val_pool.copy()
        n_train_drop = n_pv_drop = n_val_drop = 0
    else:
        tr = train_pool.loc[~train_shared].copy()
        val = val_pool.loc[~val_shared].copy()
        n_train_drop = int(train_shared.sum())
        n_pv_drop = int(pv_shared)
        n_val_drop = int(val_shared.sum())
        assert not set(tr.image) & held_imgs
        assert not set(val.image) & eval_imgs
        # Pinned to manifest f182b9c3 (v2f; same as dbf55cb6 per train2_patch_values.json). A miss means the filter changed.
        assert (n_train_drop, n_pv_drop, n_val_drop) == (498, 107, 47), (
            n_train_drop, n_pv_drop, n_val_drop)
    tile_filter = dict(
        allow_shared_tiles=bool(allow_shared_tiles),
        train_rows_dropped=n_train_drop,
        pv_rows_dropped=n_pv_drop,
        val_rows_dropped=n_val_drop,
        train_rows=int(len(tr)),
        val_rows=int(len(val)),
        train_rows_shared=int(train_shared.sum()),
        pv_rows_shared=int(pv_shared),
        val_rows_shared=int(val_shared.sum()),
    )
    print(json.dumps(dict(event='tile_filter', **tile_filter)), flush=True)
    assert_li_clean_test(M.loc[M.split.isin(['train', 'val']), 'image'])
    lists = {
        'pv': [i for i in tr.index if f'{i}_pv' in coord],
        'swh': [i for i in tr.index if f'{i}_swh' in coord],
        'hn': [i for i in tr.index if f'{i}_hn' in coord],
        'random': [i for i in tr.index if f'{i}_val' in coord],
    }
    if not lists['pv'] or not lists['random']:
        raise RuntimeError(f'PV or random-valid sampler list is empty: {[ (k, len(v)) for k, v in lists.items() ]}')
    assert set(tr.split) == {'train'}
    assert set(tr.grid).isdisjoint(HELD)
    assert set(tr.grid).isdisjoint(splits['eval'])
    return dict(M=M, M2=M2, tr=tr, val=val, coord=coord, lists=lists, cache=cc,
                data_md5=data_md5, cache_md5=md5_file(cc), tile_filter=tile_filter,
                splits={k: sorted(v) for k, v in splits.items()})


def pick(rng, lists):
    """One sampler draw. `u` is the first rng uniform, so the branch matches v2d2.

    Fall-through when a list is empty (the `and lists[...]` test fails):
      SWH empty, HN non-empty: the [0.70, 0.75) mass is absorbed by HN (HN becomes 20%).
      SWH empty, HN empty: that mass, and the HN band, become random (random becomes 30%).
      HN empty, SWH non-empty: the [0.75, 0.90) mass becomes random (random becomes 25%).
    """
    u = rng.random()
    if u < 0.70:
        return 'pv', lists['pv'][int(rng.integers(len(lists['pv'])))]
    if u < 0.75 and lists['swh']:
        return 'swh', lists['swh'][int(rng.integers(len(lists['swh'])))]
    if u < 0.90 and lists['hn']:
        return 'hn', lists['hn'][int(rng.integers(len(lists['hn'])))]
    return 'random', lists['random'][int(rng.integers(len(lists['random'])))]


class ChipDS(Dataset):
    def __init__(self, bundle, chip, degrade):
        self.tr = bundle['tr']
        self.coord = bundle['coord']
        self.lists = bundle['lists']
        self.chip = chip
        self.degrade = degrade

    def __len__(self):
        return 10 ** 12

    def __getitem__(self, k):
        rng = np.random.default_rng(int(k))
        kind, i = pick(rng, self.lists)
        key = f'{i}_pv' if kind == 'pv' else f'{i}_swh' if kind == 'swh' else f'{i}_hn' if kind == 'hn' else f'{i}_val'
        r = self.tr.loc[i]
        c = self.chip
        w, h = int(r.width), int(r.height)
        cy, cx = self.coord[key][int(rng.integers(len(self.coord[key])))]
        y0 = int(np.clip(cy - rng.integers(64, c - 64), 0, h - c))
        x0 = int(np.clip(cx - rng.integers(64, c - 64), 0, w - c))
        win = Window(x0, y0, c, c)
        with rasterio.open(r.image) as f:
            img = f.read([1, 2, 3], window=win, boundless=True, fill_value=0)
        with rasterio.open(r['mask']) as f:
            m = f.read(1, window=win, boundless=True, fill_value=255)
        img = img.transpose(1, 2, 0).astype(np.float32)
        m = m.copy()
        m[img.sum(2) == 0] = 255
        if rng.random() < 0.5:
            img = img[:, ::-1]
            m = m[:, ::-1]
        if rng.random() < 0.5:
            img = img[::-1]
            m = m[::-1]
        kk = int(rng.integers(4))
        img = np.rot90(img, kk)
        m = np.rot90(m, kk)
        img = np.clip(img * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15), 0, 255)
        return (torch.from_numpy(np.ascontiguousarray(img)),
                torch.from_numpy(np.ascontiguousarray(m)),
                bool(rng.random() < self.degrade),
                bool(r.swh_labelled))


def norm(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def prep(img, deg):
    x = img.permute(0, 3, 1, 2).cuda(non_blocking=True)
    if deg.any():
        d = deg.bool().cuda()
        c = x.shape[-1]
        s = int(round(c * 0.05 / 0.08))
        x[d] = F.interpolate(F.interpolate(x[d], size=s, mode='area'), size=c, mode='bilinear', align_corners=False)
    return norm(x)


def loss_fn(lo, m, sl, dice_w, swh_w):
    lo = F.interpolate(lo, size=m.shape[-2:], mode='bilinear', align_corners=False).float()
    lo = lo.clone()
    lo[~sl, 2] = -1e4
    cw = torch.tensor([1.0, 1.0, swh_w], device=lo.device)
    ce = F.cross_entropy(lo, m.long(), weight=cw, ignore_index=255)
    v = (m != 255).float()
    pp = lo.softmax(1)[:, 1]
    g = (m == 1).float()
    dice = 1 - (2 * (pp * g * v).sum() + 1) / ((pp * v).sum() + (g * v).sum() + 1)
    return ce + dice_w * dice, ce.detach(), dice.detach()


def dil(x, d):
    return F.max_pool2d(x.float()[None, None], 2 * d + 1, 1, d)[0, 0] > 0


def ero(x, d):
    return ~dil(~x, d)


def object_counts(pred, gt, inv=None):
    """Tile-level object scores using the evalobj hit rules.

    GT component is a hit when prediction covers >= 50% of its pixels (recall).
    A predicted 8-connected component is a TP when >= 20% of its pixels lie in
    GT dilated by 8 (evalobj's ndimage.binary_dilation iterations=8, a Manhattan
    ball, not the square max-pool used for boundary IoU).
    Ground-truth components are not size-filtered (evalobj only requires the GT object
    to be seen). Predicted components under 100 px (0.25 m^2) are ignored.
    A predicted component with >= 50% of its pixels in `inv` is dropped, matching
    evalobj's ninv/npx < 0.5 filter. `inv` is the ignore band farther than 8 px
    from GT. Pass inv=None to disable that filter.

    These components are per tile. evalobj.py instead uses polygon objects and
    merges predicted components across tile edges. The hit rules are the same;
    the objects are not.
    """
    gt_lab, ng = ndimage.label(gt.astype(bool), structure=S8)
    pr_lab, np_ = ndimage.label(pred.astype(bool), structure=S8)
    zone = ndimage.binary_dilation(gt.astype(bool), iterations=8)
    hits = n_gt = tps = n_pr = 0
    if ng:
        idx = np.arange(1, ng + 1)
        area = ndimage.sum(np.ones(gt.shape, np.int32), gt_lab, idx)
        cov = ndimage.sum(pred.astype(bool), gt_lab, idx)
        n_gt = int(ng)
        hits = int((cov >= 0.5 * area).sum())
    if np_:
        idx = np.arange(1, np_ + 1)
        area = ndimage.sum(np.ones(pred.shape, np.int32), pr_lab, idx)
        zin = ndimage.sum(zone, pr_lab, idx)
        keep = area >= OBJ_MIN_PX
        if inv is not None:
            ninv = ndimage.sum(inv.astype(bool), pr_lab, idx)
            keep = keep & (ninv < 0.5 * area)
        n_pr = int(keep.sum())
        tps = int(((zin >= 0.2 * area) & keep).sum())
    return hits, n_gt, tps, n_pr


def self_test_objects():
    canvas = np.zeros((220, 220), bool)
    gt = canvas.copy()
    gt[20:60, 20:60] = True
    pred = gt.copy()
    h, ng, tp, npr = object_counts(pred, gt, np.zeros_like(gt))
    assert (h, ng, tp, npr) == (1, 1, 1, 1), (h, ng, tp, npr)
    pred2 = np.zeros_like(gt)
    h, ng, tp, npr = object_counts(pred2, gt, np.zeros_like(gt))
    assert (h, ng, tp, npr) == (0, 1, 0, 0)
    # exactly 50% of the GT square covered -> hit; one row less -> miss
    half = np.zeros_like(gt)
    half[20:40, 20:60] = True
    h, ng, _, _ = object_counts(half, gt, None)
    assert h == 1 and ng == 1
    almost = np.zeros_like(gt)
    almost[20:39, 20:60] = True
    h, _, _, _ = object_counts(almost, gt, None)
    assert h == 0
    # pred just outside the square: 8 rows fall inside the Manhattan dilation -> 320/1600 = 0.20 -> TP
    edge = np.zeros_like(gt)
    edge[60:100, 20:60] = True
    _, _, tp, npr = object_counts(edge, gt, np.zeros_like(gt))
    assert tp == 1 and npr == 1, (tp, npr)
    far = np.zeros_like(gt)
    far[120:160, 120:160] = True
    _, ng, tp, npr = object_counts(far, gt, np.zeros_like(gt))
    assert ng == 1 and tp == 0 and npr == 1
    # GT is not size-filtered. A prediction under 100 px is ignored; 100 px is kept.
    tiny = np.zeros_like(gt)
    tiny[10:19, 10:21] = True  # 9*11 = 99
    h, ng, tp, npr = object_counts(tiny, tiny, None)
    assert (h, ng, tp, npr) == (1, 1, 0, 0), (h, ng, tp, npr)
    wide = np.zeros_like(gt)
    wide[10:20, 10:20] = True  # 10*10 = 100
    h, ng, tp, npr = object_counts(wide, wide, None)
    assert (h, ng, tp, npr) == (1, 1, 1, 1), (h, ng, tp, npr)
    # a predicted component that is entirely far-ignore is not counted
    inv = np.ones_like(gt)
    inv[20:80, 20:80] = False
    _, _, tp, npr = object_counts(far, gt, inv)
    assert tp == 0 and npr == 0
    return 'ok'


def _validate_body(model, vsel, chip):
    c, s = chip, 480
    w1 = np.hanning(c + 2)[1:-1]
    win = torch.tensor(np.outer(w1, w1) + 1e-3).float().cuda()
    ob = [(0, 4, 'lt4'), (4, 10, '4_10'), (10, 25, '10_25'), (25, 1e9, 'gt25')]
    tp = fp = fn = 0
    bi = bu = 0
    on = {k: 0 for *_, k in ob}
    oh = dict(on)
    s_tp = s_fp = s_fn = 0
    swfp_pv = 0
    sw_n = 0
    negfp = 0
    neg_area = 0
    hits = n_gt = tps = n_pr = 0
    for _, r in vsel.iterrows():
        with rasterio.open(r.image) as f:
            x = torch.from_numpy(f.read([1, 2, 3])).float().cuda()
        m = torch.from_numpy(rasterio.open(r['mask']).read(1)).cuda()
        h, w = m.shape
        p = torch.zeros(3, h, w, device='cuda')
        wt = torch.zeros(h, w, device='cuda')
        ys = sorted(set(list(range(0, h - c + 1, s)) + [h - c]))
        xs = sorted(set(list(range(0, w - c + 1, s)) + [w - c]))
        pos = [(y, xx) for y in ys for xx in xs]
        for i in range(0, len(pos), 4):
            bb = pos[i:i + 4]
            ch = norm(torch.stack([x[:, y:y + c, xx:xx + c] for y, xx in bb]))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lo = model(pixel_values=ch).logits
            pr = F.interpolate(lo.float(), size=c, mode='bilinear', align_corners=False).softmax(1)
            for k, (y, xx) in enumerate(bb):
                p[:, y:y + c, xx:xx + c] += pr[k] * win
                wt[y:y + c, xx:xx + c] += win
        p = p / wt.clamp(min=1e-6)
        cls = p.argmax(0)
        pred = cls == 1
        # Pixel metrics stay on argmax (v2d2). Object counts use P[1] >= 0.5 (evalobj).
        pred_obj = p[1] >= 0.5
        v = m != 255
        g = m == 1
        region = dil(v, 8) & (x.sum(0) > 0)
        tp += (pred & g & v).sum().item()
        fp += (pred & ~g & v & (m != 1)).sum().item()
        fn += (~pred & g & v).sum().item()
        if r.n_pv == 0:
            negfp += (pred & v).sum().item()
            neg_area += v.sum().item()
        gfull = g | (dil(g, 6) & (m == 255) & region)
        pr_ = pred & region
        bp = pr_ & ~ero(pr_, 10)
        bg_ = gfull & ~ero(gfull, 10)
        bi += (bp & bg_ & region).sum().item()
        bu += ((bp | bg_) & region).sum().item()
        gfull_np = gfull.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        pred_obj_np = pred_obj.detach().cpu().numpy()
        gc, gn = ndimage.label(gfull_np)
        if gn:
            ar = ndimage.sum(np.ones_like(gc), gc, range(1, gn + 1)) / 400.
            cv = ndimage.sum(pred_np, gc, range(1, gn + 1)) / (ar * 400.)
            for lo_, hi_, nm in ob:
                sel = (ar >= lo_) & (ar < hi_)
                on[nm] += int(sel.sum())
                oh[nm] += int((sel & (cv >= 0.5)).sum())
        inv = ((m == 255) & ~torch.from_numpy(ndimage.binary_dilation(gfull_np, iterations=8)).to(m.device)).detach().cpu().numpy()
        hh, ngt, tt, npred = object_counts(pred_obj_np, gfull_np, inv)
        hits += hh
        n_gt += ngt
        tps += tt
        n_pr += npred
        if r.swh_labelled:
            sg = m == 2
            sp = (cls == 2) & v
            s_tp += (sp & sg).sum().item()
            s_fp += (sp & ~sg).sum().item()
            s_fn += (~sp & sg).sum().item()
            sw_n += sg.sum().item()
            swfp_pv += (pred & sg).sum().item()
    obj_rec = hits / max(n_gt, 1)
    obj_prec = tps / max(n_pr, 1)
    obj_f1 = 2 * obj_prec * obj_rec / max(obj_prec + obj_rec, 1e-9) if (n_gt and n_pr) else 0.0
    return dict(
        iou=tp / max(tp + fp + fn, 1), prec=tp / max(tp + fp, 1), rec=tp / max(tp + fn, 1),
        area_bias=(tp + fp) / max(tp + fn, 1) - 1, biou=bi / max(bu, 1),
        **{f'rec_{k}': oh[k] / max(on[k], 1) for k in on}, **{f'n_{k}': on[k] for k in on},
        obj_prec=obj_prec, obj_rec=obj_rec, obj_f1=obj_f1, n_obj_gt=n_gt, n_obj_pred=n_pr,
        swh_iou=s_tp / max(s_tp + s_fp + s_fn, 1), swh_as_pv_frac=swfp_pv / max(sw_n, 1),
        neg_fp_m2_per_ha=negfp / 400 / max(neg_area / 400 / 1e4, 1e-9))


@torch.no_grad()
def validate(model, vsel, chip):
    """Score val tiles. SegNeXt NMF bases are redrawn under a fixed seed.

    fork_rng saves the training RNG (CPU and the current CUDA device) and restores
    it when the block exits, so validation does not consume training randomness.
    """
    model.eval()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(VAL_RNG_SEED)
        out = _validate_body(model, vsel, chip)
    model.train()
    return out


def sampler_report(lists, n, seed):
    start = seed * 10 ** 7
    counts = {k: 0 for k in ('pv', 'swh', 'hn', 'random')}
    for k in range(start, start + n):
        kind, _ = pick(np.random.default_rng(k), lists)
        counts[kind] += 1
    fr = {k: counts[k] / n for k in counts}
    return counts, fr


def parse_args(argv=None):
    smoke = '--smoke' in (argv if argv is not None else sys.argv)
    p = argparse.ArgumentParser(description='v2e 3-class segmentation fine-tune')
    p.add_argument('--arch', choices=ARCHS, default='mit_b2')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--iters', type=int, default=200 if smoke else 30000)
    p.add_argument('--bs', type=int, default=2)
    p.add_argument('--accum', type=int, default=2)
    p.add_argument('--chip', type=int, default=640)
    p.add_argument('--val_every', type=int, default=200 if smoke else 2500)
    p.add_argument('--val_tiles', type=int, default=5 if smoke else 300)
    p.add_argument('--max_val_tiles', type=int, default=None)
    p.add_argument('--workers', type=int, default=2 if smoke else 5)
    p.add_argument('--lr', type=float, default=None, help='default 1e-4 for convnext_t, else 6e-5')
    p.add_argument('--wd', type=float, default=None, help='default 0.05 for convnext_t, else 0.01')
    p.add_argument('--dec_mult', type=float, default=2.0)
    p.add_argument('--head_mult', type=float, default=10.0)
    p.add_argument('--llrd', type=float, default=0.9)
    p.add_argument('--dice_w', type=float, default=0.5)
    p.add_argument('--swh_w', type=float, default=2.0)
    p.add_argument('--degrade', type=float, default=0.3)
    p.add_argument('--drop_path', type=float, default=0.3)
    p.add_argument('--upernet_ch', type=int, default=512)
    p.add_argument('--val0', type=int, default=1)
    p.add_argument('--out', default=None)
    p.add_argument('--init', default=None, help='MSCAN-B .pth for segnext_b')
    p.add_argument('--no_mscan_download', action='store_true')
    p.add_argument('--allow_shared_tiles', action='store_true',
                   help='keep train rows whose image is also in val or eval, and val rows whose image is also in eval')
    p.add_argument('--smoke', action='store_true', help='defaults: iters 200, val_every 200, val_tiles 5, workers 2')
    p.add_argument('--cpu_check', action='store_true')
    a = p.parse_args(argv)
    if a.lr is None:
        a.lr = 1e-4 if a.arch == 'convnext_t' else 6e-5
    if a.wd is None:
        a.wd = 0.05 if a.arch == 'convnext_t' else 0.01
    if a.out is None:
        a.out = os.path.expanduser(f'~/capetown_eval/ds3m_ft/run_v2f_{a.arch}_s{a.seed}')
    return a


def n_val(a, n_available):
    n = a.val_tiles
    if a.max_val_tiles is not None:
        n = min(n, a.max_val_tiles)
    n = min(n, n_available)
    if n < 1:
        raise SystemExit(f'no val tiles selected (val_tiles={a.val_tiles}, max_val_tiles={a.max_val_tiles})')
    return n


def guard_out(out):
    """Refuse to start into a directory that already has a run but no resume.pt.

    log.jsonl counts. A crash before the first checkpoint still leaves a log, and
    appending to that log would mix two runs.
    """
    markers = [f'{out}/{name}' for name in ('best.pt', 'best_objf1.pt', 'last.pt', 'log.jsonl')]
    present = [p for p in markers if os.path.exists(p)]
    resume = os.path.exists(f'{out}/resume.pt')
    if present and not resume:
        raise SystemExit(
            f'refusing to start: {out} already has { [os.path.basename(p) for p in present] } and no resume.pt. '
            'This would clobber another run. Use a different --out, or put resume.pt back to resume.')
    return 'resume' if resume else 'fresh'


def check_resume(s, a):
    """Refuse a resume.pt that omits arch/seed or disagrees on bs, accum, or chip.

    The sampler start index is seed*10**7 + step*bs*accum. A different batch or
    accumulation would train on the wrong crops. chip changes the crop itself.
    """
    required = ('arch', 'seed', 'bs', 'accum', 'chip')
    missing = [k for k in required if k not in s]
    if missing:
        raise SystemExit(
            f'resume.pt is missing {missing}. Refusing to resume. '
            'arch and seed must be stored, and bs, accum, and chip must match this command.')
    if s['arch'] != a.arch or int(s['seed']) != int(a.seed):
        raise SystemExit(
            f"resume.pt is arch={s['arch']!r} seed={s['seed']!r}, not {a.arch} seed {a.seed}")
    got = {k: int(s[k]) for k in ('bs', 'accum', 'chip')}
    exp = {k: int(getattr(a, k)) for k in ('bs', 'accum', 'chip')}
    if 'iters' in s and int(s['iters']) != int(a.iters):
        raise SystemExit(f"resume.pt was saved with iters={s['iters']}, command has --iters {a.iters}; poly LR would jump")
    if got != exp:
        raise SystemExit(
            f"resume.pt was saved with bs={got['bs']} accum={got['accum']} chip={got['chip']}, "
            f"not bs={exp['bs']} accum={exp['accum']} chip={exp['chip']}. "
            'The sampler index is seed*10**7 + step*bs*accum, so a different batch or accumulation '
            'would train on the wrong crops.')


def write_config(a, bundle, model, filename='config.json'):
    cfg = dict(
        args=vars(a), arch=a.arch, seed=a.seed, data=DATA, data_md5=bundle['data_md5'],
        data_md5_8=bundle['data_md5'][:8], cache=bundle['cache'], cache_md5=bundle['cache_md5'],
        script=os.path.abspath(__file__), script_md5=md5_file(__file__), git_hash=git_hash(),
        held=sorted(HELD), splits=bundle['splits'], tile_filter=bundle.get('tile_filter'),
        sampler=dict(pv=0.70, swh=0.05, hn=0.15, random=0.10,
                     fallthrough='empty SWH -> HN if HN else random; empty HN -> random',
                     lists={k: len(v) for k, v in bundle['lists'].items()}),
        b3_stage2_block_map='j -> min(j*6//18, 5) = 0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5',
        weight_source=getattr(model, 'weight_source', None),
        mscan_init=getattr(model, 'mscan_init', None),
        cudnn_deterministic=False,
        versions=dict(torch=torch.__version__, numpy=np.__version__))
    try:
        import timm
        import transformers
        cfg['versions']['timm'] = timm.__version__
        cfg['versions']['transformers'] = transformers.__version__
    except Exception:
        pass
    path = os.path.join(a.out, filename)
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=1)
    print(json.dumps(dict(event='config', path=path)), flush=True)
    return cfg


def train(a):
    mode = guard_out(a.out)
    if a.arch == 'convnext_t' and a.bs < 2:
        raise SystemExit('convnext_t UPerNet uses BatchNorm on the 1x1 pyramid bin, so training needs --bs >= 2.')
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    print(json.dumps(dict(event='seed', seed=a.seed, arch=a.arch, out=a.out,
                          cudnn_deterministic=False, mode=mode,
                          allow_shared_tiles=a.allow_shared_tiles)), flush=True)
    bundle = load_bundle(a.allow_shared_tiles)
    print('tiles', {k: len(v) for k, v in bundle['lists'].items()}, 'cache', os.path.basename(bundle['cache']), flush=True)
    model = build_model(a.arch, init=a.init, allow_mscan_download=not a.no_mscan_download,
                        upernet_ch=a.upernet_ch, drop_path=a.drop_path, pretrained_convnext=True)
    model.cuda()
    ckpt_kind = enable_ckpt(model)
    print('grad checkpoint', ckpt_kind, flush=True)
    opt = make_optimizer(model, a.arch, a.lr, a.wd, a.dec_mult, a.head_mult, a.llrd)
    # Base lrs are taken before load_state_dict, so a resumed optimiser does not
    # freeze the learning rate at whatever value the last step had stored.
    base = [g['lr'] for g in opt.param_groups]
    val = bundle['val']
    assert set(val.split) == {'val'}
    assert set(val.grid).isdisjoint(HELD)
    assert set(val.grid).isdisjoint(set(bundle['splits']['eval']))
    if not a.allow_shared_tiles:
        eval_imgs = set(bundle['M'].image[bundle['M'].split == 'eval'])
        assert not set(val.image) & eval_imgs
        assert not set(bundle['tr'].image) & set(bundle['M'].image[bundle['M'].split.isin(['val', 'eval'])])
    # smoke: pick val tiles that contain PV so the 5-tile smoke val is informative
    # (random 5 of 1703 filtered val tiles had none). Full runs keep the natural-ratio sample.
    vpool = val[val.n_pv > 0] if a.smoke else val
    vsel = vpool.sample(n_val(a, len(vpool)), random_state=1)
    ck = f'{a.out}/resume.pt'
    step0, best = 0, -1.0
    if mode == 'resume':
        s = torch.load(ck, map_location='cpu', weights_only=False)
        check_resume(s, a)
        model.load_state_dict(s['model'])
        opt.load_state_dict(s['opt'])
        step0, best = s['step'], s['best']
        print('resumed', step0, 'best_obj_f1', best, flush=True)
        # Do not overwrite the fresh-run config.json.
        write_config(a, bundle, model, f'config_resume_{step0}.json')
    else:
        write_config(a, bundle, model, 'config.json')
    start = a.seed * 10 ** 7 + step0 * a.bs * a.accum
    loader_kw = dict(batch_size=a.bs, num_workers=a.workers, sampler=range(start, start + 10 ** 8))
    if a.workers > 0:
        loader_kw.update(pin_memory=True, persistent_workers=True, prefetch_factor=4, multiprocessing_context='fork')
    it = iter(DataLoader(ChipDS(bundle, a.chip, a.degrade), **loader_kw))
    log = open(f'{a.out}/log.jsonl', 'a')
    run = []
    if step0 == 0 and a.val0:
        v = validate(model, vsel, a.chip)
        v['step'] = 0
        print('VAL', json.dumps(v), flush=True)
        log.write(json.dumps(dict(val=v)) + '\n')
        log.flush()
    # s_per_it is training time. val0 is already finished; later val blocks are subtracted.
    t0 = time.time()
    val_seconds = 0.0
    model.train()
    for step in range(step0 + 1, a.iters + 1):
        f = min(1, step / 1000) * (1 - step / a.iters) ** 0.9
        for gq, b0 in zip(opt.param_groups, base):
            gq['lr'] = b0 * f
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            img, m, deg, sl = next(it)
            x = prep(img, deg)
            m = m.cuda(non_blocking=True)
            sl = sl.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lo = model(pixel_values=x).logits
            loss, ce, di = loss_fn(lo, m, sl, a.dice_w, a.swh_w)
            (loss / a.accum).backward()
            run.append((loss.item(), ce.item(), di.item()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            r = np.mean(run, 0)
            run = []
            elapsed = time.time() - t0 - val_seconds
            done = step - step0
            rec = dict(step=step, loss=float(r[0]), ce=float(r[1]), dice=float(r[2]),
                       lr=opt.param_groups[0]['lr'], lr_head=head_lr(opt),
                       s_per_it=elapsed / done, it_per_s=done / elapsed,
                       mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                       peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + '\n')
            log.flush()
        if step % a.val_every == 0 or step == a.iters:
            tv = time.time()
            v = validate(model, vsel, a.chip)
            v['step'] = step
            print('VAL', json.dumps(v), flush=True)
            log.write(json.dumps(dict(val=v)) + '\n')
            log.flush()
            atomic_torch_save(model.state_dict(), f'{a.out}/last.pt')
            if v['obj_f1'] > best:
                best = v['obj_f1']
                atomic_torch_save(model.state_dict(), f'{a.out}/best_objf1.pt')
            atomic_torch_save(dict(
                model=model.state_dict(), opt=opt.state_dict(), step=step, best=best,
                arch=a.arch, seed=a.seed, bs=a.bs, accum=a.accum, chip=a.chip, iters=a.iters), ck)
            val_seconds += time.time() - tv
    log.close()


def cpu_check(a):
    if torch.cuda.is_available():
        raise SystemExit('cpu_check refuses to run because CUDA is visible. Launch with CUDA_VISIBLE_DEVICES="".')
    print(json.dumps(dict(event='cpu_check', cuda=False, seed=a.seed)), flush=True)
    print('object self-test', self_test_objects(), flush=True)
    bundle = load_bundle(a.allow_shared_tiles)
    counts, fr = sampler_report(bundle['lists'], 2000, a.seed)
    report = dict(
        event='sampler', n=2000, seed=a.seed, start_index=a.seed * 10 ** 7,
        counts=counts, fractions=fr,
        lists={k: len(v) for k, v in bundle['lists'].items()},
        empty_swh=len(bundle['lists']['swh']) == 0, empty_hn=len(bundle['lists']['hn']) == 0,
        target=dict(pv=0.70, swh=0.05, hn=0.15, random=0.10))
    print(json.dumps(report), flush=True)
    assert abs(sum(fr.values()) - 1) < 1e-9
    x = torch.randn(1, 3, 640, 640)
    summaries = []
    for arch in ARCHS:
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        t1 = time.time()
        model = build_model(
            arch, init=a.init, allow_mscan_download=False,
            upernet_ch=a.upernet_ch, drop_path=a.drop_path,
            pretrained_convnext=(arch == 'convnext_t'))
        model.eval()
        with torch.no_grad():
            y = model(pixel_values=x).logits
        nparam = sum(p.numel() for p in model.parameters())
        shape = tuple(y.shape)
        print(json.dumps(dict(event='forward', arch=arch, params=nparam, logits=list(shape),
                              weight_source=str(getattr(model, 'weight_source', '')),
                              mscan_init=getattr(model, 'mscan_init', None))), flush=True)
        if shape != (1, 3, 160, 160):
            raise RuntimeError(f'{arch} logits {shape}, expected (1, 3, 160, 160)')
        # CPU backward at 64^2 with checkpointing. Batch 2 matches training: UPerNet's 1x1
        # pyramid bin is BatchNorm, which refuses a single value per channel in train mode.
        enable_ckpt(model)
        model.train()
        small = torch.randn(2, 3, 64, 64)
        out = model(pixel_values=small).logits
        out.sum().backward()
        if arch == 'convnext_t':
            ids = sorted({convnext_layer_id(n) for n, p in model.named_parameters()})
            opt = make_optimizer(model, arch, 1e-4, 0.05, llrd=0.9)
            scales = sorted({round(g['lr'] / 1e-4, 6) for g in opt.param_groups})
            expect_ids = [0, 1, 2, 3, 4, 7]
            expect_scales = [round(0.9 ** (8 - i - 1), 6) for i in expect_ids]
            if ids != expect_ids or scales != expect_scales:
                raise RuntimeError(
                    f'convnext stage_wise ids {ids} scales {scales}, expected {expect_ids} {expect_scales}')
        else:
            ids, scales = None, None
            opt = make_optimizer(model, arch, 6e-5, 0.01)
        if arch == 'segnext_b':
            wd_of = {id(p): g['weight_decay'] for g in opt.param_groups for p in g['params']}
            for n, p in model.named_parameters():
                if n.startswith('backbone.') and ('norm' in n or 'pos_block' in n):
                    if wd_of[id(p)] != 0.0:
                        raise RuntimeError(f'{n} weight decay {wd_of[id(p)]}, expected 0')
                elif n.startswith('head.'):
                    if wd_of[id(p)] != 0.01:
                        raise RuntimeError(f'{n} weight decay {wd_of[id(p)]}, expected 0.01')
            bn_m = sorted({round(m.momentum, 4) for m in model.modules() if isinstance(m, nn.BatchNorm2d)})
            if bn_m != [0.05]:
                raise RuntimeError(f'MSCAN BN momentum after checkpointing {bn_m}, expected [0.05]')
        rec = dict(arch=arch, params=nparam, logits=list(shape), ckpt_backward=list(out.shape),
                   seconds=round(time.time() - t1, 2), weight_source=str(getattr(model, 'weight_source', '')),
                   mscan_init=getattr(model, 'mscan_init', None),
                   opt_groups=len(opt.param_groups), lrs=sorted({round(g['lr'], 8) for g in opt.param_groups}),
                   convnext_layer_ids=ids, convnext_lr_scales=scales)
        print(json.dumps(rec), flush=True)
        summaries.append(rec)
        del model, opt, out, y
        gc.collect()
    return dict(sampler=report, models=summaries, bundle_lists={k: len(v) for k, v in bundle['lists'].items()},
                cache_md5=bundle['cache_md5'], data_md5=bundle['data_md5'])


def main():
    a = parse_args()
    if a.cpu_check:
        cpu_check(a)
        return
    train(a)


if __name__ == '__main__':
    main()
