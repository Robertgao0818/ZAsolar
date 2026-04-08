# Worked example: extending the XOR cipher key

This is a complete record of the 2026-04-08 key extension that took
`cipher_key.bin` from 19759 B → **23242 B**, eliminating the
"tile body > key" failure mode that affected ~0.1% of dense urban z=21 tiles
during the Joburg Batch 1 download (97 / 83579 tiles).

## Files in this directory

| File | Origin | Notes |
|------|--------|-------|
| `target_meta.json`                   | Linux  | bbox + tile coords for the chosen target tile |
| `wire_605987_447923_21_v1010.bin`    | Linux  | encrypted wire body fetched via Python (23242 B) |
| `plain_605987_447923_21.jpg`         | GEID CLI on Windows | decrypted JPG (23242 B) |

The XOR of the two files yields the cipher key bytes for positions
`[0, 23242)`. Bytes `[0, 19759)` exactly match the previous
`cipher_key.bin`, which validates that the protocol uses a fixed key.

## How to repeat for a new (bigger) tile

Useful any time a future batch hits a tile larger than the current key
length (rare past 25 KB but possible for very dense imagery).

### 1. Pick a target tile

Find the largest failed tile from any batch's manifests:

```bash
python3 -c "
import json, glob
fails = []
for mf in glob.glob('/mnt/d/ZAsolar/joburg_geid_python/G*/manifest.json'):
    m = json.loads(open(mf).read())
    for x, y in m['failed_xy']:
        fails.append((m['grid_id'], x, y, m.get('primary_imgver', 1010)))
print(f'{len(fails)} failed tiles')
"
# Then probe each to find the biggest wire body — see scripts/imagery/
# repository history for the original probe loop.
```

### 2. Save the encrypted wire body

```python
from geid_reverse_engineering.python.geid_python_batch import quadkey, USER_AGENT
import requests
SID = '<your captured SessionId>'
qk = quadkey(X, Y, 21)
r = requests.get(
    f'https://kh.google.com/flatfile?f1-{qk}-i.{IMGVER}',
    headers={
        'User-Agent': USER_AGENT,
        'Accept-Encoding': 'identity',
        'Cookie': f'$Version="0"; SessionId="{SID}"; State="1"',
    },
    timeout=15,
)
open(f'examples/key_extension/wire_{X}_{Y}_21_v{IMGVER}.bin', 'wb').write(r.content)
```

### 3. Run GEID CLI on Windows (or directly from WSL)

WSL can execute the Windows binary directly — no PowerShell wrapper, no
GUI dialogs, no Frida:

```bash
mkdir -p /mnt/c/Temp/geid_key_ext
"/mnt/c/allmapsoft/geid/downloader.exe" \
  "key_ext_${X}_${Y}" 21 21 $LON_L $LON_R $LAT_T $LAT_B "C:\Temp\geid_key_ext"
```

The bbox should bracket the target tile (a slight outward expansion is
safer than insetting). GEID will write `ges_X_Y_21.jpg` plus a few
neighbors. Copy the target one into this directory:

```bash
cp /mnt/c/Temp/geid_key_ext/key_ext_${X}_${Y}/21/${X}/ges_${X}_${Y}_21.jpg \
   geid_reverse_engineering/examples/key_extension/plain_${X}_${Y}_21.jpg
```

If the WSL-direct path doesn't work in your environment, the
`geid_reverse_engineering/windows/extend_cipher_key.ps1` wrapper does the
same thing from PowerShell.

### 4. Recover and extend

```bash
python3 geid_reverse_engineering/python/extend_cipher_key.py
```

This XORs `wire ⊕ plain`, validates that the recovered bytes match the
existing `cipher_key.bin` in the overlap region (prevents corruption from
a mismatched pair), backs up the old key, and writes the extended one to
`geid_reverse_engineering/artifacts/cipher_key.bin`.

### 5. Retry previously-failed grids

The Python batch downloader skips per-tile if the JPG already exists, so a
re-run only refetches the previously-failed tiles. With the new longer
key, all of them will succeed.

## Validation guarantees

`extend_cipher_key.py` aborts cleanly (no overwrite) if:
- The recovered key disagrees with the existing key in the overlap region
  (would indicate GEID re-encoded the JPG, or the cipher isn't fixed)
- The plain JPG is missing the SOI marker (`FF D8 FF`)
- Wire and plain have different lengths

The current `cipher_key.bin.bak.<timestamp>` files in
`../../artifacts/` are kept as rollback in case anything goes wrong.
