# GEID 6.48 Protocol — Technical Reference

Captured 2026-04-08 against Allmapsoft Google Earth Images Downloader 6.48
using Frida 17 SSL_read/write hooks on `ssleay32.dll`.

## HTTP stack inside `downloader.exe`

Despite shipping `libcurl.dll`, the GEID downloader does **not** use
libcurl for tile fetches. The actual networking stack is:

```
Delphi VCL → Indy components → OpenSSL 1.0
                                    ↓
                            ssleay32.dll  (TLS framing, SSL_write/SSL_read)
                            libeay32.dll  (crypto primitives)
```

Frida hooks must target `ssleay32!SSL_write` (and the matching `SSL_read`),
not `libcurl_easy_setopt` — that returned zero events when first attempted
(`hook_geid_curl.js` is the dead-end).

## Endpoint

```
https://kh.google.com
```

This is the legacy Google Earth Pro tile API. Still alive for backwards
compatibility with the GE Pro client even though both that client and the
broader Google Earth Pro program have been EOL'd. Modern Google Maps APIs
use entirely different URL schemes.

## Required HTTP headers

```
User-Agent: GoogleEarth/7.3.6.9345(Windows;Microsoft Windows (6.2.9200.0);en;kml:2.2;client:Pro;type:default)
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8
Accept-Encoding: identity
Content-Type: application/octet-stream
Cookie: $Version="0"; SessionId="<104-byte ASCII session id>"; State="1"
```

Any other User-Agent (browser, plain `python-requests`, etc.) returns 403.
The `Accept-Encoding: identity` is important — gzip-encoded responses
break the XOR decryption (the encoded bytes don't match the cipher key
positions).

## Request sequence to download N tiles at zoom z

```
1. GET  /dbRoot.v5?hl=en-GB&gl=US                       (DB root metadata, defines layer version IDs)
2. POST /geauth                                         (49-byte binary body → returns SessionId cookie)
3. GET  /flatfile?q2-{16-char-quadkey}-q.{quadVer}      ⌈N/16⌉ × quadtree metadata at z=16
4. GET  /flatfile?f1-{z-char-quadkey}-i.{imgVer}        N × actual JPEG image tiles
```

All requests after step 2 carry the `Cookie` header with the SessionId.

For the **current Google Earth Satellite layer** (`MapsType=59`):
- `quadVer = 1033` (q2 quadtree metadata version)
- `imgVer  = 1010` (f1 image tile baseline; varies per area — see below)

These come from the `dbRoot.v5` response and may rotate over time.

## Quadkey encoding

The (x, y, z) → quadkey conversion uses **Google Earth Pro's digit map**,
which is **not** the standard Bing/Microsoft layout. Empirically verified
by brute-force matching against 16 captured tile URLs from G0772:

```python
GE_DIGIT_MAP = (0, 3, 1, 2)   # NOT (0, 1, 2, 3) like Bing

def quadkey(x, y, z):
    qk = []
    for i in range(z - 1, -1, -1):
        x_bit = (x >> i) & 1
        y_bit = (y >> i) & 1
        qk.append(str(GE_DIGIT_MAP[(x_bit << 1) | y_bit]))
    return "".join(qk)
```

Output is a base-4 string of length z.

## Coordinate system

GE Pro uses a **linear equirectangular** projection with the world as a
`2^(z-1) × 2^(z-1)` quadtree. **NOT Web Mercator.** y axis grows **north**
(TMS-style). Empirically derived from GEID 6.48's own list1.txt task file.

```python
def bbox_to_tile_range(lon_min, lon_max, lat_min, lat_max, z):
    n = 2 ** (z - 1)
    factor = n / 360.0
    x_min = int((lon_min + 180.0) * factor)
    x_max = int((lon_max + 180.0) * factor)
    y_min = int((lat_min + 180.0) * factor)
    y_max = int((lat_max + 180.0) * factor)
    return x_min, x_max, y_min, y_max
```

Note `int(...)` not `round(...)` — GEID truncates.

## Per-area `imgVer`

Different parts of the world use different image version IDs in the f1
URL because each version corresponds to a specific Google Earth capture
date for that area. Empirically from Joburg z=21:

| Area                              | Working `imgVer` values     |
|-----------------------------------|-----------------------------|
| CBD (lat ≈ -26.22)                | `1010`                      |
| Sandton/Midrand (lat ≈ -26.13)    | `1033`, `1029`, `1024`, `1022`, `1020` |
| Northern Joburg (lat ≈ -26.00)    | `1033`, `1029`, `1020`      |

The `geid_python_batch.py` downloader auto-probes a candidate list
`(1033, 1029, 1024, 1022, 1020, 1018, 1015, 1014, 1013, 1012, 1011, 1010)`
against each grid's center tile and uses the first hit as the primary
version. Per-tile fallback through the rest of the list catches any
mid-grid version transitions.

A request with the wrong `imgVer` for an area returns **404**, not an
empty body. A request with a malformed (non-existent) version returns
**400 INVALID_ARGUMENT**.

## Wire-format encryption — fixed XOR key

Every response body (dbRoot, q2 metadata, f1 image tiles) is XOR-encrypted
with a **single fixed key** that's independent of URL, session, time, or
area. The key is recovered via a known-plaintext attack:

1. Run GEID and let it download a tile
2. Capture the wire body via Frida (`SSL_read` hook on ssleay32.dll)
3. GEID writes the decrypted JPG to disk in `<save_to>/<task>/<z>/<x>/ges_x_y_z.jpg`
4. XOR the wire body with the on-disk JPG → key bytes for `[0, len(tile))`

The key is **not periodic** (autocorrelation < 1% at any short period) and
**not stored as a literal blob** in `downloader.exe` (grep for the first
32 bytes returns 0 matches across all GEID binaries). It's likely
generated at runtime from a seed via some keystream construction, but
that doesn't matter for our purposes — the recovered bytes are stable.

To extend the key past its current length, capture a larger
known-plaintext pair. The current `cipher_key.bin` is **23242 B** as of
2026-04-08, large enough for all observed Joburg z=21 tiles. See
`examples/key_extension/README.md` for the worked example.

### Decrypted body framings

| Endpoint     | Body after XOR decrypt                                            |
|--------------|-------------------------------------------------------------------|
| `f1` tiles   | Plain JPEG (`ff d8 ff e0` SOI marker)                             |
| `q2` qtm     | `4 magic ad de 68 74` + `4 LE size` + zlib stream → ~11 KB qtm packet |
| `dbRoot.v5`  | Custom non-zlib format (Keyhole DbRoot protobuf wrapper, undecoded) |

`dbRoot.v5` decoding is **not needed** for tile downloads as long as the
`q.1033` / `i.1010` baseline versions are stable.

## geauth handshake (incomplete RE)

The `POST /geauth` body is exactly 49 bytes: 31 stable + 18 variable.

```
Offset  Length  Content                                          Notes
0       4       02 00 00 00                                      stable
4       1       03                                               stable
5       16      f8aa66c44373b042bbae748e43c3d399                 stable (client UUID, possibly derived from license SN)
21      8       0a 06 80 af 17 6f a8 ec                          stable
29      8       VARIES                                           timestamp-like
37      2       00 0b                                            stable
39      1       23                                               stable
40      9       VARIES                                           HMAC/nonce-like
```

The 18 variable bytes (offsets 29-36 and 40-48) cannot be replayed
verbatim across sessions, so this body must be **generated** to mint a
fresh SessionId. The function inside `downloader.exe` that builds this
buffer hasn't been located yet (open TODO — search for the call site that
allocates 49 bytes and writes the stable prefix `02 00 00 00 03 f8 aa 66`).

A captured SessionId is **reusable** for hours across processes / IPs
until expiry, so for batch jobs the workaround is to capture once and
reuse.

## geauth response (decoded)

```
Offset  Length  Content                              Notes
0       4 BE    0x00000258 (= 600)                   constant
4       4 BE    0x00000068 (= 104)                   length of next field
8       104     SessionId ASCII                      ← used as Cookie value in subsequent /flatfile calls
112     4 BE    0x00000078 (= 120)                   ?
116     4 BE    0x00000050 (= 80)                    ?
120     16      binary                               HMAC/sig?
```

Total body: 136 bytes.

## `.geid` task file format

Plain INI. Key sections:

```ini
[MapsType]
59         ; current Satellite layer

[AREA]
LeftLong=...
RightLong=...
TopLat=...
BottomLat=...

[Zoom]
From=18
To=21

[XY]
; computed tile coords (optional, derived from AREA + Zoom)

[FolderStyle]
0
```

Trivial to generate from Python. The task file is also accepted via the
GEID CLI mode as the first positional argument:

```
downloader.exe task_name zfrom zto L R T B savepath [date]
```

Example (verified 2026-04-08, completely headless from WSL):
```bash
"/mnt/c/allmapsoft/geid/downloader.exe" \
  "key_ext_605987_447923" 21 21 \
  28.0490913 28.0495033 -26.2174644 -26.2178764 \
  "C:\Temp\geid_key_ext"
```

GEID auto-exits when the task finishes — no GUI dialogs. CLI mode is the
**recommended fallback** when the Python client encounters a tile larger
than the current cipher key (use it to fetch the plain JPG and run
`extend_cipher_key.py`).

## Output naming

```
<save_to>/<task>/<z>/<x>/ges_{x}_{y}_{z}.jpg
```

The `geid_python_batch.py` downloader mirrors this layout exactly so that
existing GEID stitch tools / mosaicing scripts continue to work
unchanged.

## Open TODOs

1. **Capture the geauth body generation function** in `downloader.exe`
   via a Frida deeper hook. Until this is done, the Python client
   requires a captured SessionId to bootstrap.
2. **Historical imagery flow**: GEID's CLI exposes a `[date]` parameter
   for historical Google Earth captures. The protocol-level mechanism
   (probably a different `MapsType` + a date parameter on the f1 URL) is
   not yet captured. CLI invocation works as a fallback.
3. **Decode `dbRoot.v5`** to discover image version IDs programmatically
   instead of brute-force probing. Low priority — auto-probe is fast.

## Reusable artifacts in this folder

| File                                | Bytes  | Notes |
|-------------------------------------|--------|-------|
| `artifacts/cipher_key.bin`          | 23242  | XOR key — covers all observed Joburg z=21 tiles |
| `artifacts/dbRoot_v5_encrypted.bin` | ~16775 | Captured raw response, for future protobuf decoding |
| `artifacts/geauth_request_sample.bin`  | 49  | One captured handshake body |
| `artifacts/geauth_response_sample.bin` | 136 | Matching response containing a now-expired SessionId |
| `artifacts/qtm_packet_sample.bin`   | 10948  | Decrypted+inflated qtm packet, for future QTM decoder work |
