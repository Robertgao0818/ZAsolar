# Allmapsoft GEID 6.48 — Reverse Engineering & Headless Replacement

A self-contained record + working tools for downloading Google Earth Pro
satellite tiles **without the Allmapsoft GEID GUI**, by speaking the legacy
`kh.google.com` protocol directly. Built so that the ZAsolar pipeline could
run batch tile downloads on Linux/RunPod (the GEID GUI is Windows-only and
not scriptable beyond a few CLI args).

## Status (2026-04-08)

- ✓ Protocol fully understood: URL formats, User-Agent spoofing, cookie
  reuse, quadkey encoding, wire-format XOR cipher
- ✓ End-to-end Python prototype produces **byte-equal** tiles to GEID
- ✓ Concurrent batch downloader: 6 workers, ~30 dl/s, per-grid resume,
  manifest tracking
- ✓ Per-area `imgVer` auto-probing (different Joburg areas use 1010 / 1020
  / 1029 / 1033)
- ✓ XOR cipher key extended via known-plaintext attack: 19759 → 23242 B
  (now covers all observed Joburg z=21 tiles)
- ✓ Joburg Batch 1 downloaded end-to-end: **83579 / 83579 tiles, 100%, 1.1 GB**
- ⚠ Still using a captured `SessionId` for auth — geauth handshake body
  generation (49 bytes, 18 variable bytes) not yet replicated. The
  captured SessionId lasts hours; for batch jobs that's enough.
- ⚠ Historical imagery flow not implemented in Python (GEID CLI's `[date]`
  parameter handles this trivially — see `windows/extend_cipher_key.ps1`
  for the CLI invocation pattern)

## Layout

```
geid_reverse_engineering/
├── README.md                 ← this file (overview + status)
├── PROTOCOL.md               ← detailed protocol writeup
├── python/                   ← cross-platform headless client
│   ├── geid_python_prototype.py    (single-tile validator)
│   ├── geid_python_batch.py        (concurrent batch downloader)
│   └── extend_cipher_key.py        (XOR key extension via known-plaintext)
├── windows/                  ← Frida hooks + CLI helpers
│   ├── hook_geid_ssl.js            (Frida agent: SSL_read/write hook on ssleay32)
│   ├── hook_geid_curl.py           (Frida launcher; spawn / attach)
│   ├── hook_geid_curl.js           (dead-end first attempt — libcurl hook, not used)
│   ├── extend_cipher_key.ps1       (PowerShell wrapper around GEID CLI for key extension)
│   └── dump_geid_state.ps1         (recon helper: enumerate downloader.exe Win32 controls)
├── artifacts/                ← recovered protocol artifacts
│   ├── cipher_key.bin              (23242 B XOR key)
│   ├── cipher_key.bin.bak.*        (rollback backups from earlier extensions)
│   ├── dbRoot_v5_encrypted.bin     (encrypted dbRoot.v5 sample)
│   ├── geauth_request_sample.bin   (49-byte handshake request body)
│   ├── geauth_response_sample.bin  (136-byte handshake response — contains SessionId)
│   └── qtm_packet_sample.bin       (decrypted+inflated qtm packet sample)
└── examples/
    └── key_extension/        ← worked example: extending the XOR key
```

## Quick start (Linux / WSL — no Windows needed for the actual download)

You need a captured `SessionId` cookie to start. The geauth handshake is
not yet replicated, so the very first time you set this up you must
capture one with the Frida hook (see `windows/hook_geid_ssl.js`). After
that, the SessionId is reusable for hours across batches.

### Single tile sanity check

```bash
python3 geid_reverse_engineering/python/geid_python_prototype.py 605880 447914 21 \
    --session-id 'egPTzQAD...==' \
    --out /tmp/test_tile.jpg
```

Expected output:
```
[geid] quadkey(605880,447914,21) = 013313312121320212030
[geid] fetching from kh.google.com...
[geid] wire body: 9058 bytes; first 4 hex: 0791efa6
[geid] decrypted: 9058 bytes; first 4 hex: ffd8ffe0
[geid] ✓ valid JPEG SOI marker
[geid] wrote /tmp/test_tile.jpg (9058 bytes)
```

### Batch download (project-specific: Joburg grid file)

```bash
python3 geid_reverse_engineering/python/geid_python_batch.py \
    --all-batch1 \
    --output-dir /mnt/d/ZAsolar/joburg_geid_python \
    --session-id 'egPTzQAD...==' \
    --workers 6
```

For grids outside the Joburg task grid: pass `--grid-id G####` for one or
more known IDs. The script reads `data/jhb_task_grid.gpkg` from the
parent ZAsolar repo. To use this downloader for arbitrary bboxes (e.g.
Cape Town, Pretoria), the easiest path is to add a grid file in the same
format and reuse `--grid-id`.

## Key technical findings

See `PROTOCOL.md` for the full writeup. TL;DR:

1. **Endpoint**: `https://kh.google.com` (legacy Google Earth Pro API; still
   alive for GE Pro client compatibility despite the GUI being EOL)
2. **HTTP stack inside `downloader.exe`**: Delphi Indy → OpenSSL 1.0
   (`ssleay32.dll` / `libeay32.dll`), **NOT libcurl** despite shipping
   `libcurl.dll`. Frida hooks must target `ssleay32!SSL_write`, not
   `libcurl!curl_easy_setopt`.
3. **User-Agent spoofing required**: any non-GE-Pro UA returns 403
4. **Auth**: 49-byte binary `POST /geauth` returns a `SessionId` cookie
   (104 ASCII bytes) that's reused on every subsequent request via the
   `Cookie: $Version="0"; SessionId="..."; State="1"` header
5. **Tile URL**: `GET /flatfile?f1-{quadkey}-i.{imgVer}` where the quadkey
   uses the GE Pro digit map `(0,3,1,2)` — NOT standard Bing
6. **Coordinate system**: linear equirectangular, NOT Web Mercator. At zoom
   `z` the world is a `2^(z-1) × 2^(z-1)` quadtree. y axis grows north.
7. **Wire encryption**: every response body is XOR'd with a single fixed
   key (independent of URL/session). Recovered via known-plaintext attack.
8. **Per-area `imgVer`**: each geographic area has its own image version ID
   for the f1 endpoint. Must be probed per grid.

## What this folder is NOT

- Not a general-purpose Google Maps downloader — uses the legacy GE Pro
  API which only Google Earth Pro / Allmapsoft GEID still speak. Modern
  Google Maps tile URLs are completely different.
- Not a license bypass for Allmapsoft. The Frida work was for
  understanding the protocol, not for circumventing license checks.
- Not yet a fully standalone client — still depends on a captured
  `SessionId` from a real GEID handshake (or a live GE Pro client). The
  geauth body generation (49 bytes with 18 variable bytes that look like
  timestamp + HMAC) is the last piece left to RE.

## Why we did this

The ZAsolar project trains Mask R-CNN on rooftop solar installations
across South Africa, starting with Cape Town and expanding to Joburg /
Pretoria. Tile downloads were originally driven by GEID's Win32 GUI via
SendMessage automation (`scripts/imagery/windows/run_geid_tasks.ps1` in
the parent repo), which was fragile and couldn't run on RunPod.
Reverse-engineering the protocol unlocked:
- Cross-platform headless tile downloads (RunPod, Linux, WSL, macOS)
- 6-way concurrent fetching (~30 dl/s vs GEID GUI's ~3 dl/s)
- Programmatic batch management (one command for 100 grids)
- No GEID license check (only the original capture needed it)

## Credits

Reverse engineering done by Robert Gao with assistance from Claude
Opus 4.6 in the ZAsolar project, April 2026.
