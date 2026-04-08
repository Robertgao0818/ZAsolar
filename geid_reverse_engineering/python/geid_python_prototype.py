#!/usr/bin/env python3
"""End-to-end Python prototype for the Allmapsoft GEID protocol.

Proof-of-concept that bypasses GEID's GUI entirely and downloads Google Earth
Satellite tiles directly from kh.google.com using the legacy Google Earth Pro
API. Uses a captured SessionId (the geauth handshake is not yet replicated)
and the recovered fixed XOR cipher key.

What this proves:
- Tile URL format        ✓
- User-Agent spoofing    ✓
- Cookie / SessionId reuse ✓
- (x,y,z) → quadkey      ✓ (Google Earth digit map = 0,3,1,2)
- Wire-format XOR cipher ✓ (fixed key in data/geid_protocol/cipher_key.bin)

Limitations:
- SessionId expires; rerun the Frida capture to refresh
- Only the current Satellite layer (q.1033 / i.1010); historical NYI
- Tiles larger than the cipher key (~19759 B) need a longer key — extend
  the key by capturing a bigger known-plaintext tile (XOR wire body with
  the corresponding decrypted JPEG GEID writes to disk)

Usage:
    python scripts/imagery/geid_python_prototype.py 75735 55989 18 \
        --session-id 'egPTzQAD...==' \
        --out /tmp/test_tile.jpg
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

RE_ROOT = Path(__file__).resolve().parents[1]
CIPHER_KEY_PATH = RE_ROOT / "artifacts" / "cipher_key.bin"

USER_AGENT = (
    "GoogleEarth/7.3.6.9345(Windows;Microsoft Windows (6.2.9200.0);"
    "en;kml:2.2;client:Pro;type:default)"
)

# Google Earth Pro quadrant ordering (NOT standard Bing). Verified by
# brute-force matching against 16 captured (x,y,z)→quadkey pairs from G0772.
GE_DIGIT_MAP = (0, 3, 1, 2)


def quadkey(x: int, y: int, z: int) -> str:
    """Convert (x, y, z) tile coords to a Google Earth Pro quadkey."""
    qk = []
    for i in range(z - 1, -1, -1):
        x_bit = (x >> i) & 1
        y_bit = (y >> i) & 1
        qk.append(str(GE_DIGIT_MAP[(x_bit << 1) | y_bit]))
    return "".join(qk)


def xor_decrypt(wire: bytes, key: bytes) -> bytes:
    if len(wire) > len(key):
        raise ValueError(
            f"wire body is {len(wire)} B but cipher key is only {len(key)} B; "
            f"capture a bigger known-plaintext tile to extend the key"
        )
    return bytes(a ^ b for a, b in zip(wire, key))


def fetch_tile(
    x: int,
    y: int,
    z: int,
    session_id: str,
    img_ver: int = 1010,
    timeout: float = 30.0,
) -> bytes:
    qk = quadkey(x, y, z)
    url = f"https://kh.google.com/flatfile?f1-{qk}-i.{img_ver}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", USER_AGENT)
    req.add_header(
        "Accept",
        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    )
    req.add_header("Accept-Encoding", "identity")
    req.add_header("Content-Type", "application/octet-stream")
    req.add_header(
        "Cookie",
        f'$Version="0"; SessionId="{session_id}"; State="1"',
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"unexpected status {resp.status}: {resp.read()[:200]!r}")
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("x", type=int)
    ap.add_argument("y", type=int)
    ap.add_argument("z", type=int)
    ap.add_argument("--session-id", required=True, help="SessionId base64 string captured from /geauth response")
    ap.add_argument("--out", type=Path, required=True, help="Where to write the decrypted JPG")
    ap.add_argument("--cipher-key", type=Path, default=CIPHER_KEY_PATH)
    ap.add_argument("--img-ver", type=int, default=1010)
    ap.add_argument("--compare-with", type=Path, help="Optional path to a GEID-saved tile to byte-compare")
    args = ap.parse_args()

    if not args.cipher_key.exists():
        print(f"cipher key not found: {args.cipher_key}", file=sys.stderr)
        return 2
    key = args.cipher_key.read_bytes()
    print(f"[geid] quadkey({args.x},{args.y},{args.z}) = {quadkey(args.x, args.y, args.z)}", file=sys.stderr)
    print(f"[geid] fetching from kh.google.com...", file=sys.stderr)
    wire = fetch_tile(args.x, args.y, args.z, args.session_id, img_ver=args.img_ver)
    print(f"[geid] wire body: {len(wire)} bytes; first 4 hex: {wire[:4].hex()}", file=sys.stderr)
    plain = xor_decrypt(wire, key)
    print(f"[geid] decrypted: {len(plain)} bytes; first 4 hex: {plain[:4].hex()}", file=sys.stderr)
    if plain[:3] == b"\xff\xd8\xff":
        print("[geid] ✓ valid JPEG SOI marker", file=sys.stderr)
    else:
        print("[geid] ✗ NOT a JPEG — cipher key may be wrong or tile is bigger than key", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(plain)
    print(f"[geid] wrote {args.out} ({len(plain)} bytes)", file=sys.stderr)
    if args.compare_with and args.compare_with.exists():
        ref = args.compare_with.read_bytes()
        match = ref == plain
        print(f"[geid] byte-compare with {args.compare_with}: {'MATCH ✓' if match else f'DIFFER (ref={len(ref)} B, ours={len(plain)} B)'}", file=sys.stderr)
        return 0 if match else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
