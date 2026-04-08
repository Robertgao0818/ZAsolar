#!/usr/bin/env python3
"""Extend the GEID XOR cipher key using a known-plaintext attack.

Inputs (relative to this file's grandparent — i.e. the RE root):
  - artifacts/cipher_key.bin                  (current key)
  - examples/key_extension/wire_*.bin         (encrypted wire body, fetched via Python)
  - examples/key_extension/plain_*.jpg        (decrypted JPG, written by GEID CLI on Windows)

Process:
  1. XOR wire ⊕ plain  →  recovered_key (same length as both)
  2. Validate: recovered_key[:current_len] must equal current cipher_key.bin
     (proves the protocol uses a fixed key, not session-specific)
  3. Extend: new_key = recovered_key (full length)
  4. Backup current key to cipher_key.bin.bak.<timestamp>
  5. Write extended key

After running, retry any previously-failed grids and they should hit 0 failures
because tiles whose wire bodies are <= len(new_key) can now be fully decrypted.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

RE_ROOT = Path(__file__).resolve().parents[1]
KEY_PATH = RE_ROOT / "artifacts" / "cipher_key.bin"
EXT_DIR = RE_ROOT / "examples" / "key_extension"


def main() -> int:
    if not KEY_PATH.exists():
        print(f"✗ current key not found: {KEY_PATH}", file=sys.stderr)
        return 2
    current_key = KEY_PATH.read_bytes()
    print(f"current key: {len(current_key)} B  ({KEY_PATH})")

    wires = sorted(EXT_DIR.glob("wire_*.bin"))
    plains = sorted(EXT_DIR.glob("plain_*.jpg"))
    if not wires:
        print(f"✗ no wire_*.bin files in {EXT_DIR}", file=sys.stderr)
        return 2
    if not plains:
        print(f"✗ no plain_*.jpg files in {EXT_DIR} — run extend_cipher_key.ps1 on Windows first", file=sys.stderr)
        return 2

    # Match wire ↔ plain by (x,y,z) embedded in filename
    pairs = []
    for w in wires:
        # wire_X_Y_Z_vV.bin
        parts = w.stem.split("_")
        if len(parts) < 4:
            continue
        x, y, z = parts[1], parts[2], parts[3]
        plain_name = f"plain_{x}_{y}_{z}.jpg"
        plain_path = EXT_DIR / plain_name
        if plain_path.exists():
            pairs.append((w, plain_path, int(x), int(y), int(z)))
    if not pairs:
        print(f"✗ no matching wire/plain pairs found in {EXT_DIR}", file=sys.stderr)
        return 2
    print(f"matched {len(pairs)} wire/plain pair(s)")

    # Use the LARGEST pair (max key recovery)
    pairs.sort(key=lambda p: -p[0].stat().st_size)
    wire_path, plain_path, tx, ty, tz = pairs[0]
    wire = wire_path.read_bytes()
    plain = plain_path.read_bytes()
    print(f"\nbest pair: ({tx},{ty},{tz})")
    print(f"  wire:  {len(wire)} B  ({wire_path.name})")
    print(f"  plain: {len(plain)} B  ({plain_path.name})")

    if len(wire) != len(plain):
        print(f"✗ wire and plain have different lengths "
              f"({len(wire)} vs {len(plain)}); GEID may have re-encoded the JPG", file=sys.stderr)
        # Show first/last bytes for debugging
        print(f"  wire  first 16: {wire[:16].hex()}")
        print(f"  plain first 16: {plain[:16].hex()}")
        return 3

    if plain[:3] != b"\xff\xd8\xff":
        print(f"✗ plain JPG missing SOI marker (got {plain[:3].hex()})", file=sys.stderr)
        return 3
    if plain[-2:] != b"\xff\xd9":
        print(f"⚠ plain JPG missing EOI marker (got {plain[-2:].hex()}); proceeding anyway")

    # Recover key
    recovered = bytes(a ^ b for a, b in zip(wire, plain))
    print(f"\nrecovered key: {len(recovered)} B")

    # Validate against existing key
    overlap = min(len(recovered), len(current_key))
    matches = sum(1 for i in range(overlap) if recovered[i] == current_key[i])
    pct = 100 * matches / overlap
    print(f"validation: {matches}/{overlap} bytes match current key ({pct:.2f}%)")
    if pct < 99.9:
        print(f"✗ recovered key does NOT match current key in overlap region.", file=sys.stderr)
        print(f"   This means either:", file=sys.stderr)
        print(f"   - the cipher is not a fixed XOR (protocol has changed?)", file=sys.stderr)
        print(f"   - GEID re-compressed the JPG (output != original wire plaintext)", file=sys.stderr)
        # Show first divergence
        for i in range(overlap):
            if recovered[i] != current_key[i]:
                print(f"   first divergence at byte {i}: "
                      f"recovered=0x{recovered[i]:02x} current=0x{current_key[i]:02x}", file=sys.stderr)
                break
        return 4
    print("✓ recovered key matches current key in the overlap region")

    if len(recovered) <= len(current_key):
        print(f"⚠ recovered key ({len(recovered)} B) is not longer than current ({len(current_key)} B); nothing to extend", file=sys.stderr)
        return 0

    # Backup
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = KEY_PATH.with_suffix(f".bin.bak.{ts}")
    backup.write_bytes(current_key)
    print(f"\nbackup: {backup} ({len(current_key)} B)")

    # Write extended key
    KEY_PATH.write_bytes(recovered)
    print(f"✓ extended cipher_key.bin written: {len(current_key)} B → {len(recovered)} B "
          f"(+{len(recovered) - len(current_key)} B)")
    print(f"\nNext: retry the 19 failed grids:")
    print(f"  python3 scripts/imagery/geid_python_batch.py \\")
    print(f"    --grid-id G0773 G0774 G0775 G0776 G0815 G0816 G0817 G0818 \\")
    print(f"              G0854 G0855 G0888 G0889 G0922 G1152 G1153 G1154 G1189 G1190 G1218 \\")
    print(f"    --output-dir /mnt/d/ZAsolar/joburg_geid_python \\")
    print(f"    --session-id <SID> --workers 6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
