#!/usr/bin/env bash
# runpod_flatten_tiles_for_build.sh — flatten registry-layout tiles
# under /workspace into the legacy flat layout /dev/shm/tiles/<grid_id>/
# expected by SOLAR_TILES_ROOT + resolve_tiles_dir.
#
# Why this exists (Codex v3 X1.a closure for solar_zerov2 R0.5):
#   The canonical RunPod tile location is the registry layout
#   /workspace/tiles/<region>/<imagery_layer>/<grid_id>/, but
#   core.grid_utils.resolve_tiles_dir's SOLAR_TILES_ROOT env-override only
#   matches the legacy flat layout $ROOT/<grid_id>/. The chip512 build
#   needs the flat layout, so we copy (not symlink) the registry-layout
#   directories into /dev/shm to also get the inference perf win
#   (.claude/rules/05-runpod-inference.md — /dev/shm is 10-50x faster than
#   /workspace MooseFS for inference IO).
#
# This script makes that step reproducible + hash-checked, instead of the
# hidden manual `cp -r /workspace/tiles/.../G* /dev/shm/tiles/` operator
# step the R0.5 build originally used.
#
# Usage:
#   bash scripts/runpod_flatten_tiles_for_build.sh \
#     [--region ct|jhb] [--layer aerial_2025|vexcel_2024|...] \
#     [--dst /dev/shm/tiles] [--check-grids G1238,G1410]
#
# Defaults: region=ct, layer=aerial_2025, dst=/dev/shm/tiles.
# --check-grids defaults to the first two grids found.
#
# Exit codes:
#   0  flatten + verify OK
#   1  source dir missing
#   2  copy failed or hash mismatch
set -euo pipefail

REGION="ct"
LAYER="aerial_2025"
DST="/dev/shm/tiles"
CHECK_GRIDS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --layer) LAYER="$2"; shift 2 ;;
    --dst) DST="$2"; shift 2 ;;
    --check-grids) CHECK_GRIDS="$2"; shift 2 ;;
    *) echo "[error] unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Map short region alias to canonical regions.yaml key for the path.
case "$REGION" in
  ct|cape_town) REGION_PATH="cape_town" ;;
  jhb|joburg|johannesburg) REGION_PATH="johannesburg" ;;
  *) REGION_PATH="$REGION" ;;
esac

SRC="/workspace/tiles/${REGION_PATH}/${LAYER}"
if [[ ! -d "$SRC" ]]; then
  echo "[error] source tile dir not found: $SRC" >&2
  echo "[hint] available regions/layers under /workspace/tiles/:" >&2
  ls /workspace/tiles/ 2>&1 || true
  exit 1
fi

mkdir -p "$DST"
echo "[flatten] source: $SRC"
echo "[flatten] dest:   $DST"

# Count grids before copy
n_src=$(find "$SRC" -maxdepth 1 -mindepth 1 -type d | wc -l)
echo "[flatten] source grid count: $n_src"

# Copy. `cp -r` (not symlink) so /dev/shm gets the perf benefit per rule 05.
# Existing /dev/shm grids are skipped (cp -n) to make reruns cheap.
copied=0
skipped=0
for grid_dir in "$SRC"/*/; do
  grid_id=$(basename "$grid_dir")
  if [[ -d "$DST/$grid_id" ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  cp -r "$grid_dir" "$DST/$grid_id"
  copied=$((copied + 1))
done
echo "[flatten] copied=$copied  already_present=$skipped"

# Hash-check a sample of grids
if [[ -z "$CHECK_GRIDS" ]]; then
  CHECK_GRIDS=$(find "$SRC" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" | sort | head -2 | tr '\n' ',' | sed 's/,$//')
fi

mismatches=0
IFS=',' read -ra GRIDS_ARRAY <<< "$CHECK_GRIDS"
for g in "${GRIDS_ARRAY[@]}"; do
  src_chip=$(find "$SRC/$g" -maxdepth 1 -name '*.tif' | head -1)
  dst_chip="$DST/$g/$(basename "$src_chip")"
  if [[ ! -f "$src_chip" || ! -f "$dst_chip" ]]; then
    echo "[verify] $g: missing source or dest chip — SKIP"
    continue
  fi
  src_sha=$(sha256sum "$src_chip" | cut -d ' ' -f 1)
  dst_sha=$(sha256sum "$dst_chip" | cut -d ' ' -f 1)
  if [[ "$src_sha" == "$dst_sha" ]]; then
    echo "[verify] $g/$(basename "$src_chip"): OK ($src_sha)"
  else
    echo "[verify] $g/$(basename "$src_chip"): MISMATCH src=$src_sha dst=$dst_sha" >&2
    mismatches=$((mismatches + 1))
  fi
done

if [[ "$mismatches" -gt 0 ]]; then
  echo "[error] $mismatches sha256 mismatch(es); aborting" >&2
  exit 2
fi

# Final grid count check on /dev/shm
n_dst=$(find "$DST" -maxdepth 1 -mindepth 1 -type d | wc -l)
echo "[done] /dev/shm/tiles has $n_dst grids (source had $n_src)"
echo "[hint] export SOLAR_TILES_ROOT=$DST  # then re-run the build command"
