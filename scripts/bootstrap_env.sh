#!/usr/bin/env bash
# This laptop uses the system interpreter. Packages are already in the user
# site (~/.local), which /usr/bin/python3 searches with no virtualenv.
# RunPod pods still bootstrap their own environment in ct_census_setup.sh.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "System Python: $(command -v python3)"
echo "Packages: ~/.local/lib/python3.14/site-packages"
echo "Not creating ${PROJECT_ROOT}/.venv."
exec "${PROJECT_ROOT}/scripts/check_env.sh"
