#!/usr/bin/env bash
# This research machine uses the system interpreter (/usr/bin/python3).
# Packages live in the user site ~/.local, which that interpreter already
# searches. Do not create or activate .venv.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Use: source scripts/activate_env.sh"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

unset VIRTUAL_ENV
export PATH="${HOME}/.local/bin:${PATH}"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export XDG_CONFIG_HOME="$PROJECT_ROOT/.config"
export XDG_DATA_HOME="$PROJECT_ROOT/.local/share"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export JOBLIB_TEMP_FOLDER="$PROJECT_ROOT/.tmp/joblib"

if [[ -d "/usr/lib/wsl/lib" ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" \
  "$MPLCONFIGDIR" \
  "$JOBLIB_TEMP_FOLDER"

hash -r
echo "Activated project environment: $PROJECT_ROOT"
echo "Python: $(command -v python3)"
