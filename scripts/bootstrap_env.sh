#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCK_FILE="$PROJECT_ROOT/requirements.lock.txt"
REQ_FILE="$PROJECT_ROOT/requirements.txt"

mkdir -p \
  "$PROJECT_ROOT/.cache" \
  "$PROJECT_ROOT/.config" \
  "$PROJECT_ROOT/.local/share" \
  "$PROJECT_ROOT/.tmp/joblib"

export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export XDG_CONFIG_HOME="$PROJECT_ROOT/.config"
export XDG_DATA_HOME="$PROJECT_ROOT/.local/share"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export JOBLIB_TEMP_FOLDER="$PROJECT_ROOT/.tmp/joblib"

if [[ -d "/usr/lib/wsl/lib" ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$MPLCONFIGDIR"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

"$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel

if [[ -f "$LOCK_FILE" ]]; then
  if ! "$VENV_PATH/bin/pip" install -r "$LOCK_FILE"; then
    echo
    echo "Lock file install failed; falling back to requirements.txt"
    "$VENV_PATH/bin/pip" install -r "$REQ_FILE"
  fi
else
  "$VENV_PATH/bin/pip" install -r "$REQ_FILE"
fi

echo
echo "Environment ready at $VENV_PATH"
echo "Next:"
echo "  source scripts/activate_env.sh"
echo "  ./scripts/check_env.sh"
