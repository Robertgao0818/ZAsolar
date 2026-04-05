#!/bin/bash
# RunPod pod lifecycle management.
# Wraps runpodctl to start/stop/status the project pod
# and auto-update .env with SSH connection info.
#
# Usage:
#   bash scripts/runpod_pod.sh start          — start pod, wait for SSH, update .env
#   bash scripts/runpod_pod.sh stop            — stop pod
#   bash scripts/runpod_pod.sh status          — show pod status
#   bash scripts/runpod_pod.sh ssh             — SSH into the pod
#   bash scripts/runpod_pod.sh init            — start + run runpod_init.sh
#
# Config: reads RUNPOD_API_KEY and RUNPOD_POD_ID from .env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"
FMT_PY="$SCRIPT_DIR/_runpod_format.py"

# Load .env
[ -f "$ENV_FILE" ] && set -a && source "$ENV_FILE" && set +a

: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY in .env}"
: "${RUNPOD_POD_ID:=s9m75h736xh7v9}"
export RUNPOD_API_KEY

SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"

# --- Helpers ---

pod_json() {
    runpodctl pod get "$RUNPOD_POD_ID" -o json 2>/dev/null
}

pod_status() {
    pod_json | python3 "$FMT_PY" status | grep -oP 'Status:\s+\K\S+'
}

wait_for_ssh() {
    local max_wait=120
    local elapsed=0
    echo -n "Waiting for pod to be ready"

    while [ $elapsed -lt $max_wait ]; do
        local ssh_info
        ssh_info=$(pod_json | python3 "$FMT_PY" ssh 2>/dev/null || true)

        if [ -n "$ssh_info" ]; then
            echo " ready!"
            echo "$ssh_info"
            return 0
        fi

        echo -n "."
        sleep 5
        elapsed=$((elapsed + 5))
    done

    echo " timeout after ${max_wait}s"
    return 1
}

update_env_var() {
    local key="$1" val="$2"
    if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
        sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
    else
        echo "${key}=${val}" >> "$ENV_FILE"
    fi
}

update_known_hosts() {
    local host="$1" port="$2"
    # Strip user@ prefix for known_hosts
    local hostname="${host#*@}"
    ssh-keygen -R "[$hostname]:$port" 2>/dev/null || true
    ssh-keyscan -p "$port" "$hostname" >> ~/.ssh/known_hosts 2>/dev/null
}

# --- Commands ---

cmd_start() {
    local status
    status=$(pod_status || echo "UNKNOWN")

    if [ "$status" = "RUNNING" ]; then
        echo "Pod $RUNPOD_POD_ID is already running."
    else
        echo "Starting pod $RUNPOD_POD_ID..."
        runpodctl pod start "$RUNPOD_POD_ID"
        echo "Start command sent."
    fi

    # Wait for SSH and update .env
    local ssh_info
    ssh_info=$(wait_for_ssh) || { echo "ERROR: Pod did not become ready"; exit 1; }

    local host port
    # Last line of wait_for_ssh output has "host port"
    host=$(echo "$ssh_info" | tail -1 | awk '{print $1}')
    port=$(echo "$ssh_info" | tail -1 | awk '{print $2}')

    echo ""
    echo "SSH available: ssh ${host} -p ${port} -i ${SSH_KEY}"

    # Update .env
    update_env_var "RUNPOD_SSH_HOST" "$host"
    update_env_var "RUNPOD_SSH_PORT" "$port"
    echo "Updated .env (host=$host, port=$port)"

    # Update known_hosts
    update_known_hosts "$host" "$port"
    echo "Updated known_hosts"
}

cmd_stop() {
    echo "Stopping pod $RUNPOD_POD_ID..."
    runpodctl pod stop "$RUNPOD_POD_ID"
    echo "Pod stopped. (billing paused)"
}

cmd_status() {
    echo "Pod: $RUNPOD_POD_ID"
    pod_json | python3 "$FMT_PY" status
}

cmd_ssh() {
    # Use .env values or get fresh from API
    local host="${RUNPOD_SSH_HOST:-}"
    local port="${RUNPOD_SSH_PORT:-}"

    if [ -z "$host" ] || [ -z "$port" ]; then
        echo "No SSH info in .env, querying pod..."
        local ssh_info
        ssh_info=$(wait_for_ssh) || { echo "ERROR: Pod not ready"; exit 1; }
        host=$(echo "$ssh_info" | tail -1 | awk '{print $1}')
        port=$(echo "$ssh_info" | tail -1 | awk '{print $2}')
    fi

    echo "Connecting to $host:$port..."
    exec ssh -p "$port" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$host"
}

cmd_init() {
    cmd_start

    echo ""
    echo "=== Running init script ==="
    # Re-source .env to get updated values
    set -a && source "$ENV_FILE" && set +a
    local host="${RUNPOD_SSH_HOST}" port="${RUNPOD_SSH_PORT}"

    ssh -p "$port" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$host" \
        "bash /workspace/ZAsolar/scripts/runpod_init.sh"
}

# --- Main ---

CMD="${1:-status}"
case "$CMD" in
    start) cmd_start ;;
    stop)  cmd_stop ;;
    status) cmd_status ;;
    ssh)   cmd_ssh ;;
    init)  cmd_init ;;
    *)
        echo "Usage: $0 <start|stop|status|ssh|init>"
        exit 1
        ;;
esac
