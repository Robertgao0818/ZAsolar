#!/usr/bin/env bash
# Mirror docs/progress_log/ → Windows Dropbox folder so the cloud client uploads it.
#
# 设计:
#   - 镜像目标: /mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/progress_doc/
#   - 用 rsync -a 保持目标与源一致（不加 --delete，避免误删 Dropbox 上手工文件）
#   - 幂等：未变更的文件不会重传
#   - 安静模式 (--quiet) 适合 hook 调用；加 --verbose / -v 看详情
#
# 用法:
#   ./scripts/sync_progress_log_to_dropbox.sh           # 静默同步
#   ./scripts/sync_progress_log_to_dropbox.sh --verbose # 显示传输列表
#   ./scripts/sync_progress_log_to_dropbox.sh --dry-run # 预演不真改
#
# 触发场景:
#   - 任何日报写入/修改后
#   - .claude/settings.json 的 PostToolUse hook 自动调用 (匹配 docs/progress_log/)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/docs/progress_log/"
DEST="/mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/progress_doc/"

FLAGS=(-a --exclude='.git/' --exclude='__pycache__/' --exclude='*.tmp')
VERBOSE=0

for arg in "$@"; do
    case "$arg" in
        -v|--verbose) VERBOSE=1 ;;
        -n|--dry-run) FLAGS+=(--dry-run) ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

if [[ $VERBOSE -eq 1 ]]; then
    FLAGS+=(-v --stats)
fi

if [[ ! -d "$SRC" ]]; then
    echo "[sync_progress_log_to_dropbox] source missing: $SRC" >&2
    exit 1
fi

if [[ ! -d "/mnt/c/Users/gaosh/Dropbox" ]]; then
    echo "[sync_progress_log_to_dropbox] Dropbox folder not mounted" >&2
    exit 1
fi

mkdir -p "$DEST"

if [[ $VERBOSE -eq 1 ]]; then
    echo "[sync_progress_log_to_dropbox] $SRC → $DEST"
fi

rsync "${FLAGS[@]}" "$SRC" "$DEST"

if [[ $VERBOSE -eq 1 ]]; then
    echo "[sync_progress_log_to_dropbox] Done."
fi
