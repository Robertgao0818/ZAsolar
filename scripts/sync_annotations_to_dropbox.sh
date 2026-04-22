#!/usr/bin/env bash
# Mirror data/annotations/ → Windows Dropbox folder so the cloud client uploads it.
#
# 设计:
#   - 镜像目标: /mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/annotations/
#     (Windows Dropbox 客户端监听这个目录，自动上传到 Dropbox 云端)
#   - 用 rsync --delete 保持目标与源完全一致
#   - 幂等：未变更的文件不会重传，秒级完成
#   - 安静模式 (--quiet) 适合 hook 调用；加 --verbose / -v 看详情
#
# 用法:
#   ./scripts/sync_annotations_to_dropbox.sh           # 静默同步
#   ./scripts/sync_annotations_to_dropbox.sh --verbose # 显示传输列表
#   ./scripts/sync_annotations_to_dropbox.sh --dry-run # 预演不真改
#
# 触发场景:
#   - 任何 GT 更新后 (新 grid 标注、SAM FN review 输出、目录重组)
#   - .claude/settings.json 的 PostToolUse hook 自动调用 (匹配 data/annotations/)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/data/annotations/"
DEST="/mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/annotations/"

# rsync flags
FLAGS=(-a --delete --exclude='.git/' --exclude='__pycache__/' --exclude='*.tmp')
VERBOSE=0
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        -v|--verbose) VERBOSE=1 ;;
        -n|--dry-run) DRY_RUN=1; FLAGS+=(--dry-run) ;;
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
    echo "[sync_annotations_to_dropbox] source missing: $SRC" >&2
    exit 1
fi

if [[ ! -d "/mnt/c/Users/gaosh/Dropbox" ]]; then
    echo "[sync_annotations_to_dropbox] Dropbox folder not mounted: /mnt/c/Users/gaosh/Dropbox" >&2
    exit 1
fi

mkdir -p "$DEST"

if [[ $VERBOSE -eq 1 ]]; then
    echo "[sync_annotations_to_dropbox] $SRC → $DEST"
fi

rsync "${FLAGS[@]}" "$SRC" "$DEST"

if [[ $VERBOSE -eq 1 ]]; then
    echo "[sync_annotations_to_dropbox] Done."
fi
