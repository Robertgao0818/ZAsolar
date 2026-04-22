#!/usr/bin/env bash
# 更新协作仓库的 ZAsolar submodule 引用到本地最新 commit
# 用法: ./scripts/sync_to_collab.sh

set -euo pipefail

COLLAB_REPO="git@github.com:EddyZoulse01/SA_Solar.git"
TMPDIR=$(mktemp -d)
LOCAL_HEAD=$(git rev-parse HEAD)
LOCAL_MSG=$(git log -1 --format=%s)

echo "=== Syncing submodule to $LOCAL_HEAD ==="
echo "    ($LOCAL_MSG)"

# 确保本地已 push 到 origin
echo "Pushing to origin..."
git push origin main

# Clone 协作仓库，更新 submodule 引用
# 注意: 用 ./ZAsolar 而非 ZAsolar，避免 CDPATH 把 cd 跳到本地源仓库
git clone --depth=1 "$COLLAB_REPO" "$TMPDIR" 2>&1 | tail -1
cd "$TMPDIR"
unset CDPATH
git submodule update --init ZAsolar 2>&1 | tail -2

cd ./ZAsolar
git fetch origin main
git checkout "$LOCAL_HEAD" 2>/dev/null || git checkout origin/main
cd "$TMPDIR"

git add ZAsolar
if git diff --cached --quiet; then
    echo "Submodule already up to date."
else
    git commit -m "Update ZAsolar submodule: $LOCAL_MSG

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
    git push origin main
    echo "✓ Updated submodule ref in $COLLAB_REPO"
fi

rm -rf "$TMPDIR"
echo "Done."
