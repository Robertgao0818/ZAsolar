#!/usr/bin/env bash
# Pack a model_run results dir on the pod into /workspace/<run>.tar.gz, pull
# the tarball to local, and unpack into project results/<region>/<run>/.
#
# Pod side stores results on /root (overlay, ephemeral) to keep the network
# volume small; the tarball on /workspace is the durable copy that survives
# pod restarts. Delete the previous tarball manually before the next run.
#
# Usage:
#   ./scripts/pack_and_pull_pod_results.sh <region> <model_run>
#   ./scripts/pack_and_pull_pod_results.sh johannesburg v3c_vexcel_2024
#
# Env overrides (optional):
#   POD_RESULTS_PARENT=/root/results        # pod results root (parent of region)
#   ARCHIVE_DIR_LOCAL=~/zasolar_data/results_archives  # where to drop tarball

set -euo pipefail

REGION="${1:?usage: $0 <region> <model_run>}"
RUN="${2:?usage: $0 <region> <model_run>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
[[ -f "$ENV_FILE" ]] && set -a && source "$ENV_FILE" && set +a
: "${RUNPOD_SSH_HOST:?RUNPOD_SSH_HOST not set}"
: "${RUNPOD_SSH_PORT:?RUNPOD_SSH_PORT not set}"

POD_RESULTS_PARENT="${POD_RESULTS_PARENT:-/root/results}"
ARCHIVE_DIR_LOCAL="${ARCHIVE_DIR_LOCAL:-$HOME/zasolar_data/results_archives}"
TARBALL="${RUN}.tar.gz"
POD_TARBALL="/workspace/${TARBALL}"
LOCAL_TARBALL="$ARCHIVE_DIR_LOCAL/${TARBALL}"
LOCAL_UNPACK_PARENT="$REPO_ROOT/results/${REGION}"

mkdir -p "$ARCHIVE_DIR_LOCAL" "$LOCAL_UNPACK_PARENT"

SSH_OPTS=(-i ~/.ssh/id_ed25519 -p "$RUNPOD_SSH_PORT")

echo "[1/4] tar on pod: $POD_RESULTS_PARENT/$REGION/$RUN → $POD_TARBALL"
ssh "${SSH_OPTS[@]}" "$RUNPOD_SSH_HOST" "
  set -e
  cd '$POD_RESULTS_PARENT/$REGION'
  if [ ! -d '$RUN' ]; then echo 'pod source missing: $POD_RESULTS_PARENT/$REGION/$RUN' >&2; exit 1; fi
  rm -f '$POD_TARBALL'
  tar -I 'gzip -1' -cf '$POD_TARBALL' '$RUN'
  ls -lh '$POD_TARBALL'
"

echo "[2/4] aws s3 cp s3://${RUNPOD_S3_VOLUME_ID}/${TARBALL} → $LOCAL_TARBALL"
# scp via SSH proxy is rate-limited (~130KB/s); the same MFS volume is exposed
# via the S3 gateway with multipart parallel download (~12 MB/s, 90× faster).
: "${RUNPOD_S3_KEY_ID:?missing}"; : "${RUNPOD_S3_SECRET:?missing}"
: "${RUNPOD_S3_VOLUME_ID:?missing}"
S3_ENDPOINT="${RUNPOD_S3_ENDPOINT:-https://s3api-eu-ro-1.runpod.io}"
AWS_ACCESS_KEY_ID="$RUNPOD_S3_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$RUNPOD_S3_SECRET" \
AWS_DEFAULT_REGION=eu-ro-1 \
  aws s3 cp "s3://${RUNPOD_S3_VOLUME_ID}/${TARBALL}" "$LOCAL_TARBALL" \
  --endpoint-url "$S3_ENDPOINT"
ls -lh "$LOCAL_TARBALL"

echo "[3/4] unpack to $LOCAL_UNPACK_PARENT/"
rm -rf "$LOCAL_UNPACK_PARENT/$RUN"
tar -xzf "$LOCAL_TARBALL" -C "$LOCAL_UNPACK_PARENT/"
du -sh "$LOCAL_UNPACK_PARENT/$RUN"
ls "$LOCAL_UNPACK_PARENT/$RUN" | wc -l | xargs -I{} echo "  {} grid dirs"

echo "[4/4] done."
echo "  tarball on pod : $POD_TARBALL  (delete before next run: ssh ... rm $POD_TARBALL)"
echo "  tarball local  : $LOCAL_TARBALL"
echo "  unpacked       : $LOCAL_UNPACK_PARENT/$RUN"
echo
echo "next: push to Dropbox (excludes masks/, vectors/):"
echo "  ./scripts/sync_vexcel_results_to_dropbox.sh"
