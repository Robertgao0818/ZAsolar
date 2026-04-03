# upload-to-cloud Skill

Upload project data (COCO dataset, tiles, checkpoints, code) to RunPod S3 storage.

## When to use

When syncing data to RunPod for cloud training, or after generating new COCO datasets / downloading new tiles.

## Prerequisites

- AWS CLI installed at `$HOME/.local/bin/aws`
- RunPod Network Volume with S3 API enabled
- Split files prepared in `/mnt/d/ZAsolar/upload_tmp/`

## Steps

### 1. Prepare split files (if not already done)

Large data must be split into ≤4GB parts because RunPod S3 has pipe/multipart issues.

```bash
# COCO dataset (~28GB → 7 parts)
tar -cf - -C /home/gaosh/projects/ZAsolar data/coco/ | \
  split -b 4G - /mnt/d/ZAsolar/upload_tmp/coco_part_

# Tiles (~23GB → 6 parts)
tar -cf - -C /home/gaosh/projects/ZAsolar tiles/ | \
  split -b 4G - /mnt/d/ZAsolar/upload_tmp/tiles_part_
```

For small files (code, checkpoints), tar and upload directly:
```bash
tar -cf /mnt/d/ZAsolar/upload_tmp/zasolar_code.tar \
  train.py export_coco_dataset.py detect_and_evaluate.py \
  core/ scripts/ configs/ data/annotations/ requirements*.txt CLAUDE.md

tar -cf /mnt/d/ZAsolar/upload_tmp/zasolar_checkpoints.tar \
  checkpoints/best_model.pth
```

### 2. Run upload script

```bash
bash scripts/upload_to_runpod.sh
```

Features:
- **Resume support**: already-uploaded files are skipped (size match check)
- **Progress display**: `[3/7] coco_part_ac (4.0G) — uploading... ✓ Done in 245s`
- **Safe to interrupt**: Ctrl+C and re-run, completed parts won't re-upload

### 3. Reassemble on RunPod Pod

SSH into Pod, then:
```bash
cd /workspace
# COCO
aws s3 cp --recursive s3://k5r31jwc9k/coco_parts/ /tmp/coco_parts/ \
  --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io
cat /tmp/coco_parts/coco_part_* | tar -xf - -C zasolar/

# Tiles
aws s3 cp --recursive s3://k5r31jwc9k/tiles_parts/ /tmp/tiles_parts/ \
  --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io
cat /tmp/tiles_parts/tiles_part_* | tar -xf - -C zasolar/
```

### 4. Upload individual files (code, checkpoints)

For quick syncs after code changes:
```bash
# Load credentials from .env (never hardcode)
source .env

$HOME/.local/bin/aws s3 cp /mnt/d/ZAsolar/upload_tmp/zasolar_code.tar \
  s3://k5r31jwc9k/zasolar_code.tar \
  --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io
```

Or via SCP (direct TCP SSH):
```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 <file> root@<IP>:/workspace/
```

## Constraints

- Never pipe `tar | aws s3 cp -` for files >4GB — RunPod S3 multipart fails on pipe streams
- Always split to real files first, then upload each file individually
- Split files go to D drive (`/mnt/d/ZAsolar/upload_tmp/`) to avoid filling C drive
- S3 credentials are in `scripts/upload_to_runpod.sh`
- RunPod S3 endpoint: `https://s3api-eu-ro-1.runpod.io`, bucket: `k5r31jwc9k`

## S3 Config Reference

| Key | Value |
|-----|-------|
| Endpoint | `https://s3api-eu-ro-1.runpod.io` |
| Region | `eu-ro-1` |
| Bucket | `k5r31jwc9k` |
| Access Key | See `.env` (`RUNPOD_S3_KEY_ID`) |
