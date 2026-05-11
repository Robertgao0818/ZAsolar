# RunPod 大文件传输规则

## 触发条件
任何需要把 >100 MB 文件（tiles、训练集 tar、checkpoint 池、annotations 包等）从本地推到 RunPod pod，或在 pod 上准备训练集时，必须遵循以下顺序。

## 标准流程

### 1. 先看 /workspace 有没有现成的（read-before-write）
`/workspace` 是持久 network volume，跨 pod 保留；旧 run 留下的底图/tiles/COCO/checkpoint 大概率还在。
```bash
ssh "$RUNPOD_SSH_HOST" -p "$RUNPOD_SSH_PORT" \
  "ls -lh /workspace/tiles/<region>/<imagery_layer>/ ; \
   ls -lh /workspace/coco/ ; \
   du -sh /workspace/* | sort -h"
```
找到匹配的底图就直接复用，不要重传。

### 2. 大文件传输：禁止 SCP
SCP / rsync over SSH 走 SSH proxy 限速 ~130 KB/s，传几个 GB 就是几十小时，不可用。

**上传方向（本地 → pod）**：走 RunPod S3 endpoint。
```bash
aws s3 cp <local_file> s3://${RUNPOD_S3_VOLUME_ID}/<remote_name> \
  --endpoint-url https://s3api-eu-ro-1.runpod.io
# pod 端文件立刻出现在 /workspace/<remote_name>，不需要二次拉取
```
也可用 `runpodctl send / receive` 走 P2P，单文件几 GB 量级 OK。

**下载方向（pod → 本地）**：见 `scripts/pack_and_pull_pod_results.sh`，aws s3 cp 实测 12 MB/s。

**例外**：≤ 100 MB 的小文件（单个 checkpoint、config、单脚本）继续用 scp 没问题。

### 3. 训练集在 pod 上现切，不要本地切完再传
COCO/HN/clean_gt 这类训练集是从 tiles + annotations 派生出来的，本地切完打 tar 上传是双倍流量浪费。正确做法：
- 把 tiles（如果 /workspace 没有）和 annotations（小，几 MB）传到 pod
- 在 pod 上跑 `export_coco_dataset.py` / `export_v4_1_hn.py` / clean_gt builder 等脚本直接生成 `/workspace/coco/...`
- 训练脚本直接读 `/workspace/coco/<set>/`，不需要再来回搬

annotations 包本身一般 <50 MB，scp 即可；tiles 走 S3 或复用 /workspace 已有底图。

## 注意
- `/workspace` 是 MFS 网络卷，IO 慢；跑训练/推理前仍要把热数据拷到 `/dev/shm`（见 05-runpod-inference.md）
- 上传到 S3 后 pod 端才能看到，不要在 cp 没完成时就 ssh ls
- 换 pod / 重启实例后第一件事是 `ls /workspace/<expected>`，不要直接复用旧脚本套路径就开传
