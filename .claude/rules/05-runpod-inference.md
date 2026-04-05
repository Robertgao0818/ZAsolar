# RunPod 推理最佳实践

## 触发条件
在 RunPod pod 上执行任何推理任务（`detect_and_evaluate.py`、批量推理脚本等）之前，必须遵循以下流程。

## 标准流程

### 1. 数据准备：tiles → /dev/shm
```bash
# 先把需要的 tiles 拷到内存盘（network volume IO 慢 10-50x）
mkdir -p /dev/shm/tiles
for g in <GRID_LIST>; do
  cp -r /workspace/tiles/$g /dev/shm/tiles/ &
done
wait
export SOLAR_TILES_ROOT=/dev/shm/tiles
```
RTX 5090 shm = 28GB，可放约 25GB tiles。超出则分批处理。

### 2. 并行推理参数（RTX 5090, 32GB VRAM）
- **并行进程数**: 6（每个 detect_and_evaluate.py ~3-4GB VRAM）
- **DataLoader workers**: 由脚本内部管理
- **GPU 利用率目标**: >90%
- **如非 5090**（如 A100 80GB）: 可提至 12-15 并行

### 3. 后处理参数
始终使用 `--postproc-config configs/postproc/v4_canonical.json`，确保跨实验可比性。
```bash
python3 detect_and_evaluate.py \
  --grid-id $GRID \
  --model-path $MODEL \
  --postproc-config configs/postproc/v4_canonical.json \
  --force
```

### 4. 批量推理模板
```bash
PARALLEL=6
run_grid() {
  local g=$1
  SOLAR_TILES_ROOT=/dev/shm/tiles python3 detect_and_evaluate.py \
    --grid-id "$g" --model-path "$MODEL" \
    --postproc-config configs/postproc/v4_canonical.json --force \
    > "$LOG_DIR/${g}.log" 2>&1
}

running=0
for g in "${GRIDS[@]}"; do
  run_grid "$g" &
  running=$((running + 1))
  if [ $running -ge $PARALLEL ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait
```

### 5. 长任务保护
- 用 `nohup ... &` 防止 SSH 断连丢任务
- 用 `python3 -u` 或 `stdbuf -oL` 确保日志实时输出（不被缓冲）

## 注意
- `/dev/shm` 数据在 pod 重启后丢失，每次需重新拷贝
- 不要在 network volume 上直接跑推理，IO 会成为瓶颈（GPU 利用率 <30%）
