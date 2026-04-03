# RunPod SSH 自动更新规则

## 触发条件
当用户消息中出现 SSH 连接命令模式（如 `ssh root@x.x.x.x -p XXXXX`）时，自动执行以下操作：

1. 从命令中提取 `host`（含用户名）和 `port`
2. 更新项目根目录 `.env` 文件中的 `RUNPOD_SSH_HOST` 和 `RUNPOD_SSH_PORT`
3. 更新 `~/.ssh/known_hosts`（`ssh-keygen -R` 旧条目 + `ssh-keyscan` 新条目）
4. 简要确认更新完成，不需要用户手动操作

## 示例
用户输入：`ssh root@213.173.103.184 -p 29416 -i ~/.ssh/id_ed25519`
→ 自动更新 `.env`:
```
RUNPOD_SSH_HOST=root@213.173.103.184
RUNPOD_SSH_PORT=29416
```
→ 更新 known_hosts
→ 确认："已更新 .env SSH 配置 (host=213.173.103.184, port=29416)"

## 注意
- `.env` 文件已被 `.gitignore` 保护，不会提交到 git
- 同步脚本 `scripts/sync_from_runpod.sh` 和初始化脚本 `scripts/runpod_init.sh` 都从 `.env` 读取连接信息
