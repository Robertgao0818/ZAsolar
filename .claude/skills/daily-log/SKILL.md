# daily-log Skill

写日报并同步到 Dropbox。

## When to use

用户说"更新日报"、"写日报"、"同步 Dropbox"、"今天到这了"等类似意思时触发。

## Steps

### 1. 确定日期和周目录

- 日期: 当天 NZDT 日期（UTC+13），格式 `YYYY-MM-DD`
- 周目录: `docs/progress_log/week_YYYY-MM-DD`，用该周周一的日期
- 如果周目录不存在则创建

### 2. 回顾当天工作

读取对话上下文，整理当天完成的工作，包括：
- 主要活动和产出
- 新增/修改/删除的文件
- 关键决策和结论
- 待办事项

### 3. 写日报

参考前一天的日报格式，创建 `docs/progress_log/week_YYYY-MM-DD/YYYY-MM-DD.md`。

标准格式：
```markdown
# 工作记录 YYYY-MM-DD (Day)

**工作时间**: 约 X 小时

## 主题标题

### 概述

简要总结...

### 详细内容

...

### 新增/修改文件

| 文件 | 操作 | 说明 |
|------|------|------|

### 待办

- [ ] ...
```

### 4. 同步到 Dropbox

```bash
mkdir -p "/mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/progress_doc/week_YYYY-MM-DD"
cp docs/progress_log/week_YYYY-MM-DD/YYYY-MM-DD.md \
  "/mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/progress_doc/week_YYYY-MM-DD/YYYY-MM-DD.md"
```

Dropbox 路径固定为: `/mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/progress_doc/`
目录结构与本地 `docs/progress_log/` 镜像（按周分目录，文件名 `YYYY-MM-DD.md`）

### 5. 确认

- 确认日报文件已写入 `docs/progress_log/`
- 确认 Dropbox 文件存在且大小正确
- 告知用户完成

## Constraints

- Dropbox 路径不要改动，Windows 侧 Dropbox 客户端会自动云端同步
- 日报内容基于实际完成的工作，不要编造
- 如果当天日报已存在，追加内容而不是覆盖
- 保持和已有日报一致的中文风格
