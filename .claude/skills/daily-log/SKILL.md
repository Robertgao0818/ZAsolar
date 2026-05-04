# daily-log Skill

写日报，从 Claude/Codex 对话记录反推工时与时间块，同步到 Dropbox，并按需填 RA working-log Excel。

整合自：
- `~/.hermes/skills/productivity/ra-working-log-from-agent-records/SKILL.md`（agent-records → 工时/时间块 / RA log）
- `~/.hermes/skills/productivity/excel-hours-log-from-logs/SKILL.md`（Excel 写入 + Windows 锁处理）

## When to use

用户说"更新日报 / 写日报 / 同步 Dropbox / 今天到这了"或"按对话记录补工时 / 填 RA working log / 填 timesheet"等触发。

## Inputs to discover

动笔前先确认：

- 目标日期（默认当天，时区 `Pacific/Auckland`，自动 NZST/NZDT 切换；不要硬编码 UTC+13）
- 是否需要同步到 Dropbox（默认是）
- 是否需要写 RA working-log Excel（默认否；用户提到"工时表 / RA log / timesheet"才做）
- 当周周一日期 → 周目录 `docs/progress_log/week_YYYY-MM-DD/`

## End-to-end workflow

### 1. 确定日期与周目录

- 用 Python `zoneinfo.ZoneInfo('Pacific/Auckland')` 取当前本地日期；不要假设 UTC+13/UTC+12
- 周目录 = 该周周一日期，格式 `docs/progress_log/week_YYYY-MM-DD/`
- 不存在则创建

### 2. 抽取 Claude / Codex 对话记录

日志位置（WSL 本地）：

- Claude: `~/.claude/projects/-home-gaosh-projects-ZAsolar/*.jsonl`
- Codex: `~/.codex/sessions/YYYY/MM/DD/*.jsonl`（注意：Codex 按 UTC 日期分目录，跨日要包含相邻 UTC 目录后再按本地日期过滤）
- Hermes（如果用过）: `~/.hermes/sessions/session_*.json`

解析骨架（按需调整字段名）：

```python
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo('Pacific/Auckland')
TARGET = 'YYYY-MM-DD'
GAP = timedelta(minutes=45)

@dataclass
class Event:
    dt: datetime
    source: str  # claude / codex / hermes
    path: str
    text: str = ''

def parse_dt(value):
    if not value:
        return None
    if isinstance(value, (int, float)):
        if value > 10_000_000_000:
            value /= 1000
        return datetime.fromtimestamp(value, tz=ZoneInfo('UTC')).astimezone(TZ)
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(TZ)
    return None

def read_jsonl(path, source):
    events = []
    for line in Path(path).read_text(errors='ignore').splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        dt = next((parse_dt(v) for v in
                   (obj.get('timestamp'), obj.get('ts'), obj.get('time'))
                   if parse_dt(v)), None)
        if dt and dt.date().isoformat() == TARGET:
            events.append(Event(dt=dt, source=source, path=str(path)))
    return events

def cluster(events, gap=GAP):
    events = sorted(events, key=lambda e: e.dt)
    out, cur = [], []
    for e in events:
        if cur and e.dt - cur[-1].dt > gap:
            out.append(cur); cur = []
        cur.append(e)
    if cur: out.append(cur)
    return out
```

每个 cluster 记录：起止本地时间、tools（claude/codex/hermes）、关键 prompt、证据 session 路径。

### 3. 合并多源 cluster，估算工时

- 把 Claude / Codex / Hermes 的事件 **合在一条时间线** 上 cluster，不要按工具分别求和——并行 session 会重复计入
- gap 阈值默认 45 分钟；超过则切新 block
- 长时间 idle 不计入工时（即使 session 文件挂着）
- agent autonomous runtime ≠ 人在岗时间；只算"合理监督/审查"的部分
- 如果用户提到当天有 offline 工作（线下阅读、QGIS 标注、纸面会议等），作为"已声明假设"另起一块，不混进 agent 时间块

### 4. 写日报到 `docs/progress_log/week_YYYY-MM-DD/YYYY-MM-DD.md`

格式参考最近几天（如 `2026-04-26.md` / `2026-04-28.md`），头部三行是核心：

```markdown
# 工作记录 YYYY-MM-DD (Day)

**工作时间**: 约 X.X 小时（按 Claude/Codex 对话记录校准；说明合并/idle 处理口径）
**对话记录时间段**: Claude HH:MM-HH:MM、HH:MM-HH:MM；Codex HH:MM-HH:MM。说明并行/串行情况。
**证据来源**: `~/.claude/projects/-home-gaosh-projects-ZAsolar/**/*.jsonl` 与 `~/.codex/sessions/YYYY/MM/{D-1,D,D+1}/*.jsonl`，按 Pacific/Auckland 本地日期过滤。

## 概述

一句话点出当天的主线 / 战略结论。

## 工作完成

- 具体任务 1（产出 / 结论 / 当前状态）
- 具体任务 2 ...

## 时间块

- HH:MM-HH:MM NZST/NZDT - 该时段做的事；证据：Claude session / Codex session。
- HH:MM-HH:MM NZST/NZDT - ...

## Notes / uncertainty

- 工时合并口径、agent autonomous 时间是否计入、log 缺口、offline 假设等
```

如果当天日报已存在：**追加 / 修订**，不覆盖。

如果是周一并且要起新周目录，先 `mkdir -p`。

### 5. 同步到 Dropbox

```bash
mkdir -p "/mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/progress_doc/week_YYYY-MM-DD"
cp docs/progress_log/week_YYYY-MM-DD/YYYY-MM-DD.md \
   "/mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/progress_doc/week_YYYY-MM-DD/YYYY-MM-DD.md"
```

Dropbox 路径固定；目录结构与 `docs/progress_log/` 镜像。

### 6. （可选）填 RA working-log Excel

仅当用户要求填工时表时执行。

工作簿位置：`/mnt/c/Users/gaosh/Dropbox/RA_Solar/RA_Working_Hours_Log.xlsx`

步骤：

1. **先读邻近行的格式** 再写，避免风格不一致
   - sheet 名、日期列/行、时间段写法（如 `09:40–12:10; 14:05–15:35 (NZST)`）、活动简述风格、月度汇总行位置
2. **检测锁** - 同目录有无 `~$RA_Working_Hours_Log.xlsx`；保存遇 `PermissionError` 也算锁
3. 用项目 venv 装 / 调用 openpyxl：`./.venv/bin/python -m pip install openpyxl`（如果 venv 无）
4. 写入：
   - `Hour` / 总工时：合并 cluster 后的总和；可向附近行的取整惯例靠拢（3.4→3.5、3.6→4.0），但不要无证据地虚增
   - `Time period`：本地时间段，带时区标签
   - `Activities`：一句话项目/产出语言（不是"和 Claude 聊天"），与邻行一致
   - 如有月度汇总行，同步更新该列（day 1 在 B 列时，列号 = `day + 1`）
5. 锁住时：保留原文件不动，存到 `RA_Working_Hours_Log_filled_YYYY-MM-DD.xlsx`，告诉用户原文件被占用 + 副本路径
6. 写完 **重新打开校验**：用 openpyxl 读回目标单元格确认值正确

### 7. 确认收尾

- `docs/progress_log/week_YYYY-MM-DD/YYYY-MM-DD.md` 已写
- Dropbox 副本存在且大小一致（`stat` 或 `wc -c` 比对）
- 如果填了 RA log：报告最终保存路径（原文件 or 副本）以及读回校验结果
- 一句话告知用户完成 + 关键数字（总工时、时间段数）

## Integrity rules

- 不编造对话记录里没有、用户也未声明的工作
- Claude / Codex / Hermes 时间线先合并再求和，不要分别累加
- 长 idle gap 必须修剪
- 时区标签保留（NZST / NZDT），由 zoneinfo 自动判定 DST
- 取整可以保守靠拢邻行惯例，不可向上虚增
- offline 工作只在用户明确声明时记入，并标注为假设
- 区分 "agent 自主运行时间" 与 "人 RA 在岗时间"
- 如果日志证据不足，**说明不确定性**，让用户补充而非脑补

## Verification checklist

最后回复前自查：

- [ ] 日期、时区、周目录正确
- [ ] Claude + Codex 都搜过；跨 UTC 日期已包含相邻 UTC 目录后过滤
- [ ] cluster 已合并多源，剔除 idle gap
- [ ] 日报头部三行（工作时间 / 对话记录时间段 / 证据来源）齐全
- [ ] 时间块逐条带证据（session 路径或 ID）
- [ ] Dropbox 已同步且大小一致
- [ ] （如填 Excel）邻行格式已对照、读回校验通过；锁住情况下用副本并明示

## Common pitfalls

1. **分别累加 Claude 与 Codex 时长** - 并行工作会双倍计入。先合并时间线
2. **把 session 文件存活期当作工时** - 必须按 event cluster 修剪
3. **忘记时区** - Codex 时间戳常是 UTC `Z`；本地工作日跨 UTC 目录边界
4. **不看邻行就动 Excel** - 风格/取整不一致
5. **从 WSL 写打开中的 Excel** - 必转副本路径
6. **RA log 描述太啰嗦** - 工时表要项目/产出语言一句话；细节留在日报
7. **只看一个来源** - Claude / Codex / Hermes / progress log 各有覆盖盲区
8. **Dropbox 路径写错** - 固定为 `/mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/progress_doc/`，不要改
