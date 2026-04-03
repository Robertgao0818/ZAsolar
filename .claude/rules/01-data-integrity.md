---
paths:
  - "tiles/**"
  - "results/**"
  - "checkpoints/**"
  - "data/coco*/**"
---

# Data Integrity Rules

## 数据目录禁放源码
`tiles/`、`results/`、`checkpoints/`、`data/coco*/` 仅存放运行时数据/产物。禁止放置 Python 脚本或其他源码。源码应放在 `core/`、`scripts/<domain>/`、或项目根目录。

## 禁止提交二进制大文件
永远不要将 `.tif`、`.vrt`、`.jp2`、`.pth` 文件或上述数据目录提交到 git。详见 `docs/governance/repo-rules.md`。

## config.json 是参数事实源
`results/<GridID>/config.json` 记录检测/评估运行参数。脚本仅在 config.json 匹配当前配置时复用旧结果。修改检测参数后必须使用 `--force` 重新运行。
