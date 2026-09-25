# 2026-09-24 目录迁移

原项目完整保留于 `archive/legacy/`。未删除原数据库、wheel、示例或原始设计文件。

| 原路径 | 当前唯一维护路径 |
|---|---|
| `harness_retrieval/retrieval/` | `src/ssa_harness/retrieval/` |
| `Arena_competition/ssa/forecast/` | `src/ssa_harness/prompt/` |
| `Arena_competition/ssa/ensemble/` | `src/ssa_harness/ensemble/` |
| 两个旧 `tests/` | 根目录 `tests/` |
| `Arena_competition/*.md` | `docs/design/`，并保留原件 |
| 无 Context 实现 | `src/ssa_harness/context/` |

导入变更：

```python
from ssa_harness.retrieval import ResearchEngine
from ssa_harness.context import ContextBuilder
from ssa_harness.prompt import ForecastPromptCompiler
from ssa_harness.ensemble import EnsembleEngine
```

检索 CLI 从 `python -m retrieval` 改为 `python -m ssa_harness.retrieval` 或安装后的 `ssa-retrieval`。全流程使用 `python -m ssa_harness` 或 `ssa-harness`。

旧归档库保留但不自动迁移为新实验结果。修改后的 retrieval version 和完整 pipeline identity 参与 hash；请用新的 entrant/数据库开始新实验，而不是覆盖旧证据。

`docs/design/` 为原设计参考，其中已经加版本提示；当前实现约定以 `README.md`、`docs/INTEGRATION.md` 和代码测试为准。原方案 `Social Simulation Arena.md` 保留，增加当前实现索引。

`scripts/migrate_layout.py` 是一次性迁移脚本，检测到旧归档已经存在时拒绝再次执行。日常工作不需运行它。
