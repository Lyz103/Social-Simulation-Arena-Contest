> 历史设计参考：本文保留原设计。当前实现、官方字段转换、开放排名与计分约定见 `../../README.md` 和 `../INTEGRATION.md`；下文旧接口和旧验收数量不代表当前协议。

# SSA Prompt Compiler 与 Ensemble Engine 交接说明

本文档用于交接 SSA Harness 中已经实现的两个组件：

1. Forecast Prompt Compiler；
2. Ensemble Engine。

文档对应 2026-09-24 的实现状态。代码只依赖 Python 标准库，要求 Python 3.10 或更高版本。

---

## 1. 交付结论

当前代码已经实现以下能力：

- 支持 Topline、Population Profile、Ranking 三种任务；
- 将 ForecastRequest 和 ForecastContext 编译为稳定 Prompt；
- 为三种任务动态生成 JSON Schema；
- 解析并严格校验模型输出；
- Profile 强制覆盖全部 cell；
- Ranking 强制覆盖全部候选项，并根据 scores 重建顺序；
- 自动生成 persistence 和 recent-mean baseline；
- 对 Topline 和 Profile 使用混合分布公式集成 mean 与 sd；
- 对 Ranking 使用分数归一化和加权排序；
- 支持多个 LLM 成员；
- 支持字典和 dataclass 风格的上游接口；
- 记录 Prompt、Context 和 Ensemble 版本；
- 全部逻辑可离线测试。

本次交付不包含：

- Research / Retrieval Engine；
- Context Builder；
- 具体模型供应商客户端；
- Submission Adapter；
- Backtester；
- 基于历史回测学习权重；
- sd 的历史校准。

这些组件需要由总 Harness 串联。

---

## 2. 文件结构

~~~
Arena_competition/
├── ssa/
│   ├── __init__.py
│   ├── forecast/
│   │   ├── __init__.py
│   │   ├── _compat.py
│   │   ├── models.py
│   │   ├── prompts.py
│   │   ├── schemas.py
│   │   └── validators.py
│   └── ensemble/
│       ├── __init__.py
│       ├── models.py
│       ├── baselines.py
│       └── engine.py
├── tests/
│   ├── test_forecast_prompt.py
│   └── test_ensemble.py
├── context_prompt_ensemble_design.md
├── research_retrieval_engine_design.md
└── PROMPT_ENSEMBLE_HANDOFF.md
~~~

文件职责：

| 文件 | 职责 |
|---|---|
| forecast/prompts.py | 生成 system message、user message、schema 和 Prompt hash |
| forecast/schemas.py | 生成三种任务的 JSON Schema |
| forecast/validators.py | 解析并校验模型输出 |
| forecast/models.py | Prompt 输出对象和异常类型 |
| forecast/_compat.py | 兼容字典与 dataclass 输入，统一任务类型别名 |
| ensemble/baselines.py | 从历史数据生成 persistence 和 recent mean |
| ensemble/engine.py | 执行权重归一化和三种任务的集成 |
| ensemble/models.py | Ensemble 输入输出对象和异常类型 |
| tests/ | 完全离线的单元测试 |

---

## 3. 总体数据流

~~~
ForecastRequest
      +
ForecastContext
      ↓
ForecastPromptCompiler.compile()
      ↓
CompiledPrompt
      ↓
模型供应商客户端
      ↓
原始 JSON 文本或字典
      ↓
PredictionValidator.parse()
      ↓
规范化 LLM prediction
      +
BaselineFactory.build()
      ↓
EnsembleEngine.combine()
      ↓
FinalPrediction
      ↓
Submission Adapter
~~~

组件边界：

- Prompt Compiler 不调用网络，也不调用 LLM；
- Validator 不修改模型的 Topline/Profile 数值；
- Ranking Validator 会依据 scores 重新生成 ranking；
- Ensemble Engine 不读取新闻，不重新执行自然语言判断；
- Baseline Factory 假设历史已经按照时间从旧到新排列；
- 上游必须保证 history 不包含 asof 之后才可获得的真值。

---

## 4. 环境要求

最低要求：

~~~
Python >= 3.10
~~~

没有第三方依赖。

进入工程目录：

~~~
cd /data/songbo/Social_simulation/Arena_competition
~~~

验证导入：

~~~
python3 - <<'PY'
from ssa.forecast import ForecastPromptCompiler, PredictionValidator
from ssa.ensemble import BaselineFactory, EnsembleEngine

print("imports ok")
PY
~~~

如果代码被复制到另一个仓库，需要满足以下任意一种条件：

1. ssa 目录位于运行目录；
2. ssa 已作为包安装；
3. 仓库根目录已加入 PYTHONPATH。

---

## 5. 支持的任务类型

规范任务类型：

| 任务 | target_type |
|---|---|
| Topline | continuous_normal |
| Profile | profile_energy |
| Ranking | ranking_list |

兼容别名：

| 输入别名 | 归一化结果 |
|---|---|
| topline | continuous_normal |
| number | continuous_normal |
| profile | profile_energy |
| ranking | ranking_list |

建议上游正式使用规范名称，别名主要用于降低临时对接成本。

---

## 6. ForecastRequest 接口契约

实现没有强制依赖某个 ForecastRequest 类。request 可以是字典，也可以是具有同名属性的 dataclass。

### 6.1 公共必填字段

| 字段 | 类型 | 含义 |
|---|---|---|
| round_id | str | Round 唯一标识 |
| question | str | 完整预测问题 |
| target_type | str | 三种规范类型之一 |
| asof | str | 信息冻结时间 |
| history | list | 历史真值；自动 baseline 时需要 |

### 6.2 推荐字段

| 字段 | 类型 | 默认行为 |
|---|---|---|
| unit | str | 缺失时 Prompt 使用 unspecified |
| forecast_for | str | 缺失时尝试 deadline，再缺失则使用 next scheduled observation |
| minimum | number | 缺失时不设置数值下界 |
| maximum | number | 缺失时不设置数值上界 |
| deadline | str | forecast_for 的备用字段 |

### 6.3 任务专用字段

Profile：

~~~
cells: list[str]
~~~

Ranking：

~~~
candidates: list[str]
~~~

cells 和 candidates：

- 必须非空；
- 不允许重复；
- ID 会原样进入 Prompt 和 JSON Schema；
- 接收方不要在模型调用后重命名这些 ID。

### 6.4 推荐 request 示例

~~~
request = {
    "round_id": "yougov-2026-w40-approval",
    "question": "What percentage will approve in the next release?",
    "target_type": "continuous_normal",
    "asof": "2026-09-20T00:00:00Z",
    "forecast_for": "2026-09-27",
    "unit": "percentage points",
    "minimum": 0,
    "maximum": 100,
    "history": [
        {"value": 40.0},
        {"value": 41.0},
        {"value": 42.0},
    ],
}
~~~

---

## 7. ForecastContext 接口契约

context 同样可以是字典或 dataclass。

必填字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| rendered_text | str | Context Builder 生成的最终上下文 |

推荐字段：

| 字段 | 类型 | 行为 |
|---|---|---|
| round_id | str | 提供后必须与 request.round_id 相同 |
| asof | str | 提供后必须与 request.asof 完全相同 |
| context_hash | str | 用于 Prompt 归档；缺失时根据 rendered_text 计算 |

示例：

~~~
context = {
    "round_id": "yougov-2026-w40-approval",
    "asof": "2026-09-20T00:00:00Z",
    "rendered_text": """
HISTORICAL BASELINE
Latest value: 42.0
Recent mean: 41.0

RECENT DEVELOPMENTS
No strong evidence of a large change.
""",
    "context_hash": "sha256:...",
}
~~~

对接要求：

- rendered_text 不能为空；
- Context Builder 应提前控制长度；
- Context Builder 应保证没有未来信息；
- Prompt Compiler 只检查 round_id 和 asof 是否一致，不重新审计每条证据时间；
- asof 当前按字符串严格比较，建议全系统统一 UTC ISO 8601 格式。

---

## 8. Prompt Compiler 使用方法

### 8.1 基本调用

~~~
from ssa.forecast import ForecastPromptCompiler

compiler = ForecastPromptCompiler()

compiled = compiler.compile(
    request=request,
    context=context,
)
~~~

CompiledPrompt 字段：

| 字段 | 含义 |
|---|---|
| round_id | Round ID |
| target_type | 归一化任务类型 |
| system_message | 固定系统指令 |
| user_message | 题目、上下文和任务协议 |
| output_schema | 动态 JSON Schema |
| model_params | 默认 temperature=0、top_p=1 |
| prompt_version | 当前为 forecast-prompt-v1 |
| context_hash | 上游 hash 或自动计算值 |
| prompt_hash | 对 Prompt 全部关键字段计算的 SHA-256 |

转换为通用 Chat Messages：

~~~
messages = compiled.messages()
~~~

结果：

~~~
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
]
~~~

### 8.2 自定义模型参数

~~~
compiler = ForecastPromptCompiler(
    model_params={
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 1000,
    }
)
~~~

注意：

- model_params 会进入 prompt_hash；
- 修改参数后应视为不同实验配置；
- 是否支持某个参数由具体模型客户端决定。

### 8.3 自定义 System Prompt

~~~
compiler = ForecastPromptCompiler(
    system_message="Your controlled system instruction..."
)
~~~

自定义后仍会进入 prompt_hash。正式回测中应给自定义版本单独编号，或者继承类并修改 VERSION。

---

## 9. 三类 Prompt 输出契约

### 9.1 Topline

模型必须返回：

~~~
{
  "mean": 41.2,
  "sd": 1.4
}
~~~

规则：

- 只能有 mean 和 sd；
- mean 必须是有限数值；
- mean 必须满足 minimum/maximum；
- sd 必须是有限正数；
- 不允许额外解释字段。

### 9.2 Profile

模型必须返回：

~~~
{
  "cells": {
    "democrat": {
      "mean": 85.0,
      "sd": 2.0
    },
    "independent": {
      "mean": 39.0,
      "sd": 3.0
    },
    "republican": {
      "mean": 8.0,
      "sd": 1.5
    }
  }
}
~~~

规则：

- 每个要求的 cell 恰好出现一次；
- 不允许遗漏 cell；
- 不允许增加 cell；
- 不允许修改 cell ID；
- 每个 cell 必须是包含 mean 和 sd 的对象；
- 每个 cell 独立执行数值范围检查。

### 9.3 Ranking

模型内部输出格式：

~~~
{
  "scores": {
    "A": 60,
    "B": 90,
    "C": 70
  },
  "ranking": ["B", "C", "A"]
}
~~~

规则：

- 每个候选项必须恰好有一个有限 score；
- ranking 必须覆盖完整候选集合；
- 不允许额外候选；
- 分数越大，名次越高；
- Validator 最终依据 scores 重建 ranking；
- 精确同分时按 request.candidates 的原始顺序处理。

正式比赛接口如果只接受数组，由 Submission Adapter 提取：

~~~
submission = prediction["ranking"]
~~~

不要直接把带 scores 的内部对象提交给只接受数组的接口。

---

## 10. 模型客户端对接

本组件不绑定 OpenAI、Anthropic 或其他供应商。推荐适配方式：

~~~
compiled = ForecastPromptCompiler().compile(request, context)

raw_response = model_client.generate(
    messages=compiled.messages(),
    json_schema=compiled.output_schema,
    **compiled.model_params,
)
~~~

如果模型供应商支持 structured output：

- 必须优先把 compiled.output_schema 传给供应商 API；
- 不要只依赖 Prompt 中的自然语言约束；
- 仍然要在返回后运行本地 Validator。

如果供应商不支持 structured output：

- 要求模型只返回 JSON；
- Validator 能接受纯 JSON 字符串；
- Validator 兼容完整的 json Markdown fence；
- 当前实现不从普通说明文字中猜测或截取 JSON。

---

## 11. PredictionValidator

调用：

~~~
from ssa.forecast import PredictionValidator

prediction = PredictionValidator().parse(
    raw_response,
    request,
)
~~~

raw_response 可以是：

- JSON 字符串；
- 一个 Mapping 字典；
- 完整 json Markdown fence。

示例：

~~~
prediction = validator.parse(
    '{"mean": 42.0, "sd": 1.5}',
    request,
)
~~~

返回值会规范化为 Python float：

~~~
{"mean": 42.0, "sd": 1.5}
~~~

非法输出会抛出 InvalidModelOutput。

推荐失败策略：

1. 首次输出非法；
2. 最多调用一次格式修复模型；
3. 修复指令只能调整 JSON 结构，不能重新预测；
4. 再次校验；
5. 仍失败则明确标记该 LLM 成员失败；
6. 正式提交模式可以显式退化到 baseline；
7. 实验模式建议 fail closed。

当前代码不自动发起格式修复调用，因为该逻辑属于模型客户端或 Pipeline。

---

## 12. BaselineFactory

### 12.1 使用方法

~~~
from ssa.ensemble import BaselineFactory

baselines = BaselineFactory(
    recent_window=3,
    min_sd=0.25,
).build(request)
~~~

返回两个 ForecastMember：

~~~
persistence
recent_mean
~~~

### 12.2 Topline 历史格式

支持：

~~~
[40, 41, 42]
~~~

也支持：

~~~
[
    {"value": 40},
    {"value": 41},
    {"value": 42},
]
~~~

还会尝试读取以下键：

~~~
value
truth
result
prediction
~~~

历史必须按时间从旧到新排序。BaselineFactory 本身不解析日期，也不排序。

Persistence：

~~~
mean = 最后一期有效值
~~~

Recent mean：

~~~
mean = 最近 recent_window 个有效值的算术平均
~~~

Baseline sd 使用历史一阶变化的总体标准差：

~~~
deltas = [y_t - y_(t-1)]
sd = max(min_sd, pstdev(deltas))
~~~

数据不足或变化量标准差为零时，使用 min_sd。默认 min_sd 为 0.25。

### 12.3 Profile 历史格式

~~~
[
    {
        "cells": {
            "democrat": 84,
            "independent": 38,
            "republican": 8
        }
    },
    {
        "cells": {
            "democrat": 85,
            "independent": 39,
            "republican": 9
        }
    }
]
~~~

也允许 cell 值使用：

~~~
{"mean": 85}
~~~

要求每个 request.cells 至少有一个有效历史值。缺失时抛出 InvalidEnsembleMember。

### 12.4 Ranking 历史格式

~~~
[
    {"ranking": ["A", "B", "C"]},
    {"ranking": ["A", "C", "B"]}
]
~~~

要求每个被接受的历史 ranking：

- 长度与候选集合相同；
- 没有重复候选；
- 集合与 request.candidates 完全一致。

Persistence 使用最后一期排序。

Recent mean 将每期排序转换为 Borda 分数，然后计算最近若干期的平均分数。

---

## 13. EnsembleEngine 使用方法

### 13.1 自动构造 baseline

~~~
from ssa.ensemble import EnsembleEngine

final = EnsembleEngine().combine(
    request=request,
    llm_predictions=[prediction],
)
~~~

默认权重：

| 成员类型 | 权重 |
|---|---:|
| llm | 0.60 |
| persistence | 0.25 |
| recent_mean | 0.15 |

如果传入多个 LLM，它们平均分享 0.60 的类型权重。

例如两个 LLM：

~~~
llm-1 = 0.30
llm-2 = 0.30
persistence = 0.25
recent_mean = 0.15
~~~

### 13.2 显式提供 baseline

如果上游已经计算好 baseline：

~~~
from ssa.ensemble import EnsembleEngine, ForecastMember

baselines = [
    ForecastMember(
        member_id="persistence",
        member_type="persistence",
        prediction={"mean": 41.0, "sd": 1.2},
    ),
    ForecastMember(
        member_id="recent_mean",
        member_type="recent_mean",
        prediction={"mean": 41.5, "sd": 1.0},
    ),
]

final = EnsembleEngine().combine(
    request=request,
    llm_predictions=[prediction],
    baselines=baselines,
)
~~~

显式传入 baselines 后，Engine 不再读取 history 自动生成 baseline。

如果希望只运行 LLM：

~~~
final = engine.combine(
    request=request,
    llm_predictions=[prediction],
    baselines=[],
)
~~~

此时有效权重会重新归一化，LLM 权重变为 1。

### 13.3 自定义权重

~~~
engine = EnsembleEngine(
    weights={
        "llm": 0.50,
        "persistence": 0.30,
        "recent_mean": 0.20,
    }
)
~~~

规则：

- 权重必须有限且非负；
- 活跃成员权重之和必须大于零；
- Engine 会对活跃成员重新归一化；
- 未配置类型默认权重为零；
- 显式传入 weights={} 会报 InvalidEnsembleWeights；
- 每个 ForecastMember 可以用 weight 字段覆盖类型权重。

member_id 必须全局唯一。

---

## 14. Topline 和 Profile 的集成数学

对成员 i：

~~~
weight = w_i
mean = μ_i
sd = σ_i
~~~

最终均值：

~~~
μ = Σ w_i μ_i
~~~

使用混合分布的二阶矩计算方差：

~~~
Var(Y) = Σ w_i (σ_i² + μ_i²) - μ²
~~~

因此最终 sd 同时包含：

1. 每个成员自己的不确定性；
2. 不同成员均值之间的分歧。

这不是对 sd 做简单加权平均。

最终方差至少为：

~~~
min_sd²
~~~

Engine 默认 min_sd 为 0.05，它是最终数值安全下限。BaselineFactory 默认 min_sd 为 0.25，两者已经解耦。

可选最大 sd：

~~~
engine = EnsembleEngine(
    max_sd=10.0
)
~~~

发生最大值裁剪时会写入 final.warnings。

mean 超出 request.minimum/maximum 时会被裁剪，并记录 warning。

Profile 对每个 cell 使用同一套成员权重和混合公式。当前不实现总体与子群之间的约束投影。

---

## 15. Ranking 集成方法

每个成员必须提供 scores。每个成员的分数先独立执行 min-max 归一化：

~~~
normalized_score =
    (score - min_score) / (max_score - min_score)
~~~

如果一个成员给所有候选完全相同的分数，该成员的所有归一化分数设为 0.5。

最终分数：

~~~
final_score[item] =
    Σ member_weight × normalized_member_score[item]
~~~

最终 ranking 按 final_score 降序排列。

精确平局时：

1. 优先采用 persistence ranking 的相对顺序；
2. 没有 persistence 时使用 request.candidates 的原始顺序。

这种方法保证结果确定、可复现。当前 scores 是内部相对强度，不解释为概率。

---

## 16. LLM prediction 的兼容输入

llm_predictions 中每一项支持以下形式。

直接预测字典：

~~~
{"mean": 42, "sd": 1.5}
~~~

带元数据的字典：

~~~
{
    "model_id": "model-a",
    "prediction": {
        "mean": 42,
        "sd": 1.5
    }
}
~~~

使用 parsed 字段：

~~~
{
    "model_id": "model-a",
    "parsed": {
        "mean": 42,
        "sd": 1.5
    }
}
~~~

具有 prediction 或 parsed 属性的 dataclass。

也可以直接传 ForecastMember。

普通字典未提供 model_id 时，自动命名为：

~~~
llm-1
llm-2
...
~~~

---

## 17. FinalPrediction

EnsembleEngine.combine 返回 FinalPrediction。

字段：

| 字段 | 含义 |
|---|---|
| round_id | Round ID |
| target_type | 规范任务类型 |
| prediction | 最终可交给 Submission Adapter 的内部预测 |
| members | 全部参与集成的成员 |
| weights | 实际使用的归一化权重 |
| ensemble_version | 当前为 ensemble-v1 |
| warnings | 裁剪等警告 |
| degraded | 没有 LLM 成员时为 True |

转换为 JSON 时可以使用：

~~~
from dataclasses import asdict
import json

payload = asdict(final)
text = json.dumps(
    payload,
    ensure_ascii=False,
    sort_keys=True,
)
~~~

Submission Adapter 通常只提交：

~~~
final.prediction
~~~

---

## 18. 三种任务的端到端示例

### 18.1 Topline

~~~
from ssa.forecast import ForecastPromptCompiler, PredictionValidator
from ssa.ensemble import EnsembleEngine

request = {
    "round_id": "demo-topline",
    "question": "下周支持率是多少？",
    "target_type": "continuous_normal",
    "unit": "percent",
    "asof": "2026-09-20T00:00:00Z",
    "forecast_for": "2026-09-27",
    "minimum": 0,
    "maximum": 100,
    "history": [40, 41, 42],
}

context = {
    "round_id": "demo-topline",
    "asof": "2026-09-20T00:00:00Z",
    "rendered_text": "Latest value: 42. Recent mean: 41.",
}

compiled = ForecastPromptCompiler().compile(request, context)

# 用真实 model_client 调用 compiled.messages()。
raw_response = '{"mean": 43.0, "sd": 1.5}'

prediction = PredictionValidator().parse(raw_response, request)
final = EnsembleEngine().combine(request, [prediction])

print(final.prediction)
~~~

### 18.2 Profile

~~~
request = {
    "round_id": "demo-profile",
    "question": "预测各群体支持率",
    "target_type": "profile_energy",
    "unit": "percent",
    "asof": "2026-09-20T00:00:00Z",
    "forecast_for": "2026-09-27",
    "minimum": 0,
    "maximum": 100,
    "cells": ["democrat", "independent", "republican"],
    "history": [
        {
            "cells": {
                "democrat": 84,
                "independent": 38,
                "republican": 8,
            }
        },
        {
            "cells": {
                "democrat": 85,
                "independent": 39,
                "republican": 9,
            }
        },
    ],
}

raw_response = {
    "cells": {
        "democrat": {"mean": 86, "sd": 2},
        "independent": {"mean": 40, "sd": 3},
        "republican": {"mean": 9, "sd": 1.5},
    }
}

prediction = PredictionValidator().parse(raw_response, request)
final = EnsembleEngine().combine(request, [prediction])
~~~

### 18.3 Ranking

~~~
request = {
    "round_id": "demo-ranking",
    "question": "预测下周完整排名",
    "target_type": "ranking_list",
    "asof": "2026-09-20T00:00:00Z",
    "forecast_for": "2026-09-27",
    "candidates": ["A", "B", "C"],
    "history": [
        {"ranking": ["A", "B", "C"]},
        {"ranking": ["A", "C", "B"]},
    ],
}

raw_response = {
    "scores": {
        "A": 60,
        "B": 90,
        "C": 70,
    },
    "ranking": ["A", "B", "C"],
}

prediction = PredictionValidator().parse(raw_response, request)

# prediction["ranking"] 会被重建为 B, C, A。
final = EnsembleEngine().combine(request, [prediction])
~~~

历史 baseline 可能使最终 ensemble ranking 与单独 LLM ranking 不同，这是预期行为。

---

## 19. 异常类型

Prompt 包公开导出：

~~~
PromptCompilationError
UnsupportedTargetType
InvalidModelOutput
~~~

Ensemble 包公开导出：

~~~
EnsembleError
InvalidEnsembleMember
InvalidEnsembleWeights
~~~

典型错误：

| 错误 | 异常 |
|---|---|
| request/context round 不一致 | PromptCompilationError |
| request/context asof 不一致 | PromptCompilationError |
| context 为空 | PromptCompilationError |
| 不支持的 target_type | UnsupportedTargetType |
| JSON 非法 | InvalidModelOutput |
| Profile cell 不完整 | InvalidModelOutput |
| Ranking 候选不完整 | InvalidModelOutput |
| 自动 baseline 没有可用历史 | InvalidEnsembleMember |
| member_id 重复 | InvalidEnsembleMember |
| 活跃权重总和为零 | InvalidEnsembleWeights |

建议 Pipeline 捕获具体业务异常，不要捕获并吞掉所有 Exception。

---

## 20. 测试方法

完整测试：

~~~
cd /data/songbo/Social_simulation/Arena_competition
python3 -m unittest discover -s tests -v
~~~

当前预期：

~~~
Ran 15 tests
OK
~~~

语法检查：

~~~
python3 -m compileall -q ssa tests
~~~

单独运行 Prompt 测试：

~~~
python3 -m unittest tests.test_forecast_prompt -v
~~~

单独运行 Ensemble 测试：

~~~
python3 -m unittest tests.test_ensemble -v
~~~

测试覆盖：

- Prompt 确定性；
- round/asof 不匹配；
- Profile 动态 schema；
- Profile 缺失 cell；
- Profile cell 类型错误；
- Ranking 根据 scores 重建；
- JSON fence；
- 额外输出字段；
- Topline 加权均值和混合方差；
- 多 LLM 权重拆分；
- Profile 逐 cell 集成；
- Ranking 集成；
- 显式 baseline 适配；
- 常见 history 格式；
- baseline sd 下限；
- 重复 Ranking 历史拒绝；
- 空权重配置拒绝。

---

## 21. 本次交付前代码审查结果

检查项目：

| 检查项 | 结果 |
|---|---|
| Python 语法编译 | 通过 |
| 模块导入 | 通过 |
| 15 个离线测试 | 全部通过 |
| 三类任务 Prompt | 已覆盖 |
| 三类任务 Validator | 已覆盖 |
| 三类任务 Ensemble | 已覆盖 |
| 字典/dataclass 兼容 | 已实现 |
| JSON Schema | 已实现 |
| Prompt hash | 已实现 |
| 权重归一化 | 已实现 |
| 非有限数值拒绝 | 已实现 |
| Profile 完整性 | 已实现 |
| Ranking 完整性 | 已实现 |
| 排名确定性 tie-break | 已实现 |

本轮审查修复：

1. Profile cell 不是对象时，现在抛出 InvalidModelOutput，而不是底层 TypeError；
2. Ranking 历史现在同时检查长度、集合和唯一性；
3. 显式 weights={} 不再意外退回默认权重，而是按预期报错；
4. Baseline 默认最小 sd 恢复为 0.25，不再被 Ensemble 最终数值下限 0.05 覆盖；
5. 补充了相应回归测试；
6. 补齐公共异常类的包级导出。

---

## 22. 已知限制

以下不是代码错误，但接收方必须知道。

### 22.1 不负责时间泄漏过滤

Prompt Compiler 信任 Context Builder。它只检查 request.asof 与 context.asof 一致，不逐条检查证据时间。

### 22.2 history 默认已经排序

BaselineFactory 不按日期排序。上游必须确保 history 从旧到新。

### 22.3 baseline sd 尚未校准

当前 sd 是工程 MVP：

- baseline 使用历史一阶差分波动；
- ensemble 使用混合分布方差；
- 没有根据历史 CRPS 或覆盖率学习缩放系数。

正式比赛前应由 Backtester 校准。

### 22.4 固定权重尚未经过回测

0.60/0.25/0.15 是可解释的初始配置，不是已证明最优的权重。

### 22.5 Profile 没有跨 cell 约束

当前只保证 cell 完整和逐 cell 数值合法。若任务存在加权总体、和为 100 或其他约束，需要额外 Constraint Projector。

### 22.6 Ranking score 不是概率

最终 scores 只是归一化加权分数，适用于排序，不应解释为事件概率。

### 22.7 没有模型 API 实现

本组件提供 messages、schema 和 model_params。超时、重试、限流、供应商响应解析属于模型客户端。

### 22.8 没有自动格式修复

Validator 会拒绝非法输出，但不会自动再次调用模型。

### 22.9 没有自动归档

CompiledPrompt 和 FinalPrediction 已包含必要字段，但写入磁盘应由 Pipeline Archive 完成。

---

## 23. 与前两个组件的最终对接步骤

收到 Research Engine 和 Context Builder 后，建议按以下顺序对接。

### 第一步：确认 request 字段

确认至少存在：

~~~
round_id
question
target_type
asof
history
~~~

并确认 Profile/Ranking 专用字段名称分别是 cells 和 candidates。

如果上游字段不同，在 Pipeline 边界做一次映射，不建议修改 Prompt Compiler 内部。

### 第二步：确认 context 字段

确认 Context Builder 输出：

~~~
rendered_text
round_id
asof
context_hash
~~~

其中 context_hash 推荐使用带 sha256: 前缀的字符串，但当前代码不强制前缀。

### 第三步：接模型客户端

模型客户端读取：

~~~
compiled.messages()
compiled.output_schema
compiled.model_params
~~~

模型返回值交给：

~~~
PredictionValidator().parse(...)
~~~

### 第四步：确认 history schema

如果 history 不符合本文支持格式：

- 优先在 ForecastRequest.from_round 中归一化；
- 或扩展 BaselineFactory._row_value；
- 新增适配后必须补单元测试。

### 第五步：接 Submission Adapter

Topline/Profile 可以直接使用 final.prediction。

Ranking 根据官方协议通常只提交：

~~~
final.prediction["ranking"]
~~~

### 第六步：增加运行归档

至少保存：

~~~
request
research bundle hash
context hash
compiled prompt
raw model response
validated model prediction
baseline members
final prediction
actual weights
component versions
warnings
~~~

---

## 24. 接收方验收清单

接收方拿到代码后应执行：

- [ ] Python 版本不低于 3.10；
- [ ] 能从仓库根目录 import ssa.forecast；
- [ ] 能 import ssa.ensemble；
- [ ] compileall 通过；
- [ ] 15 个离线测试通过；
- [ ] 用一个真实 Topline request 完成 Prompt 编译；
- [ ] 将 schema 传给真实模型 API；
- [ ] 模型结果通过 Validator；
- [ ] history 能生成两个 baseline；
- [ ] Ensemble 输出权重和最终结果；
- [ ] Profile 所有真实 cell 均能通过；
- [ ] Ranking 官方候选集合能完整输出；
- [ ] Submission Adapter 的格式与比赛接口一致；
- [ ] 运行产物已经归档；
- [ ] 明确配置权重和组件版本。

---

## 25. 推荐后续工作

优先级从高到低：

1. 与真实 ForecastRequest 和 ForecastContext 数据结构对接；
2. 增加模型客户端 structured output；
3. 增加端到端 Pipeline 测试；
4. 用历史 round 运行 Backtester；
5. 比较 LLM only、baseline only 和 ensemble；
6. 使用 CRPS/覆盖率校准 Topline/Profile sd；
7. 使用 NDCG/Kendall tau 优化 Ranking 权重；
8. 根据 domain 决定是否使用不同权重；
9. 增加运行 archive 和 pipeline_config_hash；
10. 有明确业务约束时增加 Profile Constraint Projector。

---

## 26. 交付时建议复制的文件

必须复制：

~~~
ssa/__init__.py
ssa/forecast/
ssa/ensemble/
tests/test_forecast_prompt.py
tests/test_ensemble.py
PROMPT_ENSEMBLE_HANDOFF.md
~~~

建议同时复制：

~~~
context_prompt_ensemble_design.md
research_retrieval_engine_design.md
~~~

不要复制：

~~~
__pycache__/
*.pyc
~~~

---

## 27. 最小生产调用骨架

~~~
from ssa.forecast import ForecastPromptCompiler, PredictionValidator
from ssa.ensemble import EnsembleEngine

def forecast_round(request, context, model_client):
    compiled = ForecastPromptCompiler().compile(
        request=request,
        context=context,
    )

    raw_response = model_client.generate(
        messages=compiled.messages(),
        output_schema=compiled.output_schema,
        **compiled.model_params,
    )

    llm_prediction = PredictionValidator().parse(
        raw_response,
        request,
    )

    final = EnsembleEngine().combine(
        request=request,
        llm_predictions=[llm_prediction],
    )

    return {
        "compiled_prompt": compiled,
        "raw_response": raw_response,
        "llm_prediction": llm_prediction,
        "final_prediction": final,
    }
~~~

该函数只展示组件连接方式。生产环境还需要：

- 超时和有限重试；
- 格式修复策略；
- 明确降级；
- archive；
- 日志；
- 成本记录；
- submission；
- 回测指标。

---

## 28. 最终说明

本实现是一套可运行的 MVP，而不是已经完成回测优化的最终预测策略。

其主要价值是建立稳定工程契约：

- 上游证据如何进入 Prompt；
- 模型必须输出什么；
- 非法输出如何被拒绝；
- 简单基线如何生成；
- 三类任务如何统一集成；
- 所有结果如何保持确定性和可审计性。

后续 Prompt、权重、sd 和排名算法都可以替换，但建议保持当前公共接口稳定，并通过版本号、hash 和滚动回测管理变化。

