> 历史设计参考：本文保留原设计。当前实现、官方字段转换、开放排名与计分约定见 `../../README.md` 和 `../INTEGRATION.md`；下文旧接口和旧验收数量不代表当前协议。

# SSA Harness：Context Builder、Forecast Prompt Compiler 与 Ensemble Engine 设计方案

本文档为 Social Simulation Arena（SSA）Harness 中以下三个组件的实现设计：

1. **Context Builder**：把题目、历史序列和 Research Bundle 整理成低噪声、结构稳定的预测上下文；
2. **Forecast Prompt Compiler**：把预测上下文编译成固定协议、固定输出契约的模型请求；
3. **Ensemble Engine**：将 LLM 预测、持久性基线和简单统计基线组合为最终预测。

本文档与 `research_retrieval_engine_design.md` 配套使用。Retrieval Engine 解决“模型能够看到哪些证据”，本文档解决：

```text
如何整理这些证据
        +
如何稳定地要求模型预测
        +
如何组合和校准多个预测结果
```

第一版目标不是构造复杂的多 Agent 系统，而是建立一条**可运行、可验证、可回放、可回测**的最小预测链路。

---

## 1. 系统位置与职责边界

完整流程如下：

```text
Round metadata + Historical truth
                ↓
Research / Retrieval Engine
                ↓
          ResearchBundle
                ↓
          Context Builder
                ↓
          ForecastContext
                ↓
     Forecast Prompt Compiler
                ↓
           CompiledPrompt
                ↓
          LLM Forecaster
                ↓
          ModelPrediction
                ↓
          Ensemble Engine
                ↓
          FinalPrediction
                ↓
          Submission Adapter
```

三个组件的职责必须分开：

| 组件 | 负责 | 不负责 |
|---|---|---|
| Context Builder | 清洗、分组、截断、计算历史统计量 | 搜索网页、生成最终预测 |
| Prompt Compiler | 生成模型消息、JSON Schema 和预测协议 | 修改证据、组合多个预测 |
| Ensemble Engine | 组合预测、估计最终不确定性 | 搜索资料、自由生成解释 |

Forecasting LLM 不应自行再次搜索网页；Ensemble Engine 也不应读取原始新闻并进行第二次自然语言判断。

---

## 2. 三类任务的统一表示

SSA 当前主要包括三类任务：

```text
continuous_normal   → 单个数值的 mean + sd
profile_energy      → 多个 cell 的 mean + sd
ranking_list        → 候选项的完整顺序
```

建议统一定义请求对象：

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ForecastRequest:
    round_id: str
    question: str
    target_type: str
    board_id: str
    domain: str
    unit: str
    asof: str
    forecast_for: str
    history: list[dict[str, Any]]
    minimum: float | None = None
    maximum: float | None = None
    cells: list[str] | None = None
    candidates: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

字段要求：

- `asof`：信息冻结时间；
- `forecast_for`：被预测观测值对应的时间或窗口；
- `history`：只能包含 `asof` 前已经公布的真实值；
- `cells`：Profile 必须输出的精确 cell ID；
- `candidates`：Ranking 必须完整排序的候选集合；
- `minimum`、`maximum`：数值合法范围。

入口处必须验证请求：

```python
def validate_request(request: ForecastRequest) -> None:
    if not request.round_id or not request.question:
        raise InvalidForecastRequest("missing round identity")

    if request.target_type == "profile_energy" and not request.cells:
        raise InvalidForecastRequest("profile task requires cells")

    if request.target_type == "ranking_list" and not request.candidates:
        raise InvalidForecastRequest("ranking task requires candidates")
```

---

# Part A：Context Builder

## 3. Context Builder 的目标

Research Bundle 中可能包含重复表达、互相冲突的报道、不同来源的日期和较长 snippet。历史序列也可能包含缺失值、不规则间隔和多个 cell。

Context Builder 负责把这些输入转换成模型容易使用的固定结构：

```text
A. Task definition
B. Historical baseline
C. Recent numerical history
D. Recent developments
E. Related indicators
F. Conflicting evidence
G. Data-quality warnings
```

它的目标不是“尽可能塞入更多信息”，而是：

1. 保留影响预测的核心信息；
2. 降低上下文噪声；
3. 保证相同输入产生相同输出；
4. 保留证据来源和时间；
5. 严格限制 token 和文档数量；
6. 不在整理过程中偷偷形成新的预测结论。

---

## 4. Context Builder 数据结构

### 4.1 `HistorySummary`

```python
@dataclass(frozen=True)
class HistorySummary:
    observation_count: int
    latest_value: Any
    recent_mean: Any
    long_term_mean: Any
    recent_change: Any
    recent_volatility: Any
    minimum: Any
    maximum: Any
    recent_rows: list[dict[str, Any]]
    warnings: list[str]
```

对于 Profile，`Any` 可以是 `{cell_id: value}`；对于 Ranking，可以保存历史名次、历史分数或出现频率。

### 4.2 `EvidenceItem`

```python
@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    category: str
    title: str
    text: str
    source: str
    url: str
    published_at: str
    relevance_score: float
    temporal_score: float
    stance: str = "unknown"
```

`stance` 只表示相对于目标变化方向的粗分类：

```text
up
down
neutral
mixed
unknown
```

第一版不建议让规则系统强制推断 stance；可以全部保留为 `unknown`，或者只对明确的结构化变化赋值。

### 4.3 `ForecastContext`

```python
@dataclass(frozen=True)
class ForecastContext:
    round_id: str
    asof: str
    task_text: str
    history_summary: HistorySummary
    long_term_context: list[EvidenceItem]
    recent_developments: list[EvidenceItem]
    related_indicators: list[EvidenceItem]
    conflicting_evidence: list[EvidenceItem]
    warnings: list[str]
    rendered_text: str
    context_hash: str
    builder_version: str
```

Forecast Prompt Compiler 主要使用 `rendered_text`，但归档必须保存结构化字段。

---

## 5. 历史序列处理

### 5.1 时间过滤

Context Builder 必须再次防御性检查历史数据：

```python
rows = [row for row in request.history if row["available_at"] <= request.asof]
```

建议区分：

- `observed_at`：被测量对象对应的时间；
- `published_at`：结果的发布日期；
- `available_at`：系统确认可以获得该值的时间。

用于判断泄漏的应该是 `available_at <= asof`。

### 5.2 Topline 统计量

MVP 至少计算：

```text
latest_value
mean(last 3)
mean(last 10)
latest_value - previous_value
standard deviation(last 10)
min/max(last 10)
```

示例：

```python
def summarize_numeric(values: list[float]) -> dict:
    recent3 = values[-3:]
    recent10 = values[-10:]

    return {
        "latest_value": values[-1],
        "recent_mean": mean(recent3),
        "long_term_mean": mean(recent10),
        "recent_change": values[-1] - values[-2] if len(values) >= 2 else 0.0,
        "recent_volatility": pstdev(recent10) if len(recent10) >= 2 else None,
        "minimum": min(recent10),
        "maximum": max(recent10),
    }
```

统计窗口必须写入配置和归档，不能在不同 round 中隐式改变。

### 5.3 Profile 统计量

Profile 应该对每个 cell 独立计算同样的历史统计量，同时检查：

- 是否缺少某个 cell；
- cell ID 是否发生变化；
- 同一期是否存在重复值；
- 数值是否超出合法范围；
- 是否存在题目定义的总体—子群约束。

MVP 不应自行填补长期缺失的 cell。单点缺失可以保留 warning，并让基线使用该 cell 最近一次有效值。

### 5.4 Ranking 统计量

若有历史完整排名，建议计算：

```text
last_rank[item]
mean_rank[item]
top_k_frequency[item]
appearance_frequency[item]
rank_momentum[item]
```

为方便 Ensemble，名次可以转换为简单分数：

```python
score = candidate_count - rank + 1
```

如果有原始 pageviews、搜索指数等连续量，应优先保留连续值，而不是只保留名次。

---

## 6. 证据分类与选择

### 6.1 分类原则

第一版建议使用确定性规则，而不是额外调用 LLM 摘要：

```text
official historical release → long_term_context / related_indicators
最近 14 天新闻          → recent_developments
其他民调或外部指标       → related_indicators
明确方向相反的结构化指标  → conflicting_evidence
```

若无法可靠分类，放入 `recent_developments`，不要丢弃。

### 6.2 证据预算

建议默认预算：

```python
MAX_CONTEXT_DOCUMENTS = 8
MAX_ITEMS_PER_SECTION = 4
MAX_ITEM_CHARS = 600
MAX_CONTEXT_CHARS = 8_000
```

选择顺序：

1. 官方、结构化、与目标直接对应的数据；
2. 与目标定义相同的近期结果；
3. 可靠来源的相关指标；
4. 与预测窗口直接相关的近期事件；
5. 一般背景新闻。

不要仅按“最新”排序，否则大量新鲜但低相关的新闻会挤掉真正重要的官方数据。

### 6.3 保留冲突信息

Context Builder 不应只保留支持单一方向的证据。若两个高质量来源得出不同方向，应同时保留并明确标记来源。

MVP 可以使用以下简单策略：

```text
Top 6 overall evidence
+ 最多 2 条未进入 Top 6、但方向相反的高质量证据
```

如果无法自动判断方向，则不做 stance 平衡，只保证来源多样性。

---

## 7. Context 固定格式渲染

建议统一格式：

```text
TASK
Round: {round_id}
Question: {question}
Target: {target_type}
Unit: {unit}
Information cutoff: {asof}
Forecast target time/window: {forecast_for}

HISTORICAL BASELINE
Observation count: ...
Latest value: ...
Recent mean: ...
Long-term mean: ...
Recent change: ...
Recent volatility: ...

RECENT NUMERICAL HISTORY
- 2026-08-30: 42.0
- 2026-09-06: 41.0
- 2026-09-13: 41.0

RECENT DEVELOPMENTS
[E1] 2026-09-18 | Reuters | title
Summary text...

RELATED INDICATORS
[E2] 2026-09-17 | Official source | title
Summary text...

CONFLICTING OR UNCERTAIN EVIDENCE
[E3] ...

DATA QUALITY WARNINGS
- ...
```

规则：

- 日期统一为 UTC ISO 8601 或 `YYYY-MM-DD`；
- 浮点数统一精度；
- 证据 ID 稳定；
- 同分文档使用 URL 或内容 hash 决定顺序；
- 不输出诸如“因此预测会上升”之类的最终结论；
- 同一输入必须生成完全相同的文本。

---

## 8. Context Builder 主接口

```python
class ContextBuilder:
    VERSION = "context-v1"

    def build(
        self,
        request: ForecastRequest,
        research: ResearchBundle,
    ) -> ForecastContext:
        validate_request(request)
        self._validate_cutoff(request, research)

        history = self._summarize_history(request)
        evidence = self._normalize_evidence(research.documents)
        evidence = self._select_with_budget(evidence, request)
        sections = self._categorize(evidence, request)

        rendered = self._render(
            request=request,
            history=history,
            sections=sections,
        )

        return self._build_context(
            request=request,
            history=history,
            sections=sections,
            rendered=rendered,
        )
```

若 `research.asof != request.asof`，默认抛出异常，不能静默合并两个时间边界不同的对象。

---

# Part B：Forecast Prompt Compiler

## 9. Prompt Compiler 的目标

Prompt Compiler 将 `ForecastContext` 转换成稳定的模型调用请求。它必须同时产出：

1. system message；
2. user message；
3. JSON Schema；
4.模型调用参数；
5. Prompt 版本和 hash。

Prompt 的目标不是诱导模型输出很长的推理过程，而是规定：

- 使用哪些信息；
- 按什么预测纪律判断；
- 输出什么字段；
- 哪些输出属于非法结果。

---

## 10. `CompiledPrompt` 数据结构

```python
@dataclass(frozen=True)
class CompiledPrompt:
    round_id: str
    target_type: str
    system_message: str
    user_message: str
    output_schema: dict[str, Any]
    model_params: dict[str, Any]
    prompt_version: str
    context_hash: str
    prompt_hash: str
```

建议初始模型参数：

```python
MODEL_PARAMS = {
    "temperature": 0,
    "top_p": 1,
}
```

如果供应商支持 JSON Schema 或 structured output，应使用 API 级结构化输出，而不只是在自然语言中要求 JSON。

---

## 11. 固定 System Prompt

建议第一版 System Prompt 尽量短且稳定：

```text
You are a calibrated forecasting model.

Predict the requested future observation using only the information supplied
in the user message. Treat numerical history as the baseline and qualitative
evidence as a possible adjustment. Do not assume access to information after
the stated cutoff. Avoid large changes unless supported by strong direct
evidence. Represent uncertainty honestly.

Return only data that matches the supplied JSON schema. Do not add Markdown
or text outside the structured response.
```

不要要求输出完整 chain-of-thought。若需要解释，最多输出短的 `evidence_summary`，但正式提交结果不依赖它。

---

## 12. 通用预测协议

User Prompt 中建议固定以下协议：

```text
FORECASTING PROTOCOL

1. Establish a baseline from the latest value, recent mean, and longer-term
   historical behavior.
2. Determine whether the series is stable, trending, seasonal, or unusually
   volatile.
3. Evaluate recent evidence available before the cutoff.
4. Give more weight to direct official measurements than to indirect news.
5. Compare evidence supporting different directions.
6. Apply a conservative adjustment to the historical baseline.
7. Estimate uncertainty from historical volatility, forecast horizon,
   evidence quality, and disagreement.
8. Return the forecast using the exact required schema.
```

协议可以针对任务增加规则，但核心顺序应保持稳定，方便回测 Prompt 版本。

---

## 13. Topline Prompt

### 13.1 User Prompt 模板

```text
FORECAST TASK

Question:
{question}

Information cutoff:
{asof}

Forecast target time/window:
{forecast_for}

Unit:
{unit}

Valid range:
{minimum} to {maximum}

AVAILABLE EVIDENCE

{context.rendered_text}

FORECASTING PROTOCOL

1. Use the latest value and recent mean as the starting baseline.
2. Use longer history to estimate normal variation and mean reversion.
3. Adjust the baseline only when recent evidence is relevant and reliable.
4. Estimate a positive standard deviation for the future observation.

OUTPUT REQUIREMENTS

Return exactly one object containing:
- mean: the expected future value
- sd: the standard deviation of the predictive distribution
```

### 13.2 JSON Schema

```python
def topline_schema(request):
    return {
        "type": "object",
        "properties": {
            "mean": {
                "type": "number",
                "minimum": request.minimum,
                "maximum": request.maximum,
            },
            "sd": {
                "type": "number",
                "exclusiveMinimum": 0,
            },
        },
        "required": ["mean", "sd"],
        "additionalProperties": False,
    }
```

---

## 14. Profile Prompt

Profile 的重点是覆盖所有 cell，并保持 cell ID 完全一致。

### 14.1 模板附加规则

```text
REQUIRED CELLS
- democrat
- independent
- republican

PROFILE RULES

1. Forecast every required cell.
2. Use each cell ID exactly as written.
3. Do not omit, rename, or add cells.
4. Estimate uncertainty separately for each cell.
5. Use shared evidence consistently across cells, while respecting each
   cell's own historical baseline.
```

### 14.2 动态 JSON Schema

```python
def profile_schema(request):
    cell_schema = {
        "type": "object",
        "properties": {
            "mean": {
                "type": "number",
                "minimum": request.minimum,
                "maximum": request.maximum,
            },
            "sd": {
                "type": "number",
                "exclusiveMinimum": 0,
            },
        },
        "required": ["mean", "sd"],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "cells": {
                "type": "object",
                "properties": {
                    cell: cell_schema for cell in request.cells
                },
                "required": request.cells,
                "additionalProperties": False,
            }
        },
        "required": ["cells"],
        "additionalProperties": False,
    }
```

若任务存在确定的加总或逻辑关系，应在 Ensemble 后增加约束投影；不要只依赖 Prompt 保证一致性。

---

## 15. Ranking Prompt

Ranking 最终需要完整排序，但内部建议要求模型输出候选项分数，便于检查和集成。

### 15.1 输出格式

```json
{
  "scores": {
    "A": 82.0,
    "B": 65.0,
    "C": 41.0
  },
  "ranking": ["A", "B", "C"]
}
```

分数只要求相对可比较，不一定具有概率含义。

### 15.2 模板附加规则

```text
CANDIDATES
- A
- B
- C

RANKING RULES

1. Assign one finite score to every candidate.
2. Higher scores indicate a higher predicted rank.
3. Use every candidate exactly once in the ranking.
4. Do not introduce candidates outside the supplied set.
5. The ranking must be consistent with descending score order.
6. Break exact score ties using the candidate order supplied above.
```

程序端应根据 `scores` 重新生成 ranking，而不是完全相信模型返回的数组：

```python
ranking = sorted(
    request.candidates,
    key=lambda item: (-scores[item], candidate_index[item]),
)
```

如果比赛接口只接受数组，Submission Adapter 再删除 `scores`。

---

## 16. Prompt Compiler 主接口

```python
class ForecastPromptCompiler:
    VERSION = "forecast-prompt-v1"

    def compile(
        self,
        request: ForecastRequest,
        context: ForecastContext,
    ) -> CompiledPrompt:
        if request.round_id != context.round_id:
            raise PromptCompilationError("round mismatch")

        if request.asof != context.asof:
            raise PromptCompilationError("asof mismatch")

        if request.target_type == "continuous_normal":
            return self._compile_topline(request, context)

        if request.target_type == "profile_energy":
            return self._compile_profile(request, context)

        if request.target_type == "ranking_list":
            return self._compile_ranking(request, context)

        raise UnsupportedTargetType(request.target_type)
```

`prompt_hash` 应由以下内容规范化计算：

```text
system_message
user_message
output_schema
model_params
prompt_version
context_hash
```

---

## 17. 模型输出解析与校验

虽然解析器不属于 Prompt Compiler 本身，但两者必须配套设计。

统一输出对象：

```python
@dataclass(frozen=True)
class ModelPrediction:
    round_id: str
    target_type: str
    model_id: str
    prompt_hash: str
    raw_response: str
    parsed: dict[str, Any]
    prediction: dict[str, Any]
    generated_at: str
    warnings: list[str]
```

### 17.1 Topline 校验

```python
def validate_topline(obj, request):
    if set(obj) != {"mean", "sd"}:
        raise InvalidModelOutput("unexpected fields")

    mean_value = float(obj["mean"])
    sd_value = float(obj["sd"])

    if not math.isfinite(mean_value) or not math.isfinite(sd_value):
        raise InvalidModelOutput("non-finite value")

    if not request.minimum <= mean_value <= request.maximum:
        raise InvalidModelOutput("mean outside valid range")

    if sd_value <= 0:
        raise InvalidModelOutput("sd must be positive")
```

### 17.2 Profile 校验

必须满足：

```python
set(obj["cells"]) == set(request.cells)
```

并逐 cell 检查数值范围和 `sd > 0`。

### 17.3 Ranking 校验

必须满足：

```python
set(scores) == set(request.candidates)
len(ranking) == len(request.candidates)
set(ranking) == set(request.candidates)
len(ranking) == len(set(ranking))
```

最终 ranking 由程序按 scores 重建。

### 17.4 修复策略

优先使用供应商的 structured output。若仍失败，可以进行最多一次格式修复：

```text
Convert the supplied response to the required JSON schema.
Preserve the original numeric predictions. Do not reconsider the forecast.
```

修复调用必须保存原始响应和修复响应。第二次仍失败则抛出 `InvalidModelOutput`，不能无限重试。

---

# Part C：Ensemble Engine

## 18. Ensemble Engine 的目标

单个 LLM 预测容易受到 Prompt、采样和证据表达方式影响。Ensemble Engine 将多个可审计预测组合起来，提高稳定性。

MVP 至少组合：

```text
LLM forecast
Persistence baseline
Recent-mean baseline
```

未来可以加入：

```text
Time-series model
Multiple LLMs
Multiple Prompt variants
Structured-source model
Retrieval-free model
```

集成器不应重新读取新闻。它只处理标准化预测和历史回测得到的权重。

---

## 19. 基线预测器

### 19.1 Persistence

Topline：

```python
mean = latest_value
```

Profile：

```python
mean[cell] = latest_value[cell]
```

Ranking：

```python
ranking = latest_ranking
```

Persistence 是必须保留的强基线。任何复杂系统都应证明自己优于它。

### 19.2 Recent Mean

Topline/Profile：最近 3 期均值。

```python
mean = average(last_three_values)
```

Ranking：可以使用最近若干期平均名次或平均连续指标。

### 19.3 Baseline uncertainty

Topline/Profile 第一版可用最近 10 期的一阶变化估计：

```python
deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
sd = max(MIN_SD, pstdev(deltas))
```

若数据不足，使用按任务配置的默认 `sd`，并记录 warning。

---

## 20. 统一 Ensemble 输入输出

```python
@dataclass(frozen=True)
class ForecastMember:
    member_id: str
    member_type: str
    prediction: dict[str, Any]
    weight: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FinalPrediction:
    round_id: str
    target_type: str
    prediction: dict[str, Any]
    members: list[ForecastMember]
    weights: dict[str, float]
    ensemble_version: str
    warnings: list[str]
```

权重必须非负并归一化：

```python
weights = {name: max(0.0, value) for name, value in weights.items()}
total = sum(weights.values())

if total <= 0:
    raise InvalidEnsembleWeights("weight sum must be positive")

weights = {name: value / total for name, value in weights.items()}
```

---

## 21. Topline 集成

MVP 固定权重建议：

```text
LLM             0.60
Persistence     0.25
Recent mean     0.15
```

最终均值：

```python
final_mean = sum(
    member.weight * member.prediction["mean"]
    for member in members
)
```

最终方差应同时包含成员自身的不确定性和成员之间的分歧：

\[
\mathrm{Var}(Y)
=
\sum_i w_i\left(\sigma_i^2 + \mu_i^2\right)
-
\left(\sum_i w_i\mu_i\right)^2
\]

代码：

```python
second_moment = sum(
    member.weight
    * (
        member.prediction["sd"] ** 2
        + member.prediction["mean"] ** 2
    )
    for member in members
)

variance = max(MIN_VARIANCE, second_moment - final_mean ** 2)
final_sd = math.sqrt(variance)
```

然后将 `mean` 裁剪到合法范围，将 `sd` 限制在任务允许的合理上下界。所有裁剪都必须记录 warning。

---

## 22. Profile 集成

Profile 对每个 cell 使用相同的成员权重：

```python
for cell in request.cells:
    final[cell] = combine_normals(
        [member.prediction["cells"][cell] for member in members],
        weights,
    )
```

第一版不建议为每个 cell 单独学习权重，因为历史样本通常不足，容易过拟合。

如果任务存在约束，例如某些 cell 必须形成加权总体，可增加可选的 `ConstraintProjector`：

```text
raw ensemble means
        ↓
bounded optimization / projection
        ↓
valid profile means
```

但 MVP 仅进行：

- cell 完整性检查；
- 合法范围裁剪；
- 相同 cell 独立集成。

---

## 23. Ranking 集成

Ranking 不能直接平均字符串顺序。必须先转换成候选项分数。

### 23.1 Borda 分数

若某成员只提供完整排序：

```python
def borda_scores(ranking):
    n = len(ranking)
    return {
        item: n - rank_index
        for rank_index, item in enumerate(ranking)
    }
```

### 23.2 分数归一化

不同成员的 score 范围可能不同，应先转成 `[0, 1]`：

```python
normalized = (score - min_score) / (max_score - min_score)
```

若所有分数相同，则全部设为 `0.5`。

### 23.3 加权聚合

```python
final_score[item] = sum(
    member.weight * normalized_scores[member.member_id][item]
    for member in members
)
```

最终按 `final_score` 降序排列。精确平局时使用：

1. persistence ranking；
2. 原始候选顺序。

这保证相同输入产生相同排序。

---

## 24. 权重学习与防止过拟合

MVP 使用固定权重。只有积累足够历史 round 后，才由 Backtester 学习权重。

权重学习必须使用滚动时间验证：

```text
train on rounds <= t
select weights
evaluate on next rounds > t
```

不能随机切分时间序列，也不能使用测试 round 的真值选择权重。

建议优化目标：

| 任务 | 目标 |
|---|---|
| Topline | CRPS 为主，MAE/RMSE 为辅 |
| Profile | cell 平均 CRPS + 完整率 |
| Ranking | NDCG / Kendall tau / Top-k overlap |

权重应设置正则化或最小样本门槛，例如：

```text
历史 round < 20：固定权重
历史 round >= 20：允许在简单网格中选择权重
```

第一版不要训练高维 stacking model。

---

## 25. 不确定性校准

LLM 输出的 `sd` 不能直接视为已经校准。

建议分两阶段：

### MVP

使用混合方差公式，将历史波动和成员分歧纳入最终 `sd`。

### 有足够回测数据后

计算标准化残差：

\[
z_t = \frac{y_t - \mu_t}{\sigma_t}
\]

若实际残差明显大于预测分布，可学习全局缩放参数：

```python
calibrated_sd = scale * raw_sd
```

`scale` 只能使用当前 round 之前的真值拟合，并按任务或 domain 保存版本。

应报告：

- 50% 区间覆盖率；
- 80% 区间覆盖率；
- 95% 区间覆盖率；
- CRPS；
- 平均预测 `sd`；
- 实际 RMSE。

---

## 26. Ensemble Engine 主接口

```python
class EnsembleEngine:
    VERSION = "ensemble-v1"

    def combine(
        self,
        request: ForecastRequest,
        llm_predictions: list[ModelPrediction],
        baselines: list[ForecastMember],
    ) -> FinalPrediction:
        members = self._normalize_members(
            request,
            llm_predictions,
            baselines,
        )

        self._validate_members(request, members)
        weights = self._select_weights(request, members)

        if request.target_type == "continuous_normal":
            prediction = self._combine_topline(request, members, weights)
        elif request.target_type == "profile_energy":
            prediction = self._combine_profile(request, members, weights)
        elif request.target_type == "ranking_list":
            prediction = self._combine_ranking(request, members, weights)
        else:
            raise UnsupportedTargetType(request.target_type)

        prediction = self._final_validate(request, prediction)

        return FinalPrediction(
            round_id=request.round_id,
            target_type=request.target_type,
            prediction=prediction,
            members=members,
            weights=weights,
            ensemble_version=self.VERSION,
            warnings=[],
        )
```

---

## 27. 端到端 Orchestrator

三个组件可以通过一个简单 Pipeline 串联：

```python
class ForecastPipeline:
    def __init__(
        self,
        research_engine,
        context_builder,
        prompt_compiler,
        model_client,
        output_parser,
        baseline_factory,
        ensemble_engine,
        run_archive,
    ):
        self.research_engine = research_engine
        self.context_builder = context_builder
        self.prompt_compiler = prompt_compiler
        self.model_client = model_client
        self.output_parser = output_parser
        self.baseline_factory = baseline_factory
        self.ensemble_engine = ensemble_engine
        self.run_archive = run_archive

    def run(self, request, entrant):
        research = self.research_engine.retrieve(request, entrant)
        context = self.context_builder.build(request, research)
        prompt = self.prompt_compiler.compile(request, context)

        raw = self.model_client.generate(prompt)
        llm_prediction = self.output_parser.parse(
            request=request,
            prompt=prompt,
            raw_response=raw,
        )

        baselines = self.baseline_factory.build(request)
        final = self.ensemble_engine.combine(
            request=request,
            llm_predictions=[llm_prediction],
            baselines=baselines,
        )

        self.run_archive.save_once(
            request=request,
            research=research,
            context=context,
            prompt=prompt,
            llm_predictions=[llm_prediction],
            final=final,
        )

        return final
```

---

## 28. 运行归档与可复现性

每个 round 至少保存：

```text
runs/<round_id>/<entrant>/
├── request.json
├── research_bundle.json
├── context.json
├── context.txt
├── compiled_prompt.json
├── raw_model_response.txt
├── model_prediction.json
├── baselines.json
├── final_prediction.json
└── manifest.json
```

`manifest.json` 建议包含：

```json
{
  "round_id": "...",
  "asof": "...",
  "research_hash": "sha256:...",
  "context_hash": "sha256:...",
  "prompt_hash": "sha256:...",
  "context_builder_version": "context-v1",
  "prompt_version": "forecast-prompt-v1",
  "ensemble_version": "ensemble-v1",
  "model_id": "...",
  "created_at": "..."
}
```

同一个实验运行应绑定完整 `pipeline_config_hash`，避免旧 archive 被错误复用。

---

## 29. 异常与失败策略

建议定义：

```python
class ForecastPipelineError(Exception):
    pass


class InvalidForecastRequest(ForecastPipelineError):
    pass


class ContextBuildError(ForecastPipelineError):
    pass


class ContextBudgetExceeded(ContextBuildError):
    pass


class PromptCompilationError(ForecastPipelineError):
    pass


class UnsupportedTargetType(ForecastPipelineError):
    pass


class InvalidModelOutput(ForecastPipelineError):
    pass


class InvalidEnsembleWeights(ForecastPipelineError):
    pass


class IncompleteProfile(ForecastPipelineError):
    pass


class InvalidRanking(ForecastPipelineError):
    pass
```

建议策略：

| 情况 | 处理 |
|---|---|
| request 与 bundle 的 asof 不一致 | 终止 |
| 历史数据含未来值 | 过滤并记录；严格模式终止 |
| Context 超预算 | 确定性截断，不能随机截断 |
| LLM 输出非法 | 最多修复一次 |
| Profile 漏 cell | 拒绝该 LLM 输出 |
| Ranking 漏项/重复 | 拒绝该 LLM 输出 |
| 某个 ensemble member 失败 | 明确记录并重新归一化权重 |
| 所有预测成员失败 | 终止 |
| LLM 失败但 baseline 可用 | 正式提交模式可显式降级到 baseline |

实验模式建议 fail closed；正式比赛模式可以显式降级，但必须写入：

```json
{
  "degraded": true,
  "reason": "llm_output_invalid"
}
```

---

## 30. 离线测试方案

所有单元测试必须不依赖真实网络和真实模型。

### 30.1 Context Builder

必须覆盖：

```text
test_future_history_is_rejected
test_request_and_bundle_asof_must_match
test_history_statistics_are_correct
test_context_document_budget_is_enforced
test_context_render_is_deterministic
test_context_hash_changes_when_input_changes
test_profile_history_contains_all_cells
test_ranking_history_summary_is_stable
```

### 30.2 Prompt Compiler

必须覆盖：

```text
test_topline_prompt_contains_cutoff_and_horizon
test_profile_schema_requires_every_cell
test_profile_schema_rejects_extra_cells
test_ranking_schema_requires_every_candidate
test_prompt_is_deterministic
test_prompt_hash_changes_with_context
test_unsupported_target_type_fails
```

### 30.3 输出校验

必须覆盖：

```text
test_topline_rejects_non_positive_sd
test_topline_rejects_out_of_range_mean
test_profile_rejects_missing_cell
test_profile_rejects_extra_cell
test_ranking_rejects_duplicate_candidate
test_ranking_rejects_unknown_candidate
test_ranking_is_rebuilt_from_scores
```

### 30.4 Ensemble

必须覆盖：

```text
test_weights_sum_to_one
test_negative_weight_is_rejected_or_clipped_by_policy
test_topline_weighted_mean
test_mixture_variance_includes_disagreement
test_profile_combines_every_cell
test_ranking_borda_conversion
test_ranking_tie_break_is_deterministic
test_failed_member_weights_are_renormalized
```

### 30.5 Golden fixtures

至少准备三个固定 fixture：

1. 一个 `continuous_normal` round；
2. 一个多 cell 的 `profile_energy` round；
3. 一个至少 5 个候选项的 `ranking_list` round。

对每个 fixture 保存期望的：

```text
context.txt
compiled_prompt.json
validated_prediction.json
final_prediction.json
```

这样代码重构后可以快速识别非预期行为变化。

---

## 31. 回测与消融实验

Backtester 应能够固定 Research Bundle，只改变一个组件：

```text
Experiment A：history only
Experiment B：history + research
Experiment C：LLM only
Experiment D：persistence only
Experiment E：LLM + persistence ensemble
Experiment F：LLM + persistence + recent mean
```

Prompt 实验也必须一次只改变一类因素：

```text
base-rate instruction on/off
conflicting evidence section on/off
short context vs long context
LLM sd vs calibrated sd
scores + ranking vs ranking only
```

每次实验保存：

```text
pipeline_config_hash
round IDs
component versions
model ID
predictions
truth
metrics
cost
latency
```

不能根据最终不可见测试集调整 Prompt 或权重。

---

## 32. 推荐代码结构

```text
ssa/
├── context/
│   ├── __init__.py
│   ├── models.py
│   ├── history.py
│   ├── selector.py
│   ├── renderer.py
│   └── builder.py
├── forecast/
│   ├── __init__.py
│   ├── models.py
│   ├── schemas.py
│   ├── prompts.py
│   ├── parser.py
│   └── validators.py
├── ensemble/
│   ├── __init__.py
│   ├── models.py
│   ├── baselines.py
│   ├── numeric.py
│   ├── profile.py
│   ├── ranking.py
│   └── engine.py
├── pipeline.py
└── archive.py
```

如果第一版不希望创建太多文件，可以合并为：

```text
ssa/context.py
ssa/forecast.py
ssa/ensemble.py
ssa/pipeline.py
```

待接口稳定后再拆目录。

---

## 33. 最简 MVP 实现流程

第一版只跑通一条 Topline 链路：

```text
1. 接收 ForecastRequest
2. 获取 ResearchBundle
3. 过滤历史并计算 latest / mean3 / mean10 / volatility
4. 选择最多 8 条研究证据
5. 渲染固定 ForecastContext
6. 编译 Topline Prompt + JSON Schema
7. 调用一个 LLM，temperature=0
8. 校验 mean 和 sd
9. 生成 persistence 与 recent-mean baseline
10. 使用固定权重组合
11. 保存完整运行归档
12. 输出 {"mean": ..., "sd": ...}
```

MVP 建议固定配置：

```python
MVP_CONFIG = {
    "recent_window": 3,
    "long_window": 10,
    "max_context_documents": 8,
    "max_item_chars": 600,
    "temperature": 0,
    "weights": {
        "llm": 0.60,
        "persistence": 0.25,
        "recent_mean": 0.15,
    },
}
```

Topline 稳定后再依次增加：

```text
Profile schema 与完整性检查
        ↓
Ranking scores 与确定性排序
        ↓
多 LLM / 多 Prompt 成员
        ↓
滚动回测学习 ensemble 权重
        ↓
sd 校准和 Profile 约束投影
```

---

## 34. 推荐开发阶段

### 阶段 1：数据契约和纯函数

实现：

```text
ForecastRequest
ForecastContext
CompiledPrompt
ModelPrediction
ForecastMember
FinalPrediction
```

同时完成 JSON 序列化、规范化 hash 和异常类型。

### 阶段 2：Topline Context Builder

实现历史统计、证据预算和确定性 Renderer，使用固定 fixture 测试。

### 阶段 3：Topline Prompt 与 Validator

先用 FakeModel 返回固定 JSON，不接真实模型。

### 阶段 4：简单 Baseline 与 Ensemble

实现 persistence、recent mean、混合均值和混合方差。

### 阶段 5：端到端 Pipeline

串联现有 Research Engine，保存完整 archive。

### 阶段 6：Profile

加入动态 cell schema、逐 cell 校验和逐 cell ensemble。

### 阶段 7：Ranking

加入 scores、Borda 转换、分数归一化和确定性 tie-break。

### 阶段 8：Backtest 优化

通过滚动历史回测调整 Prompt、证据预算、权重和不确定性缩放。

---

## 35. 建议的 Pull Request 拆分

### First PR：Topline 的纯离线链路

```text
1. ForecastRequest 和公共数据结构
2. Topline HistorySummary
3. Context Builder 和确定性 Renderer
4. Topline Prompt Compiler 与 JSON Schema
5. Model output validator
6. Persistence / recent-mean baseline
7. Simple numeric ensemble
8. FakeModel 端到端测试
```

### Second PR：接入真实 Harness

```text
1. 接入 ResearchBundle
2. 接入模型客户端 structured output
3. 保存 run archive 和各阶段 hash
4. 接入 Submission Adapter
5. 增加显式 degraded fallback
```

### Third PR：Profile 与 Ranking

```text
1. Profile 动态 schema 和完整性校验
2. Profile 逐 cell ensemble
3. Ranking score schema
4. Borda / normalized-score ensemble
5. 三类任务 golden fixtures
```

### Fourth PR：基于回测的优化

```text
1. Prompt 版本实验
2. 固定权重网格搜索
3. sd calibration
4. Context 长度和证据选择消融
5. 按 domain 的配置
```

---

## 36. 第一版明确不做的内容

为了避免 MVP 失控，第一版不建议实现：

- 多轮自主 Agent 推理；
- LLM 自动修改自己的 Prompt；
- 输出或保存长 chain-of-thought；
- 每个 Profile cell 单独调用一次模型；
- 高维 stacking 或神经网络 ensemble；
- 没有时间滚动验证的自动权重训练；
- 依靠 Prompt 而不做程序输出校验；
- 在 Ensemble 阶段重新检索或阅读新闻；
- 未归档输入的在线动态摘要。

---

## 37. 验收标准

MVP 完成应满足：

1. 相同请求、Research Bundle 和模型响应产生相同最终结果；
2. Context 不包含 `asof` 后的历史或研究资料；
3. 所有 Prompt 都保存版本和 hash；
4. Topline 输出始终是合法的 `mean + sd`；
5. Profile 不可能静默遗漏 cell；
6. Ranking 不可能包含遗漏、重复或未知候选项；
7. Ensemble 权重明确、可审计、归一化；
8. LLM 失败时不会伪装成正常的 LLM ensemble；
9. 三种任务都有完全离线的单元测试；
10. Backtester 可以固定上游数据并单独替换 Prompt 或 Ensemble 版本。

---

## 38. 关键设计原则总结

### 原则 1：Context 是证据整理，不是隐藏预测

Context Builder 可以计算客观历史统计量，但不应在文本中预先给出主观最终方向。

### 原则 2：Prompt 是输出协议，不是自由聊天

Prompt 必须固定任务定义、时间边界、预测纪律和 JSON Schema。

### 原则 3：程序校验优先于语言约束

“请不要漏 cell”不够，必须通过动态 schema 和集合比较保证完整性。

### 原则 4：简单基线必须保留

Persistence 和 recent mean 不只是 ensemble 成员，也是判断复杂系统是否有效的最低标准。

### 原则 5：不确定性必须回测校准

`sd` 不是装饰字段。最终需要使用历史残差检查区间覆盖率和 CRPS。

### 原则 6：一次只优化一个组件

固定 Research Bundle 后测试 Context/Prompt；固定成员预测后测试 Ensemble，避免无法归因。

### 原则 7：所有中间产物均可冻结和回放

Request、Context、Prompt、模型原始响应、成员预测和最终输出都必须归档并带版本。

---

## 39. 最终最小接口

三个组件对外只需要稳定暴露以下接口：

```python
context = context_builder.build(
    request=request,
    research=research_bundle,
)

prompt = prompt_compiler.compile(
    request=request,
    context=context,
)

final = ensemble_engine.combine(
    request=request,
    llm_predictions=[model_prediction],
    baselines=baseline_predictions,
)
```

只要这三个接口稳定，内部就可以逐步替换：

- 证据选择方法；
- Prompt 模板；
- 模型供应商；
- baseline；
- ensemble 权重；
- 不确定性校准方法。

这使 SSA Harness 能够从一个最小可用预测系统，逐步演化为可系统回测和优化的完整预测框架。
