> 历史设计参考：本文保留原设计。当前实现、官方字段转换、开放排名与计分约定见 `../../README.md` 和 `../INTEGRATION.md`；下文旧接口和旧验收数量不代表当前协议。

# SSA Harness：Research / Retrieval Engine 设计方案

本文档为 Social Simulation Arena（SSA）Harness 中第二部分 **Research / Retrieval Engine** 的实现设计。

目标是：在预测截止时间之前，为 Forecasting / Reasoning Engine 收集一组**受时间约束、格式统一、可缓存、可复现、可审计**的研究证据。

本文档主要面向以下工作：

- 设计 Retrieval Engine 的模块边界；
- 在现有 `ssa/adapters/search.py` 和 `ssa/harness.py` 基础上重构；
- 支持 Topline、Profile、Ranking 三类题目；
- 保证不会把截止时间之后的信息泄露给预测模型；
- 为后续加入 YouGov、Civiqs、Wikipedia、Google Trends 等专用数据源留下接口。

---

## 1. 背景与目标

SSA 的一个参赛 Agent 大致经过以下流程：

```text
收到 round 请求
    ↓
解析题目、时间、人群和答案类型
    ↓
Research / Retrieval Engine
    ├── 生成查询
    ├── 调用搜索或结构化数据源
    ├── 过滤截止时间之后的信息
    ├── 去重、排序、清洗
    ├── 生成统一 research bundle
    └── 将 bundle 冻结并归档
    ↓
Forecasting / Reasoning Engine
    ↓
生成 mean / sd、profile 或 ranking
    ↓
提交预测
```

Retrieval Engine 的职责是**获取和组织证据**，不是直接做最终预测。

例如：

```text
Retrieval Engine：找到最近的民调、经济数据和相关事件
Forecast Engine：根据历史序列和这些证据，预测支持率为 42%，sd 为 2.5
```

这种职责分离有三个好处：

1. 可以单独测试搜索质量；
2. 可以替换搜索引擎而不改预测器；
3. 可以记录模型究竟看到了哪些证据。

---

## 2. 必须遵守的 SSA 约束

### 2.1 时间边界

所有研究资料必须满足：

```text
published_at <= asof
```

其中 `asof` 是该 round 的信息冻结时间，通常是调用窗口开始的时间，而不是网页当前时间。

不能在回测时直接访问已经公布的真实答案，否则会造成未来信息泄露。

### 2.2 固定预算

建议采用和仓库现有搜索适配器一致的初始限制：

```python
MAX_QUERIES = 4
RESULTS_PER_QUERY = 3
SNIPPET_CHARS = 700
SEARCH_DEPTH = "basic"
TOPIC = "news"
DAYS = 14
```

含义是：

- 每个 round 最多 4 个查询；
- 每个查询最多返回 3 条结果；
- 每条结果最多保留 700 个字符；
- 只使用最近 14 天的新闻结果；
- 默认只进行一轮查询。

预算必须由程序强制执行，不能只写在 prompt 中要求模型“尽量少搜”。

### 2.3 可复现和可审计

搜索结果会随时间变化，因此必须同时记录：

- 查询内容；
- 搜索参数；
- 返回结果；
- 获取时间；
- 信息截止时间；
- 最终发送给模型的文本；
- bundle 的哈希值。

### 2.4 失败时关闭（fail closed）

搜索失败时不能静默地把 web 条件退化成无搜索条件。

错误示例：

```python
try:
    documents = search(query)
except Exception:
    documents = []

return forecast_without_research()
```

这种行为会导致一个名义上的 `web` entrant 实际上变成 `recent10` entrant。

正确策略是：

- 如果有已经冻结的缓存，使用缓存；
- 如果没有缓存且检索失败，记录明确错误并不提交预测；
- 不要伪造空的研究上下文。

---

## 3. 模块总体架构

推荐的模块结构如下：

```text
ssa/research/
├── __init__.py
├── models.py          # 请求、文档、bundle 的数据结构
├── planner.py         # 查询规划器
├── retriever.py       # Retriever 抽象接口和 Tavily 实现
├── sources.py         # YouGov/Civiqs/Wikipedia 等结构化源
├── filters.py         # 截止时间过滤、来源过滤、去重
├── ranker.py          # 证据排序
├── renderer.py        # 固定格式文本渲染
├── archive.py         # query cache 和 round archive
├── engine.py          # ResearchEngine 主流程
└── errors.py          # 明确的异常类型
```

如果暂时不想新增目录，也可以先在 `ssa/adapters/search.py` 中实现，再逐步抽取到上述结构。

当前仓库已有的相关组件：

- `ssa/adapters/search.py`：Tavily 搜索、缓存、round 归档、渲染；
- `ssa/adapters/newsdigest.py`：固定新闻摘要的归档；
- `ssa/harness.py`：生成查询 prompt、解析查询、调用搜索并拼接预测 prompt；
- `tests/test_search.py`：搜索组件的无网络测试；
- `tests/test_newsdigest.py`：新闻摘要归档测试。

---

## 4. 核心数据结构

### 4.1 `ResearchRequest`

统一描述一次研究请求。

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class ResearchRequest:
    round_id: str
    question: str
    target_type: str
    board_id: str
    unit: str
    domain: str
    asof: str
    deadline: str
    history: list[dict[str, Any]]
    cells: list[str] | None = None
    ranking_spec: dict[str, Any] | None = None
```

字段说明：

| 字段 | 含义 |
|---|---|
| `round_id` | 具体 round 的唯一 ID |
| `question` | 完整自然语言题目 |
| `target_type` | `continuous_normal`、`profile_energy` 或 `ranking_list` |
| `board_id` | `topline`、`profile` 或 `ranking` |
| `unit` | 数值单位 |
| `domain` | `public-opinion`、`consumer`、`attention` 等 |
| `asof` | 信息冻结时间 |
| `deadline` | 参赛预测截止时间 |
| `history` | 可用于预测的历史序列 |
| `cells` | Profile 题需要预测的 cell ID |
| `ranking_spec` | Ranking 题的长度、候选集、评分参数等 |

建议提供一个转换函数：

```python
@classmethod
def from_round(cls, round_data, history, asof):
    return cls(
        round_id=round_data["round_id"],
        question=round_data["question"],
        target_type=round_data["target_type"],
        board_id=round_data.get("board_id", "topline"),
        unit=round_data["unit"],
        domain=round_data.get("domain", "unknown"),
        asof=asof,
        deadline=round_data.get("deadline") or round_data["lock_at"],
        history=history,
        cells=round_data.get("cells"),
        ranking_spec=round_data.get("ranking"),
    )
```

### 4.2 `RetrievedDocument`

所有来源的结果都应该归一化成统一格式。

```python
@dataclass
class RetrievedDocument:
    title: str
    url: str
    snippet: str
    source: str
    query: str
    rank: int
    published_at: str | None = None
    modified_at: str | None = None
    retrieved_at: str | None = None
    source_score: float = 0.0
    temporal_score: float = 0.0
    relevance_score: float = 0.0
    verified_time: bool = False
    metadata: dict[str, Any] | None = None
```

建议保留 `query` 和 `rank`，因为它们属于审计信息：可以知道某条证据是由哪个查询、以什么排名返回的。

### 4.3 `ResearchBundle`

这是 Retrieval Engine 的主要输出。

```python
@dataclass
class ResearchBundle:
    round_id: str
    entrant: str
    asof: str
    queries: list[str]
    documents: list[RetrievedDocument]
    rendered_text: str
    params: dict[str, Any]
    source_count: int
    filtered_count: int
    warnings: list[str]
    bundle_hash: str
```

Forecasting Engine 只需要使用：

```python
bundle.rendered_text
```

而不需要知道底层是 Tavily、RSS、Wikimedia API 还是其他数据源。

---

## 5. Query Planner：查询规划器

### 5.1 职责

Query Planner 负责决定：

> 为了回答当前题目，需要向检索器提出哪些查询？

它不负责：

- 发起网络请求；
- 判断最终预测值；
- 直接修改历史数据；
- 处理缓存文件。

接口：

```python
class QueryPlanner:
    def plan(self, request: ResearchRequest) -> list[str]:
        raise NotImplementedError
```

### 5.2 模板规划器

第一版建议先实现不依赖 LLM 的模板规划器。

```python
class TemplateQueryPlanner(QueryPlanner):
    def plan(self, request):
        question = request.question

        if request.domain == "public-opinion":
            return [
                f"{question} latest poll",
                f"{question} recent trend",
                f"{question} official release",
            ]

        if request.domain == "consumer":
            return [
                f"{question} latest release",
                f"{question} historical trend",
                f"{question} recent economic news",
            ]

        if request.domain == "attention":
            return [
                f"{question} upcoming events",
                f"major events relevant to {question}",
                f"recent news relevant to {question}",
            ]

        return [question]
```

模板规划器的优点：

- 可预测；
- 容易测试；
- 不产生额外 LLM 调用；
- 适合作为 fallback。

### 5.3 LLM 查询规划器

SSA 当前 harness 的设计是让模型先返回：

```json
{
  "queries": [
    "Trump approval latest polling",
    "Economist YouGov approval September 2026",
    "Trump approval recent political news"
  ]
}
```

然后由程序统一执行查询。

建议限制输出格式：

```text
Before answering, decide what information would change your estimate.
Return exactly one JSON object:

{"queries": ["query 1", "query 2"]}

You may provide at most 4 queries. Do not answer the forecast yet.
```

解析时：

```python
def parse_queries(text: str, limit: int = 4) -> list[str]:
    obj = extract_first_json_object(text)
    raw = obj.get("queries")

    if not isinstance(raw, list):
        raise ValueError("response does not contain a query list")

    out = []
    seen = set()

    for item in raw:
        if not isinstance(item, str):
            continue

        query = " ".join(item.split()).strip()
        key = query.lower()

        if not query or key in seen:
            continue

        seen.add(key)
        out.append(query)

    if not out:
        raise ValueError("query list is empty")

    return out[:limit]
```

### 5.4 查询规划原则

对于 Topline 民调题，查询应覆盖：

1. 同一调查机构的最新结果；
2. 同一指标的近期趋势；
3. 可能影响未来结果的事件。

对于 Profile 题，应优先搜索完整调查报告，而不是为每个 cell 单独搜索。

对于 Ranking 题，应搜索未来测量窗口内可能引起关注的事件，而不是搜索已经发布的最终榜单。

---

## 6. Retriever：检索器

### 6.1 统一接口

上层不应直接调用 Tavily 或具体供应商。

```python
from typing import Protocol


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        asof: str,
        max_results: int = 3,
    ) -> list[RetrievedDocument]:
        ...
```

### 6.2 Tavily 实现

现有 `ssa/adapters/search.py` 已经提供了可复用的实现思路：

```python
class TavilyRetriever:
    endpoint = "https://api.tavily.com/search"

    def search(self, query, *, asof, max_results=3):
        response = requests.post(
            self.endpoint,
            timeout=45,
            json={
                "api_key": os.environ["TAVILY_API_KEY"],
                "query": query,
                "topic": "news",
                "search_depth": "basic",
                "max_results": max_results,
                "days": 14,
                "include_answer": False,
                "include_raw_content": False,
            },
        )

        response.raise_for_status()
        payload = response.json()

        return [
            normalize_tavily_result(
                item,
                query=query,
                rank=i + 1,
            )
            for i, item in enumerate(payload.get("results", []))
        ]
```

不建议打开 `include_answer`，因为搜索 API 的自动摘要会成为一个额外的不可控推理层，难以审计。

### 6.3 结构化数据源

除了普通网页搜索，应该逐步加入结构化来源：

| 题目类型 | 推荐来源 |
|---|---|
| YouGov | YouGov 官方页面或官方数据文件 |
| Civiqs | Civiqs dashboard |
| Morning Consult | Morning Consult tracker |
| Michigan | University of Michigan SCA |
| SCE | New York Fed |
| AAII | AAII sentiment survey |
| Wikipedia | Wikimedia API / Pageviews API |
| Google Trends | Google Trends 数据接口或归档数据 |

接口示例：

```python
class StructuredSourceRetriever:
    def fetch(self, source: str, request: ResearchRequest):
        raise NotImplementedError
```

结构化来源的优先级通常高于新闻摘要，因为它们更接近最终评分数据。

### 6.4 任务专用检索器

建议提供以下专用检索器：

```text
WebSearchRetriever
PollingRetriever
ConsumerIndexRetriever
WikipediaResearcher
GoogleTrendsResearcher
```

调用层可以根据 `domain` 和 `round_id` 选择：

```python
def select_retriever(request):
    if request.domain == "attention" and request.ranking_spec:
        return wikipedia_researcher

    if "yougov" in request.round_id or "civiqs" in request.round_id:
        return polling_retriever

    return web_search_retriever
```

不要让每种题型都强行使用普通新闻搜索。

---

## 7. 截止时间过滤

这是 Retrieval Engine 最重要的正确性约束。

### 7.1 基本过滤

```python
from datetime import datetime


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def is_before_cutoff(doc, asof: str) -> bool:
    timestamp = doc.published_at or doc.modified_at

    if not timestamp:
        return False

    return parse_time(timestamp) <= parse_time(asof)
```

### 7.2 无发布时间的页面

建议分成三种模式：

```python
TIME_POLICY = {
    "competition": "reject",
    "debug": "keep_with_warning",
    "backtest": "reject",
}
```

比赛和回测模式下，如果无法验证时间，默认过滤掉。

### 7.3 时间可信度

网页结果中的 `published_date` 可能不可靠，因此应该保留：

```python
verified_time: bool
temporal_score: float
```

建议：

| 时间来源 | `temporal_score` |
|---|---:|
| 官方发布时间 / RSS | 1.0 |
| 页面 `datePublished` | 0.9 |
| 页面 `dateModified` | 0.7 |
| 搜索引擎推断日期 | 0.5 |
| 无时间 | 0.0 |

### 7.4 不能只依赖搜索 API 的 `days`

搜索参数 `days=14` 只是一个粗过滤，不能替代程序自己的 `asof` 检查。

必须同时执行：

```text
搜索 API 的时间范围过滤
        +
本地基于 asof 的逐条过滤
```

---

## 8. 文档清洗与去重

### 8.1 URL 规范化

```python
from urllib.parse import urlparse, urlunparse


def canonical_url(url: str) -> str:
    parsed = urlparse(url)

    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path.rstrip("/"),
        "",
        "",
        "",
    ))
```

这样可以把带 tracking 参数的 URL 视为同一页面。

### 8.2 内容标准化

```python
import re


def normalize_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\\s+", " ", text)
    return text.strip().lower()
```

### 8.3 去重策略

按以下顺序判断重复：

1. canonical URL；
2. 标题标准化；
3. snippet 内容哈希。

```python
def deduplicate(documents):
    seen_urls = set()
    seen_content = set()
    output = []

    for doc in documents:
        url = canonical_url(doc.url)
        content = normalize_text(doc.snippet)
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        if url in seen_urls or content_hash in seen_content:
            continue

        seen_urls.add(url)
        seen_content.add(content_hash)
        output.append(doc)

    return output
```

---

## 9. 证据排序

搜索引擎的原始排名不一定适合预测任务。

建议使用一个简单、可解释的证据评分：

```text
evidence_score =
    0.30 × source_authority
  + 0.25 × freshness
  + 0.20 × query_relevance
  + 0.15 × content_quality
  + 0.10 × temporal_certainty
```

### 9.1 来源权重

```python
SOURCE_WEIGHTS = {
    "yougov.com": 1.00,
    "civiqs.com": 1.00,
    "sca.isr.umich.edu": 1.00,
    "newyorkfed.org": 1.00,
    "aaii.com": 0.90,
    "reuters.com": 0.85,
    "apnews.com": 0.85,
    "wikipedia.org": 0.80,
    "unknown": 0.40,
}
```

### 9.2 新鲜度

```python
import math


def freshness(age_days: float) -> float:
    return math.exp(-age_days / 7.0)
```

### 9.3 查询相关性

MVP 可以先用词重合度：

```python
def lexical_relevance(query: str, text: str) -> float:
    query_terms = set(query.lower().split())
    text_terms = set(text.lower().split())

    if not query_terms:
        return 0.0

    return len(query_terms & text_terms) / len(query_terms)
```

未来可替换成 embedding 相似度，但第一版不建议过早引入复杂模型。

---

## 10. 两级缓存和归档

### 10.1 Query cache

用途：避免多个 entrant 对相同查询重复付费。

目录：

```text
search/cache/<sha256>.json
```

cache key 必须包含查询和检索参数：

```python
def cache_key(query, params):
    raw = json.dumps(
        {
            "query": query.strip().lower(),
            "params": params,
        },
        sort_keys=True,
    )

    return hashlib.sha256(raw.encode()).hexdigest()
```

不能只根据 query 做 hash。否则改变 `max_results` 或 `search_depth` 后仍会错误地复用旧结果。

### 10.2 Round archive

用途：记录一个 entrant 在一个 round 中实际看到的研究资料。

目录：

```text
search/rounds/<round_id>/<entrant>.json
```

建议格式：

```json
{
  "round_id": "yougov-2026-w40-approval",
  "entrant": "my-agent",
  "asked_at": "2026-09-26T10:00:00Z",
  "asof": "2026-09-26T14:00:00Z",
  "params": {
    "engine": "tavily",
    "max_queries": 4,
    "results_per_query": 3,
    "days": 14,
    "snippet_chars": 700
  },
  "queries": [
    "Trump approval latest polling"
  ],
  "results": [
    {
      "title": "Example poll",
      "url": "https://example.com/poll",
      "published_at": "2026-09-22T10:00:00Z",
      "snippet": "...",
      "query": "Trump approval latest polling",
      "rank": 1
    }
  ],
  "warnings": [],
  "bundle_hash": "sha256:..."
}
```

同一个 round archive 写入后应视为不可变。

### 10.3 缓存有效期

Query cache 可以有有效期，但 round archive 不应因为时间过去而自动更新。

建议：

```text
query cache：不超过一个 filing window
round archive：写入一次，永久保留
```

跨 round 复用查询缓存时，必须检查缓存年龄，不能把上周的搜索结果直接提供给本周题目。

---

## 11. 固定格式渲染

模型最终看到的文本必须由同一个渲染器生成。

```python
def render_documents(documents, asof):
    if not documents:
        return "(no verified research results)"

    lines = [
        f"Research results retrieved at {asof}.",
        "",
    ]

    current_query = None

    for doc in documents:
        if doc.query != current_query:
            current_query = doc.query
            lines.append(f'Query: "{current_query}"')

        date = ""
        if doc.published_at:
            date = f" ({doc.published_at[:10]})"

        lines.append(f"- {doc.title}{date}")
        lines.append(f"  URL: {doc.url}")
        lines.append(f"  {doc.snippet}")

    return "\n".join(lines)
```

必须统一：

- 字段顺序；
- 日期格式；
- URL 是否显示；
- 文本截断长度；
- 结果数量；
- 空结果处理。

否则不同模型之间的差异可能来自格式，而不是模型能力。

---

## 12. `ResearchEngine` 主流程

推荐主接口：

```python
class ResearchEngine:
    def __init__(self, planner, retriever, archive, ranker):
        self.planner = planner
        self.retriever = retriever
        self.archive = archive
        self.ranker = ranker

    def retrieve(self, request: ResearchRequest, entrant: str):
        existing = self.archive.load(
            round_id=request.round_id,
            entrant=entrant,
        )

        if existing is not None:
            return existing

        queries = self.planner.plan(request)
        queries = normalize_queries(queries)
        queries = queries[:4]

        documents = []

        for query in queries:
            documents.extend(
                self.retriever.search(
                    query,
                    asof=request.asof,
                    max_results=3,
                )
            )

        before_filter = len(documents)

        documents = deduplicate(documents)
        documents = [
            doc for doc in documents
            if is_before_cutoff(doc, request.asof)
        ]

        filtered_count = before_filter - len(documents)

        documents = self.ranker.rank(
            documents,
            request=request,
        )

        if not documents:
            raise EmptyCorpus(
                f"no valid research result for {request.round_id}"
            )

        rendered = render_documents(
            documents,
            asof=request.asof,
        )

        bundle = build_bundle(
            request=request,
            entrant=entrant,
            queries=queries,
            documents=documents,
            rendered=rendered,
            filtered_count=filtered_count,
        )

        self.archive.save_once(bundle)
        return bundle
```

### 12.1 `build_bundle`

`bundle_hash` 应该由规范化后的 JSON 计算：

```python
def build_bundle(...):
    payload = {
        "round_id": request.round_id,
        "entrant": entrant,
        "asof": request.asof,
        "queries": queries,
        "documents": [asdict(doc) for doc in documents],
        "rendered_text": rendered,
        "params": params,
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()

    digest = hashlib.sha256(raw).hexdigest()

    return ResearchBundle(
        round_id=request.round_id,
        entrant=entrant,
        asof=request.asof,
        queries=queries,
        documents=documents,
        rendered_text=rendered,
        params=params,
        source_count=len(documents),
        filtered_count=filtered_count,
        warnings=[],
        bundle_hash=f"sha256:{digest}",
    )
```

---

## 13. 与现有 `ssa/harness.py` 的接入

当前仓库的流程大致是：

```text
build_query_prompt()
        ↓
调用模型生成 queries
        ↓
parse_queries()
        ↓
search.gather()
        ↓
search.record_round()
        ↓
build_prompt(..., search=search)
        ↓
调用模型生成预测
```

可以先保留原流程，逐步把细节迁移到 `ResearchEngine`。

目标是让 harness 只保留类似下面的代码：

```python
def _retrieve(entrant, round_data, history):
    asof = batches.window_opens_at(round_data["lock_at"])

    request = ResearchRequest.from_round(
        round_data,
        history=history,
        asof=asof,
    )

    bundle = research_engine.retrieve(
        request=request,
        entrant=entrant,
    )

    return {
        "asof": bundle.asof,
        "queries": bundle.queries,
        "results": bundle.documents,
        "text": bundle.rendered_text,
        "bundle_hash": bundle.bundle_hash,
    }
```

Forecast prompt 只使用：

```python
research["text"]
```

不应在 Forecasting Engine 内部再次自行搜索网页。

---

## 14. 三类题目的检索策略

### 14.1 Topline 数值题

例如：

```text
Economist/YouGov weekly tracker:
percent of US adult citizens who approve of Donald Trump's job performance
```

建议检索三类信息：

1. 同一调查机构的最近结果；
2. 同一指标的近期趋势；
3. 可能影响下一次结果的事件。

示例查询：

```text
Economist YouGov Trump approval latest poll
Trump approval polling trend September 2026
Trump approval recent political news September 2026
```

Forecast Engine 再将：

```text
历史序列 + persistence + research bundle
```

组合成最终分布。

Retrieval Engine 不应自行把新闻转化为具体数值。

### 14.2 Profile 题

例如 YouGov 16-cell 题。

推荐查询：

```text
Economist YouGov Trump approval demographic crosstabs latest
Trump approval Democrats Republicans age race gender YouGov
```

不要为 16 个 cell 分别执行 4 个查询，否则会产生：

- 64 次查询；
- 不同 cell 看到不同时间边界；
- 结果之间缺少一致性；
- 预算和费用不可控。

应该尽量检索完整调查表，再交给 Profile Forecast Engine 生成所有 cell 的分布。

### 14.3 Ranking 题

例如 Wikipedia Top 10。

研究内容建议包括：

1. 最近几周排名历史；
2. 未来测量窗口内的重要体育、电影、政治和娱乐事件；
3. 可能引起 Wikipedia 浏览量的突发事件；
4. 页面或主题的历史出现频率。

注意不能搜索已经发布的目标周最终榜单，否则会泄露答案。

不要把以下结果直接当作预测答案：

```text
Wikipedia top 10 this week
```

更合理的是搜索：

```text
major events September 28 October 4 2026
upcoming sports movies politics events next week
celebrity events scheduled September 2026
```

---

## 15. 异常类型和失败策略

建议定义清晰的异常：

```python
class ResearchError(Exception):
    pass


class NoSearchKey(ResearchError):
    pass


class SearchTimeout(ResearchError):
    pass


class SearchRateLimited(ResearchError):
    pass


class EmptyCorpus(ResearchError):
    pass


class CutoffViolation(ResearchError):
    pass


class ArchiveCorruption(ResearchError):
    pass
```

建议策略：

| 情况 | 处理 |
|---|---|
| 没有 API key，且无缓存 | 拒绝生成 web forecast |
| API 超时 | 有限重试，仍失败则终止 |
| HTTP 429 | 延迟、退避后重试 |
| 查询无结果 | 可使用备用查询，但不能静默当作无新闻 |
| 文档没有时间 | 竞赛模式下过滤 |
| round archive 已存在 | 直接复用，不重复搜索 |
| 文档晚于 `asof` | 丢弃并记录 warning |
| JSON 缓存损坏 | 标记损坏并重新获取 |
| 所有结果都被过滤 | 抛出 `EmptyCorpus` |

---

## 16. 测试方案

测试必须完全支持离线运行，不能依赖真实搜索 API。

### 16.1 查询去重

```python
def test_duplicate_queries_are_removed():
    queries = [
        "Trump approval",
        "Trump approval",
        " trump approval ",
    ]

    assert normalize_queries(queries) == ["Trump approval"]
```

### 16.2 查询数量限制

```python
def test_query_budget_is_enforced():
    queries = [f"q{i}" for i in range(10)]
    assert len(normalize_queries(queries, limit=4)) == 4
```

### 16.3 截止时间过滤

```python
def test_future_documents_are_rejected():
    docs = [
        make_doc("2026-09-20T00:00:00Z"),
        make_doc("2026-09-28T00:00:00Z"),
    ]

    kept = filter_documents(
        docs,
        asof="2026-09-25T00:00:00Z",
    )

    assert len(kept) == 1
```

### 16.4 相同查询只请求一次

```python
def test_query_cache_hit():
    retriever = FakeRetriever()
    engine = make_engine(retriever)

    engine.retrieve(request, "agent-a")
    engine.retrieve(request, "agent-a")

    assert retriever.call_count == 1
```

### 16.5 不同 round 不复用过期研究结果

缓存需要包含 round 的时间边界，或必须检查缓存年龄。

```python
def test_cache_does_not_cross_asof_boundary():
    ...
```

### 16.6 round archive 不可变

```python
def test_round_archive_is_immutable():
    first = engine.retrieve(request, "agent-a")
    second = engine.retrieve(request, "agent-a")

    assert first.bundle_hash == second.bundle_hash
```

### 16.7 空结果拒绝

```python
def test_empty_corpus_fails_closed():
    retriever = EmptyRetriever()
    engine = make_engine(retriever)

    with pytest.raises(EmptyCorpus):
        engine.retrieve(request, "agent-a")
```

### 16.8 渲染稳定

```python
def test_render_is_deterministic():
    first = render_documents(docs, asof)
    second = render_documents(docs, asof)
    assert first == second
```

### 16.9 三种 round 类型

至少准备三个 fixture：

1. `continuous_normal`；
2. `profile_energy`；
3. `ranking_list`。

这和官方接口测试同时覆盖三种输出形态的原则一致。

---

## 17. 开发顺序

建议按以下顺序实现。

### 阶段 1：数据结构

实现：

```text
ResearchRequest
RetrievedDocument
ResearchBundle
```

并完成 JSON 序列化和哈希。

### 阶段 2：Mock Retriever

先不接真实网络：

```python
class MockRetriever:
    def search(self, query, *, asof, max_results):
        return [
            RetrievedDocument(
                title="Mock result",
                url="https://example.com",
                snippet="mock evidence",
                source="mock",
                query=query,
                rank=1,
                published_at="2026-09-20T00:00:00Z",
            )
        ]
```

先把 Engine、缓存、归档和渲染流程跑通。

### 阶段 3：缓存和归档

完成：

```text
search/cache/
search/rounds/
```

实现写入一次、读取复用和 hash 校验。

### 阶段 4：接入 Tavily

保持 `Retriever` 接口不变，上层不感知供应商细节。

### 阶段 5：截止时间过滤

这是上线前不可省略的功能。

### 阶段 6：查询规划器

先模板，再接 LLM 生成查询。

### 阶段 7：接入 harness

让 `harness._retrieve()` 只负责构造 `ResearchRequest` 和调用 `ResearchEngine`。

### 阶段 8：增加专用数据源

建议顺序：

1. 通用网页检索；
2. YouGov / Civiqs；
3. Wikipedia；
4. Google Trends；
5. Michigan / AAII / NY Fed。

---

## 18. 推荐的 MVP 范围

如果第一版时间有限，建议只实现以下功能：

```text
Research Engine MVP
├── ResearchRequest
├── RetrievedDocument
├── ResearchBundle
├── TemplateQueryPlanner
├── TavilyRetriever
├── cutoff filter
├── URL/content deduplication
├── query cache
├── round archive
├── deterministic renderer
├── EmptyCorpus fail-closed
└── offline unit tests
```

主接口：

```python
bundle = engine.retrieve(
    request=request,
    entrant="my-agent",
)
```

预测模块只使用：

```python
bundle.rendered_text
```

第一版不建议同时做：

- 复杂多轮 Agent 搜索；
- embedding-based reranking；
- 自动网页全文解析；
- 每个 subgroup 单独搜索；
- 多个搜索供应商自动切换；
- 搜索结果自动生成二次摘要。

这些功能可以在 MVP 稳定后再加入。

---

## 19. 关键设计原则总结

### 原则 1：检索和预测分离

Retrieval Engine 提供证据，Forecast Engine 生成分布或排名。

### 原则 2：时间边界优先

任何文档都必须能回答：“它在 `asof` 时刻已经存在吗？”

### 原则 3：每个 round 的研究资料冻结

同一个 entrant 在同一个 round 不应因为 refresh 或重试看到不同的研究结果。

### 原则 4：实验条件统一

统一搜索引擎、查询上限、结果数、截断长度、渲染格式和时间策略。

### 原则 5：可审计

必须记录查询、来源、时间、结果和 bundle hash。

### 原则 6：失败时不要伪造成功

没有合法研究结果就不要把 web forecast 当作普通 forecast 提交。

### 原则 7：结构化来源优先

已知官方数据源时，应优先使用官方数据而不是依赖泛化新闻搜索。

---

## 20. 建议的第一个提交（First PR）

如果需要把任务拆成一个可审查的第一个 Pull Request，可以只包含：

```text
1. 新增 ssa/research/models.py
2. 新增 ssa/research/filters.py
3. 新增 ssa/research/renderer.py
4. 新增 ssa/research/archive.py
5. 新增 MockRetriever
6. 新增离线单元测试
7. 暂时不改变生产 harness 的调用路径
```

第二个 PR 再包含：

```text
1. 接入 TavilyRetriever
2. 将现有 search.py 的缓存逻辑迁移或封装
3. 接入 ResearchEngine
4. 修改 harness._retrieve()
5. 运行完整 harness 测试
```

第三个 PR 再增加：

```text
1. 结构化数据源
2. Wikipedia 专用研究器
3. Google Trends 专用研究器
4. evidence ranking
5. 搜索质量评估指标
```

这样可以把“数据结构正确性”“网络检索正确性”和“预测效果优化”分开审查，降低一次性改动的风险。

