# MIT SSA 使用说明

本文档说明当前 `ssa_harness` 的安装、配置、接口、四阶段预测流程、数据源和测试方式。它面向本地联调与比赛前的受控演练，不代替比赛官方文档，也不会自动报名或提交预测。

## 1. 项目做什么

项目把一次预测请求串成一个可回放的流水线：

```text
官方请求 / ForecastRequest
        ↓
Research / Retrieval Engine       检索证据、时间过滤、去重、排序、归档
        ↓
Context Builder                   历史规范化、证据摘要、上下文分段
        ↓
Forecast Prompt Compiler          固定系统提示、任务提示、输出 Schema
        ↓
LLM 成员 + Baseline 成员
        ↓
Ensemble Engine                  校验、加权、分布合成、最终响应
        ↓
官方 response envelope + SQLite 运行归档
```

当前主包位于 `src/ssa_harness/`。`archive/legacy/` 是迁移前项目，仅用于保留历史，不是当前运行入口。

当前流水线直接支持三种任务：

| `target_type` | 输出 | 用途 |
|---|---|---|
| `continuous_normal` | `{mean, sd}` | 连续值预测，使用正态分布 |
| `profile_energy` | `{profile: {cell: {mean, sd}}}` | 多个 cell 的联合预测 |
| `ranking_list` | `{ranking: [...]}` 或固定候选的 `scores + ranking` | 固定候选或开放 Top-N 排名 |

官方 Schema 还列出其他类型，但当前 `ForecastRequest` 和 Pipeline 只接受上述三种类型。

## 2. 安装与基本验证

在仓库根目录 `MIT_SSA` 中运行：

```powershell
python -m pip install -e ".[test,server]"
python -m pytest -q
```

离线四种任务演示不需要密钥，也不调用网络：

```powershell
python -X utf8 -m ssa_harness demo --output-dir data/demo
```

演示会生成 Topline、Profile、固定 Ranking 和开放 Ranking 的 envelope、corpus、回放响应、配置以及完整运行记录。

## 3. 密钥和环境变量

密钥不写入 JSON、源代码、SQLite、日志或 Git。当前项目不会自动读取 `.env` 文件。

PowerShell 临时设置：

```powershell
$env:TAVILY_API_KEY = "你的 Tavily key"
$env:SSA_LLM_API_KEY = "你的 LLM 或中转 key"
```

永久保存到当前 Windows 用户：

```powershell
[Environment]::SetEnvironmentVariable("TAVILY_API_KEY", "你的 Tavily key", "User")
[Environment]::SetEnvironmentVariable("SSA_LLM_API_KEY", "你的 LLM 或中转 key", "User")
```

永久设置后需重新打开终端。只能检查变量是否存在，不要打印密钥内容：

```powershell
python -c "import os; print('Tavily:', bool(os.getenv('TAVILY_API_KEY'))); print('LLM:', bool(os.getenv('SSA_LLM_API_KEY')))"
```

Tavily 需要 `TAVILY_API_KEY`。YouGov 公共 tracker 和 Civiqs dashboard 当前适配器不需要 API key。LLM 密钥变量名由 `models[].api_key_env` 指定，因此第三方中转也可以使用 `RELAY_API_KEY` 等自定义名称。

## 4. 配置文件

仓库提供模板 `configs/live.example.json`。不要直接把模板当作个人配置；复制一份：

```powershell
Copy-Item configs/live.example.json configs/live.json
```

配置文件路径是 `configs/live.json` 时，里面的相对路径按 `configs/` 目录解析，例如 `../data/research/live.sqlite3` 会落到仓库的 `data/research/`。

### 4.1 Pipeline 配置

```json
"pipeline": {
  "entrant": "ruc-team-01",
  "mode": "live",
  "research_policy": "required",
  "allow_baseline_fallback": false,
  "repair_attempts": 1,
  "max_elapsed_seconds": 480,
  "sd_scale": 1.0
}
```

- `entrant`：运行身份，只能使用字母、数字、`.`、`_`、`-`，且以字母或数字开头。
- `mode`：`snapshot` 或 `live`。实时测试使用 `live`；严格回测使用 `snapshot`。
- `research_policy`：`required` 强制检索；`disabled` 明确关闭检索。
- `allow_baseline_fallback`：是否允许所有 LLM 失败后仍用 baseline 输出。正式运行建议保持 `false`，避免错误被隐藏。
- `repair_attempts`：非法模型 JSON 的格式修复次数，当前最多 1 次。
- `max_elapsed_seconds`：整个 Pipeline 时间预算，范围为 1 到 540 秒。
- `sd_scale`：最终标准差校准倍数，默认为 1。

### 4.2 Research 配置

Tavily 实时检索：

```json
"research": {
  "provider": "tavily",
  "db": "../data/research/live.sqlite3",
  "config": {
    "max_queries": 4,
    "results_per_query": 3,
    "max_elapsed_seconds": 150,
    "allow_partial": false
  }
}
```

`max_queries: 1` 可用于单 query 联调。`allow_partial: false` 会在查询失败时直接报错；只有明确接受不完整证据时才改为 `true`。

可选检索提供者：

| `provider` | 作用 | 是否联网 | 密钥 |
|---|---|---:|---|
| `snapshot` | 读取本地 corpus JSON | 否 | 无 |
| `structured` | 搜索 source package | 否 | 无 |
| `tavily` | Tavily Web Search API | 是 | `TAVILY_API_KEY` |
| `civiqs` | Civiqs dashboard 专用适配器 | 是 | 无 |
| `yougov` | YouGov 公共 XLSX 专用适配器 | 是 | 无 |

Civiqs 示例：

```json
"research": {
  "provider": "civiqs",
  "tracker": "approve_president_trump_2025",
  "filters": {},
  "user_agent": "MIT-SSA-your-team/0.1 (contact: you@example.org)",
  "config": {"max_queries": 1, "results_per_query": 3}
}
```

YouGov 示例：

```json
"research": {
  "provider": "yougov",
  "tracker": "donald-trump-approval",
  "user_agent": "MIT-SSA-your-team/0.1 (contact: you@example.org)",
  "config": {"max_queries": 1, "results_per_query": 3}
}
```

YouGov 的公共数据许可限制自动化抓取。正式使用前应核对并取得许可；没有许可时，建议手动下载 XLSX 后保存为 source package。Civiqs 没有公开 REST API，适配器读取结果页面内嵌的 Remix loader 数据；云运行器可能收到 403，应准备离线归档或可访问该网站的运行器。

### 4.3 Context 配置

```json
"context": {
  "strict_snapshot": false,
  "max_chars": 14000,
  "max_token_upper_bound": 16000
}
```

实时检索使用 `strict_snapshot: false`，保留 live 证据的时间不确定性警告。严格历史回测使用 `snapshot` provider 和 `strict_snapshot: true`。

### 4.4 LLM 配置

```json
"models": [
  {
    "member_id": "llm-primary",
    "kind": "chat_completions",
    "model": "你的中转模型名称",
    "endpoint": "https://relay.example.com/v1/chat/completions",
    "api_key_env": "SSA_LLM_API_KEY",
    "timeout": 90,
    "max_attempts": 2,
    "strict_schema": false
  }
]
```

第三方中转必须兼容 OpenAI Chat Completions：接受 `model`、`messages`、`response_format`、`stream: false`，并返回 `choices[0].message.content`。当前客户端使用 `Authorization: Bearer <key>`，要求 HTTPS endpoint，不能把 key 放进 URL 或 query 参数。

如果中转商的环境变量名是 `RELAY_API_KEY`，同时修改：

```json
"api_key_env": "RELAY_API_KEY"
```

`strict_schema: false` 使用 `json_object` 加本地严格校验，通常更适合中转服务；只有确认模型支持 OpenAI `json_schema` 才设为 `true`。模型参数写在：

```json
"prompt": {"model_params": {"temperature": 0, "top_p": 1}}
```

### 4.5 Ensemble 配置

```json
"ensemble": {
  "failure_policy": "drop_invalid",
  "weights": {"llm": 0.6, "persistence": 0.25, "recent_mean": 0.15},
  "baseline": {"recent_window": 3, "min_sd": 0.25}
}
```

`llm`、`persistence`、`recent_mean` 是成员类型权重。`drop_invalid` 会排除非法成员并继续；`strict` 会在成员失败时直接失败。权重会对当前可用成员重新归一化。

## 5. 请求接口规范

官方 HTTP 请求是一个 JSON 对象，不带额外 wrapper。完整官方 Schema 位于：

```text
src/ssa_harness/schemas/agent-api-request.schema.json
```

顶层结构：

```json
{
  "schema_version": "ssa-agent-api-v2",
  "request_id": "request-20260925-001",
  "round": {
    "round_id": "round-001",
    "board_id": "topline",
    "target_type": "continuous_normal",
    "question": "What will the approval rating be next week?",
    "unit": "percentage points",
    "lock_at": "2026-12-31T00:00:00Z",
    "context": {},
    "ranking": {}
  },
  "optional_crosstabs": []
}
```

关键字段：

- `schema_version` 必须为 `ssa-agent-api-v2`。
- `request_id` 用于请求幂等和签名绑定。
- `round.round_id` 必须是小写字母、数字和连字符组成的标识。
- `board_id` 当前使用 `topline`、`profile` 或 `ranking`。
- `target_type` 当前 Pipeline 支持 `continuous_normal`、`profile_energy`、`ranking_list`。
- `question` 是任务问题，也是查询规划的主要输入。
- `unit` 必须明确，例如 `percentage points`、`index points`。
- `lock_at` 是参赛端实际截止时间。
- `context` 保存官方冻结历史；不能把截止时间误当成预测目标日期。
- `optional_crosstabs` 是可选的维度和 groups 列表。

内部 CLI 测试也可以直接使用 `ForecastRequest` JSON。最小连续值请求示例：

```json
{
  "round_id": "manual-test-001",
  "question": "What will Donald Trump's approval rating among US adults be next week?",
  "target_type": "continuous_normal",
  "unit": "percentage points",
  "asof": "2026-09-24T00:00:00Z",
  "deadline": "2026-12-31T00:00:00Z",
  "history": [
    {"date": "2026-09-21", "available_at": "2026-09-21T12:00:00Z", "value": 42.0},
    {"date": "2026-09-22", "available_at": "2026-09-22T12:00:00Z", "value": 42.5},
    {"date": "2026-09-23", "available_at": "2026-09-23T12:00:00Z", "value": 43.0}
  ],
  "source_name": "Public polling sources",
  "population": "US adults",
  "metric": "approval percentage"
}
```

历史记录必须带 `available_at` 或 `published_at`。只有 `date` 不能证明当时已经可用；超过 `asof` 的记录会被排除。

## 6. 响应接口规范

完整官方响应 Schema 位于：

```text
src/ssa_harness/schemas/agent-api-response.schema.json
```

连续值：

```json
{
  "schema_version": "ssa-agent-api-v2",
  "forecast": {"mean": 43.2, "sd": 1.4}
}
```

Profile：

```json
{
  "schema_version": "ssa-agent-api-v2",
  "forecast": {
    "profile": {
      "Democrat": {"mean": 18.0, "sd": 2.0},
      "Republican": {"mean": 72.0, "sd": 2.5}
    }
  }
}
```

Ranking：

```json
{
  "schema_version": "ssa-agent-api-v2",
  "forecast": {"ranking": ["Item A", "Item B", "Item C"]}
}
```

`sd` 必须为正数。点预测不能代替分布预测。内部固定排名可以保留 scores，但官方 response adapter 会按协议输出 ranking。

## 7. CLI 运行方式

### 7.1 运行一条实时测试问题

创建 `data/manual-test/request.json` 后：

```powershell
python -X utf8 -m ssa_harness forecast `
  --config configs/live.json `
  --request data/manual-test/request.json `
  --full-run `
  --output data/manual-test/run.json
```

不加 `--full-run` 时只打印最终 response。完整运行文件包含：

```text
research          检索查询、证据、排除项、时间保证
context           Context Builder 产物
compiled_prompt   Prompt Compiler 产物和 prompt_hash
model_runs        原始模型输出、修复结果、校验结果
final             Ensemble Engine 产物和成员权重
response          官方格式响应
```

### 7.2 运行官方 envelope

```powershell
python -X utf8 -m ssa_harness forecast `
  --config configs/live.json `
  --envelope input-envelope.json `
  --full-run `
  --output data/runs/forecast-full.json
```

`--asof` 可以显式覆盖默认的 endpoint 时间策略：

```powershell
python -m ssa_harness forecast `
  --config configs/live.json `
  --envelope input-envelope.json `
  --asof 2026-09-24T00:00:00Z
```

### 7.3 只运行检索模块

```powershell
python -m ssa_harness.retrieval run `
  --request research-request.json `
  --provider tavily `
  --entrant manual-test `
  --db data/research/manual.sqlite3 `
  --output data/research/manual-bundle.json
```

### 7.4 离线回测、评分和校准

```powershell
python -m ssa_harness backtest --config configs/snapshot.json --cases cases.json
python -m ssa_harness evaluate --records records.json --output data/evaluation.json
python -m ssa_harness calibrate `
  --cases examples/calibration-cases.synthetic.json `
  --validation-start 2026-08-01T00:00:00Z `
  --test-start 2026-09-01T00:00:00Z `
  --output data/calibration.json
```

### 7.5 结构化来源导入

通用 CSV：

```powershell
python -m ssa_harness import-source --kind series-csv `
  --input poll.csv `
  --manifest poll-manifest.json `
  --output data/sources/poll.json
```

Civiqs：

```powershell
python -m ssa_harness import-source --kind civiqs `
  --tracker approve_president_trump_2025 `
  --output data/sources/civiqs-approval.json
```

YouGov：

```powershell
python -m ssa_harness import-source --kind yougov `
  --tracker donald-trump-approval `
  --output data/sources/yougov-approval.json
```

Wikipedia 每周聚合：

```powershell
python -m ssa_harness aggregate-wikipedia-week `
  --packages day1.json day2.json day3.json day4.json day5.json day6.json day7.json `
  --week-start 2026-09-07 `
  --length 10 `
  --output data/sources/wiki-week.json
```

source package 包含 `manifest`、`history` 和 `corpus`。配置 `history_sources` 可以把口径完全匹配的历史合并到任务；配置 `research.provider: "structured"` 并列出 `packages` 可以把它们作为检索证据。

## 8. HTTP 服务接口

启动服务：

```powershell
python -X utf8 -m ssa_harness serve `
  --config configs/live.json `
  --keys keys.json `
  --host 127.0.0.1 `
  --port 8787
```

服务端点：

- `GET /health`：返回 `{"status":"ok"}`。
- `POST /forecast`：接受官方 request envelope，返回官方 response envelope。
- `OPTIONS /forecast`：CORS 预检。

默认要求 Ed25519 请求签名：

```text
X-SSA-Key-Id
X-SSA-Timestamp
X-SSA-Signature
```

签名内容是 `timestamp + b"." + raw_request_body`，时间窗口默认 300 秒。服务部署时应放在 TLS reverse proxy 后面；不要直接把本地 HTTP 服务暴露到公网。`--no-verify` 只用于明确的本地调试。

`--keys` 文件只保存公钥，不保存私钥，格式为：

```json
{
  "keys": [
    {"key_id": "entrant-key-1", "public_key": "base64-encoded-ed25519-public-key"}
  ]
}
```

## 9. 归档、缓存和重放

运行结果写入 `run_db` 指定的 SQLite，检索 bundle 写入 `research.db`。归档包含：

- 原始请求和 normalized request；
- 研究查询、证据、排除原因和 temporal assurance；
- Context、compiled prompt、prompt hash；
- 模型原始响应、格式修复、validated prediction；
- ensemble 成员、权重、warnings 和最终响应；
- pipeline、provider、模型和配置 hash。

相同 `(entrant, round_id, request_hash, pipeline_config_hash)` 会重放已有结果，不会重新调用网络或模型。要测试新问题或新配置，应使用新的 `round_id` 或新的实验身份。不要覆盖已有不同内容的输出文件。

## 10. 时间和证据规则

历史值必须有可用时间。系统会在 `asof` 之前过滤、按观测时间排序、按 cell 保留截止时间前最新版本，并拒绝同一观测/可用时间的冲突版本。

实时 Tavily 的 `published_at` 只是供应商元数据，不能证明网页在过去 cutoff 时的内容；实时 bundle 会标记 `publication_metadata_only`，不能当作严格历史回测。严格回测只能使用当时保存的 snapshot 或 source package。

当前 YouGov/Civiqs 适配器在下载时写入当前 `captured_at` 和内容哈希。后来下载的文件不能被手动改写成过去的捕获时间。

## 11. 当前功能和限制

已实现：

- 官方 request/response Schema 校验与 envelope 适配；
- Topline、Profile、固定/开放 Ranking；
- Tavily 实时检索、本地 snapshot、结构化 source package；
- YouGov 公共 tracker XLSX 解析；
- Civiqs dashboard HTML Remix payload 解析；
- CSV、Wikimedia、Google Trends 联合 basket 导入；
- 上下文整理、Prompt 编译、JSON 校验、一次格式修复；
- persistence/recent mean baseline 和多模型 Ensemble；
- SQLite 运行归档、幂等重放、HTTP `/forecast` 服务；
- CRPS、Profile Energy、RBO、Kendall、滚动回测和校准工具。

限制：

- 当前测试数据和校准示例主要是合成数据；不能据此宣称真实预测效果；
- Tavily 和真实 LLM 的可用性、余额、模型兼容性取决于外部服务；
- YouGov 自动化抓取需要遵守其数据许可；
- Civiqs 可能对部分云 IP 返回 403；
- Civiqs 是会修订历史曲线的模型化 tracker，比赛使用前必须建立自己的不可变归档；
- 没有配置真实 API 时，live Pipeline 不能完成真实联调；
- 代码不会替你注册比赛、部署公网 endpoint 或提交预测。

## 12. 故障排查

| 错误 | 常见原因 |
|---|---|
| `missing model credential environment variable` | `api_key_env` 与当前终端环境变量名不一致 |
| `Tavily HTTP 401` | key 错误、过期，或运行程序的终端没有该变量 |
| `model HTTP 404` | 中转 endpoint 没有写完整的 `/chat/completions` |
| `model HTTP 400` 与 `response_format` 有关 | 保持 `strict_schema: false`，检查中转是否支持 `json_object` |
| `invalid model response envelope` | 中转没有返回 `choices[0].message.content` |
| `EmptyCorpus` | 所有检索结果被时间、域名或来源过滤 |
| `asof cannot be in the future` | 请求的 cutoff 晚于当前 UTC 时间 |
| `request_conflict` | 相同参赛者和 round 已绑定了不同请求 |
| `Civiqs HTTP 403` | 当前出口 IP 被 Civiqs 拒绝，改用归档或可访问的运行器 |

## 13. 相关文件

- [README.md](../README.md)：项目概览和快速入口。
- [INTEGRATION.md](INTEGRATION.md)：接口边界、source package 和时间规则。
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)：实现状态和验证边界。
- [live.example.json](../configs/live.example.json)：实时配置模板。
- [agent-api-request.schema.json](../src/ssa_harness/schemas/agent-api-request.schema.json)：官方请求 Schema。
- [agent-api-response.schema.json](../src/ssa_harness/schemas/agent-api-response.schema.json)：官方响应 Schema。
