# Retrieval Engine/Context Builder/Forecast Prompt Compiler/Ensemble Engine
# MIT SSA Forecasting Harness

完整使用说明见 [docs/USAGE.md](docs/USAGE.md)，包含接口规范、配置字段、密钥、实时联调、数据源和 HTTP 服务。

预测输入、官方平均 skill、可选统计基线与按任务校准的更新见 [docs/FORECAST_OPTIMIZATION.md](docs/FORECAST_OPTIMIZATION.md)。真实旧题重跑仅用于开发验证，不代表独立测试集或比赛成绩。

统一的 Social Simulation Arena 预测工程，Python 3.10+。检索、上下文、Prompt 和集成模块位于同一级目录。当前提供可运行的离线全链路、官方接口适配、可配置模型 HTTP 客户端、HTTP endpoint、运行归档、计分和按时间划分的校准工具。

**验证边界：离线/本地 HTTP 测试不等于真实预测效果。** 本地样例与校准样例均为合成数据；尚未使用真实模型/Tavily 密钥联调，也没有宣称在真实留出集上击败 Persistence。

## 目录

```text
MIT_SSA/
├── src/ssa_harness/
│   ├── retrieval/       # 信息搜索、快照检索、过滤、缓存、证据归档
│   ├── context/         # 历史摘要、证据分区、预算、context hash
│   ├── prompt/          # 固定协议、JSON Schema、模型输出校验
│   ├── ensemble/        # 基线、数值分布和固定/开放列表集成
│   ├── contracts.py     # ForecastRequest、官方请求/响应适配
│   ├── history.py       # 所有下游共用的历史时间过滤与规范化
│   ├── ranking.py       # 开放排名候选发现与标题规则
│   ├── sources.py       # 结构化 CSV/JSON/Wikimedia/Trends 适配
│   ├── llm.py           # 模型客户端和明确标记的离线回放客户端
│   ├── pipeline.py      # 完整预测流水线
│   ├── archive.py       # SQLite 全运行归档与幂等
│   ├── evaluation.py    # 官方口径指标、滚动回测、权重/sd 校准
│   ├── server.py        # /forecast、签名验证、CORS
│   ├── cli.py           # 统一命令行
│   └── schemas/         # 固定版本官方 Schema 和来源指纹
├── tests/               # 原有测试 + 契约/回归/端到端测试
├── configs/             # 可配置运行示例
├── examples/            # 合成校准数据
├── scripts/             # 官方核验脚本、一次性迁移脚本
├── docs/                # 接口、实施状态、迁移说明
├── reports/             # 验证结果、合成校准报告
├── data/                # 本地运行产物；数据库不进入 Git
├── archive/legacy/      # 两个原项目完整保留；不作为当前代码运行
├── pyproject.toml
└── Social Simulation Arena.md
```

包名使用 `ssa_harness`，避免与官方仓库自身的 `ssa` 包冲突。以前的 `retrieval`、`ssa.forecast`、`ssa.ensemble` 导入分别改为 `ssa_harness.retrieval`、`ssa_harness.prompt`、`ssa_harness.ensemble`。

## 安装与测试

在 MIT_SSA 根目录运行 PowerShell：

```powershell
python -m pip install -e ".[test,server]"
python -X utf8 -m unittest discover -s tests -v
# 或
python -m pytest -q
```

如果暂不安装包，可在当前终端设置 `$env:PYTHONPATH='src'`；正式运行建议安装，避免依赖工作目录。

## 一次跑通四种任务

```powershell
python -X utf8 -m ssa_harness demo --output-dir data/demo
```

它对 Topline、Profile、固定候选 Ranking、开放 Ranking 分别运行：

```text
官方格式请求 → 适配 → 历史规范化 → 本地证据检索 → Context
→ Prompt → 合成模型回放 → 校验 → 基线与集成 → 官方格式响应 → 归档
```

每种任务在输出目录产生 envelope、corpus、model responses、config 和完整 run JSON。再次执行使用相同归档，模型不再调用。不同输入/配置不能覆盖同一 entrant/round；迭代策略时使用新的实验 entrant 或新的数据库。

**演示模型是 ReplayClient，不是实际 LLM。** `live` 模式禁止使用它。

## 真实模型与搜索

1. 复制 `configs/live.example.json` 为自己的配置。
2. 设置真实 provider 的 HTTPS `/chat/completions` 地址和 model ID。
3. 在环境变量中设置 `SSA_LLM_API_KEY` 与 `TAVILY_API_KEY`。不要写入配置、日志或仓库。
4. 根据模型支持的参数调整 `prompt.model_params`；不支持 temperature/top_p 时可明确设为 `{}`。
5. 提供真实官方 envelope，检查各题的来源、population、单位与目标窗口；可通过 `task_metadata_by_round` 补充。

```powershell
python -X utf8 -m ssa_harness forecast `
  --config configs/live.json `
  --envelope input-envelope.json `
  --output data/forecast-response.json
```

加 `--full-run` 输出完整运行记录。默认只输出可供官方 endpoint 返回的响应对象。

客户端支持 `json_object` + 本地严格校验，或配置 `strict_schema: true` 使用 JSON Schema 请求。不同 provider/model 支持的 schema 关键字不同；本项目不在失败时偷偷切换模型或协议。真实 provider 支持情况必须通过联调确认。每个成员有限 HTTP 重试，非法模型输出最多再修复一次，所有原始文本和修复结果均归档。

`research_policy: "disabled"` 明确运行仅历史模式；搜索失败不会被暗中改为空上下文。`allow_baseline_fallback: true` 才允许所有 LLM 失败后的基线预测，结果标记 `degraded`。

## 时间边界

- 官方 endpoint envelope 默认按固定版本的 call-window-open 政策取 `asof = lock_at - 24h`。此规则有版本标记，不适用于任意新赛季或其他提交协议；调用方可以明确传 `--asof`。
- 原始历史必须有 `available_at` 或 `published_at`。只有观测日期不构成可用性证明。
- 官方 frozen context 的可用性标为 `official_frozen_context_asserted`；如指定更早 cutoff，没有更精确可用时间的行会被排除。
- 历史先过滤、按观测时间排序、同观测时间保留 cutoff 前最新版本，再由 Context 与 baseline 共用。
- `deadline`、`target_start/end`、`forecast_for`、`release_at` 不混用。官方 envelope 未提供结构化目标时间时保留未知，并使用题目原文，不猜测截止日期就是预测日期。
- snapshot Context 默认拒绝 publication-only 的 live 搜索。live 示例显式设置 `strict_snapshot: false`，会保留时间保证不足的警告，不能拿它当严格历史回测。

## HTTP endpoint

下载当前官方 `keys.json`（仅公钥）后：

```powershell
python -X utf8 -m ssa_harness serve `
  --config configs/live.json --keys keys.json `
  --host 127.0.0.1 --port 8787 --enable-probe-fixtures
```

部署时用 TLS reverse proxy 暴露最终 HTTPS `/forecast` URL。代码不代办报名、不部署公网、不提交比赛预测。

签名验证默认开启，校验收到的原始字节和 300 秒时间窗口；重复请求返回相同完整预测，复用 request_id 改变内容返回 409。`--no-verify` 是明确的关闭选项。健康检查是 `/health`，支持 onboarding 所需 OPTIONS/CORS。

`--enable-probe-fixtures` 只为官方三个保留的 `ssa-contract-test*` 非计分测试题生成固定协议答复，避免 2099 年的假截止时间进入真实研究流程。它不对普通 round 生效，不证明真实模型能够预测。独立真实流程的 HTTP 测试使用正常 envelope 和完整 Pipeline。

核验官方工具（联网下载固定 commit 的测试脚本，只请求本机）：

```powershell
python scripts/verify_official_probe.py
python scripts/verify_official_metrics.py
```

## 结构化来源

通用民调/指标 CSV（列 `date,value,available_at`，人口表再带 `cell`）：

```powershell
python -m ssa_harness import-source --kind series-csv `
  --input poll.csv --manifest poll-manifest.json --output data/poll-source.json
```

人口表加 `--cell-column cell`。`wikipedia-json` 支持 Wikimedia daily top 原始 JSON，过滤 Main_Page 和非文章命名空间；`trends-csv --items A B C` 支持同一批次联合归一化的 basket 数据，`--shares` 输出份额。

manifest 至少包含 `source_name, source_url, captured_at, unit, population`，可携带 `content_hash`、`available_at`、`published_at`。合并到目标历史还必须有与题目完全相同的 `metric`。Trends 需要 `joint_basket: true`，禁止把各自独立缩放的序列直接当可比指数。

导入结果包含 `manifest/history/corpus`。配置的 `history_sources` 可以列出这些文件；系统严格核对 source/population/unit/metric 后并入历史，再执行统一时间过滤。若要搜索这些材料，`research.provider` 设为 `structured` 并列出 `packages` 文件路径。相关但口径不同的材料只进入证据，不合并目标历史。

`sources.fetch_wikipedia_daily()` 可采集当前 API 响应及捕获时间。现在获取过去某天的数据，不会伪装成当时已有的历史快照。

YouGov/Civiqs 已有专用适配器。它们写入同样的 `manifest/history/corpus` source package，并保留捕获时间、原始内容哈希和来源元数据：

```powershell
python -m ssa_harness import-source --kind civiqs `
  --tracker approve_president_trump_2025 `
  --output data/sources/civiqs-approval.json

python -m ssa_harness import-source --kind yougov `
  --tracker donald-trump-approval `
  --output data/sources/yougov-approval.json
```

配置实时检索时可将 `research.provider` 设为 `civiqs` 或 `yougov`，并填写 `tracker`、可选的 `filters` 和描述性 `user_agent`。两者当前公共入口都不要求 API key：Civiqs 从 dashboard HTML 的 Remix loader 数据读取，YouGov 从公共 XLSX 下载端点读取。YouGov 公共数据许可限制自动化抓取；正式启用前应取得书面许可，或由维护者手动抓取后提交不可变 source package。Civiqs 页面可能对云运行器返回 403，因此比赛运行建议使用已归档 package 或能访问该站点的受控机器。Tavily 仍单独需要 `TAVILY_API_KEY`，模型调用需要 `SSA_LLM_API_KEY`（或兼容 chat-completions 的配置）。

Wikipedia 周榜历史不能直接使用日榜。`aggregate-wikipedia-week --packages day1.json day2.json ... --week-start 2026-09-07 --length 10 --output week.json` 要求完整七天日榜，汇总已归档的日榜流量；会明确记录日榜截断带来的覆盖限制。日榜可用于研究候选，只有 `frequency: weekly` 的包能并入周榜目标历史。

## 回测与校准

```powershell
python -m ssa_harness calibrate `
  --cases examples/calibration-cases.synthetic.json `
  --validation-start 2026-08-01T00:00:00Z `
  --test-start 2026-09-01T00:00:00Z `
  --output data/calibration-demo.json
```

只在 validation 上选择权重和 sd scale；test 完全留出。validation 真值必须在 test 开始前已可用。不同单位、来源、population、cell 定义需分别校准。合成示例只用于验收软件，**不要把示例选择出的权重用于比赛**。

`evaluate --records file.json` 评估已产生预测；`backtest --config snapshot.json --cases cases.json` 在快照模式逐题运行。样本真值只交给评分器，不传给 predictor。指标是 CRPS、完整 Profile Energy score、开放列表 RBO loss 和固定候选 Kendall distance。Profile 使用官方相同的 400 点确定性样本和 seed，正态分位数的独立实现与固定官方样例在 1e-7 容差内比对。

与 Snapshot/Backtester 同学的接口及真实数据验收要求见 `docs/INTEGRATION.md`。实施状态见 `docs/IMPLEMENTATION_STATUS.md`。
