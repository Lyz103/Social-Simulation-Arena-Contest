# 团队接口与验收约定

更新时间：2026-09-24。当前包名是 `ssa_harness`，与官方仓库的 `ssa` 分离。

## 1. 入口数据

推荐接官方 envelope：

```python
from ssa_harness.contracts import ForecastRequest
request = ForecastRequest.from_envelope(envelope, metadata={
    "source_name": "具体来源",
    "population": "具体总体",
    "metric": "具体目标变量",
    "minimum": 0,
    "maximum": 100,
    "forecast_for": "题目给出的目标波次或窗口"
})
```

参数只填有依据的字段。不要统一把所有指标限制为 0..100；单位、范围、测量窗口以各题为准。

如果 Snapshot Builder 直接产生内部请求，使用 `ForecastRequest.from_dict()`。Topline history 样例：

```json
{
  "date": "2026-09-17",
  "available_at": "2026-09-18T10:00:00Z",
  "captured_at": "2026-09-18T10:05:00Z",
  "value": 41.2,
  "source_url": "https://source.example/report",
  "content_hash": "sha256:...",
  "provenance": "source_capture_asserted"
}
```

Profile 的 value 为 `{"cell_a": 10, "cell_b": 20}`；允许每条只有一个 cell，会独立保留每个 cell 的最新有效观测。Ranking 的 value 是有序字符串列表，附加 `views` 或 `index` 可保留连续指标。未知发布时间不能用观测日期代替。

`normalize_history()`：过滤 cutoff 后公布或捕获的材料；排序；合并完全一致重复行；相同观测时间的版本只保留 cutoff 前最后可用版本；同一观测/可用时间数值冲突则报错。所有 baseline 和 Context 都调用它，不各自猜测时间规则。

## 2. Snapshot Builder 的交付

提供以下任意一种：

- `ResearchRequest(mode="snapshot")` + 捕获时间合格的 corpus JSON；
- 已封存且可校验的 `ResearchBundle`；
- `import-source` 产生的结构化 source package，其 history 并入任务、corpus 用于检索。

每个快照保留原始字节、实际捕获时间、来源 URL、内容哈希、版本与外部归档出处。哈希只验证内容一致性，不能证明伪造的旧时间戳是真的。项目不会替调用方制造历史可用性证明。

严格快照回测不使用当前 Tavily 搜索。可复用回测期间封存的证据，但不能将现在下载的网页随意标记为过去的 captured_at。

## YouGov 与 Civiqs

`ssa_harness.source_adapters` 提供 `YouGovRetriever`、`CiviqsRetriever` 以及离线 package 生成器。两者的当前公共入口均不使用 API key；适配器会在下载时记录 `captured_at` 和 `sha256`，并拒绝历史模式下把实时网页/XLSX 当成严格快照。YouGov 端点返回 OOXML XLSX，解析时按 workbook relationships 映射 sheet，避免工作簿重排导致 subgroup 错位。Civiqs 端点是结果页面内嵌的 Remix loader JSON，不是公开 REST API；解析器使用括号平衡扫描并验证 tracker payload。

配置示例：

```json
{"research":{"provider":"civiqs","tracker":"approve_president_trump_2025","config":{"max_results":3}}}
```

提交比赛前应把抓取结果导出为 source package 并提交/保存其原始哈希。YouGov 自动化下载须先核对其 public-data license 并取得书面许可；没有许可时使用人工下载的 XLSX 归档。Civiqs 无 key，但云 IP 可能遭到 403，需准备可访问站点的运行器或离线归档。

## 3. Pipeline 使用

```python
result = pipeline.run(request)
official_response = result["response"]
final_prediction = result["final"]["prediction"]
```

`ForecastPipeline` 的所有 I/O 都是可注入依赖；运行配置参与 hash。无研究模式、成员失败剔除和基线降级必须显式配置。

所有 LLM 共享本次固定 `CompiledPrompt` 和相同证据。客户端不自行搜索。模型原始输出在解析前写入事件表，格式修复的原始和修复回复均保留；成功模型结果可在进程失败后的重试中复用。整个结果封存后，即使模型配置端点外部发生变化也不会再次调用；改变本地配置则要求新实验身份。

不可覆盖的主键为 `(entrant, round_id)`。`request_id` 与原始 envelope hash 另外绑定，防止同一个请求 ID 被换题复用。数据库在同一本地文件系统内协调；跨主机应替换为支持事务的中心存储。

同步 HTTP 使用超时和有限重试，阶段之间检查总预算。Python 不能强制中断任意自定义同步回调；生产部署还需进程级超时和资源限制。归档 lease 比预算多 60 秒，定制阻塞组件不能无限运行。

## 4. 开放排名

固定篮子：`ranking_spec.items` 是官方给出的全集，最终排序包含所有条目。相同分数按官方输入次序打破并列。

开放 Top N：`ranking_spec.length` 是输出长度，`candidates` 只是建议池，不是封闭全集。建议池由历史频率/名次、结构化候选字段和 Wikipedia 证据 URL 形成；模型可根据给定证据提出新项。不同成员列表通过加权倒数排名合并，按历史顺序、再按名称处理并列。Wikipedia 主页面和非正文命名空间被排除；空格转换下划线、首字母标准化不等于在线 redirect/alias 解析。需要可靠别名映射时由来源 adapter 提供，历史映射也应有时间边界。

模型不能凭空联网寻找额外资料。候选池过小有警告；模型凑不齐合法输出会失败或按已配置策略使用 baseline，不填造假的条目。

## 5. 交给 Backtester 的记录

`walk_forward(cases, predictor)` 和 CLI `backtest` 的 case：

```json
{
  "request": {"...": "完整 ForecastRequest"},
  "truth": 42.0,
  "truth_available_at": "2026-09-24T10:00:00Z",
  "persistence": {"mean": 41.0, "sd": 1.0}
}
```

Profile truth 是精确 cell map，Ranking truth 是最终有序列表。真值在 predictor 外部保留，不进入 context 或 request.metadata。`truth_available_at` 必须晚于 `asof`。

`persistence` 若用于声称官方 skill，应由该 round 官方冻结 null 提供。我们的 BaselineFactory 是候选策略之一，其默认 sd 未必等于官方 null 的 sd，不应混称。

评分：Topline CRPS；Profile 按 request.cells 的固定次序计算完整 Energy；开放榜单 RBO loss（默认 p=0.9，可在 ranking_spec.rbo_p 指定）；固定篮子 Kendall distance。不混合不同量纲的原始损失。

校准 case 另提供成员列表：

```json
{
  "request": {"...": "完整 ForecastRequest"},
  "truth": 42.0,
  "truth_available_at": "2026-09-24T10:00:00Z",
  "members": [
    {"member_id": "model-a", "member_type": "llm", "prediction": {"mean": 42, "sd": 1}},
    {"member_id": "persistence", "member_type": "persistence", "prediction": {"mean": 41, "sd": 1}}
  ]
}
```

当前是显式网格选择，不训练未知数据集。仅 validation 选择 weights/sd_scale，test 用于最终评估，且 validation 真值在 test_start 前必须已公开。调 Prompt 的轮次也必须绑定固定验证集，不能看 test 后继续反复选择最优配置。

模型训练数据污染不由检索时间过滤解决。真实历史评估还应记录 model ID/版本、已知训练截止边界、回测时期和证据可信度。合成测试或模型可能记住的旧题不构成无污染预测成绩。

## 6. 正式上线前仍须提供

1. 真实模型 endpoint、model ID 和环境变量密钥；本地未设置密钥。
2. 可信历史快照、真值发布时间与官方 null；目前附带的校准数据全部是合成样例。
3. 计划覆盖的来源/题目定义映射。通用 CSV 和 snapshot 接口已经实现，来源专用 PDF/XLSX/网页抽取需要根据实际源文件继续接入。
4. HTTPS 部署域名和公钥更新策略；本次只启动过临时 loopback 测试服务，未部署公网。

这些是外部输入与效果验收条件，不应由代码编造。提供后可以使用现有命令直接开展真实联调和滚动回测。
