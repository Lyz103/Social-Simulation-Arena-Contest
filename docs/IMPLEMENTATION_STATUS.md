# 八步实施与验证状态

日期：2026-09-24。官方协议固定至 `e80c8b42c157dadc018441da2a9691b44b20affc`；来源和 Schema SHA-256 见 `src/ssa_harness/schemas/provenance.json`。

| 步骤 | 当前交付 | 状态与验证边界 |
|---|---|---|
| 1. 统一契约与官方适配 | ForecastRequest、三类历史映射、Profile/Ranking 响应转换、官方 Schema | 已实现并验证 |
| 2. 历史规范化与 Context | cutoff 检查、排序、版本去重、各 cell 摘要、固定分区、来源预算、hash | 已实现并验证 |
| 3. 修复 Prompt/Ensemble | 独立目标时间、context 身份、稳定方差、权重归一化、同分规则、显式失败策略 | 已实现，含原问题的回归测试 |
| 4. 模型客户端与 Pipeline | 可配置 chat-completions HTTP、有限重试、一次修复、多模型调用、baseline fallback、CLI/HTTP | 已实现；真实 provider 和 Tavily 未联调，无环境密钥 |
| 5. 开放 Ranking | 候选发现、规范标题/排除规则、开放 Top N、不同候选列表的加权集成 | 已实现；对 redirect 别名未做在线自动解析 |
| 6. 完整归档与幂等 | 请求、research、context、prompt、raw/repair/validated model response、成员失败、最终预测、配置 hash | 已实现；重放、冲突、损坏检测、成功成员恢复测试通过 |
| 7. 契约与端到端测试 | 109 个测试、四种任务完整 demo、本地 HTTP、固定版本官方探针、wheel | 已验证本机 Python 3.12；多系统 CI 配置已提供但尚未在 GitHub 执行 |
| 8. 回测校准与来源适配 | CRPS/Energy/RBO/Kendall、覆盖率、滚动回测接口、时间划分网格选权重/sd、CSV/Wikimedia/Trends 接入 | 软件工具与合成验收完成；真实历史校准与预测效果验收待团队数据 |

## 已修复的审核问题

- Profile history_by_cell 和 Ranking recent_weeks 均转换成统一 history。
- 正式响应把内部 cells 转为 profile、剥离 ranking 内部 scores，并加入完整协议 envelope。
- 基线不会读取 availability 超过 cutoff 的记录；历史按时间处理而不是信任输入次序。
- Context 显式渲染历史，Prompt 不把 deadline 当预测目标。
- 缺 round/asof、研究任务口径不匹配、context 被修改等情况显式拒绝。
- 固定候选与开放列表使用各自契约；开放模型可引入证据支持的新条目。
- 空历史/非法成员按显式策略处理；所有模型失败时不会偷偷使用 baseline。
- 大均值小方差不再因大数相减降至最小 sd；超大权重先缩放再归一化。
- 大幅正负 ranking 分数归一化避免 high-low 溢出。
- 模型 ID 不被错误重命名；固定候选同分规则在 compiler/validator/ensemble 一致。
- 当前快照下载时间不会伪装成历史 captured_at；Wikipedia 日榜不能当周榜真值。

## 验证结果

- `python -m pytest -q`：109 passed。
- 原有 53 个离线测试全部保留并通过；新增契约、时间泄漏、源码缺陷、修复/重试、运行恢复、HTTP 签名等测试。
- 官方非计分 probe：三种形状、幂等、错误签名拒绝均通过。live-key 探针只能证明未知/无效签名被拒绝，不能证明收到真实 live 私钥签名的生产请求；实际本地 HTTP 验签使用临时生成密钥测试。
- 完整四种流程使用固定 ReplayClient 与合成快照，验证流水线和协议；不代表真实模型预测。
- 官方计分对照的固定样例中 Energy 误差约 1.2e-12，RBO/Kendall 一致；正常容差为 1e-7，不声明任意数值上的字节一致。
- 真实 provider 密钥和 TAVILY_API_KEY 均未配置，未消耗模型或搜索费用。

机器可读报告：`reports/official_probe.json`、`reports/official_metrics.json`、`reports/demo.json`、`reports/backtest.synthetic.json`、`reports/calibration.synthetic.json`、`reports/verification.json`。

## 真实完成验收仍需的外部输入

1. 实际模型 provider、model ID 与环境密钥，以及 Tavily key 或可信研究快照。
2. 具备可验证可用时间的历史数据、真实未来目标、每轮官方 persistence null。
3. 各来源的准确字段和测量口径。已加入 YouGov 公共 tracker XLSX 与 Civiqs dashboard HTML 的专用适配器；仍需在比赛前完成来源许可、归档和真实运行器网络可达性确认。
4. 若正式参赛：部署 HTTPS endpoint 后，以真实 round 执行 rehearsal、再报名。这里没有部署、报名或发出比赛预测。

不能把第 8 步的软件实现/合成回测报告说成已完成真实权重校准；也不能把第 4 步的本地模拟 HTTP 测试说成真实供应商联调通过。获取上述输入后无需重搭主架构，按 README 命令即可继续。
