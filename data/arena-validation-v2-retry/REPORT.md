# Arena 真实题目验证

已结算题型与数量：{'continuous_normal': 1}。
只使用官方 answer-window 历史，准备的输入最多数值 24 点、每群体 12 点；模型实际可见窗口由运行归档的 context.config 决定。预测阶段不读取答案和其他参赛者预测。
本批为已分析过的开发集，不能作为独立测试集。统计候选使用预先配置的参数，并不依据本批结果自动选优。
下表 mean skill 按官方逐题平均口径计算，0 为 Persistence；aggregate_skill 仅保留为历史兼容字段，不能当官方均分。

| 策略 | 题型 | 题数 | mean skill ×100 | 胜 / 平 / 负（对官方 Persistence） |
|---|---|---:|---:|---|
| official_persistence | crps | 1 | 0.00 | 0 / 1 / 0 |
| persistence | crps | 1 | 33.99 | 1 / 0 / 0 |
| recent_mean | crps | 1 | 43.62 | 1 / 0 / 0 |
| baseline_ensemble | crps | 1 | 55.49 | 1 / 0 / 0 |
| ensemble | crps | 1 | 53.46 | 1 / 0 / 0 |
| llm | crps | 1 | 32.23 | 1 / 0 / 0 |

## 数值题明细

mean ± sd 不是置信区间；CRPS 越低越好。

| 题目 | 真值 | 基线集成 mean ± sd | 官方 Persistence CRPS | 基线集成 CRPS | LLM mean ± sd | LLM CRPS | 最终集成 CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| aaii-2026-09-24 | -15.4 | -18.275 ± 14.768 | 8.2537 | 3.6737 | -6.000 ± 11.000 | 5.5936 | 3.8413 |

## 分群题明细

Energy score 越低越好，每题有 16 个 cell。

| 题目 | 官方 Persistence | 基线集成 | 纯 LLM | 最终集成 |
|---|---:|---:|---:|---:|

## 模型运行

返回模型：deepseek-flash；成功题数：1。
成功运行记录中的 token：17680（包含客户端已归档的长度重试；不包含失败轮次及被替代配置的请求，不是账单总额）。
最终成员权重和是否降级以逐题 runs/*.json 的 final 字段为准。

## 验证边界

- 本地回溯验证，不是比赛提交或独立测试集成绩；不排除模型训练数据包含历史答案。
- 原始快照目前从官方仓库下载，历史可用时间依赖官方归档声明。
- 46 道已结算题因冻结历史不满足本批条件而排除，原因见 manifest.json。
- 本批没有排名题，也没有实时检索；结果不能代表完整线上研究流水线。
- 官方 Persistence 的均值与输入最后观测全部一致；仓库 baseline 的标准差算法不同。

## 复现

```bash
.venv/bin/python scripts/validate_arena.py predict --data-dir data/arena-validation-v2-retry
# 可选的 EWMA、局部趋势及滚动残差统计候选：
.venv/bin/python scripts/validate_arena.py predict --baseline-suite expanded --data-dir data/arena-validation-v2-retry
# 将 SSA_LLM_API_KEY 设置在环境中后：
.venv/bin/python scripts/validate_arena.py predict --config <matching-experiment-config.json> --data-dir data/arena-validation-v2-retry
.venv/bin/python scripts/validate_arena.py score --data-dir data/arena-validation-v2-retry
```
