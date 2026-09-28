# Arena 真实题目验证

已结算题型与数量：{'continuous_normal': 12, 'profile_energy': 2}。
只使用官方 answer-window 历史，准备的输入最多数值 24 点、每群体 12 点；模型实际可见窗口由运行归档的 context.config 决定。预测阶段不读取答案和其他参赛者预测。
本批为已分析过的开发集，不能作为独立测试集。统计候选使用预先配置的参数，并不依据本批结果自动选优。
下表 mean skill 按官方逐题平均口径计算，0 为 Persistence；aggregate_skill 仅保留为历史兼容字段，不能当官方均分。

| 策略 | 题型 | 题数 | mean skill ×100 | 胜 / 平 / 负（对官方 Persistence） |
|---|---|---:|---:|---|
| official_persistence | crps | 12 | 0.00 | 0 / 12 / 0 |
| official_persistence | energy | 2 | 0.00 | 0 / 2 / 0 |
| persistence | crps | 12 | 6.82 | 8 / 0 / 4 |
| persistence | energy | 2 | -0.44 | 1 / 0 / 1 |
| recent_mean | crps | 12 | -36.08 | 8 / 0 / 4 |
| recent_mean | energy | 2 | -0.11 | 1 / 0 / 1 |
| baseline_ensemble | crps | 12 | -0.45 | 8 / 0 / 4 |
| baseline_ensemble | energy | 2 | 3.93 | 1 / 0 / 1 |
| ensemble | crps | 12 | -15.66 | 8 / 0 / 4 |
| ensemble | energy | 2 | 21.52 | 2 / 0 / 0 |
| llm | crps | 12 | -29.96 | 5 / 1 / 6 |
| llm | energy | 2 | 29.50 | 2 / 0 / 0 |

## 数值题明细

mean ± sd 不是置信区间；CRPS 越低越好。

| 题目 | 真值 | 基线集成 mean ± sd | 官方 Persistence CRPS | 基线集成 CRPS | LLM mean ± sd | LLM CRPS | 最终集成 CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| yougov-2026-w38-generic | 12.0 | 9.750 ± 2.465 | 0.6071 | 1.3429 | 7.700 ± 2.500 | 2.9766 | 2.2125 |
| mc-2026-w38-approval | 41.94 | 41.817 ± 1.250 | 0.3623 | 0.2969 | 42.000 ± 1.600 | 0.3748 | 0.3442 |
| aaii-2026-09-17 | -24.5 | -2.150 ± 12.504 | 22.3537 | 15.6639 | -3.000 ± 7.500 | 17.2777 | 16.3885 |
| civiqs-2026-w38-angry-share | 30.2 | 29.688 ± 0.251 | 0.4164 | 0.3749 | 29.700 ± 0.500 | 0.3012 | 0.3152 |
| civiqs-2026-w38-approval | -27.7 | -24.337 ± 1.032 | 2.0841 | 2.7806 | -24.200 ± 1.500 | 2.6637 | 2.6963 |
| civiqs-2026-w38-econ-direction | -38.5 | -38.037 ± 1.069 | 0.4164 | 0.3284 | -38.000 ± 1.500 | 0.4164 | 0.3832 |
| civiqs-2026-w38-econ-now | -34.8 | -31.675 ± 0.894 | 2.3714 | 2.6207 | -31.700 ± 1.000 | 2.5363 | 2.5692 |
| civiqs-2026-w38-family-finances | -30.9 | -30.137 ± 0.722 | 0.6071 | 0.4631 | -29.900 ± 1.200 | 0.5949 | 0.5388 |
| civiqs-2026-w38-inflation-concern | 59.0 | 59.262 ± 1.214 | 0.4450 | 0.3063 | 58.900 ± 1.700 | 0.3996 | 0.3592 |
| yougov-2026-w39-generic | 14.0 | 11.125 ± 1.989 | 1.2809 | 1.8847 | 11.000 ± 2.700 | 1.8392 | 1.8419 |
| mc-2026-w39-approval | 41.19 | 41.801 ± 1.259 | 0.4971 | 0.4104 | 42.000 ± 1.300 | 0.4989 | 0.4619 |
| aaii-2026-09-24 | -15.4 | -18.275 ± 14.768 | 8.2537 | 3.6737 | -10.000 ± 14.000 | 4.0925 | 3.5932 |

## 分群题明细

Energy score 越低越好，每题有 16 个 cell。

| 题目 | 官方 Persistence | 基线集成 | 纯 LLM | 最终集成 |
|---|---:|---:|---:|---:|
| yougov-xtab-2026-w38 | 7.9932 | 6.0677 | 5.1466 | 5.5164 |
| civiqs-profile-2026-w38 | 9.9516 | 11.5662 | 7.6250 | 8.7530 |

## 模型运行

返回模型：deepseek-flash；成功题数：14。
成功运行记录中的 token：90512（包含客户端已归档的长度重试；不包含失败轮次及被替代配置的请求，不是账单总额）。
最终成员权重和是否降级以逐题 runs/*.json 的 final 字段为准。

## 验证边界

- 本地回溯验证，不是比赛提交或独立测试集成绩；不排除模型训练数据包含历史答案。
- 原始快照目前从官方仓库下载，历史可用时间依赖官方归档声明。
- 46 道已结算题因冻结历史不满足本批条件而排除，原因见 manifest.json。
- 本批没有排名题，也没有实时检索；结果不能代表完整线上研究流水线。
- 官方 Persistence 的均值与输入最后观测全部一致；仓库 baseline 的标准差算法不同。

## 复现

```bash
.venv/bin/python scripts/validate_arena.py predict --data-dir data/arena-validation
# 可选的 EWMA、局部趋势及滚动残差统计候选：
.venv/bin/python scripts/validate_arena.py predict --baseline-suite expanded --data-dir data/arena-validation
# 将 SSA_LLM_API_KEY 设置在环境中后：
.venv/bin/python scripts/validate_arena.py predict --config <matching-experiment-config.json> --data-dir data/arena-validation
.venv/bin/python scripts/validate_arena.py score --data-dir data/arena-validation
```
