> 2026-09-24 实施更新：当前统一工程入口见 [README.md](README.md)，四个核心模块位于 `src/ssa_harness/{retrieval,context,prompt,ensemble}`。旧项目保留在 `archive/legacy/`；接口约定见 [docs/INTEGRATION.md](docs/INTEGRATION.md)。Ranking 已区分固定候选与开放 Top N；计分采用 CRPS / Energy / RBO / Kendall，不能把全部任务视为同一普通回归损失。

# Social Simulation Arena

三个任务感觉本质上还是预测的任务，和以往不同的是: 

1\.训练集并非固定好的，需要我们自己去造训练集，但是是有真值的

2\.测试集现在是不可见的，其实还是和我们以前训练的模型一样的，因为严格意义上 我们的测试集也是不可见的。

3\.所以感觉并不是一个社会模拟问题，还是结果导向的 没有中间过程的预测问题



**Number / Topline \(回归任务\)**

- 预测一个数值 

- 例如：下周某项支持率是多少 

- 返回：`mean + sd`

- 例：`{"mean": 41.2, "sd": 1.4}`

**Population Profile \(回归任务\)**

- 一次预测多个群体/单元格 

- 例如：Democrat、Independent、Republican，或者不同年龄、性别群体各是多少 

- 每个 cell 都要返回 `mean + sd`

- 重点是：**必须把题目要求的所有群体都预测完整**

**Ranking \(learn to rank 任务\) **

- 预测一组项目的顺序 

- 例如：下周 Wikipedia Top 10、热门品牌/搜索对象排名 

- 返回完整排序 

- 例：`["A", "B", "C"]`



回归任务和rank任务是同一类的问题 rank问题其实就是对ABC的分数进行回归

所以可以采用同样一套的解决思路



# Strong LLM \+ harness的方法来做

## Snapshot Builder：时间快照构造器 @fangyuan

这是整个 Harness 最重要的组件之一。

对于时间点 t：

**Snapshotₜ = 截至 t 真正可获得的信息**

包括：

```Plain Text
Long-term information
+
Short-term information
+
Historical ground truth
+
Task metadata
```

而且必须严格保证：

**information\.timestamp ≤ t**

这样历史回测不会有未来信息泄漏。

## Research / Retrieval Engine：信息搜索器 @songbo \& boyi

负责帮 LLM 找资料。

例如：

```Plain Text
Search Web
Search official source
Fetch historical series
Fetch related polls
Fetch news
Fetch Google Trends
Fetch Wikipedia / attention data
```

但不是“搜得越多越好”。

需要：

```Plain Text
query generation
↓
search
↓
deduplicate
↓
filter by timestamp
↓
rank by relevance
↓
compress
```

最终给 LLM 的应该是一个干净的信息包。

## Context Builder：上下文整理器 @songbo \& boyi

搜索出来的东西不能直接全塞进去。

建议分成：

```Plain Text
A. Long-term context
B. Recent developments
C. Numerical history
D. Related indicators
E. Conflicting evidence
```

例如：

```Plain Text
LONG-TERM
过去一年支持率均值、波动、长期趋势

RECENT
最近 7 天发生的重要事件

NUMERICAL
最近 10 期真实值

EXTERNAL
其他相关民调 / 经济指标

CONFLICTS
支持上升的证据
支持下降的证据
```

这个组件本质上是在帮助 LLM **减少 context noise**。



## Forecast Prompt Compiler：预测 Prompt 构造器 @songbo \& boyi 

不要每次随便问一句：

> “你觉得下周多少？”
> 
> 

而要固定一个 forecasting protocol。 \(可以根据不同任务不固定\)

例如要求模型依次做：

```Plain Text
1. Establish base rate
2. Analyze historical trend
3. Identify recent changes
4. Compare supporting/opposing evidence
5. Estimate magnitude
6. Produce forecast
7. State uncertainty
```

## Ensemble Engine：集成器 @songbo \& boyi 

如果跑了多个预测：

```Plain Text
LLM A = 41.0
LLM B = 42.1
LLM C = 41.7
Time-series baseline = 41.3
Persistence = 41.1
```

Harness 负责组合。



## Backtester：历史回测器 调整prompt 以及不同部分的策略 @ fangyuan

这个也是必需品。

它负责在过去制造很多：

```Plain Text
timestamp t
↓
冻结信息
↓
运行完整 Harness
↓
得到预测
↓
和真实 yₜ₊₁ 比较
```

也就是：

**Harness\(t\) → ŷₜ₊₁**

然后记录：

```Plain Text
prediction
truth
error
CRPS
model disagreement
research sources
```

没有这一层，你根本不知道 Prompt 改了以后是变强还是变弱。







# 

# 

