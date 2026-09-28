# 预测修复与可验证优化

本轮修复模型输入与评分口径，增加可选统计预测方法和任务政策。14 道旧题已经用于误差分析，属于开发集；同题重跑用于诊断与工程验收，不是独立测试集，也不据此自动选择上线权重。

## 评分目标

`score_records()` 新增 `mean_skill`：逐题 `1 - loss / official_persistence_loss` 的算术平均。官方 null 损失为零时 skill 为零，未提供 null 的记录不参加 skill 均值。

原有 `aggregate_skill` 是总损失之比，保留用于兼容，并标记 `aggregate_skill_kind: loss_weighted_legacy`。两者可能一正一负，不应混用。

`by_group` 按来源、单位、人口、目标指标、cells 及评分指标分组；跨目标的 `by_metric.mean_loss` 仅是兼容输出，`raw_loss_comparable: false` 明确提示不可作同口径平均损失。与其他参赛者比较时须使用共同题集。

## 模型能看到的信息

- Prompt 分别写明信息截止、提交截止、计划发布时间、已知测量窗口。发布时间不等于观测日期；未知窗口保持未知，不从 round ID 或截止日期猜测。
- Context 增加来源、目标指标、距最新观测天数、典型观测间隔、距计划发布天数。后者不是统计模型的预测步数。
- 原始历史展示长度独立于统计摘要：默认 `history_points: 24`、`profile_history_points: 12`。摘要仍由 `recent_window`、`long_window` 控制，并显式记录摘要使用点数。
- 分群历史用公共日期轴和每群体的值数组表示，缺失值为 null，不填充。模型看不到每个值的完整可用时间，但所有值都已经过 cutoff 过滤，原始时间和来源保留在归档中。
- 排名历史保留已有 `views/index`，避免只传名次而丢失流量强度。
- 必需历史超过预算时仍明确失败，不静默截断。长历史未必改善预测，需时间留出实验验证。

## 可选统计基线

默认成员与数值行为继续兼容原来的 `persistence/recent_mean`。可在 `ensemble.baseline` 开启更多候选：

```json
{
  "numeric_members": ["persistence", "recent_mean", "ewma", "local_trend"],
  "recent_window": 3,
  "ewma_alpha": 0.35,
  "trend_window": 6,
  "trend_damping": 0.8,
  "sd_method": "rolling_residual",
  "residual_window": 20,
  "min_residuals": 3,
  "min_sd": 0.25,
  "forecast_steps": 1
}
```

这是参数说明示例，未经比赛数据校准，不是推荐最优值。

- EWMA 递归平滑历史；local_trend 根据近期斜率做带阻尼外推。
- `forecast_steps` 是明确指定的观测步数，不自动从发布日、截止日推断。不同数据频率和跨多个发布波次的题目需要正确配置。
- `rolling_residual` 对每个策略和每个群体，使用历史前缀预测后续 h 步，以误差 RMSE 估计标准差，保留系统偏差。
- 残差不足时回退到历史差分估计并记录原因。已知上下界同时用于历史残差计算和最终均值裁剪。
- 这是当前已知历史版本上的前缀误差估计，不重建每个历史起点当时的数据修订版本。最终效果仍需真实时间留出验证。
- 排名成员保持原来的两个基线。新增数值成员必须在 `ensemble.weights` 中获得正权重才会影响集成，遗漏类型的权重为零。

## 按任务配置权重和标准差校准

`ensemble.task_policies` 支持精确匹配 `target_type/source_name/metric/population/unit`：

```json
{
  "weights": {"llm": 0.6, "persistence": 0.25, "recent_mean": 0.15},
  "task_policies": [
    {
      "match": {"target_type": "continuous_normal", "source_name": "YOUR_SOURCE", "metric": "YOUR_METRIC"},
      "weights": {"persistence": 1.0},
      "sd_scale": 1.0
    }
  ]
}
```

这是展示“允许 LLM 权重为零”的政策语法，不意味着所有数值题应使用此配置。

匹配字段最多的政策优先；同具体度多个匹配直接报错。政策的 weights 完整覆盖默认 weights，未列成员类型为零。`FinalPrediction.policy` 记录所选规则、最终归一化权重和缩放；有效标准差缩放为 `policy.sd_scale × pipeline.sd_scale`，仅应用一次。

Python API 中显式 `ForecastMember.weight` 仍优先于配置，保持兼容；Pipeline 创建的成员不设置显式 weight。实际选中政策的 LLM 权重为零时，Pipeline 跳过模型调用、记录 `model_skip`，不会因模型不可用而阻断明确选择的统计路线。完全不配置模型时，构造器仍要求显式 `allow_baseline_fallback: true`。

所有新配置纳入 pipeline identity。更改模型输入、策略或权重必须使用新的 entrant/归档；不要复用旧轮次主键覆盖实验。

## 输出截断与审计

Chat Completions 的模型配置可设置 `truncation_retry_tokens: 32768`。默认关闭；只有服务端明确返回 `finish_reason: length`、该上限高于原 `max_tokens` 且剩余时间足够时，才使用相同模型、相同消息提高额度重试一次。HTTP 请求总数最多为 `max_attempts + 1`，仍共用原总时间预算；拒绝、网络故障和格式错误不触发提高额度。

`transport_attempts` 保存每次返回的完成原因、模型、用量、文本、响应 ID 和请求 token 上限。成功的 `usage` 累加截断尝试的 token；失败轨迹保存于归档的 `model_failed` 事件。这里记录的是 API 返回用量，不能代替账单。

## 校准与留出验证

`calibrate --objective mean_skill` 使用官方逐题平均 skill 选择权重与标准差倍数。默认 `--objective mean_loss` 保持原行为。

每个 case 应明确提供官方 null：

```json
{
  "request": {"...": "完整请求"},
  "truth": 42,
  "truth_available_at": "2026-09-24T10:00:00Z",
  "persistence": {"mean": 41, "sd": 1.5},
  "members": [{"member_id": "llm-a", "member_type": "llm", "prediction": {"mean": 42, "sd": 2}}]
}
```

mean_skill 模式要求 validation/test 都有显式 `persistence`，缺失会拒绝；不以自家 baseline 冒充官方 null。校准还检查目标 metric，避免同来源、同单位但不同目标混在一起；测试期开始前必须已知所有校准真值，测试标签不参与选优。

## 本轮可复现产物

- [修复与实验对比报告](../data/arena-validation-v2/COMPARISON.md)：233 项测试、同题前后对比、统计候选成绩和独立补跑说明。
- 原实验：`data/arena-validation/`、`configs/arena-validation.deepseek.json`。
- 输入修复实验：`data/arena-validation-v2/`、`configs/arena-validation.deepseek.v2.json`。模型、输出预算、60/25/15 权重与原有效实验一致；没有启用新统计方法作为模型集成成员，便于区分输入变化。
- 失败题补跑：`data/arena-validation-v2-retry/`、`configs/arena-validation.deepseek.v2-retry.json`。单独配置一次 32k 长度重试，不混入原实验配置。执行结果见比较报告。
- `scripts/validate_arena.py --data-dir ...` 可将不同实验写入不同目录；已存在且内容不同的预测文件拒绝覆盖。评分核对输入与配置哈希，拒绝混合配置和同名策略覆盖。
- `predict --baseline-suite expanded` 单独列出新统计候选；不会根据结果自动选模型。
- 旧导出若缺身份字段，先运行 `verify-legacy-exports --data-dir ...`；模型结果必须与归档一致，统计预测必须可精确重算，验证后才补充哈希，预测值保持原样。

```bash
.venv/bin/python scripts/validate_arena.py prepare \
  --source-dir data/arena-validation --data-dir data/arena-validation-v2
.venv/bin/python scripts/validate_arena.py predict --data-dir data/arena-validation-v2
.venv/bin/python scripts/validate_arena.py predict \
  --baseline-suite expanded --data-dir data/arena-validation-v2
# SSA_LLM_API_KEY 只放在环境变量中
.venv/bin/python scripts/validate_arena.py predict \
  --config configs/arena-validation.deepseek.v2.json --data-dir data/arena-validation-v2
.venv/bin/python scripts/validate_arena.py score --data-dir data/arena-validation-v2
```

本轮没有启用实时搜索，也没有新增未经快照验证的历史新闻。下一轮效果验证应使用冻结后的新波次，分别比较统计预测、修复输入的 LLM、证据检索和任务政策的增量。

以上原实验配置用于解释保存结果。代码升级也会改变 pipeline identity；再次调用模型应复制配置并使用新的 entrant、run_db 和数据目录。评分现有归档不需要重新调用模型。
