# 验证看板提交记录

入口：http://36.140.148.11:18765/

按 ssa-validation SKILL.md 提交既有历史实验，两个 POST 均成功。服务端响应原样保存在 *-response.json，包括 source_commit、分板覆盖数/mean_skill/mean_loss 和逐题 scores。没有自行提交分数。

覆盖均为 14/60：数值 12/51、分群 2/6、排名 0/3。标为历史实验导入；不是新的 API-only 重跑，也不是完整集或赛前盲测。API 上下文与原实验的差异记录在 input-audit.json 并写入公开 notes。

| 方案 | 服务端 ID | 数值 mean_skill | 分群 mean_skill |
|---|---|---:|---:|
| deepseek-flash-v2-archive-14-20260925 | da82f58358e7 | -0.190432 | 0.193934 |
| local-trend-rolling-residual-archive-14-20260925 | 294a94961228 | 0.183274 | 0.005591 |

mean_skill 直接抄录服务端响应，0 为官方 Persistence。网页和对话中可用 ×100 的显示方式；原始响应保持不变。
