# H593 后 LLM 归因回放：已启动

2026-09-25收口更新：全部593张、50片均完成。剩余独立核算由5162949并行处理，再恢复5162894串行汇总，两者均COMPLETED/0。共同响应477、patch剩余430、固定内容命中469，原生478；详见[全量归因解释](REPORT_POSTLLM_H593_DECOMPOSITION_INTERPRETATION_20260925.md)。以下保留启动时记录。

解释目标：原POST_REAL的54次纠错是否主要由共同响应保留，是否需要patch特异响应或重新选择reference token。全593张原五折、原C128、原适配器与INTERNAL3头固定；不训练、不重跑编码器/RoMa、不申请GPU。

七条回放为CONSTANT、NATIVE、COMMON_ONLY、SPATIAL_ONLY、RECOMPOSED、NATIVE_FIXED_ARGMAX、COMMON_FIXED_ARGMAX。逐patch分数、命中索引、共同向量与剩余范数继续保存。SPATIAL_ONLY沿用旧程序名，含义仅为patch去均值剩余响应。

## 已完成

- `5162891`：首图、完整128候选、七条回放及独立核算，dev_cpuonly，COMPLETED/0，Slurm用时53秒。实际回放阶段约16秒。
- 独立patch分数误差最大5.55e-17，头logit误差最大1.07e-14。
- 原生及恒定条件的CPU/GPU动作一致；最大L误差约2.43e-5，最大logit误差约0.00224，不宣称位级相同。
- 重组回放通过。此次首图按无标签manifest第一项选取，不依成功/失败选择。
- 续跑脚本的成功、预算退出、超时、真实错误与续跑上限五种分支已实际执行mock验证；提交脚本与Slurm保存脚本字节一致。

## 自动后续

`5162892` 已在dev_cpuonly完成自动续提，COMPLETED/0，用时33秒。后续已实际提交并核对：

- `5162893_[0-49%50]`：50片、最多并行50、cpuonly，每项4CPU/16GB/10分钟；每片11或12张，候选级断点续跑。最新检查PENDING，原因Priority。
- `5162894`：独立CPU汇总，依赖`afterok:5162893_*`；PENDING，原因Dependency。

各worker不并发写总汇、不读取held标签。两份提交脚本均与Slurm保存版本逐字节一致；ReqTRES均无GPU。

实际后续Job ID及提交资源以[调度回执](../results/rc_postllm_h593_decomposition_v1/dispatch/chain.json)为准。程序仅在首图完整独立验收后释放全量；任务完成和科学结论分开判断。

全量报告需同时列总正确、误伤、逐折、原54救回保留及逐图决策变化。共同方向能量大不等于它解释全部纠错；出现CPU/GPU动作差异时必须单列，不能按CPU结果声称完整复现GPU478。

[固定计划](../plan/RC_POSTLLM_H593_DECOMPOSITION_V1_20260925.md) · [首图独立核算](../results/rc_postllm_h593_decomposition_v1/audit_rows/0000.json) · [冻结协议](../results/rc_postllm_h593_decomposition_v1/protocol.json)
