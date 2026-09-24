# 内部 M v3：先归档上轮，再启动新实验

2026-09-24。用户已确认“保留 RoMa，内部学会使用 M”，并授权先推送上轮，再开始下一轮。

## 上轮 GitHub 归档已完成

仓库：https://github.com/yinaiden6-dev/new_HYP

提交：[`d465b9f41d332cd72f146edea5e652918c5dc75c`](https://github.com/yinaiden6-dev/new_HYP/commit/d465b9f41d332cd72f146edea5e652918c5dc75c)。`git ls-remote`已确认远端main与本地提交完全一致，不是仅有本地commit。

3832个变更文件；清单覆盖约409MB文本源码、协议、报告、标量头参数、最终逐候选分数及复核证据。包括内部M v2与补充CPU对照、TRAIN16终点复核、简化头外部迁移、H593 Qwen质量联合校准及三项收口报告。不含原图、二进制模型／适配器权重、token缓存和err/out日志。本次实验代码是在这次上轮归档之后准备并提交，未冒充已完成结果纳入该提交。

自动审批最初两次拒绝推送，用户再次明确确认该仓库、提交及文件范围后，推送获批并完成。归档清单在导出仓库`backup/internal_m_and_three_claims_20260924/files.json`。

## 新实验已提交

| Job | 阶段 | 实际资源／依赖 | 启动核查 |
|---|---|---|---|
|5161636|CPU共同初始化与外部对照|dev_cpuonly，8CPU、32GB、10分钟，无GPU|COMPLETED 0:0，用时55秒|
|5161637|真实LLM首步工程验证|accelerated，1GPU、8CPU、64GB、10分钟；afterok:5161636|COMPLETED 0:0，用时1分40秒；实际首步PASS|
|5161638|PRE_CONSTANT训练|accelerated，1GPU、8CPU、64GB、10分钟；afterok:5161637|已RUNNING，hkn0412|
|5161639|PRE_REAL训练|accelerated，1GPU、8CPU、64GB、10分钟；afterok:5161637|已RUNNING，hkn0435；从pilot第1步继续|
|5161640|CPU最终汇总|cpuonly，无GPU、16GB、10分钟；afterok:5161638:5161639|PENDING / Dependency|

首步已通过实际梯度、原缓存回放、参数／优化器持久化等检查，随后自动提交PRE_CONSTANT与PRE_REAL两条训练及依赖两者的CPU汇总。真实模型首步完整C128、适配器和头均更新、末端不读M、参数与优化器恢复核对均通过。整个训练尚未完成，不宣称内部M学习已带来收益。

此次内部三参数头仅仅读取RAW差、sym(Lθ)、bias；真实M只通过LLM前适配器进入，适配器与读出共同训练。主干、原检索LoRA及reference tokens冻结，保持完整自然C128，检索身份标签为新增训练监督。外部加性／乘积读出使用相同TRAIN16；共同初始化由该TRAIN16拟合，没有使用旧完整TRAIN头来冒充16图训练。

两内部臂固定128步／8遍，16与128步全TRAIN评估，64步仅留checkpoint；128步追加内部常量／错绑M诊断。CPU外部头使用2000次完整TRAIN16均值更新，内部为128次逐query更新，标签预算相同但优化预算与计算量不同。每个GPU任务10分钟，可按同Job断点续跑，最多48次，属于硬上限而非预计耗时。

本阶段是TRAIN16拟合能力实验；不自动评估已打开probe，不作跨组或外部确认，不改写原COST1成绩。M改变条件化相似度也不自动证明局部注意力或空间ownership。

## 已完成的提交前检查

- 16项独立测试通过：完整C128、COST1／HOLD、无末端M泄漏、共同起点、联合VJP、共享前向、参数与优化器精确恢复、首步回执中断恢复，以及真实shell启动脚本的正常／超时／失败分支。
- 生产预检48项（3种头×16 TRAIN行）通过：直接autograd与分段VJP一致，内部头不直接读取M。
- 实际Slurm脚本5161637与冻结launcher逐字节一致。
- CPU三种读出另行从既有独立FRESH_ZERO内容缓存重算6096个logits与48个动作，最大误差2.78e-15。仅在TRAIN16上，无M共同头8/16、外部ADDITIVE4和PRODUCT5均12/16；这是拟合对照，不是留出成绩。
- authority：`7c3e6066c760d3cdb3a838a07b138fadef190a601169970c9ea22f8a267b6fb1`。

[执行协议](../plan/RC_COLNOMIC_INTERNAL_M_LEARNED_USE_V3_EXECUTION_20260924.md) · [authority](../registry/rc_internal_m_learned_use_v3_authority_20260924.json) · [提交记录](../results/rc_internal_m_learned_use_v3/submission.json) · [独立测试](../results/rc_internal_m_learned_use_v3/prefreeze_tests.json) · [生产预检](../results/rc_internal_m_learned_use_v3/preflight.json) · [实际脚本核对](../results/rc_internal_m_learned_use_v3/submitted_spool_validation.json)

[真实首步验证](../results/rc_internal_m_learned_use_v3/pilot_validation.json) · [自动续提记录](../results/rc_internal_m_learned_use_v3/dispatch/pilot_PRE_REAL.json) · [CPU独立核算](../results/rc_internal_m_learned_use_v3/cpu_independent_recount.json)
