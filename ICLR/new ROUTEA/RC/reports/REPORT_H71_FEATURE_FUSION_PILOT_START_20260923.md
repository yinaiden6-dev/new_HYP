# 71张融合收益先导已提交

2026-09-23 11:14 UTC核对。首轮数组 **5159608** 已提交，**5159609** 为 afterany 自动续跑回调。当前尚未产出收益结论。

- 面板：固定执行索引0–70，71张、37个component；原五折held数量21/15/15/5/15。
- 对照：COL_ONLY、COARSE、FINE、COARSE_FINE、NO_ADAPTER × FREE/ROMA_WEIGHTED，共10臂×5折=50配置；COST1、seed17，各240次联合更新，全自然RAW C128。
- 初始化：复用原同折、同评分方式的全量TRAIN初始小头；适配器输出零初始化。本轮是71张子集微调，不能称为从零只用71张训练。
- 主比较：融合臂相对同mode的COL_ONLY适配器、NO_ADAPTER；同source比较有无RoMa权重。附同面板RAW及原COST1、救回/损失、MRR、margin和分组方向，全部臂报告。
- 缓存：现有593特征库已验收；本轮只读取71张及其全部候选。无编码器重新前向、无GPU作业。
- 留存：参数及优化器断点、逐步loss、完整C128分数、127 logits、逐token贡献、匹配位置及融合token/gate分析副本。

## 实际执行与调度修复

原v1计划将50任务同时送cpuonly/dev_cpuonly，被dev的MaxSubmitJobsPU=4拒绝；该次未创建训练Job，也没有模型更新。另存执行v2授权，训练源码、训练授权、数据与科学配置均未修改。

实际5159608_[1-49%50]在cpuonly；5159608_0转到dev_cpuonly补位。所有fit申请8CPU、32GB、10分钟。5159609在cpuonly申请1CPU、2GB、5分钟，依赖`afterany:5159608_*`。提交脚本与Slurm spool逐字节一致，资源和依赖已实时核对。原全量200配置的50个GPU子任务继续暂挂；原30项pair-quality CPU实验不受本轮修改。

回调仅对本实验未完成配置续提，最多32轮，保留240步目标；全部50配置预测验收后自动汇总。某片缺少正常退出回执或独立核算失败时停止并记录，不把超时或缺结果当作科学负结果。

科学授权：`registry/rc_h71_feature_fusion_pilot_authority_v1_20260923.json`。

执行授权：`registry/rc_h71_feature_fusion_execution_authority_v2_20260923.json`。

状态：`results/rc_h71_feature_fusion_pilot_v1/dispatch/status.json`。

最终结果位置：`results/rc_h71_feature_fusion_pilot_v1/result.json`、`validation.json`、`summary_zh.md`（运行完成后生成；本次记录时尚不存在）。

本轮是已打开面板、单种子、240步先导。正向信号不等于独立外部确认；无收益不能证明ColNomic缺少必要信息，也不自动触发扩大样本或扫描超参数。

## 后续调度更新：accelerated分区，CPU计算

用户随后明确要求申请空载GPU转accelerated。5159608全部50项已原JobID迁移：每项1GPU、8CPU、32GB、10分钟，ArrayTaskThrottle=50。逐项核对分区及GPU字段通过。训练程序仍设置CUDA_VISIBLE_DEVICES为空，继续CPU FP64计算，不增加编码器采集，不修改240步、面板、模型或检查点。

旧CPU续跑回调5159609在未运行时取消，替换为5159623（cpuonly，afterany:5159608_*）。执行v3令后续训练片和最终worker汇总使用accelerated+1GPU；轻量续提回调仍cpuonly。该明确用户指令覆盖原执行层“不申请GPU”设置，科学授权和原始源码封存保持。

当前执行授权：`registry/rc_h71_feature_fusion_execution_authority_v3_20260923.json`；迁移前后逐项记录：`results/rc_h71_feature_fusion_pilot_v1/dispatch/migration_v3_complete.json`。本段取代上文关于当前分区与回调ID的旧快照。
