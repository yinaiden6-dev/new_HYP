# 内部模型外部验证：已提交

用户指定的最后一组补充：将**已经训练完成的H593 fold0 POST_REAL**及配套头固定，用于原GroZi480自然C128。H593五折478/593结果保留，本轮没有重做H593、没有重新训练fold0，也没有使用GroZi挑选源模型。

H593五折包含五套参数，478/593是各自留出组的合并成绩。本轮预先固定编号0的一套参数，是固定单模型的跨数据集验证，不能将其源成绩写成该五折汇总，也不构成在外部数据上的第六折训练。

| 任务 | 内容 | 资源与依赖 |
|---|---|---|
| 5163009 | 首片8张采集和冻结回放 | dev_accelerated，1 GPU、8 CPU、32 GB、10分钟；已COMPLETED，3分17秒，8张六臂预测及缓存等价验收通过 |
| 5163010_[0-11%12] | 其余472张，12路承接59个旧缓存片 | accelerated，同资源；首片依赖已满足，当前Priority排队 |
| 5163011 | 全480封存、独立NumPy核算、分组统计与报告 | cpuonly/dev_cpuonly，8 CPU、16 GB、10分钟，无GPU；afterok前两组 |

模型参数、源TRAIN的M标准化与0切换阈值全部冻结。复用原reference tokens及全部RoMa M，仅新增外部query的3584D隐藏表示；每张图一次冻结ColNomic前向，无reference或RoMa重算。真实M、同模型恒定M、同模型错绑M，以及不加适配器、外部加性/乘积头均保存完整候选分数。

结果产生前不宣称外部内部模型验证成功；完成后按实际数据收口，不自动展开新训练。

[固定方案](../plan/RC_POSTLLM_GROZI_EXTERNAL_V1_20260925.md) · [冻结协议](../registry/rc_postllm_grozi_external_authority_v1_20260925.json) · [任务记录](../results/rc_postllm_grozi_external_v1/jobs.json) · [提交后资源核对](../results/rc_postllm_grozi_external_v1/scheduler_validation.json)

## 2026-09-25 双分区调度调整

用户要求dev_accelerated与accelerated共同排队。整组12项加入dev被QOS拒绝；现场核对dev限制MaxSubmitJobsPU=4、MaxJobsPU=1，汇总任务已占一个dev提交名额。已将5163010_0、_1、_2改为dev_accelerated,accelerated；_3–11继续accelerated。各子任务仍为1GPU、8CPU、32GB、10分钟，原Job ID、依赖和数组并发上限12保留。此调整只改变调度资格，不改变冻结实验协议。

## 2026-09-25 汇总校验修复

GPU首片与12个数组任务均COMPLETED，480张缓存与六臂预测齐全。原汇总5163011在`NATURAL_C128`校验失败：5533物理图片仅有5532个身份（Biogen_21在714、715重复），原RAW排名按身份保留最高分代表；新汇总器误将其当成5533物理全排列。独立复核全部480份候选轴与原自然C128一致。

原v1与冻结协议保留，v2增加身份完整覆盖及从完整5533分数稳定排序的独立重建；5项回归检查通过。仅重新提交CPU汇总5163071，dev_cpuonly/cpuonly共同排队，8CPU、16GB、10分钟，无GPU。候选、模型、预测、阈值及统计规则未改。[修复记录](../registry/rc_postllm_grozi_join_repair_v2_20260925.json)。

## 最终完成与独立核算

修复汇总5163071在dev_cpuonly以1分42秒完成，COMPLETED、0:0。480张、60个分片、6臂预测全部通过原输入SHA、自然候选轴、原M、token等价与独立NumPy动作核对，最大logit误差2.132e-14。另一次从封存预测与原标签独立重算的480逐图/2880动作与11项分组统计一致。

冻结POST_REAL：RAW321→339/480，18救/0损，等视频组增益95%区间[0.950,3.788]个百分点；恒定314、错绑308，源fold0配套外部头342。全部结果已进入统一收口与论文草稿；无外部训练，不新增后续实验。[正式结果](REPORT_POSTLLM_GROZI_EXTERNAL_V1_20260925.md) · [第二次独立核算](../results/rc_postllm_grozi_external_v1/operations/independent_final_recount_20260925.json)。
