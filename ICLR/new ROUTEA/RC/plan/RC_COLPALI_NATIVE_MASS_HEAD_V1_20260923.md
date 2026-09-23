# 原生ColPali C128的整体质量校准

前置：rc_colpali_native_quality_v1 全593质量ready。候选来自ColPali全图库FP64 MaxSim自然Top128，按公开图库身份去重。不得使用ColNomic排名作为候选。所有原图和tokens复用已有缓存，质量仅对原生新增图像对补算。

保持原辅助对照的CONTENT7与MASS5公式、FP64训练、原5折、seed17、AdamW lr0.03 wd0.001、2000步、COST1/CE。按各TRAIN自己的自然召回重算有效训练集；每张held query包括召回失败都进入593分母。冻结预测后汇总标签。

主要比较：原生ColPali RAW、CONTENT7_COST1、MASS5_COST1，报告95%组件bootstrap、救回/损失、MRR_C128及自身召回上界；CE与固定质量头绑定打乱为补充。仍为简化全局M校准，不是完整局部七参数模型，也非外部数据。

自动依赖链：质量补齐汇总→prepare→缓存token评分首片→其余49片→5折训练→汇总。前置失败则阻止后续，不用不完整数据产出结果。
