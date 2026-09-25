# LISTWISE_UNIT1 执行定义

用户已明确“继续 / go on”，授权执行既定下一步方案。本次仅一个新头，旧ORIGINAL7和TRAIN_UNIT_COST7及其160图预测直接复用。不是旧头重训、不是新编码器或新RoMa。

保持原PAIR64+FULL32、原顺序、six native features、七参数、FP64、seed17、AdamW .03/.001、2000步和两池均值1:1。以已做unit-cost函数为源码基准，只替换FULL分支为logsumexp([RAW0,127 challenger logits])−target logit；PAIR仍为cost1 BCE。所有预测及新参数fresh重训重放后封存，再开启原EVAL32/128身份标签。

原RAW C128、完整127 action、HOLD阈值0、physical-row同分规则不变。主要单因素对照为LISTWISE_UNIT1对TRAIN_UNIT_COST7；强性能对照是原ec7 28/32、99/128。两个面板分别报告救回/损失/净增，允许损失；回到原28/99不算超过原头。

每种REAL与CBIND都报告，但不以CBIND已有依赖重新包装新理论。历史数据已打开，属于开发复测；593没有新结果，原五折447不参与本次训练或选头。EVAL数学诊断、solver系数、D1-MI、GroZi和formal392均不得用于拟合。

只提交一个10分钟、CPU8/16G、开发加速分区优先且普通加速分区备选的任务，包含fit/fresh replay、预测封存和独立readout。分区会申请1GPU以遵守既有队列方式，实际学习与评分CPU执行。

候选目标使用与最终argmax相同的候选集合，但受限七参数、有限数据及PAIR/FULL异质分布下不保证增益。损失形式改变也改变梯度分配/尺度，不能未经对照把任何增益都归为抽象一致性理论。标准交叉熵不是原创理论。

研究依据与未做区别见 RC_FULL_CANDIDATE_IDENTITY_LOSS_NEXT_STEP_V1_20260912.md。本文件是执行的固定规格；没有温度、权重、epoch或head选择扫描，不根据EVAL另选条件。
