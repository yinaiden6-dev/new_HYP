# H593：质量×内容的作用位置隔离

2026-09-22用户授权继续剖析、后续实验，并要求中间数据留存。复用封存ColNomic tokens与原RoMa overlap，无新编码器/RoMa forward，无新增人工标注；CPU FP64，原H593五折、自然RAW C128、593分母含23缺失。原COST1为主，CE辅助。此分支不改变并行运行的坐标精度实验。

三个因子：M为整体sqrt(mean u * mean v)质量缩放；Q为query加权汇聚；R为reference权重参与MaxSim。固定这些变量原值，枚举2^3组合；去除Q/R使用全1局部权重，但M单独保留原始总质量（或设1）。原生111按字面FP64原算术及roll控制重放；控制中原质量对被roll侧的mean重新计算以保持原生逐位一致，关闭一侧局部权重后该侧控制严格等于real。

第九臂FIXED_FREE_ARGMAX：保留M、Q和reference权重数值，但固定每个query token读取自由ColNomic最大相似位置，不允许reference权重重选位置。reference总作用可拆成“在自由命中处改变证据数值”和“另选位置带来的增量”；两项逐token记录并核算恒等式。重选增量对每个候选非负，并不意味着它提高身份区分，必须看target-vs-wrong差值及最终救回/损失。

这改变具体评分算子，不是此前S/M/L/Q/R输入列置零。固定头对照说明当前模型依赖性，不能排除分布移动；同协议2000步AdamW、原折/原训练顺序重训检验调整后的可达效果。全部九臂完整报告，不能凭heldout挑选最佳机制作为新未触碰确认。

所有候选label-blind计算，每query保存128轴、九臂四统计、全部127×6特征，另存u/v、自由和加权命中位置、局部相似度、逐token贡献及reference作用的两项分解；原图及token通过原来源SHA绑定复用。五折预测及独立复核完成前不打开heldout身份标签。旧纯ColNomic统计小头仍保留作为对照，其未修改token表示。

先shard00（按源缓存路径确定，不看标签；最多8图）资格试运行；通过后其余74片可CPU并行。当前dev QoS最多1项运行、4项提交，dev_cpuonly与dev_accelerated共享；CPU主批用cpuonly以免阻塞GPU的开发队列。每项15分钟并可恢复。

后续纯ColNomic token级可靠性/表示适配器的结构，应与本次定位出的主要作用路径对应；本实验不预先承诺该新模型会超过原头，也不以解释成立代替性能提升。
