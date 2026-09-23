# ColPali自己的自然C128：候选协议纠正

用户明确要求ColPali自己的128候选。此前293→388/593固定了ColNomic C128，只是辅助机制对照，不能作为ColPali检索管线的迁移结果。

1. 复用验收完成的593张ColPali query tokens及5413张reference tokens，不重新编码。
2. 对每张query遍历全部5413张reference，以ColPali MaxSim选自己的自然Top128，绝不插入target。FP64/all1030 tokens/no renormalization与前次诊断保持相同；精确同分按物理行升序。此为FP64 MaxSim协议，不能冒充历史FP16实现逐bit复现。
3. 50片、最大并行50、accelerated、15分钟短片，可从每256个reference和每张query断点续跑。首片通过后其余49片自动开始。保存全部5413分数及完整排序，边界128/129和winner独立NumPy核对。
4. 所有候选先封存再读标签，汇总ColPali自己的RAW、全图库MRR、C128召回和缺失数。新候选有效TRAIN由自己的召回重算，沿用原五折身份/组件隔离；不能照搬ColNomic召回过滤后的TRAIN列表。
5. 与旧RoMa候选逐图逐reference对齐，只对同一原图对复用原M，列出缺失配对。禁止按候选位置复用、禁止补target、禁止用ColNomic排名选择新候选。
6. 新配对RoMa尚未齐全之前不训练或发布新的联合质量头结果。完整局部七参数映射与整体M简化头是不同实验，后者不能冒称前者；局部网格改变时不能混用不同池化定义的M。
7. 当前第一段自动链：原生全图库检索→候选封存→缓存交集和缺失任务清单→新RAW与新召回统计。补齐缺失M并重训为后续阶段，未完成时明确保留缺口。

该面板为已打开H593开发集，不是新的外部确认。旧实验及缓存全部保留。与rerank共享优先级；其他采集/融合继续暂挂。
