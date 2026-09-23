# ColPali整体质量迁移：固定ColNomic C128

用户优先推进ColPali。593张编码已由历史模型环境验证，与5413行旧reference缓存的三个固定锚点通过兼容检查。图库旧缓存不含全部原图SHA，保留此历史来源限制。

复用相同593 query、原ColNomic自然C128与五折分组；不计算ColPali自己的C128，不声称完整ColPali检索管线泛化。23张目标不在C128的样本仍计入。

ColPali基线是1030个缓存token的FP64 MaxSim求和，不重新归一化。自由内容统计只读前1024个已确认的图像token，FP64单位归一化：前向MaxSim均值、反向均值、两侧top1-top2间隔、全配对相似度均值。其余6个提示token不赋予空间权重。

原RoMa每候选全局M不变。简化联合证据为S=M×L，L为图像token自由MaxSim；四个挑战者差分量RAW/S/M/L加偏置，共五个有效参数。同监督内容头为RAW加上述五个内容统计和偏置，共七参数。各自COST1/CE训练；原TRAIN顺序、2000步、FP64 AdamW lr.03 wd.001；重算ColPali自己的winner和HOLD标签。主对照MASS5_COST1 vs COLPALI_RAW_C128及CONTENT7_COST1。

绑定控制固定同一头，仅将每query的128个M沿候选物理轴循环64位，保留内容与RAW分数。所有held预测封存后才能读全局身份标签，报告救回/误伤、分组区间、候选内MRR和逐query决策。

这首先回答整体匹配质量能否补充另一冻结检索器。它不是完整局部u/v七参数机制迁移，也不以失败判定ColPali缺信息；后者需独立验证网格映射。没有额外编码器或RoMa前向。

评分50片GPU矩阵运算、15分钟可续跑；首片通过后放行其他49片。保留token源SHA、全部128 RAW分数、内容统计、M、前向/反向最大值与argmax，以及头参数和127挑战者logit。完整相似度矩阵由固定tokens可重建，不复制保存。每张第一候选独立NumPy复算。CPU五折训练/汇总10分钟分段。
