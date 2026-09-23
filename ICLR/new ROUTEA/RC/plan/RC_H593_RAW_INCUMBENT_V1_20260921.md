# H593：RAW原答案内容领先信息的受控读出

2026-09-21，用户授权“进行下一步”。固定原COST1、H593 grouped nested OOF5、自然RAW C128、20个封存内层基头。GAP_BIAS2为492/593；刚完成的直接净增GAP_NET2为491，不自动替换492。本轮不使用GAP_NET2挑选的外层参数，也不以0721/0809/0529选规则。

## 唯一机制变量

原六输入头的第一项为 `x_c=(RAW_c-RAW_w)/max(std_C128,1e-12)`。RAW_w为原答案；x_c非正，已存在于封存X中。原基头已读取当前challenger的x_c，不能声称此前完全没有考虑原答案。

当前最后一层只读取合成分数m及127个替代候选的分差d。检验重新读取RAW内容领先信息是否有跨组增量：

| 臂 | 额外输入r | 新门 | 参数数 |
|---|---|---|---:|
| GAP_BIAS2，原492精确复现 | 无 | m+beta*d+bias | 2 |
| GAP_RAWPAIR3，同参数数量信息对照 | 原最高challenger的x_c | (m+gamma*r)+beta*d+bias | 3 |
| GAP_RAWNEAR3，预定主臂 | max_{c!=w} x_c，即RAW最近竞争者的差分 | 同上 | 3 |

gamma、beta>=0，bias自由。r越负，表示RAW的内容领先越大；gamma*r越负，降低替换倾向。r只是检索分差，不是已校准正确概率。PAIR重读已被基头压缩的当前候选比较；NEAR聚合完整127轴，可能与当前最高纠错候选不同。既不新增图像编码，也不称为全新原始信息。

原最高challenger与原78次正SWITCH全部冻结。只在原HOLD上计算新门；g>0只替换原最高challenger分数，g<=0保留原127分数。不能用本轮修复原错误正SWITCH或C128缺失。

## 学习和单变量对照

保持492的训练配方：仅内层OOF原HOLD且delta非零记录，delta=I(challenger正确)-I(RAW正确)；均值logistic+0.001/2参数L2（含暂态bias），L-BFGS-B零初始化、maxiter2000、ftol1e-12、gtol1e-8。斜率固定后，在全部内层原HOLD（含双方都错delta0）上精确选择FP64偏置决策区间，最大净增、并列少改动、bias近0；无正净增则禁用。此处不声明斜率和bias联合经验净增最优。

本轮有意保持492训练法，隔离输入变化；不同时引入新损失、非线性结构、分组权重或新的超参数搜索。每折重新计算无新输入GAP_BIAS2，参数、优化向量及内层计数须逐位复现旧值。

原已有RAW2_CE=426和纯内容CONTENT7_COST1=427不读取RoMa联合评分及该HOLD竞争门，不能替代本轮对照。历史J端点实验是COST4=440基头上的不同特征。本轮不重复宣称“所有内容模型均无效”。原S_GAP3=486另作历史同参数数量参考。

## 标签和输入保护

每折拟合仅读本折TRAIN身份与同折20头中的四个内层OOF预测。已有内外层identity/component/source_image隔离继续检查。RAW特征来自75片无标签、原候选轴缓存；REAL/CBIND的RAW列须逐位相同。PAIR索引使用该条内层预测的original_top，不能用外层训练头给训练query重新选候选。

五折预测、新进程重训验证、独立输入索引/损失梯度/KKT/偏置全端点/127分数验证均通过并封存后，汇总才能读取curator外层身份。D1-MI、formal392及外部数据不读。不使用照片类型、框、mask、剂量手工标签；仍为retrieval-only监督。

## 报告和执行

主比较GAP_RAWNEAR3对GAP_BIAS2；同三参数的PAIR对照必须一并报告。完整593分母、23个自然C128缺失、64组件等权bootstrap（seed20260920，10000次）；报告R@1、MRR、救/损、逐折计数、gamma/beta/bias、训练净增及已有SWITCH逐位保护。不能将两臂较好者事后改名为预定主臂。区间和单次计数区分；H593已打开，此轮是开发验证而非外部确认。

代码、计划、launcher和来源SHA在拟合前冻结。五折最多5并发，accelerated分区，每项8CPU/16G/1GPU、10分钟，实际计算CPU；复用缓存，无编码器/RoMa前向。afterok汇总。没有自动部署变更。
