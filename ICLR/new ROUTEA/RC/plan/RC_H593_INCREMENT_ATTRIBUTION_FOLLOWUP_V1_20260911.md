# 正向593结果的针对性归因跟进

原593 CONDITIONAL4对BASE7_ALL为5救0损（440→445）；对SMALL128为3救2损（444→445），对CONSTANT1仅1救0损。五个新增均是原top challenger已经为target但未过HOLD。新端点符号分支没有提升，其RELATIVE2的J_BIND反而更高。本轮保留正向主线，关闭ENDPOINT2作为当前性能候选，补足对收益来源的控制。

这是看到已完成OOF后的探索性跟进，不冒称原预注册主检验或未触碰外部确认。固定593图/68身份/64组/原五折、所有C128、原BASE7_ALL/CONSTANT1/CONDITIONAL4参数。原实验结果和部署不改。

唯一新增拟合是BIAS1：z_new=z_BASE+b，b一个共享无约束标量，零初始化；只用该折原训练身份标签，复用原FULL检索损失、FP64、seed17、AdamW lr=.03/weight_decay=.001、2000步。它是与统一内容补偿同为一参数的全局偏置控制，不能将其失败外推为所有阈值选择方法都无效。

固定三模式：REAL；INCREMENT_BIND（原BASE六特征保持REAL，只把新增补偿所读的dF及D/reference均重上下文按原CBIND映射移位）；CBIND（全部原证据和增量一同移位）。BASE和BIAS1在INCREMENT_BIND须逐位等于REAL；已有BASE/CONST/COND在REAL/CBIND须逐位复现原fold输出。

主归因比较：CONSTANT1对BIAS1（同1个新增参数），以及CONDITIONAL4的新增5救回在INCREMENT_BIND下保留多少。全部模型总正确/救损、完整MRR、64组差与区间同时报告，不挑选有利控制，不因为shuffle结果高就采用shuffle模型。

BIAS1仅读当前fold训练角色和当前fold父参数；不得读取其他fold标签/参数、rc_opened_、原593/端点分支最终成绩作为拟合输入。全部五折新参数和预测通过重放后再join。缺席target的23张仍在完整593评价分母中。

资源沿用用户已授权的accelerated占1GPU、CPU8/16G、CUDA屏蔽，5折并行，申请10分钟/折；join与独立复核各10分钟。此处不做任何GPU模型计算。UTC9月11日16:00截止不变。
