# new HYP：TRAIN128 reference选择分歧缓存

目标是为一次预先固定的TRAIN-only条件化补偿检验准备输入，不新增模型或EVAL结果。
继续用户授权的retrieval-only研究，截止北京时间2026-09-11 24:00。

复用已资格化的TRAIN128 tokens/maps及自然RAW C128，32训练正例身份/32来源组。
只读TRAIN worker和已封存的32个token/map payload及其receipt/validation；不读取
curator、EVAL、oracle证书，不训练、不运行encoder/RoMa，不改变C128或原七参数。

对每个query和其全部128个reference，保持原query权重wq，FP64计算：

- F = sum_i wq_i * max_j cosine(q_i,r_j) / clamp(sum_i wq_i,1e-12)。
- j*(i) = argmax_j [wr_j * cosine(q_i,r_j)]，并沿用first-index平局。
- A = sum_i wq_i * cosine(q_i,r_j*(i)) / 同一分母。
- D = F-A；另存reference平均权重mean_wr及query平均权重mean_wq。

严格保持先乘/求和、后除法，不重排为先归一化wq的E[p*cos]。所有ref tokens均
保留完整轴。D仅刻画在固定当前query权重时，reference权重改变匹配位置所造成的
cosine差异；不是全部geometry损伤，也不是身份正确概率。不能由D大就判某候选错误。

同一cosine矩阵必须逐bit复现原S/M/SQ/SR，包括重新计算roll后的均值；共65536标量。
保存F/A/D/mean_wr/mean_wq [128,128]、query IDs/ordinal、物理candidate轴和原C4。
新进程以独立NumPy argmax/索引提取及canonical pooling复核，并另用math.fsum检查；
不以高精求和替换生产FP64值。参数更新、图像读取、encoder/RoMa calls均为0。

后续条件化检验以另一个固定OOF4计划为准；cache PASS不等于模型有收益。使用
8CPU/16G/20分钟CPU作业，原数学输入不变，尚不授权访问EVAL。
