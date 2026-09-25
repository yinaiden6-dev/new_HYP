# TRAIN 内检验照片内容条件化：固定四折最小对照

用户在逐图讨论后明确继续，授权这一项有界的retrieval-only检验。此前9月11日截止不被修改；本文件记录9月12日新继续指令下的单项工作。无外部确认或部署步骤，不恢复所有历史试验。

问题：仅query的内容表示，能否帮助共享小头选择六类既有证据的权重，超过统一权重和同容量无正确query绑定的对照？这不是已验证的照片类型分类器，也不保证读懂数字或恢复被压缩丢失的信息。

数据固定为原TRAIN128/32身份/32来源组、已有四折，每折留出8组；复用已验证各折BASE7参数，原头训练更新0。新残差仅用同折TRAIN的FULL C128身份标签训练。旧EVAL32/128、593其他图、EVAL视觉标注和oracle系数禁止进入。本次128是TRAIN分组留出，不是原99/128的EVAL面板。

令x为原六特征，a=[x,1]，z0为冻结各折BASE7的原logit。所有新臂共用

    z(q,c)=z0(q,c)+sum_{j,k} a_j(q,c) B_{j,k} h_k(q,c)

RAW自身logit固定0，仍遍历全部127 challenger，max>0才SWITCH，按原physical顺序打破平局。初始B=0，初始动作与BASE一致；允许新的正负残差。

四臂一次性固定，无超参数搜索：

| 臂 | h | 新参数 |
|---|---|---:|
| GLOBAL7 | 常数1 | 7 |
| STATS28 | 1＋旧条件门的3个统计量，经本折TRAIN均值/标准差规范化 | 28 |
| QUERY28（预定主臂） | 1＋query内容的3维主成分，经本折TRAIN规范化 | 28 |
| QUERY_PERM28 | 同QUERY28，但在TRAIN和heldout各自内部，分别按固定query hash循环置换内容描述 | 28 |

STATS使用RAW分差、自由/加权匹配分歧差、reference平均可见性权重差；不新增dF项。与QUERY的相同28参数结构比较条件输入，不把此前CONDITIONAL4的不同结构/不同损失当单因素对照。

query描述固定为原runtime全部有效image tokens的FP64均值（128维），不读取文件名、身份、类型标注；先确认token数量等于image grid乘积。只在每折TRAIN的描述上求PCA前三维，固定特征向量符号，按TRAIN投影标准差缩放，保存解释方差与特征值间隔。不用留出样本拟合PCA、归一化或任何选择。三维是预定容量约束；失败只排除这个粗内容摘要及当前训练配方，不能排除全部照片条件化。

置换对照是固定一次的负对照，不是有统计保证的独立性检验；不跨TRAIN/heldout搬运描述，不使用标签生成置换。QUERY28部署路径仅需要本图描述，QUERY_PERM28仅作批内打乱诊断。

新训练四臂同样FULL-C128交叉熵：logsumexp([0,z1,...,z127])-z_target；图片等权，不再沿用cost4的错误惩罚。AdamW，lr=.03，decoupled decay=.001，FP64，B全零，2000更新，末步，CPU8。四臂数据、基头、初始化、损失和预算一致。GLOBAL7是本次新目标的匹配统一读出，不将其训练称为重新复现原头。

工程检验：有效token/图片SHA/自然C128绑定；Torch与NumPy独立均值比较；原BASE logits与历史封存逐bit相同；合成逐项公式与向量化loss/gradient比较；四折新参数fresh-process重放；全部128预测封存后独立NumPy重算logits/动作，然后才打开heldout标签。

读出固定：QUERY28对BASE7、GLOBAL7、STATS28、QUERY_PERM28分别报告rescue/loss/net、完整gallery MRR、逐来源组差和组bootstrap区间。观察净增以rescue>loss计，允许损失；条件输入贡献要求超过同容量STATS与PERM，统计不确定性另报，不把零损失当门。报告每折/每组，防止一个身份产生大量相似图掩盖不稳定性。

本项不自动对旧EVAL重测或晋级。若query摘要无收益，应结合本折解释方差和错误变化判断局限；不自动改维度、改seed或追加模型。候选细粒度差异的内容读出是后续不同输入问题，尚未由本项实现。
