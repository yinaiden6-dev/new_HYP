# ORIGINAL7：GAP对照的独立探索性性能跟进

本轮来自已打开TRAIN OOF对照的事后选择，不是原主臂过门后的晋级。CONTENT_HOLD1的全部预设门失败、父authority不自动开放EVAL的状态原样保留，禁止把其结论改成GO。

GAP_HOLD1对照在TRAIN128四折留组中为110/128，相对每折BASE108新增2正确、0损失；它只提供需要后续评价的性能线索。这里的110不是冻结ec7在EVAL128上的99。本计划仅探索原联合模型HOLD边界的检索监督校准，不证明自由内容补偿、D估计、空间HYP或ownership。

用户授权的持续研究截止北京时间2026-09-11 24:00（UTC 16:00）。本计划独立授权仅本轮的已打开内部旧EVAL32与EVAL128评价；原其他P/V/ownership及formal392权限不变，不读取D1-MI、GroZi或新受保护推理。不使用新encoder、RoMa、F计算或人工空间标注。

## 固定模型与唯一训练

冻结ORIGINAL7/NATIVE7 ec7的六系数及bias，参数seal：
registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json，
file SHA42c8e503874cb807ca7085a39c802c016de2b1b8c6124bb17a56ed6f06bd174b，
parameter SHAec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263。

在全TRAIN128（固定32正例身份/组）的已资格化REAL原六特征上，只拟合一个全数据共享的非负有限FP64 GAP系数alpha。使用上一试验冻结的相同精确求解规则：TRAIN所有原ec7正确完整保留，在约束内最大化TRAIN新增正确，再选择实现该最大值的最小alpha；无可救则alpha=0。独立fresh进程用另一FP64位序二分重算阈值、参数及全部保护证书。没有多个候选系数、checkpoint或超参供EVAL选择。

仅TRAIN worker能读取TRAIN身份标签；它禁止读取EVAL预测/标签。所有训练参数、输入SHA、经验最优证书及fresh验证先封存，再运行EVAL worker。

## 预测规则与数据

每套评价使用自然RAW C128、原全部127 challenger logits和physical-row并列规则。优先直接复用已经独立资格化并封存的原ec7全127 logits，避免重新计算编码器/匹配器/特征；必须绑定原seal与validation，核对参数身份、query/候选轴和两套评价的完整覆盖。

BASE原max logit m>0时，全部原logits及SWITCH动作不变；原HOLD时，仅将原最高challenger的m变为fl(m+alpha)，其它126个原值不变。alpha=0直接返回原位值（含负零）。然后执行同一max>0 SWITCH，否则HOLD。

该动作等价于正向平移原top-logit决策门槛；不能包装成新增身份排序机制。它结构上保持原最高challenger身份及全部原SWITCH（无论其原来对错），但可能损失原HOLD正确，需要评价逐例核验。

完整评价两套，均不得剔除候选缺席、困难样本或表现不利的组：
- 旧matched EVAL32：RAW25/32，原ec7 28/32；
- 新EVAL128：RAW88/128，原ec7 99/128，24正例身份/21组，7个C128缺席保留。

两套EVAL参数、预测先封存、独立重放，再由reducer读取各自既有身份/组/RAW rank ledger。完整报告RAW、原ec7、ec7+GAP的正确、MRR、RAW救回/损失、相对原正确集合的救回/损失、每组及每套全部结果，不挑表现较好的一套作为结论。MRR分别沿用各自既有口径：旧EVAL32为自然C128内排名，新EVAL128为完整5412身份图库排名；均按原move-to-front最终选择与并列规则计算，输出显式rank_scope，不把旧32的C128 MRR写成full-gallery，也不合并两套MRR。

## 判断与科学范围

预先固定本轮探索性性能目标：
1. 新EVAL128正确数严格大于99，并保留原ec7全部99张正确；
2. 旧EVAL32保留原ec7全部28张正确；
3. 独立原baseline回归、参数/预测封存、完整候选轴及结果统计复核全部通过。

成功也仅登记为探索性内部性能候选，不自动覆盖原冻结模型或宣称new HYP全部成立。任一目标失败则明确报告并保留原模型，不从EVAL回调alpha、不改门、不错把TRAIN110称为EVAL提升。

固定各组等权、10000次bootstrap seed20260910、精确双侧group sign-flip；区间与p均为探索性描述，不作为额外挑选条件。此前多轮方法探索及两套EVAL反复打开产生适应性选择偏差；本轮再次封存可防止逐例调参，但不能恢复未触碰测试集属性。后续独立外部确认如需开展，必须另行设计与授权。

检索身份关系是本任务训练监督，不新增框、mask、点、人工crop或分割teacher；冻结基础模型预训练照实披露。若性能改善，支持的有限解释是原联合评分中已有身份排序信息，HOLD边界有可由检索监督校准的余量，不推翻此前M/ell与候选绑定的计算证据。
