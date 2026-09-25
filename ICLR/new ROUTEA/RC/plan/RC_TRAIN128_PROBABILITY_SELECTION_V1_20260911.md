# 固定FIT候选池的概率损失选择：未来净增目标

用户最新要求为准确率/可靠净增优先，允许EVAL128原正确损失一两张。之前原正确全保的门和所有NO-GO保持历史原样。本轮仍只访问TRAIN128的既定outer4fold，不授权真实EVAL或部署。

上一选择试验的候选在SELECT前三折都没有新增正确，原BASE却已29/31、32/35、34/35正确；二值准确率无法在多数持平候选间提供选择信号。现在只检查连续身份预测损失是否提供有用区分。没有新候选训练、LP、切组、温度、特征、encoder或RoMa。

## 不变输入

完整复用rc_train128_group_held_selection_v1的FIT-only候选池、FIT fresh验证、16FIT/8SELECT/8outer元数据、旧BASE7与原六维X。必须确认每个pool是在SELECT标签打开前封存及复算。SELECT只对增量拟合隔离，BASE可能见过其中原FULL/PAIR；outer仍对该折基头及增量排除。outer已打开，是开发复用，不称独立外部确认。

四臂都是同一池的成员，BASE必含。仅SELECT标签用于选择，outer标签必须在全部新预测和fresh验证后打开。禁止任何rc_opened_*（含hash）、真正EVAL数据、D1-MI/GroZi/formal392推理。

## 唯一新选择量

将原RAW winner的策略分数0和全部127 logits作为128个类的能量，温度固定1，不拟合温度。每个SELECT query的CE为logsumexp([0,z])-z_target；RAW target取z_target=0。当前TRAIN128全部target自然在C128，必须核验；否则本次选择资格失败，不插入target。

各query CE先FP64计算，再以其binary64值的Fraction在每组内平均、八组等权。它衡量分配给正确reference的概率，不把更低CE当成更高实际准确率或泛化保证。

SELECT原正确损失预算固定ceil(2*n_SELECT/128)，本批每折均1张；这只是按用户最终128图预算预先给出的选择过滤，不保证outer或真实EVAL也最多损2张。不得事后调整。

- BASE7：原基头。
- ACC_BUDGET：SELECT损失不超过预算，要求组准确率严格大于BASE，依组准确率/正确数/L1/ordinal原顺序选，未满足回BASE。
- CE_FREE：CE严格低于BASE者中依CE、实际参数L1、ordinal选；无改进回BASE。
- CE_BUDGET（primary）：与CE_FREE同样选择，但先过滤SELECT原正确损失超过上述预算的候选。

CE相同使用准确Fraction比较，L1为已有FP64参数相对BASE的精确端点差；BASE在完全相同选择指标下优先。所有theta都必须逐bit等于池中theta，SELECT阶段无LP或更新。

## outer读出与下一阶段

四折参数/选择分数/全部127 logits先封存并fresh复算，再打开outer角色。报告四臂正确、RAW救损、对BASE paired救损、全32组与各fold，及SELECT的CE/损失预算/选择原因。

探索性EVAL资格门：primary相对BASE净增>0、原正确损失<=2、等权组平均增益>0。这只是进入另行冻结真实EVAL比较的资格，不称可靠GO。固定seed20260910、10000次组bootstrap和精确双侧sign-flip，并报告增益/损失组数、删除任一组后的最小净增。置信区间跨0或增益集中一组必须明确为可靠性未建立。

真实EVAL将来必须单独冻结全TRAIN选择与读出，完整报告同一新EVAL128相对原99的新增/损失以及旧EVAL32相对原28的变化；本计划不预先开放。任何已打开EVAL的适应性选择偏差照实保留，不用证书系数或EVAL调参。

本轮失败则停止这项选择规则，不挑控制改写primary。原模型和已有结果不修改。只用检索身份标签，无新增空间标注/分割teacher，基础模型预训练披露。CPU8核最多20分钟，截止UTC2026-09-11 16:00（北京时间11日24时）。
