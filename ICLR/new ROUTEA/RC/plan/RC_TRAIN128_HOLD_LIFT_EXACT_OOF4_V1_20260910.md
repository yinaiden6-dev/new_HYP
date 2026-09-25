# TRAIN128：保留原SWITCH与候选排序的内容辅助HOLD校准

冻结时间：2026-09-10。研究截止：北京时间2026-09-11 24:00（UTC 16:00）。

## 动机与范围

前一四参数补偿的完整TRAIN OOF结果为BASE108、CONSTANT108、CONDITIONAL107；三者RAW break均1，但两次损失均丢掉BASE已有rescue。0389原top已经是target，只因logit<0而HOLD；0087被负内容补偿压回HOLD；0102被不同补偿改变了top竞争。因而当前试验缩小动作修改范围，不再对全部候选任意加减内容补偿。

这是一项新的内容辅助HOLD校准试验，不是D估计成功、空间HYP、ownership或新身份排序网络。GAP控制本质为原top logit的全局门槛平移，CONTENT为依赖自由内容对比的条件门槛；不能把这种等价性隐藏成新模型原理。检验的是内容是否帮助决定何时采纳原联合模型已选定的reference。

原最优ec7与EVAL128的99/128原样保留。本计划禁止最终ec7训练、EVAL推理/标签读取、D1-MI、GroZi、formal392推理以及任何新encoder/RoMa。仅复用已资格化TRAIN缓存、固定四折和原各折BASE参数；当前TRAIN及OOF先前结果已打开，是开发期复用，不称未触碰独立确认。

## 固定输入与基头

使用rc_train128_disagreement_oof4_v1的固定四折：32个query正例身份/组，各折8组，图数35/27/32/34。每折BASE7此前只用其它组的原FULL32/PAIR64训练，参数和原所有127个heldout logits已经封存及fresh重训复核。本次直接复用这些BASE参数；必须验证旧authority、fold manifest、各fold参数seal/validation的SHA与原基头训练来源，不复用旧补偿参数。

用rc_original7_train128_inputs_v1的REAL六特征与rc_train128_disagreement_features_v1的F、候选轴；原RAW自然C128、完整127 challenger不变。BASE logits仍按原X@w+b算序计算，heldout BASE逐bit对照旧封存预测。标签只允许读取当前fold的train_roles；四折新参数/预测及fresh replay都通过后才打开heldout roles做join。共享gallery正负reference资源照旧，正例query身份/组隔离不等同于图库reference从未作为训练负例出现。

## 两个一参数臂

每个query由BASE的全部127 logits选原最高challenger t，按physical row处理并列；令m=z_t、w为RAW winner。

- BASE7：完整保留原动作。
- GAP_HOLD1：h=1。
- CONTENT_HOLD1（primary）：h=max(sym(F_t,F_w),0)，sym(a,b)=(a-b)/(|a|+|b|+1e-12)，FP64算序固定。

两新臂均有每折一个共享非负有限FP64系数alpha。

若m>0，BASE已经SWITCH：全部127 logits与动作原样输出。
若m<=0，BASE原来HOLD：仅将原最高challenger改为

    z'_t = fl(m + fl(alpha*h))

其余126个logits原样输出；仍对全部127个logits执行原max>0 SWITCH，否则HOLD。禁止重选top-K、插入target、根据标签选择候选或修改候选轴。

结构性质：alpha,h非负，所以HOLD时原最高challenger仍最高；已SWITCH的所有样本完全不变。因此0087/0102这类原SWITCH决策不会被该修改破坏。原HOLD正确仍可能被新切换破坏，必须在TRAIN约束和OOF正确集合中直接检验，不宣称普遍无损。该族不能修复BASE已错误SWITCH，也不能修复原top challenger不是target的身份竞争错误。

## 精确的检索监督拟合

不用AdamW，不选择步数、学习率或多个checkpoint。两臂在同一fold TRAIN身份标签上求解同一有限一维问题：

1. 硬约束：保留该fold全部TRAIN BASE正确的样本。
2. 在约束下，最大化TRAIN新增正确数。
3. 多个alpha得到同一最大救回数时，取最小非负可表示FP64 alpha；无可救回样本则alpha=0，精确回到BASE。

BASE已SWITCH样本不依赖alpha。BASE HOLD且h=0也不依赖alpha。对每个m<=0、h>0的TRAIN行，以canonical fl(m+fl(alpha*h))>0为谓词，在非负有限binary64 bit序上二分，求首次过零的最小可表示alpha（最多63次比较）；若最大有限alpha仍不能使其过零，则无有限过零点。

TRAIN原BASE正确且HOLD的样本必须不过零，其安全上界是该首次过零alpha的前一个浮点数；所有保护样本取最小上界，空集合不添加上界。TRAIN原BASE错误且原top=target、h>0的样本，过零点不超过安全上界时可救；取所有可救点的最大值即覆盖所有可救样本的最小alpha。该单调结构给出限定类的全局经验最优；不代表一般模型最优或泛化保证。其它wrong-to-wrong变化单独报告。

实数容量解释可用U=min(-m/h)及严格跨零条件，但实际选参及保护核验必须依据上述FP64两步算序，不能用一次浮点除法近似后宣称边界安全。正浮点bit序不含NaN；训练与heldout输出均要求有限，源特征h须在[0,1]且有限（GAP为1，CONTENT由sym保证）。因此有限alpha与h的乘积有限，非正m与非负乘积相加也不会溢出。alpha=0必须直接返回原完整logits及动作，以保留负零位值。对选中的alpha逐例重算完整动作、全部TRAIN BASE正确集合保留与rescue最优性/最小性证书。不会根据OOF标签修改alpha。

## 执行、对照与门

prepare/preflight仅元数据与合成边界；自然fit/replay/join须单独hash绑定authority、实际Slurm执行和用户截止检查。合成测试覆盖alpha=0、m=0、subnormal、h=0、同一阈值下保护/救回不可兼得、无保护样本、无可救回样本、物理序并列及原SWITCH完全保留。fresh进程必须独立重算参数与全部新127 logits，再按旧流程封存后join。

四折合并全部128图，固定primary CONTENT，只有以下全满足才可另行冻结最终ec7/EVAL试验：

- CONTENT正确数严格大于BASE和GAP；
- CONTENT等权组准确率严格大于BASE和GAP；
- CONTENT保留完整OOF BASE正确集合，即相对BASE的break=0。

同时报告RAW rescue/break、相对BASE与GAP的paired变化、各fold/全部32组、结构锁住的原SWITCH数、原HOLD新切换数、wrong-to-wrong、TRAIN安全上界/可救数/精确alpha。固定seed20260910、10000次共同组bootstrap和精确双侧group sign-flip，p只报告，不进门。

任一门未满足，停止这个受限HOLD校准族，不打开EVAL；若满足，也仅成为另行冻结全TRAIN拟合/EVAL验证的资格，尚无超过99/128的性能结论。BASE7/OOF108始终与原ec7/EVAL99明确分开。

任务训练仅query/reference身份标签；基础模型既有预训练监督照实披露，无额外框、mask、点、人工crop或SAM。
