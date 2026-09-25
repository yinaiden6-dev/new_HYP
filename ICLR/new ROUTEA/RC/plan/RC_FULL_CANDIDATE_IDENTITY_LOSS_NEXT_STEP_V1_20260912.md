# 下一步候选：完整候选身份损失与接口解释分开验证

状态：DESIGN_ONLY，未提交训练任务，不是新理论或性能结论。

## 为什么提出这一项

刚完成的原因隔离发现：原mixed96记录损失在预定参数盒中已接近最小值，继续降低它仍28/99；错误惩罚4→1仍27/99（128为5救5损）。六类已知可行的改进动作区域，其原TRAIN风险下界均高于原头。这个结果没有证明所有目标都失败，也没有证明全部信息缺失，但不支持继续仅压低同一个目标或只改那个成本系数。

本次限定源码/报告核查未找到：原native6/7param、原PAIR64+FULL32、RAW0和127个challenger共同softmax的FULL交叉熵目标。已有PAIR_RANK只改最强wrong减target，仍有独立target过零项和cost4；unit-cost只改三个4→1，FULL仍为SIGN/max-wrong。P-only family CE是不同模型/接口，不冒充本项。

## 唯一新条件 LISTWISE_UNIT1

保持原RAW自然C128、全部127challenger、six native features、七参数头、原96图和顺序、seed17、AdamW lr.03/weight_decay.001、2000步、PAIR/FULL均值1:1、FP64及最终HOLD/SWITCH/tie。旧头不重训。

设z_RAW=0，z_c=w^T x_c+b，target为y：

    L_FULL = logsumexp([0,z_1,...,z_127]) - z_y

若target就是RAW，z_y=0。PAIR只含原RAW和唯一challenger，使用cost1普通BCE，即匹配的二类log-loss。

主要单因素对照为LISTWISE_UNIT1对刚完成的TRAIN_UNIT_COST7：PAIR已同为cost1、相同优化器和数据，只替换FULL损失形式。强性能参照仍是原ec7的28/32、99/128，不能只超过较弱的27便称为超过原模型。

原推理等价于argmax([0,z])，零分边界优先RAW、正分同分按既有physical-row规则。因此新目标使用与决策相同的候选集合。这个结构一致性不保证七参数受限函数类、有限数据和混合PAIR分布上必然改善。

FULL损失形式改变也会改变梯度分配和相对尺度；若有增益，先归因于这项整体目标替换，不能未经额外控制就专归于某个抽象概率性质。不增加温度/损失系数扫描，不按EVAL挑epoch。

## 读数与边界

- 参数和全部160图预测先封存，再读原EVAL32/EVAL128标签；两个面板分别报告，对原ec7和对UNIT_COST7的救回、损失、净增均列出。
- 当前用户允许损失、要求净增，不再加零损失门。不能因机制检验未充分成立而抹掉已观察净增，也不能从一两例净增倒推出统一理论。
- 与既有六项风险冲突的关系可作后续解释，但那些EVAL诊断系数/病例挑选不得成为训练或选择输入；不把该分析条件变成额外性能门。
- 这些都是历史已打开开发面板，不是外部确认；593若要测试需另按其五折协议完整重训对照，不能把旧99与447混为同一头。
- Softmax/交叉熵本身不是原创。多类别代理损失与分类一致性已有文献，例如 [Tewari and Bartlett, JMLR2007](https://www.jmlr.org/papers/v8/tewari07a.html)。本文若要强化new HYP，需要具体说明保留何种reference证据、目标改动改变了什么，以及其可复现实证增量，不能只把标准损失换名。

## 接口线的合理位置

旧P缺完整reference质量、旧模型依赖M、无损P/V回放是已证解释性贡献。当前ec7本来含M和完整reference搜索，继续补M不会超越它。

完整质量/atom代理 × full-reference MaxSim/RoMa硬对应的四格归因尚未核实完成，但原RAW64与EVAL128 visibility缓存没有完整warp或hard indices，不能承诺纯缓存零forward。其它fresh-D1/P支持轴不能拼接，缺401对也不能删。它适合量化历史接口损失，不应作为下一次准确率突破的默认赌注。

精确保留决策的自适应计算也可作为工程备选，但普通pruning/caching已是成熟方法，后置MaxSim节省不了已经付出的RoMa前向，端到端收益尚未测量，不作为new HYP主张。

## 依据

- programs/run_routea_matched_three_arm_common3_native7_crossfit_v1.py:434
- programs/run_rc_native7_training_objective_factorial_v1.py:40
- programs/run_rc_unit_cost96_cause_v1.py:78
- reports/REPORT_CONVEX_CAUSE_ISOLATION_V1_20260912.md
- reports/NOTE_INTERFACE_PRESERVATION_BREAKPOINT_EVIDENCE_20260912.md
