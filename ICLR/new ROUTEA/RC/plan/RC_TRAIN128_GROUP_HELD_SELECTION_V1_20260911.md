# TRAIN128：增量拟合与选择分组隔离的同池检验

## 目的和当前事实

上一轮PROTECTED投影在每折同一24组上产生、保护并选择候选。TRAIN保护均成立，但outer留组新增6正确、损失4正确；全部新损失都是原正确HOLD被错误SWITCH。选模优先TRAIN收益，L1仅后置排序，曾选择L1位移18.89/29.31的更新；较小位移的另一折仍有损失，不能据此断言简单限步长即可修复。

本轮只改变增量候选的拟合/选择分工，检验选择保护能否迁移。不是更换encoder、添加特征、读取EVAL证书或宣称HYP成立。原真实EVAL仍为ec7旧28/32、新99/128。研究截止北京时间2026-09-11 24:00（UTC16:00）。

## 固定分组与角色

outer仍是既定TRAIN128的四折、每折留出8个正例身份/来源组（图数35/27/32/34）。原outer BASE7参数不重训，来源/全部outer127 logits须回归。

每折outer训练24组按sha256('RC_TRAIN128_GROUP_HELD_SELECTION_V1_20260911|'+group)及group字符串排序，前8组为SELECT，其余16组为FIT。seed与分割不随任何结果改变。metadata prepare只读取原outer train_roles的query/identity/group元数据，生成独立FIT和SELECT角色文件及公开query-role清单；不读取特征、预测或正确性来选组。

重要限定：原BASE可能已使用SELECT组中的原FULL/PAIR标签训练。因此SELECT只与增量LP拟合隔离，不称基头从未见过SELECT。outer8组同时未参与该折基头、FIT、SELECT；outer此前在开发中已打开，本轮仍为开发复用，不称未触碰外部确认。

## FIT阶段：真正只用16组产生候选

只加载FIT身份标签，不能调用旧load_fold先把24组target装入再筛选。原24组候选池不能复用为‘未见SELECT’候选。

原六维特征、自然RAW C128、全部127 challenger、RAW winner策略分数0和physical-row规则保持。复用上一保护投影的固定数值学习规则：FIT中原BASE正确构成P；FIT原错且target在C128构成E；从FIT原P的精确正margin求delta。每个E仅一次PROTECTED7最小L1改动LP，约束P+e的完整动作；delta仅数值搜索目标，最终精确正margin及实际FP64 P全保、e修复方可入池。原BASE必入池，无合格候选或无正有限delta时精确回退。

固定highs-ds、presolve、30秒和1e-9原容差，不新扫参。FIT候选池、拟合分数、拒绝账本及参数封存；fresh进程只读FIT角色重新求解，核验候选及分数/选择确定性后，才解锁SELECT角色。

## 同一候选池的四臂

所有新参数只能来自已封存FIT候选池，SELECT阶段不得新拟合。

- BASE7：原outer基头。
- FIT_SELECT：原FIT等权组准确率、总正确数、最小实际Fraction L1、source error ordinal的词典序选择。
- SELECT_RANK：先要求SELECT等权组准确率严格高于BASE，再在合格者中用同样SELECT组均值/正确数/L1/ordinal排序；无严格增益则BASE。
- SELECT_PROTECT（primary）：与SELECT_RANK共用严格SELECT组均值增益要求，再加‘SELECT原BASE正确集合完整保留’过滤；其余排序完全相同。无合格者则BASE。

这样FIT_SELECT与SELECT_RANK比较增量选择数据变化；SELECT_RANK与SELECT_PROTECT比较额外正确集合保护。不能将不同作用混成一个已证明原因。P_FIT硬保护不外推到SELECT或outer；SELECT保护也不外推到outer。

SELECT阶段单独封存评分、入选/拒绝原因、参数及全部outer127 logits，并由fresh进程复算。只有四个outer折全部封存、资格化，才解锁outer标签并统一汇合128query。

## 固定读出与门

报告RAW和四臂的正确数、RAW救回/损失、相对BASE及同池控制的paired变化、完整正确集合、各outer折与全32组。主性能门保持：SELECT_PROTECT比BASE总正确数严格增加、完整保留BASE正确集合；等权组平均增益>0作逻辑一致性和效应量检查。两个选择控制不额外阻断primary性能门，也不根据outer表现事后改主臂。

固定seed20260910、10000次组bootstrap和精确双侧group sign-flip，p只描述。训练/选择/outer三层数字必须分别标记，不将TRAIN128成绩写成原EVAL128的99得到提升。

未过主门则停止本配方、不进入真正EVAL、不重切组或改规则。若过，最终拟合/评估也须另行冻结：原全32 TRAIN组同一hash顺序前8 SELECT、其余24 FIT，固定ec7原头；同一旧EVAL32/新EVAL128完整报告，不能更换系数或选择集求好结果。本计划不提前授权该最终阶段。

## 输入、执行与解释边界

程序硬拒绝所有rc_opened_*路径，包括hash；容量witness、系数、EVAL错例筛选不进入FIT/SELECT。没有D1-MI、GroZi、formal392推理、新F/encoder/RoMa、SAM或任务框/mask/点标注。只有TRAIN query/reference身份关系作为任务监督，基础模型预训练照实披露。原模型及旧结果不修改。

模型选择本身也可能过拟合有限数据，可参考[Cawley与Talbot，JMLR 2010](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf)。本轮只是检验上述具体分工，不继承未触碰nested-CV的无偏保证，也不承诺基头已见的SELECT完全独立。

CPU8核、最多59分钟；实际Slurm ID与UTC截止均需检查。只在新目录results/rc_train128_group_held_selection_v1追加产物。提交后不重复运行完整有效的上游缓存，不监控其它分支作业。
