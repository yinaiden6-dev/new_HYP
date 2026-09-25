# TRAIN128四折：条件化内容补偿是否可学习

这是一项TRAIN-only检验。要回答的是：固定query权重后的reference-token选择分歧，
能否帮助学习比统一补偿更可靠的候选判断。它不预设完整geometry可信度理论成立。
本轮不读EVAL特征、标签、结果或rc_opened_eval_strict_*证书，不将其数值用于参数或选型。

## 数据和组隔离

使用已冻结TRAIN128及其F/A/D/mean_wr缓存，保持自然C128、原六列、全部127 challenger。
32个训练正例身份与32来源组一一对应。按
SHA256('RC_TRAIN128_DISAGREEMENT_OOF4_V1_20260910|'+group)
排序，以排序位次mod4分配，每折8组。不得根据目标、错误、分数或类别效果重新分折。
图库仍共享，隔离指query正例身份/来源组，不称reference图像全未见。

每折的基头使用原FULL32和PAIR64中不属于留出组的记录，从零按原train_head配方训练；
原PAIR64的历史特征/控制/标签保持不变。若FULL或PAIR为空则阻断。
基头训练后冻结。两种补偿只使用扩展TRAIN128中非留出组的完整FULL记录和检索身份标签。
本折留出组不得进入任何一层拟合；fold curator投影与heldout角色分离，参数与预测封存
后再计算留出指标。不能把固定ec7在其训练组上的表现冒充整个模型的OOF。

## 三个固定条件

BASE7：本折基头的精确原动作，无补偿。
CONSTANT1：z = z_base + softplus(beta0)*dF，1个学习参数。
CONDITIONAL4（主）：

    dF = (F_c-F_w)/(abs(F_c)+abs(F_w)+1e-12)
    dWR = (mean_wr_c-mean_wr_w)/(abs(mean_wr_c)+abs(mean_wr_w)+1e-12)
    alpha = softplus(beta0 + beta1*(-xRAW) + beta2*(D_w-D_c) + beta3*dWR)
    z = z_base + alpha*dF

w为RAW winner，c遍历全部127 challenger；xRAW为原六列中的标准化RAW分数差。
上述量均无需target标签。它是受分歧上下文调节的一步内容补偿，并非独立新V/空间网络。
没有额外归一化或按全体数据拟合尺度；原RAW每query的自然C128标准化保持不变。
softplus在有限参数下不等于0，所以BASE7单独保留，不能称它是补偿的有限参数初始化。

两种补偿beta0均为0，条件臂3个斜率也为0。两臂使用相同优化器AdamW(lr=.03,
weight_decay=.001)、seed17、2000步、最后一步；不挑checkpoint。基头均冻结，
补偿训练只用FULL原query loss：winner正确用4*softplus(maxwrong)；target非winner
用softplus(-target)+4*softplus(maxotherwrong)，然后query均值。两种补偿都不额外
加入PAIR loss，此变化只属于共同的补偿学习规则。保持原max ties的梯度约定；优先
复用逐query loss，不引入未说明的dim-max tie改变。

每折源文件/训练行/参数/留出全C128预测均封存，独立新进程从零重训与完整预测重放。
不改变RAW winner策略分数0，仍max logit>0才SWITCH，physical-row升序tie。
原始测试模型ec7不被修改；OOF各基头是排除留出组后重新拟合的模型。

## 固定放行条件

四折合并完整128个OOF query，按原组等权报告，不挑折：
1. CONDITIONAL总正确数严格超过BASE和CONSTANT。
2. CONDITIONAL对BASE和CONSTANT的等权组平均准确率差都严格为正。
3. CONDITIONAL相对RAW的break数不高于BASE相对RAW的break数。

同时报告相对BASE全部rescue/break及原正确集合保持，不把群体净增混称逐样本无损。
报告各模型末步损失/全部2000步有限性（原训练函数不改动，不承诺逐步曲线）、32组统计及固定seed20260910的10000次组bootstrap和精确
双侧sign-flip；本轮不以p值选择模型或修改放行条件。

未全部通过则停止这项四参数方案，保留原模型，不打开EVAL。通过也仅允许另行冻结
最终全TRAIN拟合/评估：最终保持原ec7不变，在全TRAIN128上学习补偿，先封存参数与
预测再join旧EVAL32及新EVAL128。较弱OOF基头上的收益不保证能改善完整ec7。
此处不预先授权最终拟合或EVAL计算，更不自动部署、进入P0或ownership。

所有学习只用检索身份标签，无任务box/mask/point/SAM或分割teacher；基础模型预训练
照实披露。不读D1-MI、GroZi或受保护formal392推理。截止北京时间2026-09-11 24:00。
