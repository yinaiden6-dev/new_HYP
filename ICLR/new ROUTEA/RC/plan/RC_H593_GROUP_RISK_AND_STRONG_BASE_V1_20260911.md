# 统一假说的最小实证检验：风险权重与强基线补偿

理论稿：reports/NEW_HYP_UNIFIED_DECISION_HYPOTHESIS_V3_20260911.md。用户明确目标是理论统一与突破；本轮只检验两个已定位的混杂，不重新搜索特征、不做大范围规模曲线。

固定原593图、68身份、64来源组、原五折。既有ALL/SMALL基头与ALL统一/条件补偿直接复用各fold已验证参数，不修改历史SMALL采样（四折少一身份仍披露）。RAW候选、完整127 challenger、FP64、共享identity-only监督不变。

新增：
1. 在完全相同ALL训练行上，按损失有效行的component计数，使每个非空训练component总权重1/G；仅改变数据损失聚合。得到GROUP_BASE，并用同组权重训练GROUP_CONST、GROUP_COND。
2. 冻结原强SMALL基头，使用与原ALL补偿完全相同的ALL训练行、图片均值损失，训练SMALL_CONST、SMALL_COND。

共复用4个模型、新拟合5个模型每fold。统一7参数基头零初始化；补偿仍1或4参数，原AdamW lr=.03、weight_decay=.001、2000步，不早停或选seed/checkpoint。group损失是每query原SIGN loss的非负加权和，权重合计为1；有效组计数只来自当前fold训练行。缺席target不伪造正项，全部593仍进入OOF分母。

预定主比较GROUP_COND对既有ALL_COND；强对照GROUP_COND对新补齐的SMALL_COND。GROUP_BASE对ALL_BASE用于分解权重效果；GROUP_COND对GROUP_CONST、SMALL_COND对SMALL_CONST判断条件化的独立价值。全部9模型均报告，不根据结果改主臂。若SMALL补齐后更好，应如实接受强对照，不能继续只拿440作基线。

REAL、INCREMENT_BIND和CBIND按既有定义执行，作为标准控制，不当作新的绑定发现。既有4模型REAL/CBIND需逐位回归原fold动作/logits。新拟合和所有预测经fresh重算后，全部五折封存才join身份标签。

拟合过程禁止rc_opened_、父实验最终结果和其它fold参数/训练标签。该方案受已打开开发结果启发，不是未触碰确认。报告微平均净增、救/损、等组差及不确定性；不以宏平均单独改善或正净增自动宣布突破。

资源沿用用户授权的accelerated上1GPU占位、CPU8/16G、5折并行，拟合10分钟每fold；join/review10分钟。无GPU模型计算、新encoder/RoMa或额外任务空间标注。截止UTC2026-09-11 16:00不变。
