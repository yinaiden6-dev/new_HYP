# GLOBAL7统一微调：全TRAIN128拟合与固定32/128面板检验

用户在TRAIN分组留出GLOBAL7 108→114结果后明确授权继续，检验统一微调能否改善原EVAL32/EVAL128。本文件仅授权这一固定配方，不自动替换部署或扩展其他试验。

只拟合GLOBAL7_T128。冻结原ec7六权重和bias，使用原TRAIN128全部128图/32身份/32来源组、原顺序及完整RAW C128。初始化七维残差delta=0，保持上一轮GLOBAL7：

    z0 = X @ original_weight + original_bias
    z = z0 + [X,1] @ delta
    loss = mean(logsumexp([RAW0,z_1,...,z_127]) - z_target)

直接复用run_rc_query_content_routing_oof4_v1.py中的design、objective、optimize；FP64，AdamW lr=.03、decoupled decay=.001，2000步、最后一步，无阈值、温度、epoch或参数搜索。新生产计算保留上述两项相加顺序，不将合并参数后浮点计算冒称逐bit相同。残差衰减不是修改旧checkpoint。旧原头训练更新0。

本次将四折中“排除留出组的基头＋本折TRAIN残差”推广为“冻结全原训练ec7＋全TRAIN128残差”。不重训基头；同一配方改变的是最终训练范围，不保证OOF收益必然迁移。

评价只用原EVAL32和EVAL128，已打开的开发面板。复用rc_full_candidate_identity_loss_v1的ORIGINAL7、LISTWISE_UNIT1预测和参数，原候选/RAW排序/127 challenger/physical tie/HOLD-SWITCH规则不变。新臂REAL与已有CBIND计算均封存后再打开评价身份标签；训练与评价的query、原query ID、图片SHA、identity、group、component逐项检查无交集。训练128的T128 ID以原query ID和图片SHA桥接到已有H593训练元数据，仅用于检查组归属，不扩大训练图。

每面板分别报告RAW、ORIGINAL7、LISTWISE_UNIT1、GLOBAL7_T128正确数、完整gallery MRR，以及新臂相对原头和LISTWISE的rescue/loss/net及来源组统计。当前用户标准为观察净增，允许损失；是否两面板同时改善、是否超过当前101/128及不确定性另列，不事后拼接不同头的最佳数字。

工程验证：训练原六特征/候选轴重放；训练元数据只投影128图；旧ec7二进制参数核对；原GLOBAL7训练函数源码绑定；合成独立CE loss/gradient与一步AdamW参数核对；新参数fresh-process重放；160图三模型两模式121920 logits独立NumPy核算；原模型预测和逐图结果逐bit/逐项保持一致。评价标签不参与拟合或选头。

旧114/128保留为TRAIN四折结果，本轮结果不会追溯改写它。本项即使通过观察净增也只是固定开发面板结果，不直接证明新HYP普遍理论或外部泛化。失败按实际救损收口，不自动换配方补跑。
