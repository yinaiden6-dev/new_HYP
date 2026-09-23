# H593 固定范围CE数值精度闭合

5154655/5154656已完成；内容仍496，平方491。旧证书有效，但差距上界5.07e-6至3.14e-5，未达到原1e-6标准。不得据此宣布已闭合最优求解。

沿用原H593五折、593分母、自然RAW C128（570入围/23缺失）、原FULL CE、固定13/18参数结构、FP64特征和[-64,64]范围。每折从封存的CONTENT_BOX_CE18/DIAG_BOX_CE13出发，仅用TRAIN检索标签进行数值修正。采用固定30次上限的带边界活跃集阻尼Newton，投影梯度阈值1e-11、固定50次回溯、标准化Hessian最小二乘rcond=1e-14，无附加损失惩罚、无范围搜索、无评测选择。分数/动作与原主路相同。

保留全部旧对照与496最好点数。全部证书沿用Fraction精确FP64端点、dyadic单纯形和50位区间计算；原差距门1e-6不修改。求解器停止仍非证书，未达精度继续标记未决。每折新进程重放证书、Torch CE梯度与全部分数。合成核对Hessian、边界KKT投影、HOLD tie与封存起点。

五折预测全部封存后才打开held标签。性能信号仍须超过冻结496/CE486的正确数及等组件均值；主目标是完成先前数值诊断，不把优化精度GO写成识别性能GO，不证明无范围约束下最优或最优准确率。内容头原F系数五折均在+64边界，必须报告此限制。

Slurm五折array并行5、accelerated、每折10分钟；afterok汇总10分钟。无登录节点自然拟合。论文COST1保留，不进入ownership、D1-MI、formal392、GroZi。原来的5154655提交于accelerated但最终会计记录为dev_accelerated，完成记录如实登记，不推断是谁迁移。

入口programs/run_rc_h593_box_ce_polish_v1.py；authority registry/rc_h593_box_ce_polish_authority_v1_20260921.json。
