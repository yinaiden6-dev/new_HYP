# Qwen乘积项完整结果与内部M V3阶段快照

快照时间：2026-09-24 08:06:29 UTC。文件映射保持原workspace相对目录；SHA256与排除清单见 [files.json](files.json)。本次不上传模型／适配器二进制权重、token缓存、原图、err/out日志、锁或未完成输出副本。

## 已完成：Qwen乘积项

原H593五折、ColNomic自然C128、同一冻结Qwen logits和RoMa配对质量；593张全部计入，目标在C128内570张。QWEN_ML5是M和L两列加性输入。QWEN_ML_PRODUCT6额外加入 `sym(M_g*L_g, M_w*L_w)`，共六参数，保留原加性结果。

| 损失 | 原加性ML5 | 新乘积PRODUCT6 | 对原ML5救回／误伤 |
|---|---:|---:|---:|
| CE（本追加对照主比较） | 536/593 | 537/593 | 2／1 |
| COST1（次比较） | 504/593 | 504/593 | 0／0 |

10/10拟合、五折汇总及独立复核通过。CE净增1但分组区间跨零，尚未建立稳定乘积增量；COST1正确集合不变。这是已使用H593上的开发性分组检验，不是新增外部确认。

- [结果与分组边界](../../ICLR/new%20ROUTEA/RC/results/rc_h593_qwen_quality_product_v1/report_zh.md)
- [完整五折结果目录：参数、全部127挑战者分数、逐query动作和验收](../../ICLR/new%20ROUTEA/RC/results/rc_h593_qwen_quality_product_v1/)
- [实验方案](../../ICLR/new%20ROUTEA/RC/plan/RC_H593_QWEN_QUALITY_PRODUCT_V1_20260924.md)
- [命名与公式澄清](../../ICLR/new%20ROUTEA/RC/reports/NOTE_NEW_HYP_HEAD_NOMENCLATURE_20260924.md)

## 进行中：内部M V3

本快照保存 PRE_CONSTANT 的128/128步与最终TRAIN16评估（8/16），以及 PRE_REAL 的第1–88步和第16步评估。真实M线尚未完成；不把阶段快照写成最终对比结果，不作泛化结论。

该实验保留RoMa的真实M，由LLM前适配器使用M，联合学习内部适配器与三参数读出；末端读出没有直接M列。固定TRAIN16、完整C128、128更新，主干冻结。保存代码、协议、测试、CPU对照、逐步数值、已完成endpoint分数与来源记录。模型／适配器checkpoint及token traces仍只保留在workspace，仓库中的指纹不代表这些二进制文件已上传。

- [V3执行方案](../../ICLR/new%20ROUTEA/RC/plan/RC_COLNOMIC_INTERNAL_M_LEARNED_USE_V3_EXECUTION_20260924.md)
- [代码](../../ICLR/new%20ROUTEA/RC/programs/run_rc_internal_m_learned_use_v3.py)
- [阶段数据](../../ICLR/new%20ROUTEA/RC/results/rc_internal_m_learned_use_v3/)

绝对workspace路径及原始封存hash保持不变。重新执行仍需恢复本地模型与未上传缓存；本仓库不是即克隆即运行的完整环境。
