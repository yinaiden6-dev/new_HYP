# 2026-09-24：ColQwen base、简单融合对照与蒸馏

## ColQwen base：五折全部完成

本地colqwen2.5-base（无检索适配器），自有全图库自然C128，H593原五折。召回438/593；155张缺失仍计分。

|方法|正确/593|对RAW救/损|
|---|---:|---:|
|RAW|227|0/0|
|纯内容CONTENT7 COST1|240|13/0|
|纯内容CONTENT7 CE|260|46/13|
|MASS5 COST1|322|96/1|
|MASS5 CE|333|117/11|

这是base对照，不是训练版ColQwen。质量MASS5小头逐折重训，不是原参数零样本迁移。

- [最终报告](<ICLR/new ROUTEA/RC/reports/REPORT_COLQWEN_BASE_NATIVE_FINAL_20260924.md>)
- [逐query结果](<ICLR/new ROUTEA/RC/results/rc_colqwen_base_native_v2/head/result.json>)
- [候选分数、五折参数与127个logits](<ICLR/new ROUTEA/RC/results/rc_colqwen_base_native_v2/head>)
- [独立复算451866个logits](<ICLR/new ROUTEA/RC/reports/rc_colqwen_base_final_independent_check_20260924.json>)

## 简单解释对照：完成

ColNomic自然C128、H593同五折，固定零阈值：COST1加性4参数、乘积5参数、两个同参数量平方项加性头均481/593（57救2损）；当前结果不支持乘积项对该简化机制不可缺少。不能用旧EVAL32上的优势代替这个较大面板结果。

内部验证选择阈值后的PRODUCT5为477（66救15损），简单门控为434（25救17损）；报告保留全部预算和CE敏感性分析，既不只展示有利阈值，也不宣称胜过所有门控。整体比较为已打开开发数据的机制诊断。

- [全部结果与预定比较](<ICLR/new ROUTEA/RC/reports/REPORT_H593_SIMPLE_EXPLANATIONS_V1_20260924.md>)
- [逐候选分数、参数与选择过程](<ICLR/new ROUTEA/RC/results/rc_h593_simple_explanations_v1>)

## 蒸馏：部分完成，尚无新held119结论

64张TRAIN四项拟合门全部通过，排序一致率83.57%；8张开发probe55.04%，未通过。续提失败已修复为cpuonly链。后续457拟合/分组选择状态以该快照中的原始产物为准，不能把n64通过当作跨组成功。

- [拟合与泛化修复记录](<ICLR/new ROUTEA/RC/reports/REPORT_M_DISTILL_FIT_AND_GENERALIZATION_REPAIR_20260924.md>)
- [产物快照](<ICLR/new ROUTEA/RC/results/rc_m_distill_generalization_v1>)

本次保存源码、计划、封存协议、原始JSON结果及小头参数。未新增二进制模型、训练checkpoint、token缓存、原图或err/out日志。绝对路径保留来源，不代表clone即可完整运行。[文件SHA清单](backup/final_controls_20260924/files.json)。

`head/rows.json`约151MB，未作为Git单文件提交；全部逐query源JSON已收录，可运行`python tools/rebuild_colqwen_base_head_rows.py`逐字节重建，脚本核对原authority SHA256。
