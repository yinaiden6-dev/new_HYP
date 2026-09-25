# 无 RoMa 联合关系读出：64/32 开发试验

固定原 ColNomic 自然 C128，所有候选保留。仅训练身份标签；最终固定八遍端点。
SET/SPATIAL 等参数、等样本与更新次数，计算时间另列；COMPRESSED 是历史算子的同末端适配，参数量不同。
这是已打开的 H593 fold0 子集，不能称全593或外部确认。

|臂|角色|干预|正确/总数|RAW|救回|损失|组数|
|---|---|---|---|---|---|---|---|
|BASE|train|native|42/64|42|0|0|32|
|BASE|held|native|26/32|26|0|0|9|
|COMPRESSED|train|native|53/64|42|11|0|32|
|COMPRESSED|held|native|26/32|26|0|0|9|
|SET|train|native|42/64|42|0|0|32|
|SET|held|native|26/32|26|0|0|9|
|SET|held|shuffle|26/32|26|0|0|9|
|SET|held|rotate|26/32|26|0|0|9|
|SPATIAL|train|native|42/64|42|0|0|32|
|SPATIAL|held|native|26/32|26|0|0|9|
|SPATIAL|held|shuffle|26/32|26|0|0|9|
|SPATIAL|held|rotate|26/32|26|0|0|9|
|ORIGINAL_COST1|held|native|25/32|26|1|2|9|

独立 NumPy 候选logit核算最大误差：9.77e-15。
完整候选分数、关系证据、模型和优化器断点在 `results/rc_colnomic_relation_reader_v1/`。
结果支持程度需结合训练拟合、SET/SPATIAL配对和坐标干预分析；单次负结果不证明token缺信息。
