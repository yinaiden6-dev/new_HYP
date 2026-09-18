# GLOBAL7统一微调：原固定32/128面板结果

原各折TRAIN OOF 114/128是先前开发结果；本页评价最终全TRAIN128残差。原ec7冻结、七维残差按固定2000步配方拟合。RAW C128及完整127 challenger HOLD/SWITCH不变。

| 面板 | RAW | 原ec7 | LISTWISE_UNIT1 | GLOBAL7_T128 | 对原头救/损/净 | 对LISTWISE救/损/净 |
|---|---:|---:|---:|---:|---|---|
| EVAL32 | 25 | 28 | 26 | 27 | 0/1/-1 | 1/0/+1 |
| EVAL128 | 88 | 99 | 101 | 98 | 3/4/-1 | 2/5/-3 |

允许损失，按每面板rescue-loss报告观察净增，来源组不确定性及CBIND结果保存在机器结果。两套已打开开发面板不是外部确认。没有改写原114 TRAIN留出、28/32或99/128原模型成绩，没有自动替换部署或宣称new HYP GO。

[机器结果](result.json) · [独立核算](result_validation.json)
