# H593 保留曲率、移除额外RAW惩罚：嵌套五折开发结果

同一 COST1、自然 RAW C128、593 张分母；23 张 target 不在候选内。原排序及已有 SWITCH 冻结。

| 模型 | 正确/593 | 对 COST1 救/损 | SWITCH |
|---|---:|---:|---:|
| CE_FULL | 486 | 14/9 | 117 |
| COST1_FULL | 481 | 0/0 | 78 |
| GAP_NET1 | 484 | 13/10 | 115 |
| RAW | 426 | 3/58 | 0 |
| S_NET1 | 485 | 8/4 | 100 |
| S_SAFE1 | 482 | 3/2 | 87 |
| ZERO_S | 487 | 14/8 | 119 |
| BIAS1 | 484 | 13/10 | 115 |
| S_FIXED_BIAS2 | 486 | 11/6 | 106 |
| S_BIAS2 | 485 | 10/6 | 108 |
| GAP_BIAS2 | 492 | 16/5 | 112 |
| S_GAP3 | 486 | 12/7 | 112 |
| GAP_RAWPAIR3 | 486 | 12/7 | 107 |
| GAP_RAWNEAR3 | 482 | 15/14 | 119 |
| GAP_RAW_INTERACT4 | 486 | 16/11 | 116 |
| GAP_CURVE4 | 489 | 20/12 | 122 |
| GAP_CURVE3 | 490 | 13/4 | 103 |

主臂GAP_CURVE3对GAP_BIAS2（492）及GAP_CURVE4（489）；同三参数历史对照RAWNEAR3（482）。原COST1排序和正SWITCH冻结。
保留d*d曲率、gamma固定0，重训beta/eta/bias；RMS、监督、损失、精确偏置校准不变。原COST1中的RAW内容输入仍保留。

- GAP_BIAS2__to__GAP_CURVE3: 3 救/5 损；等组差 -0.004371，bootstrap95% [-0.017857142857142856, 0.008049242424242424]。
- GAP_CURVE4__to__GAP_CURVE3: 12 救/11 损；等组差 0.000249，bootstrap95% [-0.02594061760594389, 0.02388422947303921]。
- GAP_RAWNEAR3__to__GAP_CURVE3: 14 救/6 损；等组差 0.018499，bootstrap95% [0.0015500992063492065, 0.0381890261431277]。
- COST1_FULL__to__GAP_CURVE3: 13 救/4 损；等组差 0.018256，bootstrap95% [-0.009699364475903999, 0.04594109465519161]。
- CE_FULL__to__GAP_CURVE3: 9 救/5 损；等组差 0.006770，bootstrap95% [-0.008235465706168833, 0.022761093073593072]。
- ZERO_S__to__GAP_CURVE3: 9 救/6 损；等组差 0.001287，bootstrap95% [-0.010493290888798702, 0.012179807114607203]。

预定信号核算：{"primary_net_vs492": -2, "primary_net_vs489": 1, "primary_count_improvement": false, "curvature_without_extra_raw_development_signal": false, "both_descriptive_intervals_positive": false, "formal_external_GO": false}
区间仅描述当前开发数据，不校正历史反复选择。

H593 已打开；方法设计参考历史开发结果，本轮不构成新的外部确认，不自动替换部署。
