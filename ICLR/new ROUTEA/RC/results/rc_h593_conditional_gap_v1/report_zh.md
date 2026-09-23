# H593 条件化竞争读出：嵌套五折开发结果

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

主臂GAP_RAW_INTERACT4对GAP_BIAS2（492）；同四参数对照GAP_CURVE4；结构对照GAP_RAWNEAR3（482）。原COST1排序和正SWITCH冻结。
新增一个训练HOLD行RMS缩放的交互项r*d或平方项d*d，新增系数自由；既有m/r/d、监督、损失与精确偏置校准不变。

- GAP_BIAS2__to__GAP_RAW_INTERACT4: 5 救/11 损；等组差 -0.016743，bootstrap95% [-0.03539243579770924, -0.00046955737787539596]。
- GAP_RAWNEAR3__to__GAP_RAW_INTERACT4: 5 救/1 损；等组差 0.006127，bootstrap95% [-0.0007726332720588175, 0.01485906862745098]。
- COST1_FULL__to__GAP_RAW_INTERACT4: 16 救/11 损；等组差 0.005884，bootstrap95% [-0.021473816503400602, 0.0318145946233607]。
- BIAS1__to__GAP_RAW_INTERACT4: 10 救/8 损；等组差 -0.001544，bootstrap95% [-0.022902102365567438, 0.01891665846025909]。
- CE_FULL__to__GAP_RAW_INTERACT4: 9 救/9 损；等组差 -0.005602，bootstrap95% [-0.02128767072643451, 0.007369162508753496]。
- ZERO_S__to__GAP_RAW_INTERACT4: 14 救/15 损；等组差 -0.011085，bootstrap95% [-0.03136307890817311, 0.006743509311868677]。
- GAP_BIAS2__to__GAP_CURVE4: 11 救/14 损；等组差 -0.004621，bootstrap95% [-0.03154947336415627, 0.024077404162422537]。
- GAP_RAWNEAR3__to__GAP_CURVE4: 13 救/6 损；等组差 0.018250，bootstrap95% [-0.0003567947065189703, 0.04090696470349502]。
- COST1_FULL__to__GAP_CURVE4: 20 救/12 损；等组差 0.018007，bootstrap95% [-0.004864404450227059, 0.042650881020021646]。
- BIAS1__to__GAP_CURVE4: 16 救/11 损；等组差 0.010579，bootstrap95% [-0.016313349736201296, 0.03997529166732981]。
- CE_FULL__to__GAP_CURVE4: 15 救/12 损；等组差 0.006520，bootstrap95% [-0.01585412870167006, 0.03210455311385917]。
- ZERO_S__to__GAP_CURVE4: 18 救/16 损；等组差 0.001038，bootstrap95% [-0.02491726697781385, 0.029429063647239195]。
- GAP_CURVE4__to__GAP_RAW_INTERACT4: 8 救/11 损；等组差 -0.012123，bootstrap95% [-0.035112833130411254, 0.006890837123201544]。

预定信号核算：{"primary_net_vs492": -6, "primary_net_vs_curve": -3, "primary_count_improvement": false, "interaction_development_signal": false, "both_descriptive_intervals_positive": false, "formal_external_GO": false}
区间仅描述当前开发数据，不校正历史反复选择。

H593 已打开；方法设计参考历史开发结果，本轮不构成新的外部确认，不自动替换部署。
