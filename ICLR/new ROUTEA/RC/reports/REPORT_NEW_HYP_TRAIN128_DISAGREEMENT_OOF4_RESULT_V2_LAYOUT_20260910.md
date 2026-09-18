# new HYP：TRAIN128 按来源组 OOF4 条件化检验

**预设 OOF 门：未通过。本次条件化机制按计划停止，不进入 EVAL。**

128 张图、32 个身份、32 个来源组，4 折各留出 8 组；所有候选均来自自然 RAW C128。每折 BASE7 仅用排除留出组后的原 FULL32／PAIR64 检索监督重新训练，再冻结基头训练补偿。这不是原 ec7 头；本轮 OOF 数字不能与原 ec7 的 EVAL128 99/128 当作同一模型、同一数据比较。

RAW 正确 86/128；自然 C128 缺失 target 0 张，全部保留统计。

| 模型 | 正确 | 对 RAW 救回／损失 | 等权来源组准确率 |
|---|---:|---:|---:|
| BASE7 | 108/128 | 23 / 1 | 83.125% |
| CONSTANT1 | 108/128 | 23 / 1 | 82.604% |
| CONDITIONAL4 | 107/128 | 22 / 1 | 81.979% |

BASE7 不加补偿；CONSTANT1 学习一个全局非负补偿强度；CONDITIONAL4 根据 RAW 差、匹配位置分歧差和 reference 平均权重差学习条件化强度。所有超参数和末步取值规则预先固定。

| 成对比较 | 新增正确／损失正确 | 净增图数 | 等权组差（百分点） | 组 bootstrap 95% 区间 | 组 sign-flip 双侧 p |
|---|---:|---:|---:|---:|---:|
| CONDITIONAL4 − BASE7 | 1 / 2 | -1 | -1.146 | [-3.958, +1.042] | 0.5 |
| CONDITIONAL4 − CONSTANT1 | 0 / 1 | -1 | -0.625 | [-1.875, +0.000] | 1 |
| CONSTANT1 − BASE7 | 1 / 1 | +0 | -0.521 | [-3.125, +1.562] | 1 |

统计单位为全部 32 个来源组，等权计组；bootstrap 10,000 次，固定 seed 20260910。双侧 sign-flip p 为精确组级计算，零差组保留。p 值不是本轮晋级门。

预设门要求 CONDITIONAL4 的正确图数和等权组准确率均严格超过 BASE7 与 CONSTANT1，并且对 RAW 的 break 不多于 BASE7。

- RAW break 不多于 BASE7：通过。
- 正确图数超过 BASE7：未通过。
- 正确图数超过 CONSTANT1：未通过。
- 等权组准确率超过 BASE7：未通过。
- 等权组准确率超过 CONSTANT1：未通过。

各折参数如下（显示值取 9 位有效数字，完整 FP64 十六进制值保存在参数产物）。BASE7 顺序为 RAW、S、M、L、Q、R、bias；CONDITIONAL4 顺序为截距、−xRAW、D_w−D_c、sym(mean_wr_c, mean_wr_w)。

| 折 | 留出图数 | BASE7／CONSTANT1／CONDITIONAL4 正确数 |
|---|---:|---:|
| 0 | 35 | 30 / 30 / 29 |
| 1 | 27 | 19 / 20 / 20 |
| 2 | 32 | 28 / 27 / 27 |
| 3 | 34 | 31 / 31 / 31 |

折 0：

- BASE7：`[0.940941313, -3.30868038, 4.78451666, 5.60907648, -0.0504513444, -0.155660305, -1.40815957]`
- CONSTANT1 α：`[25.4660407]`
- CONDITIONAL4 β：`[30.7845973, -10.0972216, -58.0628548, 33.9903982]`

折 1：

- BASE7：`[1.00162945, -3.89581481, 6.83195515, 5.57499998, -0.347657359, -0.432601056, -1.86393134]`
- CONSTANT1 α：`[22.1574188]`
- CONDITIONAL4 β：`[24.0669219, -4.78124875, -53.4272967, 24.0247934]`

折 2：

- BASE7：`[0.827863549, -2.17470883, 3.59545974, 5.4628095, -0.331964814, -0.409559551, -1.74428951]`
- CONSTANT1 α：`[29.8133483]`
- CONDITIONAL4 β：`[27.5296607, -3.01061349, -45.4021906, 22.3083437]`

折 3：

- BASE7：`[1.66914304, -4.63603222, 8.93552601, 6.90915227, -0.169039186, -0.0270573979, -0.901491455]`
- CONSTANT1 α：`[39.0269247]`
- CONDITIONAL4 β：`[31.3140095, -3.70996881, -27.1880683, 36.9930854]`

[全部32组结果图](/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC/results/rc_train128_disagreement_oof4_display_v2_layout/all32_group_results.png) · [全组逐项表](/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC/results/rc_train128_disagreement_oof4_display_v2_layout/all32_group_metrics.csv) · [完整各折参数](/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC/results/rc_train128_disagreement_oof4_display_v2_layout/fold_parameters.json)

[来源结果](/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC/results/rc_train128_disagreement_oof4_v1/result.json) · [独立复核](/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC/results/rc_train128_disagreement_oof4_v1/validation.json)

此报告只展示已封存结果，没有重新训练、读取 EVAL、挑选参数或筛掉负向／零增益组。OOF 门通过也只是为后续独立检验提供资格，不等于已证明普遍增益或零 break。
