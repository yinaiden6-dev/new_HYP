# 全128归因链：因素、候选区分与外部／内部传播

本轮主问题：哪些计算输入／属性支持 M 对正确与错误候选的相对区分，这种区分如何通过原冻结外部头和 POST_REAL 内容调制传播；删除输入后，同结构读出能否补偿。

**全128自然候选面板是主结果。正确数为次要机制终点，不以某一正确数增长代替归因。**

三个层级分别是：直接 J/P 入口；P 内部相位与幅度属性；经 M 接口向下游传播。前两者不能相互替代。

## 冻结传播的连续效应

下表是加入该通路的条件效应（例如 P_with_J = A1J1P1 − A1J1P0）。每个指标先在 query 上取固定 target−wrong 差，再按 component 等权汇总。方括号为探索性组重采样区间。

| 面板／条件效应 | log M差 | 原外部NATIVE7 action差 | 简化M_FREE action差 | POST内容差 | POST action差 |
|---|---|---|---|---|---|
| FULL128/P_with_J | +0.613814 [+0.410939, +0.826052] | +0.617612 [+0.414815, +0.82684] | +0.60968 [+0.37753, +0.838597] | +0.0485649 [+0.0372881, +0.0604755] | +2.77518 [+2.10242, +3.48317] |
| FULL128/P_without_J | -0.07675 [-0.123063, -0.0338851] | -0.12027 [-0.186752, -0.0587506] | -0.132181 [-0.211373, -0.0587046] | -0.00746975 [-0.0113539, -0.00368187] | -0.341338 [-0.491646, -0.195631] |
| FULL128/J_with_P | +1.2692 [+0.988815, +1.56276] | +1.53137 [+1.20859, +1.86155] | +1.62431 [+1.29206, +1.96165] | +0.0577135 [+0.0419965, +0.0744804] | +3.74656 [+2.8401, +4.69347] |
| FULL128/J_without_P | +0.578641 [+0.44005, +0.728256] | +0.793489 [+0.601854, +0.999135] | +0.882446 [+0.680107, +1.09903] | +0.00167886 [-0.00338685, +0.00729874] | +0.63005 [+0.25752, +0.999009] |
| FULL128/J_P_interaction | +0.690564 [+0.465541, +0.917653] | +0.737882 [+0.503387, +0.977047] | +0.74186 [+0.472999, +1.00654] | +0.0560346 [+0.0419161, +0.0710386] | +3.11651 [+2.33919, +3.95286] |
| GROUP_DISJOINT12/P_with_J | +0.508387 [-0.0877596, +1.23373] | +0.305783 [-0.129607, +0.744071] | +0.228341 [-0.279493, +0.703646] | +0.053722 [+0.0221921, +0.087328] | +3.20311 [+1.25967, +5.31004] |
| GROUP_DISJOINT12/P_without_J | -0.121248 [-0.235858, -0.0205394] | -0.181328 [-0.351159, -0.0319757] | -0.217445 [-0.421891, -0.0373023] | -0.0116918 [-0.019804, -0.00420563] | -0.497947 [-0.796492, -0.20245] |
| GROUP_DISJOINT12/J_with_P | +1.39889 [+0.542098, +2.337] | +1.54207 [+0.650616, +2.43347] | +1.66163 [+0.717897, +2.61624] | +0.0638445 [+0.0274928, +0.10294] | +3.90628 [+1.80408, +6.13215] |
| GROUP_DISJOINT12/J_without_P | +0.769254 [+0.468675, +1.05297] | +1.05496 [+0.607071, +1.48198] | +1.21584 [+0.75222, +1.64591] | -0.00156925 [-0.00713057, +0.00461629] | +0.205219 [-0.727229, +1.14441] |
| GROUP_DISJOINT12/J_P_interaction | +0.629635 [-0.00651506, +1.36009] | +0.48711 [-0.0391851, +1.01128] | +0.445786 [-0.175426, +1.05858] | +0.0654138 [+0.0271298, +0.106298] | +3.70106 [+1.47352, +6.07877] |

POST_REAL 接收真实 M 条件；其末端 INTERNAL3 头不直接读 M。外部路径所有依赖 M 的统计量同步更新，其余局部形状和自由内容固定。这是 **M-only中介传播**，不是重新运行整个 RoMa 与编码器的总干预。
所有新消融相对于同次 COARSE A1J1P1；ORIGINAL_HR1仅为原模型重放锚点。COARSE与HR1的差异不混入 J/P删除效应。

## 属性恢复及新机制组重复

旧14对固定 target/wrong 的联合恢复已接到同一外部／POST模型。若 RAW winner 是第三个候选，固定其原 HR1 M/L 锚点；因此只能解释两候选差值，不生成完整C128排序或准确率。

| 条件恢复 | log M差 | NATIVE7 action差 | M_FREE action差 | POST内容差 | POST action差 |
|---|---|---|---|---|---|
| phase_restore_after_amplitude_damage | +0.0379368 [+0.00901402, +0.0657666] | +0.0386395 [+0.00810293, +0.0715633] | +0.0365848 [+0.0039746, +0.0687208] | +0.00228321 [+0.000484645, +0.00423197] | +0.11078 [+0.0138122, +0.210562] |
| amplitude_restore_under_local_damage | +0.0635971 [-0.0429016, +0.197599] | +0.0703077 [-0.05172, +0.226991] | +0.0518709 [-0.0971946, +0.20581] | +0.014085 [+0.0034152, +0.0253722] | +0.670049 [+0.123365, +1.29602] |
| interaction | -0.0113152 [-0.0282638, +0.00839644] | -0.0128096 [-0.0310499, +0.00918456] | -0.00957019 [-0.0302859, +0.0192659] | -0.00243787 [-0.00460855, -0.000447727] | -0.109962 [-0.212706, -0.00810225] |

历史相位桥接使用 LOCAL−GLOBAL；本轮恢复使用 GLOBAL−LOCAL，符号已经明确记录。
新组复验从此前F71以外的9组各选最小ordinal，共9张；每张仅预先固定的target／自由内容最强wrong。18臂全部在同一CPU后端计算；没有把旧救回案例9/21/32当作独立确认。

| 新组条件恢复 | target logM效应 | 固定wrong logM效应 | target−wrong logM效应 |
|---|---|---|---|
| phase_restoration_after_amplitude_permutation | +0.0284987 [+0.00632286, +0.0486608] | +0.00432073 [-0.0236841, +0.0326756] | +0.024178 [+0.00761779, +0.0404502] |
| amplitude_restoration_under_local_phase | +0.261316 [+0.149915, +0.379053] | +0.243839 [+0.122917, +0.383227] | +0.017477 [-0.0924877, +0.126123] |
| factorial_interaction | -0.0484903 [-0.0614228, -0.0350523] | -0.0419124 [-0.0524502, -0.0326596] | -0.00657794 [-0.0180138, +0.00407127] |
| phase_restoration_native_amplitude | -0.0199916 [-0.0295517, -0.0113759] | -0.0375916 [-0.0590672, -0.0166414] | +0.0176 [+0.00189784, +0.0351864] |

新组实际target-present分母为9组；候选缺失不补入，也不从下一张替换。

## 同结构补偿：直接入口层

20项训练仅比较删除 J/P入口后的 PRODUCT5补偿：原五折、相同COST1、步数、参数量与固定0阈值。它检验这套读出能否利用剩余信息，不是相位或幅度属性的删除后重训。
下表是删除通路模型相对原通路重训模型的 OOF target−strongest-wrong action margin，负值表示当前配方补偿不完全；不能据此推出信息不存在。

| 面板 | 删除输入 | 连续margin差及探索性区间 |
|---|---|---|
| FULL128 | A1J1P0 | -0.774495 [-1.00564, -0.541566] |
| FULL128 | A1J0P1 | -1.08199 [-1.32895, -0.834355] |
| FULL128 | A1J0P0 | -1.16872 [-1.44004, -0.892741] |
| GROUP_DISJOINT12 | A1J1P0 | -0.657045 [-1.27995, -0.146472] |
| GROUP_DISJOINT12 | A1J0P1 | -0.944009 [-1.61818, -0.328284] |
| GROUP_DISJOINT12 | A1J0P0 | -1.06033 [-1.75279, -0.390972] |

每折的TRAIN拟合损失、拟合正确数及收敛轨迹仍保留在原结果中；未充分拟合时，不能将OOF失败解释成信息缺失或不可替代。

## 四步因果矩阵

| 属性层级 | 干预 | 条件恢复 | 同族补偿 | 组隔离重复 |
|---|---|---|---|---|
| Direct correspondence-distribution input P | Measured on full128: frozen direct P deletion, with J present and absent, and same-intervention M-to-external/POST propagation. | The full J-by-P factorial measures conditional entry effects and their interaction; this is not recovery of a uniquely identified geometric property. | Completed: all128 original five folds, same PRODUCT5 family and COST1 recipe, including P-deleted inputs; interpret against recorded TRAIN fitting quality. | Locked twelve queries/nine groups absent from prior F71 are reported separately for frozen propagation; data remain opened H593. |
| Direct cross-image context input J | Measured on full128: frozen J deletion, P present/absent; P may still carry information produced by the native interaction. | Conditional J-by-P effects measured; J deletion is not deletion of every cross-image computation. | Completed: J-deleted and J/P-deleted same-family PRODUCT5 compensation with matched recipe. | Same locked group-disjoint12 frozen-mechanism subset, not an untouched external dataset. |
| Cross-patch phase consistency of P | Historical balanced LOCAL/GLOBAL perturbations preserve patch norm/frequency power and match signed perturbation dose. | Completed fixed14 conditional restoration: GLOBAL versus LOCAL with AMP_PERMUTE retained; propagated to the same frozen external and POST paths. | NOT COMPLETED: deleting the entire P input and refitting PRODUCT5 does not test phase-specific deletion and compensation. | Completed separate fixed-pair replication: minimum ordinal from each of nine groups absent from F71, eighteen same-CPU arms; exact target-present denominator retained below. |
| Binding of patch amplitude/concentration to position | Historical amplitude permutation preserves the norm distribution but changes which patch receives which amplitude. | Completed fixed14 restoration with LOCAL damage retained; report measured direction and uncertainty rather than assuming stable recovery. | NOT COMPLETED: whole-P entry removal does not isolate amplitude-binding removal and compensation. | Completed amplitude-specific conditional restoration within the same eighteen-arm, nine-selected-group fixed-pair replication; not downstream full-C128 replication. |

## 单图可分性与配对项

pair_structure是无标签输入结构检查：检验query侧因子是否跨候选恒定，以及 log M(q,r) 是否能由单图加性项解释。若发现非零矩形循环残差，它排除了该面板上严格的单图可分表达；它本身不证明这些非可分项就是身份收益的唯一来源。
完整arm统计和循环审计保存在 complete_stage_results.pair_structure，不将理论代数恒等式与测得的数值残差混为一谈。

| 输入通路 | query侧跨候选极差最大值 | reference侧跨query极差最大值 | logM矩形循环RMS |
|---|---:|---:|---:|
| A1J1P1 | +0.369854 | +0.265223 | +0.874114 |
| A1J1P0 | +0.145721 | +0.0480021 | +0.635324 |
| A1J0P1 | +0.38177 | +0.346777 | +0.163532 |
| A1J0P0 | +0 | +0 | +1.63793e-16 |


## 次要决策结果

| 面板 | M来源 | POST | 原外部NATIVE7 | M_FREE |
|---|---|---:|---:|---:|
| FULL128 | A1J0P0 | 93/128 | 93/128 | 93/128 |
| FULL128 | A1J0P1 | 93/128 | 93/128 | 93/128 |
| FULL128 | A1J1P0 | 94/128 | 100/128 | 100/128 |
| FULL128 | A1J1P1 | 100/128 | 102/128 | 102/128 |
| FULL128 | ORIGINAL_HR1 | 101/128 | 102/128 | 102/128 |
| GROUP_DISJOINT12 | A1J0P0 | 7/12 | 7/12 | 7/12 |
| GROUP_DISJOINT12 | A1J0P1 | 7/12 | 7/12 | 7/12 |
| GROUP_DISJOINT12 | A1J1P0 | 7/12 | 8/12 | 10/12 |
| GROUP_DISJOINT12 | A1J1P1 | 9/12 | 9/12 | 9/12 |
| GROUP_DISJOINT12 | ORIGINAL_HR1 | 9/12 | 9/12 | 9/12 |

## 可主张的范围与未闭合部分

本轮能够给出：特定计算入口／属性的条件作用、正确／错误候选相对证据的变化、这份证据经冻结外部和内部路径的传播，以及当前同结构读出的可补偿程度。
**不能据此主张 M 是唯一原因、身份充分统计量、RoMa 普遍不可替代，或全部空间信息都由P入口承载。** A/J中仍可能包含空间信息；固定内部模型只读取M也是设计事实，不是充分性证明。
相位／幅度属性的删除后补偿仍未由本链完成，必须与本轮已经完成的J/P入口补偿分开。
数据是已打开H593的前128。12张／9组是相对于此前F71的机制组隔离复验，不是新外部数据，也没有重新定义旧593、32或128基准的最优成绩。

独立阶段验收与其直接输出封存SHA已重新检查；没有再次运行编码器、RoMa或训练。所有完整summary、逐query结果及来源封存在同目录result.json。
