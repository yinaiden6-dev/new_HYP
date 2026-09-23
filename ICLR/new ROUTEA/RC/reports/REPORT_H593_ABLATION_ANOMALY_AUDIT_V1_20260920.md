# H593 六项消融反常结果：独立重算与机制分析

本报告回答：S 置零为什么从 COST1 的 481/593 提高到 487/593；为什么旧面板下降；其他删项结果为什么与直觉不同。范围为已打开的 H593 grouped OOF5、自然 RAW C128、既有六统计和 HOLD/SWITCH。23 张 target 不在 C128 的图继续计入全部 593 张准确率。本次不训练、不选择新参数、不改变部署。

## 1. 独立实现检查结果

新脚本 `programs/audit_rc_h593_ablation_independent_v1.py` 不导入原工程的 scorer、trainer 或 join。直接读取并核对 75 个封存特征分片、五折参数、候选物理行及修正后的 gallery identity，重新计算全部分数和决策。

- 593 queries、26 种头、1,958,086 个 challenger 分数全部通过。
- NumPy 逐项乘加与封存分数最大差为 `7.105427357601002e-15`，所有选中 reference 和正确数一致。
- 全部模型中最小第一/第二动作分数差为 `9.641232251000886e-06`，远大于重算误差。
- 五折 query、identity、component、原图 SHA 均无训练/测试交叉。
- FULL 参数与原 `loss_binding` 封存头相同；ZERO_S 精确仅把参数索引 1 置零，保留其余权重与 bias。
- 128 候选、127 challenger 和原 winner 的索引关系完整；HOLD=0，严格正分数才 SWITCH；没有漏掉 23 张候选缺失图。

在上述范围内未发现解释此次提升的实现错误。这不是从原图重新运行 encoder/RoMa 的全链路复验，也不宣称排除了所有上游问题。

证据：`results/rc_h593_ablation_independent_audit_v1/validation.json`；原始 `results/rc_h593_six_feature_ablation_v1/result.json` 和五折 `payload.json`。

## 2. 完整结果：置零与重训回答不同问题

完整头：COST1 **481/593**，CE **486/593**；RAW **426/593**。

| 删除的直接输入项 | COST1 固定头置零 | COST1 删项重训 | CE 固定头置零 | CE 删项重训 |
|---|---:|---:|---:|---:|
| RAW | 456 | 467 | 418 | 467 |
| S | 487 | 478 | 486 | 483 |
| M | 426 | 474 | 442 | 472 |
| L | 435 | 479 | 478 | 480 |
| Q | 480 | 480 | 486 | 483 |
| R | 481 | 480 | 486 | 485 |

ZERO 保留原坐标系和其余参数，检验既有头如何依赖该直接贡献。REFIT 删除一列后，从零重训其他权重和 bias，检验剩余坐标在同一训练配方下能否补偿。它们不是同一实验的两种写法。

## 3. S 提升的逐动作原因已经定位

对 COST1 FULL→ZERO_S：

- 原有 **78 次 SWITCH 全部选择相同 reference**。
- 41 个改变全部是 **HOLD→SWITCH**，并无原 SWITCH 改候选或退回 HOLD。
- 新开的 41 次 SWITCH 中：**14 救回、8 损失、19 仍然错误**，因此净增 6。
- 14 张新救回图的 target，在 FULL 下就已是分数最高的 challenger，只是该分数小于 0，无法越过 HOLD。
- 所有这些新选中候选的 S 直接贡献均为负；去掉这一贡献使其跨过 0。

因此，对这次 14 张救回，缺口明确发生在动作校准：模型已经把 target 排在 challenger 首位，但没有选择出手。这不等于所有余下错误都只是阈值问题，也不等于 S 置零等价于统一移动 bias。

逐例分数、S 贡献、old/new reference 和动作已保存到 `results/rc_h593_ablation_independent_audit_v1/s_changed_cases.json`。

## 4. 为何重训反而只有 478：现有目标确实不偏好 487 那组参数

独立使用 NumPy 重算原 COST1 surrogate，未调用原 loss 实现、未更新参数：

| 参数 | 五折 TRAIN 加权平均损失 | OOF 损失 | OOF 正确/593 |
|---|---:|---:|---:|
| FULL | 0.430356 | 0.440650 | 481 |
| ZERO_S | 0.508681 | 0.510968 | 487 |
| REFIT_S | 0.433444 | 0.441476 | 478 |

每一折中，REFIT_S 的 TRAIN 损失均小于 ZERO_S。ZERO_S 的 TRAIN 正确数加总也比 FULL 多，但损失更大。因此不能简单解释成只有测试分布偶然偏好 ZERO_S：原 surrogate 在训练集上就会偏好某些准确率较低、但分数惩罚较小的参数。

这里的训练正确数加总是五折重复使用的训练样本，不是额外独立数据。损失遵循原协议，只计算 target 在 C128 的样本：OOF 570 张；训练五折合计 2,280 次样本出现。准确率仍报告完整 593 张，未改变分母。

COST1 对错误置信度和 margin 计算连续惩罚，R@1 则每张图只计对/错一次。两者偏好不必一致。此次损失比较解释了为什么原配方没有自动保持 ZERO_S 的较高准确率；它不构成优化器全局收敛证明，也不是全部历史失败的统一根因。

证据：`surrogate_diagnostics.json`、`surrogate_diagnostics.csv`，均在独立审计结果目录。

## 5. 六列并非六份独立信息

原始候选统计满足 `L_g = S_g / max(M_g, eps)`。若均为正且忽略 epsilon，将 challenger 与 winner 做对称归一化后，记三列为 s、m、l，则：

\[
s \simeq \frac{m+l}{1+ml}.
\]

对全部 75,311 个 challenger 向量核算，关系最大误差约 `1.05e-8`，中位误差约 `3.53e-11`；corr(S,M)=0.9141，corr(S,L)=0.9623，corr(M,L)=0.8778。这个关系由定义产生，不是从 target 标签拟合出来的规律。

所以，删掉 S 的直接输入没有删掉所有 S 派生信息：L、Q、R 仍由原 real_score 计算。删除 M 或 L 的直接输入也没有消除所有相应信息。由于耦合关系非线性，剩余线性头又未必能完全补偿。

参数也给出直接证据：COST1 FULL 的 S 系数五折均负（约 -2.094 到 -0.999）；删除 M 后重训，S 系数五折全正（1.180 到 2.137）；删除 L 后也全正（0.158 到 0.438）。CE 同样出现这一变化。负 S 系数是其他坐标存在时的条件修正，不能解释成“内容证据有害”。

## 6. 其他反常现象如何理解

**M 置零回到 426，但重训回到 474。** 固定置零后 593 张全部 HOLD，最大 challenger 分数范围为 [-6.235,-0.467]，远离数值边界。原头失去了正贡献，又保留了既有负贡献和 bias。重训后其他耦合项接替部分作用。它支持“原头依赖 M 这一贡献”，不支持“M 的信息删掉以后还能无损恢复”。

**L 置零 435，重训 479。** 同样体现既有参数依赖与其他坐标补偿的区别。

**RAW 置零误伤很多。** 六统计中的 RAW 是 challenger 相对原 RAW winner 的分差，全部 75,311 行均非正；原头 RAW 权重为正。因此置零 RAW 精确移除了每个 challenger 的非正约束，使其更容易出手。COST1 SWITCH 78→142，RAW 误伤 3→36；CE SWITCH 117→228，误伤 11→86。这不是 RAW-only 实验，也没有去掉 RAW 候选与原 winner。

**Q/R 正确数变化小。** 单项额外增益确实较小，但正确数相同不等于动作相同：COST1 ZERO_R 有 1 救、1 损和 1 次错→错；CE ZERO_Q 有 2 救、2 损和 7 次错→错；CE ZERO_R 有 3 次错→错。RoMa 信息仍存在 S/M/L 中，不能据此说空间信息无用。

## 7. 历史下降与本次上升并不矛盾

| 协议/头 | FULL | S 直接项置零 |
|---|---:|---:|
| 旧 EVAL32 / ORIGINAL7 | 28/32 | 26/32 |
| 旧 EVAL128 / 同一个 ORIGINAL7 | 99/128 | 98/128 |
| difficult90 / FROZEN_C | 69/90 | 68/90 |
| H593 grouped OOF5 / COST1 | 481/593 | 487/593 |
| H593 grouped OOF5 / CE | 486/593 | 486/593 |

旧 32/128 与本轮不是同一个头；训练协议和评测群体也不同。旧结果没有被改写。仅在 H593 相同群体内，把 COST1 换成 CE，S 置零就变为 5 救5损、净增0，说明该干预依赖整个头的校准，不能认定 S 普遍有害。

历史来源：`rc_original7_six_feature_decisions_v1/result.json`、`rc_original7_eval128_full_evidence_v1/result.json`、`rc_frozen_group_effect_difficult90_v1/result.json`（均在 results 下）。

## 8. 可以确立和不能确立的结论

可以确立：此次 487 的计数和直接评分实现可信；新增正确来自原已排第一但被 HOLD 压住的 challenger；训练 surrogate 与准确率存在实际观察到的偏好差异；六统计存在由定义产生的耦合，因此单项消融不能视作六种独立视觉信息的删除。

尚不能确立：S 普遍有害、六项各自不可缺少、487 能在新数据稳定超过原头、调整阈值即可解决所有剩余错误。

S-zero 净增按五折为 [-1,+6,-1,+1,+1]；64 component 等权增益的 bootstrap95% 区间约 [-1.191,+4.545] 个百分点，仍含0。原消融表的单侧 p 检验方向是 FULL 优于 ZERO，不能拿它直接当成 ZERO 提升的 p 值。本轮是开发后分析，不替换部署结果。

后续优化应明确区分 challenger 排序与 HOLD/SWITCH 决策，再检验 TRAIN 内选择的校准是否能跨组改善净正确数；不应把这次观察到的 487 同时用于挑选模型和宣称独立验证。

## 9. 审计期间完成的 learned ColNomic-only 对照

本节是在新结果及验证文件完成后加入；与前述消融使用相同 H593 grouped OOF5、自然 RAW C128、训练顺序、COST1/CE 配方和动作规则。CONTENT7 使用六输入七参数，与完整头容量一致，但所有输入来自无 RoMa 权重的 ColNomic tokens：RAW gap、双向 MaxSim、双向第一/第二匹配差、整体相似度均值。

| 模型 | 正确/593 | 对 RAW 救/损 |
|---|---:|---:|
| RAW | 426 | 0/0 |
| 普通 PATCH_MAXSIM | 424 | 1/3 |
| learned ColNomic-only / COST1_CONTENT7 | 427 | 1/0 |
| learned ColNomic-only / CE_CONTENT7 | 431 | 5/0 |
| RAW+前向 MaxSim / COST1_MAXSIM3 | 426 | 0/0 |
| RAW+前向 MaxSim / CE_MAXSIM3 | 426 | 0/0 |
| 原联合证据 / FULL_COST1 | 481 | 58/3 |
| 原联合证据 / FULL_CE | 486 | 71/11 |

75 个分片全部完成，每个 query×candidate 的五项内容统计均有 NumPy 重算；五折训练有新进程重放。另用新脚本 `programs/audit_rc_h593_content_diagnostics_v1.py` 独立重算缓存统计到五种头的 376,555 个 OOF logits，最大误差 `7.105427357601002e-15`，动作与正确数一致。结果及训练/OOF损失保存为独立审计目录的 `content_surrogate_diagnostics.json`。

纯内容对照弱于完整头不是因为只比较了未训练模型。主比较 COST1 下，完整头对 CONTENT7 为 57 救3损、净增54；64 component 等权差 +10.636 个百分点，bootstrap95% [6.490,15.302]。CE 下完整头为67救12损、净增55；区间为 [5.446,16.563] 个百分点。这支持联合证据相对本轮预定纯内容统计和相同容量小头的额外价值，不证明所有纯内容模型都不能替代。

为什么 MAXSIM3 全部 HOLD：在每个 query 的 C128 内，前向无权重 MaxSim F 与 RAW 分数相关系数中位为 0.999453，584/593 的第一名相同。75,311 个 F challenger 对比值仅16个为正，最大正值也只有0.003067。F 大体重复原 RAW 的排序，未提供足够新的翻转依据。MAXSIM3 两种损失的最大 OOF challenger 分数均严格小于0，分别最高 -0.49009、-0.82517。

更丰富的 CONTENT7 不是完全无效：COST1 出手3次，1救0损2仍错；CE出手12次，5救0损7仍错。CE 的5次救回均有正的 Gq 贡献和正的 U 贡献（U对比为负、权重也负），而前向 MaxSim F 贡献均负。这里有“更突出的局部匹配峰值、较低的普遍背景相似度”组合的正信号。OUTCOME-0416 是 CE_CONTENT7 正确而 FULL_CE 仍错的样本；不能据一个已打开例子直接拼接 oracle 融合。

训练集上纯内容配方的收益也较小：五折重复训练出现合计 2,280 次，RAW 正确1,704，COST1_CONTENT7为1,712，CE_CONTENT7为1,729；原完整头对应1,922、1,956。说明弱提升不只是到了 OOF 才全部消失，但仍未通过本分析隔离纯内容表示能力与其优化/目标问题。

这些结果全部属于已经打开的 H593 开发分析；不构成新的外部确认、ownership 证明或自动部署替换。普通 PATCH_MAXSIM 本轮使用 FP64，旧免训练对照是 FP32，应按来源区分。

来源：`results/rc_h593_learned_colnomic_only_v1/result.json`、`validation.json`、`report_zh.md`、全部 feature/fold payload；独立审计目录的 `content_surrogate_diagnostics.json` 与 `content_redundancy_diagnostics.json`。
