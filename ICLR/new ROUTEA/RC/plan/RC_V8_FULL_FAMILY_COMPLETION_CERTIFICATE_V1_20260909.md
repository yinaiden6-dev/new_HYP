# 完整合法区域的可区分性与缺失项补全界

2026-09-09。用户授权继续推进。一次 opened TRAIN32 分析，不是 V9 拟合、
新识别结果、P GO 或正式评估。原 V8 NO-GO 保留；正式392的P/RoMa结果、V、
action、HOLD/SWITCH、ownership封闭，不读改或依赖D1-MI、GroZi和其它任务。

## 已验收的依据，以及当前还不能推出什么

Job5137551完成，完整矩阵及独立artifact验证通过。20个target-H1失败中，
19个target在其原H_T上胜过原strongest-wrong，7个在H_T和H_W上都胜但原对角
比较仍失败。对完整C128，固定H_T的32条分解为：8条完整同域优势、14条
没有已知blocker但存在UNKNOWN、7条有已知blocker、3条target-H0。

8+14=22只是固定H_T的正向区分证书数量的乐观上限，不是新排序方法准确率
的上限。原矩阵仅包含1672个已选H，未检查2691个完整合法regions，因此
不能据此认定仅改变选区无用，也不能直接把相对分数公式再拟合一遍。

本次同时回答两个必要问题：

1. 不将缺失视为匹配或不匹配，已观测成本能否排除缺失项藏着更强解释？
2. 保持冻结V8参数和全部原合法regions，是否存在可区分完整C128的区域？

## 冻结的完整范围

沿用原InputBundle、原32×C128、REAL的全部2691个cycle-closed maximal
components、最终V8 REAL head及FP64残差计算。无训练、无新模型、无新增
region、无删点、无top-K、无面积/温度/阈值扫描。只计算REAL；本诊断没有
重做matched AP/Q、C_BIND/P_COORD自然门，因此不能产生P GO。

每个owner的每个合法H，在同一source query cell集合上检查全部C128
reference的原RoMa assignment。所有原selected-H的完整交叉分数、能量必须
逐位重放Job5137551产物；原对角支持和physical witnesses必须一致。
完整区域与候选轴全部保存，不能只保存有利区域或最强错误reference。

## 唯一分析公式：对缺失补全的保守能量下界

沿用原V8计算每个已观测atom的FP64 scalar costs a_i、b_i。对已观测的g个
reference groups，A_j=max a_i，B=max b_i；m是该完整query支持内缺失的格数。

把这些已经计算出的FP64 costs视为精确二进制有理数。任何保留已知assignment、
将m个缺失格配到任意有效reference token的假想补全，均满足：

`E_completion* >= L* = sum_j A_j / (g+m) + B`。

理由：补全不能减小已有group maxima或global max；最终group数不超过g+m。
本次固定分母g+m，不再尝试reference轴上限等另一公式。全缺失时L*=0。
缺失格的实际成本没有设为0，也不使用域外坐标的nearest-token placeholder。
零成本只出现在放宽的理论下界中，绝非观测或补全结果。

完整项用`E*=sum_j A_j/g+B`。用Fraction进行分子、分母和符号判断，不使用
exp近似来证明界。同步保留原PyTorch的energy/K及理想标量表达式的差异。
这不是对全部PyTorch reduction/mean/exp浮点误差的严格运行时包络，也不是
新的FP64模型或bit-exact优化。不能把解析certificate替代冻结自然score。

UNKNOWN继续表示原assignment不可用；补全界只是约束假想解释集合。
假想补全可以不满足geometry，放宽只会使排除竞争者更难。certificate不证明
补全实际存在、缺失格可见，或存在可解释的语义冲突。

## 逐region结论与全family汇总

owner合法H的所有atoms有效，故E_owner*完整。对全部其他candidate：

- 所有竞争者的完整E*或缺失L*均严格大于E_owner*：CERTIFIED。
- 任一完整竞争者E*<=E_owner*：REFUTED，保存已知blocker。
- 其余：UNRESOLVED；保存缺失项及界，不将其归为失败或成功证据。

先封存全部32条、全部family及C128矩阵，再读已打开TRAIN32 roles。汇总
target是否存在CERTIFIED region、是否至少存在未REFUTED region、所有wrong
候选是否也有CERTIFIED region，以及target是唯一已证实候选的query数。
后一个数量仍不能排除UNRESOLVED wrong，必须另存wrong可能性并区分严格
唯一性。H0单独计数，保留全部32分母和原V8成功/失败分层。

这是标签关联后的固定参数、完整family可区分性诊断，不是新selector的
accuracy、rescue、性能上界或自然GO。多个candidate可能各有可区分区域，
不能把“目标有一个证书”冒充“无需目标信息就选中了目标”。

## 如何使用结果

仅在完整family中确有足够区分证据、且错误候选的同类证据受到约束时，才
有依据另外设计目标无关的区域比较机制。达到26条证书也只是后继设计依据，
不是训练授权自动触发或GO；26来自保留22/32强对照所需的净增4参考目标。
不足26只约束当前冻结head/当前family，不证明换参数、特征或所有HYP不可能。
如果全部family仍缺少目标优势，应追查assignment/聚合/身份信息，而非继续
仅从旧fixed-pair改善预期识别突破。任何后继都另行冻结且保留原全部对照。

## 工程与核验

隔离worker、authority及输出根。4CPU、64GiB、59分钟、torch单线程、0GPU；
逐query原子封存，计时3000秒后不启动新query，超时/信号只报INCOMPLETE，
不自动重排队或重提。完成后同一job中另起进程核验保存的源哈希、完整轴、
原交叉矩阵逐位一致记录、Fraction公式、缺失状态、postjoin计数。明确
artifact验证不等于原始残差独立重算。

提交前用合成数据检查完整/全缺失、已有group合并及新group的补全界、
小型枚举补全、严格不等号/平局、已知反例、UNKNOWN不冒充观测、来源轴和
不可变产物篡改拒绝。验证实际空环境启动块，使用脚本文件，不使用嵌套
`bash -c`启动测试。按用户要求，只做一次提交激活核查，随后退出不监控。
