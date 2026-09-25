# V8：真实配对的残差证据与有界affinity

2026-09-08。一次显式科学后继，尚未拟合；不重命名V7的失败。
仅opened TRAIN32 development。正式392、V、HOLD/SWITCH与ownership保持封闭。

## 已有依据与执行前置

V7已独立重放确认REAL7/32、AP11/32、QUERY_ONLY14/32，对旧强对照net−15。
源代码按合同执行，失败不能归为未完成训练或错误checkpoint。

两处设计问题需要分别处理：
1. V7先对每个channel取不同atom的最大残差，再相加。这可能构造没有任何
   单一真实配对支持的残差向量。正权重下，它严格上界真实配对的最坏能量。
2. 三个target-H0在原group-balanced margin −30.03444302726264中贡献
   −29.88299053306123；其余29条按同一原分母仍贡献−0.15145249420141416。
   固定−256结构sentinel支配了跨query数值，但这不改变原7条成功的失败结论。

Job5137441用原V7最终head、全部原candidate与原selected H，完整诊断上述
额外惩罚；不重选coherent winner。先验收完整产物与固定pair证据，再记录
是否支持本后继的开发拟合。诊断crossing不是新模型accuracy/rescue或GO。
这不是参数/阈值/温度/区域尺度扫描；V8只包含以下两个事先写明的改变。

## 改变一：先整合真实配对，再取最坏项

沿用原V7的全部source、FP64单位q/r tokens、RoMa source对应、原cycle-closed
maximal connected family和256个零初始化参数；不改C128或候选生成。
令e_i,d=(q_i,d−r_j(i),d)^2，w=128 softmax(theta)，分成w_mu与w_M两个128维块。

`a_i=sum_d w_mu,d e_i,d`；`b_i=sum_d w_M,d e_i,d`。
`E(H)=mean_{unique reference cell j} max_{i:j(i)=j} a_i + max_i b_i`。

每个max选择一个真实atom，平局选最小source query index；梯度沿该实际
atom流动，不把不同channels的最坏项拼成虚构配对。保存这些witness。

对固定H与固定w，E_V8 <= E_V7，这是max/sum次序不等式；不是排名改进保证。
在theta=0时，线性审计值1−E/2等于“每ref cell最低cosine的均值”与“全H最低
cosine”的平均，范围[-1,1]。审计值仅用于解释，不是本方法部署score。
worst-atom仍可能惩罚视角/照明等nuisance；不能将大残差等同语义冲突。

## 改变二：明确候选证据的分数域

唯一候选affinity为 `K(H)=exp(−E(H)/2)`，固定单位尺度，无可学习/扫描的temperature。
非空K∈(0,1]；candidate取其完整family的max K，canonical同分规则不变。
空family的H0取这个分数域底值0；H1选择仍与score阈值无关。

它是Gaussian-inspired affinity，不声称概率校准或区域上的Mercer kernel。
缺少target region仍然是保守失败，不是UNKNOWN自动变成正确匹配。

对同一能量与固定参数，K与线性1−E/2的candidate/H排序相同，H0都位于非空
候选之下；它不能自己把原错误排名翻正。但它改变跨query/group margins、
C_BIND margin-drop的比较及训练梯度。因此这是额外明确的score/loss定义，
不是“纯数值修复”或旧V7的新GO。E0保留C_BIND drop符号可变化的反例。
分数[0,1]下C128单target CE有约3.865365的理论最低值，不因loss不趋零调温度。

## 对照、训练与验证不放宽

REAL、ALLPATCH、TRUE_QUERY_ONLY各256参数，统一使用上述coherent affinity。
ALLPATCH在原全部valid atoms上分组；P_COORD逐位不变。
QUERY_ONLY先仅从query选一个四格连通H，variation先按w_mu/w_M整合成每atom
scalar，再以mean+max ENERGY选H（不是用affinity选低变化区）。随后遍历完整
有效reference token轴作固定per-atom appearance argmin，并用同一coherent
K评分。C_BIND换真实donor；P_COORD在CURRENT query网格重新选区，经inverse
置换读source tokens。C128的query permutation必须共享。没有RoMa-missing弱对照。

保留原V6的冻结共同轴QUERY_ONLY强对照22/32，不重训、不删除。
三个新arm共用已冻结29条结构可训练query；全部32评估。原V6的512位置顺序、
seed17、AdamW lr.03/betas(.9,.999)/eps1e-8/wd0均保持。3条masked query的位置
仍执行零当前梯度AdamW步骤并记录动量，不能称为无更新。

用本方法唯一K执行train/checkpoint/resume/deploy；沿用relative full-C128 CE、
rank、C_BIND、P_COORD损失，AP坐标项常数。fresh/resume512及完整AdamW在256/512
逐字节一致，全部原输入、候选affinity、region vectors、Q selector vectors和
physical witnesses都需独立重放。不能用原V7参数直接当作V8训练完成。

冻结gate仍为：coverage>=26/32；rank>=21/32；group margin>0；相对新AP/Q及
旧V6强对照分别paired net>=4且rescue>break；C_BIND/P_COORD各>=21下降且
group drop>0。保留全部同时失败。额外旧强对照意味着至少26/32的真实排名成功。
P_COORD全部REAL family为空造成margin-drop退化，继续显式披露，不当独立
identity-specific spatial causality证据。只有完整development GO及独立验证
后，才允许准备严格未打开的正式面板；正式门与来源另需绑定V8，不能复用
只接受V7 GO的authority或冒称正式像素端到端已验证。

不添加backbone，不读/改/依赖D1-MI或GroZi，不触碰其它运行任务。
只写隔离V8根。提交前记录Job IDs及单writer；按用户要求不循环监控任务。

独立固定支持诊断验收：results/rc_v7_residual_envelope_artifact_validation_v1/result.json。
该验收重算已保存标量、原分数向量、anchors和统计；没有假称从未保存的raw
residual数组独立重算全部energy/argmax。原运行的source与结构证书保留。
