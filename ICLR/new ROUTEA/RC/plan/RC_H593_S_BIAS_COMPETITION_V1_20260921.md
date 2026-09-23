# H593：S 偏置与候选竞争的嵌套留组实验

日期：2026-09-21。用户已授权开始实验。唯一新分支：`rc_h593_s_bias_competition_v1`。

## 问题与动机

同一 H593 / grouped OOF5 / 自然 RAW C128，原 COST1 为481/593，上一轮 S_NET1 为485/593（对 COST1 8救4损），CE为486，事后 ZERO_S 为487。S_NET1 是开发信号，组置信区间包含0，不能称为稳定新最优。

S_NET1 仅通过 `m + alpha * S_plus` 放行原 HOLD，其中 m 是原 COST1 最高 challenger logit。当前验证两点：独立的 HOLD 放行偏置能否改善校准；加入最高与次高 challenger 的分差 d，能否区分单靠 m、S 无法区分的错误。

已有开发数据指出同折 OUTCOME-0794 的 m 和 S 都高于 OUTCOME-0389，非负单调 S 与常数偏置不能同时保住前者、救回后者；两者 d 不同。这仅用于提出本实验，不用于选择阈值或拟合新参数。不能据此声称分差一定可泛化。

对原归一化 S 线性输入加常数，与原全局偏置代数冗余。本轮学习的是原 COST1 权重冻结后的 **HOLD 子集放行偏置**，并用仅偏置对照区分它与新条件证据的作用。

## 固定模型和五臂

对原127个 challenger logits 取第一个最大值 j，定义 `m=z[j]`、`S_plus=max(X[j,S],0)`，`d=z[j]-max(z[k], k!=j)`。全程FP64、原候选顺序、原C128不变。

放行分数使用显式标量运算顺序：

`g = ((m + (alpha * S_plus)) + (beta * d)) + bias`。

alpha、beta非负，bias自由；不使用 target 身份、人工图片类型、框或mask作为推理输入。

| 新臂 | 自由量 | 目的 |
|---|---|---|
| BIAS1 | bias | 只调 HOLD 放行门的对照 |
| S_FIXED_BIAS2 | bias；alpha固定为本折上轮S_NET1，beta=0 | 严格检验“只给已有S头加偏置” |
| S_BIAS2 | alpha,bias；beta=0 | S方向和偏置共同拟合 |
| GAP_BIAS2 | beta,bias；alpha=0 | 竞争分差本身的作用 |
| S_GAP3 | alpha,beta,bias | 预先指定的主臂 |

主要配对比较：S_GAP3 对 S_BIAS2。S_GAP3 对 GAP_BIAS2检验额外S信息；S_FIXED_BIAS2 对 S_NET1检验纯偏置收益；与COST1/CE/ZERO_S比较保留完整谱系，不事后挑最优臂改写主臂。

所有新臂只能作用于原 m<=0 的 HOLD。只有新 g>0 时才用g替换原最高challenger logit并SWITCH。否则127个分数完整保留。原 m>0 的SWITCH所有分数逐位保留。原排序的第一名身份、非第一名126项始终保留；因此本轮无法修复原最高challenger本身选错的情况。

## 固定训练规则

复用已封存的上轮20个内层 COST1 头与内层留组预测。本轮每个外层只读取本外层 parent payload；不读取其余外层 payload，因为它们的 calibration 可能含本折外层标签。每个内层模型训练于另外3折，内层留出覆盖外层TRAIN。外层fit不读 curator、parent result 或任何外层正确性。

对原HOLD记录，delta = I(top challenger正确) - I(RAW正确)。delta=+1为救回、-1为损失、0为中性。利用检索身份标签，无额外空间监督。

S_BIAS2、GAP_BIAS2、S_GAP3与BIAS1采用相同两步规则：

1. 固定m为offset，仅对delta非零的内层HOLD最小化平均 `log(1+exp(-delta*g)) + .001/2 * ||theta||^2`；正则包括自由bias。alpha、beta约束非负，全部零初始化，SciPy L-BFGS-B，maxiter=2000、ftol=1e-12、gtol=1e-8。要求success、目标不升、独立重算的投影梯度无穷范数<=1e-5。
2. 固定已学斜率，对全部内层HOLD（包括delta=0）精确枚举binary64偏置的所有决策区间，最大化救回数减损失数。并列优先改动更少HOLD、再最小|bias|、再较小bias；无正净增时显式禁用。S_FIXED_BIAS2直接使用此步骤。

这是“斜率用凸 surrogate 学习，偏置按经验净增校准”，**不是三参数经验准确率的联合全局最优**。相较上轮直接对一个alpha精确最大化净增，斜率学习目标发生变化；S_FIXED_BIAS2专门避免此混淆。中性delta=0在旧兼容字段名 `both_wrong` 中计数，数学解释为neutral。

偏置精确优化证书：有限u的第一正分数偏置是 `nextafter(-u,+inf)`，前一可表示数仍不为正；相同阈值的事件同时计数。枚举区间内最接近0的可行数，保存每项阈值、区间计数、参数hex和哈希。独立程序另外枚举所有阈值/前驱/0进行核验，不调用训练器的门或校准实现。

## 数据与评测

沿用H593的593张query、68身份、64组件、五折分组。候选是冻结的自然RAW C128；23张target不在候选内仍留在分母。每折检验内外层identity/component/image不交叠。仅汇总阶段在五折预测通过双验证并封存后读取外层标签。

报告正确数、R@1分子/593、MRR、SWITCH数、逐折正确数，对RAW/COST1及预先指定比较的rescue/break/changed，64组件等权差及10000次固定种子20260920 bootstrap95%。bootstrap为描述性开发不确定性，不将跨臂选最优的区间当作独立确认。保留损失数，不强制零损失门。

工程完成与科学成功分开：五折及汇总双验证通过才称完整实验；主臂有增益也不能自动宣称稳定突破或外部GO。H593已反复打开、方法设计使用了历史结果；嵌套拟合隔离不能消除历史方法选择。

## 执行与文件

- 基线：`results/rc_h593_s_net_hold_optimization_v1`，authority SHA256 `44dcf7e919ff62f63c06c444149c4f088ac9a05e61aa2b4255bb7a92474ddde2`。
- 本轮结果：`results/rc_h593_s_bias_competition_v1`。
- 代码、计划、launcher、测试、独立验证器及数据源由新authority固定SHA256；冻结后不原地修改。
- synthetic preflight包括梯度、边界、阈值并列、中性项、disabled及127分数保护；自然数据训练仅在Slurm执行。
- accelerated五折数组最大并行5，每项申请10分钟、8CPU、16G、1GPU（计算实际使用CPU）；afterok依赖汇总10分钟。超时前540秒timeout，不requeue。
- 保存提交回执并验证实际partition、time limit、依赖及Slurm spool与冻结launcher字节一致。
- 不启动encoder/RoMa，不重训20个内层基头，不改旧实验、正式部署、D1-MI/formal392或外部数据。
