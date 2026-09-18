# new HYP：reference权重的选点效应与条件校准

本页把已验证的旧模型机制、已定位的一例失败与当前待检验的修正连接起来。当前最优仍是ORIGINAL7/NATIVE7：RAW自然C128、全部127 challenger、固定HOLD/SWITCH；旧EVAL32为25→28（3救0损），新EVAL128为88→99（12救1损）。两套样本分别报告，均属于已打开的内部评价。本文不把正在运行的TRAIN OOF试验写成新性能结果。

## 可精确区分的两个作用

固定某一query与candidate reference的完整token相似度矩阵c、原query权重u与reference权重v。令

\[
Z=\max(\sum_i u_i,\epsilon),\quad
j_v(i)=\arg\max_j v_jc_{ij},\quad a_i=\max_jc_{ij}.
\]

用同一个query支持定义三个实数量：

\[
F=\frac{\sum_i u_i a_i}{Z},\qquad
A=\frac{\sum_i u_i c_{i,j_v(i)}}{Z},\qquad
L_* =\frac{\sum_i u_i v_{j_v(i)}c_{i,j_v(i)}}{Z}.
\]

于是

\[
F-L_* = \underbrace{F-A}_{D:\text{选点造成的自由内容损失}}
       +\underbrace{A-L_*}_{\text{同一选点的幅度变化}}.
\]

在u非负、Z正、两个argmax比较同一完整有效token轴且数值有限的条件下，每项a_i不小于c_i,jv，所以D≥0。D=0仅说明有权重的query token没有因reference加权而选到更低的cosine；它不保证身份正确，也不保证权重幅度合理。generic文字或重复版式同样可以产生较高F、较小D。

第二项不保证非负，即便v在[0,1]：负cosine乘小权重会向0移动。例c=(-0.4,-0.8)、v=(1,0.1)、u=1，则F=-0.4、A=-0.8、L*=-0.08，D=0.4，而A-L*=-0.72。因此D不能叫总加权评分损失，更不能直接解释为身份错误概率。

旧头读取的ell=S/max(M,epsilon_M)也不能无条件等同于L*。在实数下S=M L*，故ell=[M/max(M,epsilon_M)]L*；只有M不低于该下限时才还原L*。生产代码继续保留原FP64乘法、归约与除法次序；上式是数学解释，不授权代数改写checkpoint或声称逐bit恒等。新缓存F/A也固定先加权求和、后除clamp分母。

## 它解释了什么、还没解释什么

新EVAL128的0337中，保留各自query权重时，target自由内容F约0.707504，高于wrong约0.594628。由v×cosine重新选点后，裸内容A反转为target约0.443245、wrong约0.475781；再乘所选reference权重，错误优势进一步增大。这里能直接定位一个计算失败环节：权重先改变读入内容，再影响其幅度。数值来自已打开两reference诊断，其归约实现与新缓存可有末位差异，不输入当前模型拟合。

这不是删除geometry的理由。原固定模型在新EVAL128的12次救回，分别去掉M或ell列都会全部消失；而已有FREE8统一补自由内容的重训旧EVAL仅27/32，低于原28/32。冻结原头加单一全局自由内容系数的精确后验容量证书也无法同时保0220、修0337。该证书隔离，未用于当前系数、阈值或折分。

所以需要验证的有限命题是：同一共享模型能否从当前候选与RAW winner的证据冲突中，学习何时补偿自由内容，而不是把D较大直接判为wrong。

## 当前唯一新增性能试验

TRAIN128、32个query正例身份/组，四折每折留出8组。每折原7参数基头排除留出组后重训，再冻结；同一组隔离下比较统一1参数补偿与条件4参数补偿。条件项读取RAW差距、winner与challenger的D差、reference平均权重对比；补偿作用于同一自由内容对比。两臂同初始化约定、2000步与保护损失，不读取EVAL标签。

只有条件臂完整OOF总正确数与等权组准确率都严格超过两个对照，且不增加相对RAW的break，才有资格另行冻结最终全TRAIN拟合与旧/新EVAL评估。条件臂即使过门，也不保证能提升原完整ec7；若不过门，只排除这一四参数配方的当前训练结果，不否定已有99/128的识别净增或所有条件模型。

这仍是retrieval-only任务训练：query/reference身份关系是监督，不新增框、mask、点、人工crop、SAM或空间teacher；冻结基础模型的既有预训练来源照实披露。new HYP的机制解释与性能改进并行推进，空间ownership不在本分支目标中。

来源：
- [原128图机制结果](REPORT_NEW_HYP_ORIGINAL7_EVAL128_MECHANISM_RESULT_V1_20260910.md)
- [0337选点反转](REPORT_NEW_HYP_0337_REFERENCE_WEIGHT_IDENTITY_REVERSAL_V1_20260910.md)
- [扩容及FREE8失败定位](REPORT_NEW_HYP_TRAIN128_RESULT_AND_FAILURE_MECHANISM_V1_20260910.md)
- [固定OOF计划](../plan/RC_TRAIN128_DISAGREEMENT_OOF4_V1_20260910.md)
- [已资格化TRAIN缓存](../results/rc_train128_disagreement_features_v1/result.json)
