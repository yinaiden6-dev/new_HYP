# 联合校准是否需要绝对证据尺度

用户已授权继续retrieval-only试验至北京时间2026-09-11 24:00（UTC
2026-09-11 16:00），覆盖此前“最后一次后停止”的安排。旧结果、原
冻结协议和旧H/P-only状态不变。本试验不要求superregion、细粒度绑定
或ownership作为解释，不添加空间监督、SAM、encoder或RoMa forward。

## 一个具体且尚未验证的问题

原NATIVE7已含S=M*L及M、L等相对比较，不能把简单添加M*L称为新机制。
其五项证据比较使用(a-b)/(|a|+|b|+epsilon)，主要保留相对比例。
忽略epsilon时，两端相同的正尺度变化不可见；保留原epsilon时通常
仅产生数值级差异。绝对尺度可能是需要保留的置信信息，也可能只是
不应建模的干扰变量。本次只检验它是否提高识别，不预设它是失败根因。

## 三个固定head

只使用原C_PAIRED soft visibility × full-reference image-token
weighted MaxSim标量，原RAW C128和127challenger action。原NATIVE7
的六个特征保持原样，三模型如下：

- NATIVE7：原六特征+bias，重新按原协议拟合并要求参数/预测回归。
- ABS12（唯一新方法/主模型）：原六特征，加五个未作pairwise对称比值
  归一化的差：deltaS、deltaM、deltaL、delta(S-Sq)、delta(S-Sr)。
  L严格沿用S/max(M,1e-12)。共11特征+bias。
- REL12（同参数数量、只使用原信息的对照）：原六特征，加原五项
  非RAW相对特征各自的x*abs(x)。共11特征+bias，不能获得绝对尺度。

新增五列分别除以其TRAIN-only RMS；RMS从原PAIR64及FULL TRAIN32×127
按固定顺序拼接的4128行求得，不作均值中心化。严格零RMS列用1作除数，
保留并报告退化；不删列或换特征。两扩展模型都采用该固定处理，
原六列不重标定。报告design rank，参数数相同不冒称有效容量必然相同。

不挑一种更有利的head冒充主方法。ABS12若只优于NATIVE7而未优于REL12，
只能说明当前扩展表达可能有用，不能把收益专归绝对尺度信息。

## 数据、训练和控制

全部沿用原RAW FULL64（TRAIN32/EVAL32）和历史PAIR64。查询、身份、
supergroup切分复用已验证qualification，不增加训练样本，不用EVAL
选择RMS、参数、特征或checkpoint。不涉及D1-MI、GroZi、正式392。

复用原train_head完整损失与算术：seed17、FP64 Linear全零初始化、
AdamW(lr=.03,weight_decay=.001,betas=.9/.999,eps=1e-8)，2000固定更新；
PAIR原序，FULL TRAIN按execution升序，每步完整batch。PAIR标签0权4、
标签1权1的BCE；FULL base正确时4softplus(maxwrong)，base错误时
softplus(-target)+4softplus(maxother)；总loss为PAIR与32-query均值之和。
不新增loss、earlystop、温度、阈值、seed或学习率扫描。

EVAL只计算REAL与原完整evidence shift64 C_BIND。两者保留同一RAW轴；
C_BIND先换完整candidate evidence，再重算相对和绝对特征，不能只换
最终分数。三head的参数、TRAIN-only RMS、完整EVAL预测全部封存之后
再读取EVAL labels和旧outcomes。训练期间可读取原TRAIN/PAIR监督。

NATIVE7原参数与REAL/C_BIND预测必须回归，否则工程ABORT，不解释新
模型成绩。验证独立从原scalars重建特征/RMS、复算训练与全部预测，
并用原public action规则汇总；不要以工程PASS宣布科学GO。

## 报告与后续判断

EVAL32当前原NATIVE7为28/32、RAW25/32；FROZEN_C旧27与opened90旧69
保持独立lineage。本试验仅在同一已打开EVAL32中比较三head，报告相对
NATIVE7、REL12、RAW的逐query rescue/break/net、MRR、groups及C_BIND
保留；TRAIN32与PAIR64另列，不合并分母。

内部可继续候选要求ABS12比NATIVE7有严格正净增且不破坏原NATIVE7
正确样本；绝对尺度特有解释还要求ABS12比REL12有严格正净增，并失去
其C_BIND下的新增救回。若未满足，分别说明是新模型无收益、容量对照
同样有效、或候选绑定未解释收益。不是旧P-only门或外部确认GO，不
凭一个已打开32-query结果部署或宣称普遍规律。

执行前检查时间截止；训练、验证只用CPU。若提交Slurm，仅作一次激活
核对，不监控队列。程序和结果append-only，旧模型与当前部署不变。
