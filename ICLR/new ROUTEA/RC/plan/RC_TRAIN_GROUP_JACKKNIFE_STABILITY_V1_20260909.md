# new HYP：TRAIN supergroup逐组删除稳定性复核

继承现有retrieval-only授权、截止北京时间2026-09-11 24:00，以及提交后
继续独立工作要求。本轮为训练敏感性分析，不是寻找更好的seed或训练
子集，不改变或替换目前原NATIVE7模型。

## 依据和固定问题

5138852已确认固定同协议四头EVAL32为JOINT4=26、PRODUCT5=27、
RESPONSE6=26、ORIGINAL7=28。完整模型胜过删减模型的方向为正，但
部分动作logit接近零。固定组效应已把直接贡献与重优化贡献分开。
这些少数纠错/保护是否高度依赖某个TRAIN group，尚未检验。

全部原FULL TRAIN32只含12个既有supergroup；PAIR64另有不相交的20组，
EVAL32为不相交11组。依原split qualification核验。按supergroup字符串
排序，依次只删除FULL TRAIN中的一个组，覆盖全部12组，无挑选。
PAIR64始终完整且原顺序保留。其余TRAIN按原execution顺序，EVAL完整
32条和C128不变。每组删除仅用于敏感性复核；不输出可选部署头或
把最佳删除组提升为主模型。

## 固定模型与训练

每个条件训练同一四头：JOINT4[0,2,3]、PRODUCT5[0,1,2,3]、
RESPONSE6[0,2,3,4,5]、ORIGINAL7[0..5]。

先做一个FULL_TRAIN基线，必须与5138852的四头参数、EVAL REAL/C_BIND
完整动作和全部预测逐bit一致。之后12个删组条件各训练四头。
总52次拟合，每次原FP64全零初始化、seed17、AdamW lr=.03、
weight_decay=.001、betas=.9/.999、eps=1e-8、2000更新。原PAIR loss
与剩余FULL TRAIN行上的mean sign loss，原函数不改。没有warm start、
早停、checkpoint选择、阈值/温度/seed扫描或新特征。

## 封存与完整输出

按预定顺序完成所有条件/四头参数和EVAL32 REAL/C_BIND完整127
challenger预测，再join EVAL标签。原TRAIN身份用于按既有group划分，
不以EVAL表现决定删哪组或选哪个头。全部条件都报告。

每个删组条件记录被删query、剩余TRAIN数量、四头准确数、paired救/损
和等权EVAL group差。四项factorial边保持与5138852一致。汇总各边的
正/零/负方向次数、净增min/median/max、各head正确数min/median/max、
对FULL_TRAIN同head的各query正确频率、HOLD/SWITCH一致率和最终
reference一致率（相同SWITCH也可能选择不同candidate）；重点病例也
不能替代全32条汇总。

所有删组稳定性频率以12个删除条件为分母，FULL_TRAIN基线单列，
不把它当第13次删组或独立重复样本。

新进程使用同一固定优化器重建并重训全部52次，参数与预测逐项复核，
用独立的指标计算复现全部报告。FULL_TRAIN原四头基线必须精确回归。

## 解释限制

PAIR的20个组始终未被扰动；FULL被删组大小不同，剩余行mean损失
相应重新加权也是本干预的一部分，不能声称是只移除一个等权观测。

这是既有TRAIN组影响的descriptive jackknife/sensitivity，不是12个
独立外部测试集，不生成未经资格的p值，也不证明对所有训练分布稳健。
若方向随删组翻转，需要收窄“稳定机制”的措辞；若方向一致，只能支持
当前内部固定输入/训练协议的稳定性。不能把某个删组得到的较高成绩
当作超过原28的新模型或在EVAL上挑选训练数据。

不删除EVAL样本，不进入D1-MI、GroZi、正式392、SAM、superregion或
ownership。没有新编码器/匹配器forward，监督仍只有检索身份和配对。

## 执行

单作业dev_cpuonly、8CPU、4G、59分钟；按执行时间检查能覆盖训练和
新进程完整重训，原每4头训练+验证约2分18秒，13条件约30分钟上限内
合理安排，实际59分钟为硬资源上限。截止UTC2026-09-11T16:00也生效。
提交后继续理论与证据整合，不反复查询调度器。
