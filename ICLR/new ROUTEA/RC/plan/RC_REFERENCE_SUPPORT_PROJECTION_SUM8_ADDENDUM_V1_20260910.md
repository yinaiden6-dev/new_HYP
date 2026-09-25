# 执行前补充：用户提出的对角线投影联合证据

主计划：RC_REFERENCE_SUPPORT_MAXMIN_READOUT_V1_20260910.md。
本补充在全量支持缓存及新读出自然执行前提出，不改变缓存或原三个
arm，不替换原主比较；增加一个明确、同容量的次比较SUM8。

## 来源及有限核对

用户观察TRAIN pilot图后提出：“对着斜线做垂线，垂足距离原点最远的点，
大概率是target？”图中x为原精确归一化支持margin，y为优化支持的
精确可达下界。投影到y=x的垂足为((x+y)/2,(x+y)/2)，沿右上方向的
有符号坐标为(x+y)/sqrt(2)。距离本身为abs(x+y)/sqrt(2)，会把左下
方向也计作强值；本次明确测试右上方向，即不取绝对值的x+y。

仅对已打开的4条TRAIN、全部512项作数值核对：目标按x排名为3/1/1/1，
按y为7/1/2/1，按x+y为6/1/1/1。左下图NDV2-007-P01相加胜过单独y，
但相加与单独x都选对相同3条；没有新增识别收益的证明。左上图仍有
5个错误候选的组合值严格更高。数据属于3个TRAIN身份/组，不是新EVAL。

## 唯一新增定义

模型实际输入沿用主计划中原FP64 J与T=float(exact L)，定义：

    U_g = binary64_add(J_g, T_g)
    SUM8第七列 = symmetric(U_challenger, U_winner)。

这不是对exact绘图横坐标作无舍入替换；J仍为已冻结原FP64/clamped
池化margin，T仍为按唯一预定规则就近舍入的L。保存U的hex和附加列
hash。正负U均保留，不取绝对值，不截零，不调整角度或权重。

SUM8与FIXED8、MAXMIN8均为8参数，保留原六列、原检索监督/优化顺序、
所有127个challenger、原HOLD/SWITCH规则及tie-break。C_BIND转移整份
证据，EXTRA_BIND只转移新增列；共享一个SUM8训练参数，不训练控制头。

## 比较及边界

原主比较MAXMIN8 vs ORIGINAL7保持；SUM8 vs ORIGINAL7/FIXED8/MAXMIN8
列为用户提出的TRAIN启发次比较。分别报告MAXMIN8与SUM8是否保留原28
并新增至少1个；不按EVAL结果选择一个arm冒充唯一预设成功结果。

四头共同在原TRAIN32+PAIR64拟合，所有FULL64预测同一阶段封存后才
读取EVAL角色及旧结果；独立进程完整重训四头。旧ORIGINAL7/FIXED8
精确复现门不变。无额外参数搜索、阈值搜索、投影方向搜索或EVAL子集。
结果若有增益，只按已打开内部开发证据报告。用户关于“retrieval-only
用于身份判断”的澄清保持：支持无需单独分类，由检索监督学习共享
判定规则。无额外空间监督、identity专属分类参数或ownership要求。

主计划的三头数量由本补充更新为四头，读出资源仍为8CPU/4G/15分钟。
完整缓存仍是8,320项；本补充不新增LP、内容编码或RoMa调用。读出
authority同时绑定主计划和本补充；已完成的旧预检记录保留，不覆盖。
