# Cycle-closed connected region：先修复P生成语义

2026-09-08。状态：FROZEN_SINGLE_GENERATION_FEASIBILITY_TEST。
范围：唯一一次generation-only E0与opened TRAIN32 natural coverage gate。
不是完整P-only GO；不执行新head拟合、formal392、V、action或ownership。

## 依据和停止的问题

前轮固定九例可视化与连续RoMa轨迹证明：五个target选中的手/桌面支持，
四个atoms全部回不到原四格。旧legal只检查前向邻近、query四连通和至少
两个reference cells，cycle-quality作为soft特征，不能保证局部对应闭合。
同品牌错规格却可有非常准确的往返对应，所以可靠几何不等于身份正确。

本阶段只回答：纠正前者后，正确reference是否仍能自然生成足够多的
candidate-conditioned connected multi-patch regions？不可因覆盖不足放松规则。
旧严格双向visibility/reprojection coverage=0与AS1 synthetic-only状态保留。
这里使用RoMa当前原始连续返回和native query格子，不能冒称旧严格定义修复。

## 唯一新legal生成函数

输入仍是原始ProposalAtomBank，完整q和specific reference内容形成的既有
RoMa/ColNomic atoms。所有原atoms在输入中保留；不按P score、token norm、
c正负、rank、target、filename、softmax/sparsemax筛选。

对atom a，先把原RoMa source_reverse_query_xy的归一化坐标映射回原query
native cell：`floor((x+1)*gw/2), floor((y+1)*gh/2)`。约定[-1,1)半开域；
域外显式不闭合，不clamp。cycle资格要求其返回cell等于source_query_indices[a]。
这是一次固定的native-cell equality检验，不扫描像素误差阈值或置信阈值。
不强制一对一reference assignment。

只在原valid且cycle-qualified atoms上建图。边保留既有双图局部规则：
CURRENT query Manhattan距离1，CURRENT reference Chebyshev距离<=2。
枚举所有maximal connected components，至少4个query atoms且至少2个
distinct reference indices。每个component就是一个可重放的H1候选；输出
全部components，按canonical current query-index tuple稳定排序。没有top-K。
输入native网格有限，component数最多floor(valid_atom_count/4)，所以family有限。
这将原“固定4格seed”改为数据决定大小的连通region，但没有按科学结果选尺度。

H0仅在这个family为空时出现，reason为CYCLE_CLOSED_REGION_FAMILY_EMPTY。
这是structural H0，不是semantic no-match或身份absent判定；不使用score零点。
缺少semantic null/calibration的边界沿用已授权legal-family successor。

## 直接可证的有限性质

1. 每个输出region是query四连通且paired图连通的multi-patch集合。
2. 每个被纳入atom的原source roundtrip回到自身native query cell。
3. 对任意全部atoms未通过cycle检验的旧支持，该支持不可能作为本family的H1。
4. 成员资格不依赖head、target、候选rank/order、分数符号或identity参数。
5. REAL/C_BIND/P_COORD使用同一函数；训练/部署消费者未来只能使用该family。

这些是生成器结构性质，不证明视觉身份正确、真实物体mask或空间ownership。
不保证所有背景atoms都被消除；也不保证同品牌错误候选被淘汰。

## P_COORD和C_BIND

cycle检验使用source token的返回身份；P_COORD改变的是当前token-coordinate
association。source cycle资格在这种重标记下保持不变，CURRENT坐标的图连通
必须重算。不得用P_COORD里为几何重放设置的canonical reverse点伪造完美cycle，
也不得把source reverse和新start坐标错误相减制造额外干预。
C_BIND继承已验证before-RoMa/assignment donor bank；不挪动封存score来替代。
candidate reorder按key对齐应完全一致。

## 执行和gate

先完成独立synthetic E0：identity warp、collapsed return、域外/边界、单patch
拒绝、>=2reference cells、多个components、任意feature改变不影响生成、
序列化重放、P_COORD改变连通但不改source cycle资格。

然后完整TRAIN32×C128×{REAL,C_BIND,P_COORD}生成并写入无target的proposal
records，封存以后才打开既有opaque postjoin roles做coverage归约。
主要门是target自然存在且合法region coverage>=26/32。target缺失仍归
candidate recall；此输入TRAIN32的候选来源不变、不插target。
同时报告每个candidate组件数、大小、H0比例、C_BIND内容绑定、P_COORD和
重排一致性，明确这些未取代后续margin、matched allpatch/query-only门。

coverage不足：记录GENERATION_COVERAGE_NO_GO并停止本生成器，不放宽cycle
规则或region最小大小。coverage通过：仅允许设计/冻结共享selector和完整
P evidence优化合同，不能据此自动提交正式P0。

只写新的rc_cycle_closed_region_generation_v1程序/core/tests/results/authority。
源图、旧V3/GISC/V6结果和其它对话任务不变；protected数据完全不读。
采用accelerated、单GPU（本阶段CPU计算）、4CPU、32GiB、最多59分钟。

## 有界文献依据

- RoMa v2的precision只在covisible且误差<8pixels处训练，overlap是对应共同
  可见性而非身份真值：https://arxiv.org/html/2511.15706v1
- NCNet在匹配空间检查邻域一致性：https://arxiv.org/abs/1810.10510
- SuperGlue区分partial assignment与事后outlier rejection，dustbin需要真实
  unmatched监督：https://arxiv.org/html/1911.11763v2

本生成器是受这些原则启发的可检验工程假设，没有声称实现了论文算法或
发现了新颖性，更没有从这些论文推导本项目的自然GO。
