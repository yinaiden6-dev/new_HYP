# new HYP：理论定义、命题与当前证据说明

归档日期：2026-09-19。本文件把理论定义、五条形式化性质、决策目标及当前证据放在一起，完整历史原文另附于 `theory_sources/`，其SHA与仓库原件一致。

**当前名称为 new HYP — Reference-conditioned Joint Evidence Calibration for Fine-grained Retrieval。** 研究对象是候选reference条件下的身份判断；旧空间/superregion Reference HYP的失败保留，ownership不属于当前已经完成的主张。实现仍须披露RoMa、ColNomic和共享七参数头，不能把抽象命名等同于跨编码器证明。

任务训练只用query/reference身份或正负检索关系，冻结基础模型的预训练另行披露。五条性质是带条件的结构、信息或代数结论，不是普遍识别保证，也不把基础代数或共享参数可扩展性宣称为首次发现。

以下“定义与性质”完整摘入2026-09-10 V2第1–6节；其中实验语句保留其历史时点。随后列出已完成的新证据，避免把旧文中的“当前”“尚未完成”当成2026-09-19状态。

## 1. 定义与实际实现

给定query q、自然候选集C、reference R_g，质量通道G产生候选条件的
匹配可靠性，内容通道C_content产生内容相容性。联合证据经共享校准
函数A_theta与基础检索先验B决定候选。theta不含identity专属参数。

当前实现的状态可完整记录为E_g=(M_g,S_g,S_g^Q,S_g^R)，其中
L_g=S_g/max(M_g,epsilon)。在数学上忽略浮点归约差异且分母高于数值保护下限时：

    local_i = max_j [v_j * c_ij]
    M = sqrt(mean(u) * mean(v))
    L = sum_i u_i*local_i / sum_i u_i
    S = M*L

实际实现采用冻结的完整FP64次序，不能用代数等价擅自替换。RoMa和
ColNomic分别是当前G与内容通道的实现，方法部分必须披露。new HYP
本身不要求二维连通区域，也不声称具有独立的新假设生成网络。

任务训练监督只有query-reference身份/正负检索关系。冻结基础模型的
预训练来源单独披露；不使用本任务mask、box、point、人工crop或分割
teacher。HOLD表示保留RAW winner，不表示reference不存在。

## 2. 性质一：reference登记与任务参数学习可分离

假设G、内容编码器、校准函数均共享参数，reference身份只作为资源
索引，不进入identity专属可训练参数。则登记一个新的reference、
计算其表示并纳入候选后，原theta仍可直接用于评分。

证明：将新reference代入同一个共享函数即可，参数维度和定义不变。
这是结构上的可执行性，不保证新reference会被召回、正确排名或在
任意遮挡下可辨认，也不是本文独创的数学定律。它给出“外部reference
记忆可扩展”的技术基础，不能直接等同于普遍的过目不忘保证。

## 3. 性质二：丢失必要统计量的接口不能普遍恢复原评分

设完整状态E、原评分f(E)、压缩接口T(E)。若存在E1/E2满足
T(E1)=T(E2)但f(E1)不等于f(E2)，则不存在只读取T的函数能对所有E
重建原f。否则同一个输入值须输出两个不同值，矛盾。

具体构造：一个query token、两个reference tokens，c=(1,0)，
u=(1)，v=(1,a)，0<=a<=1。若接口只保存获胜的第一个reference位置，
它不随a改变；但L=1，M=sqrt((1+a)/2)，S=M随a改变。因此仅保存那个
局部匹配不充分。该构造证明评分可恢复性的问题，不证明预测一定变差。

当前项目已发现旧P压缩缺少完整reference质量统计，并通过无损接口
恢复了原标量和决策。恢复统计量不自动保证new HYP解释或识别增益。

## 4. 性质三：相对比较对共同尺度近似不敏感

令D_e(a,b)=(a-b)/(|a|+|b|+e)，k>0，A=|a|+|b|>0，则

    |D_e(ka,kb)-D_e(a,b)|
      = |a-b|*e*|1-1/k| / ((A+e)*(A+e/k)).

e=0时严格尺度不变；e>0时误差大小由上式给出，A很小时不能声称可忽略。
这解释了原相对特征弱化绝对尺度的方式，不说明绝对尺度必然有用。
实际ABS12试验将绝对差补回后EVAL26，原NATIVE7为28；同参数REL12也26。
因此“额外信息存在”与“该信息提高泛化”必须分开。

## 5. 性质四：正确action与所有错误logit为负不是同一条件

固定RAW winner w，定义z_w=0、其它候选z_g为旧action logit。排除
依赖精确平局的情况，正确目标t为w时需max_wrong z<0；t不为w时需
z_t>0且z_t>max_wrong z。它不要求每个错误logit都小于0。

例如z_t=2、z_wrong=1时action正确，但错误绝对压负条件未满足。
因此两种训练代理目标不同。当前线性头下，上述严格条件都是线性
不等式；共享解是否存在可用可行性证书验证。

项目已用原缓存浮点特征的精确有理数表达核验12套TRAIN约束系统。
这些证书限定了所测试特征与线性头的训练能力，不是EVAL结果、所有
模型的上界或32/32资格门。随后固定NATIVE7的2×2目标对照中，原方案
EVAL28，三种改法均27。因此“目标条件不同”没有自动带来更好识别。

## 6. 性质五：乘积对比改变线性读出的函数表达范围

在正M、正L、S=M L、无epsilon/floor的理想实数条件下，令
x=(M_c−M_w)/(M_c+M_w)，y=(L_c−L_w)/(L_c+L_w)，可得

    dS=(S_c−S_w)/(S_c+S_w)=(x+y)/(1+xy).

x、y落在(-1,1)，分母为正。该函数在一般开域上不是x、y的仿射组合：
原点和两条坐标轴迫使仿射式为x+y，而x=y≠0即给出反例。因此在相同
底层质量/内容信息上，[RAW,x,y,dS]的线性评分函数类严格包含
[RAW,x,y]的线性评分函数类。这个结论针对函数表达，不是训练成功率
或泛化保证，也不是本项目新创的数学定律。

保留e>0时，设A=M_c+M_w、B=L_c+L_w、
x=(M_c−M_w)/(A+e)、y=(L_c−L_w)/(B+e)，则实数恒等式为

    dS=[A*y*(B+e)+B*x*(A+e)]
       /[A*B+x*y*(A+e)*(B+e)+2e].

这时精确关系仍含A/B，不能无条件称dS只由两个归一化对比恢复。实际
L由S/max(M,e)得到；触发floor、非正内容或浮点舍入时须另行分析。
生产路径继续直接读取原FP64列，公式只解释该列的函数作用。

原PAIR64和FULL TRAIN32的4128次比较全部M/L/S>0且无M floor，含e
公式对原dS最大残差4.44e-16，独立重执行一致。dS对简单x+y最大差
0.93939表明二者数值不同；这个数不是准确率增益或因果贡献。
完整推导、可复现程序及源绑定见
reports/NEW_HYP_PRODUCT_CONTRAST_MATHEMATICAL_REVIEW_V1_20260909.md。


## 统一对象与任务

new HYP中的假说是“候选reference g解释query的身份”，不是必须连通的空间区域。reference组织匹配可靠性、内容相容性及竞争证据；空间支持只是证据组织的一种实现。RoMa/ColNomic是当前实现，论文必须披露，不能把去掉模型名当通用性证明。

令w为RAW答案，Z为推理时全部合法候选及其可见证据。切换到g相对保留w的条件价值为

    Delta_g(Z) = P(Y=g | Z) - P(Y=w | Z).

HOLD的相对价值为0，理想决策取最大正Delta，否则保持w。任意固定系统相对RAW的准确率增量，恰等于rescue概率减去loss概率。query-reference身份标签Y已足以监督这一任务，不必增加框、mask、匹配正确性或ownership标注。

这是决策目标的定义；当前七参数logit并未被证明是Delta或身份后验的校准估计。旧SIGN损失是代理目标，不能把定义当成其一致性保证。HOLD保留RAW答案，也不等于未知类拒识。


## 当前证据与理论的对应（2026-09-19）

| 理论层次 | 已完成的对应证据 | 边界 |
|---|---|---|
| 共享reference条件证据与校准 | 同一full-H593固定COST1头，GroZi480为RAW321→355，34救0损；ISIC537为466→500，34救0损 | GroZi为正式外部确认；ISIC为已打开队列的探索。不是任意新身份保证 |
| 信息接口是否保留原评分 | 缺失完整reference质量统计的接口不能普遍重建；补齐后可无损恢复原评分 | 恢复原评分不自动提升识别，也不证明旧空间P成立 |
| 动作目标与错误绝对压负的差异 | 同H593五折、同七参数，COST4 440→COST1 481，44救3损；CE486 | 训练成本单因素有净增，CE相对COST1优势的组区间跨0；不宣称普适最优成本 |
| 实际参数学习的价值 | 同H593自然C128，手填等权／平衡均459，COST1 481、CE486 | 对所测两种固定规则成立；COST4仅440，不能说一切训练版都优于免训练 |
| 高基线保持回归 | processed128：RAW121、COST1 122、CE123；保住原121正确 | 合成处理回归；净增区间包含0，不是外部GO或普遍零损失定理 |
| 候选绑定与空间所有权 | 已有整包候选证据错绑对照；processed128中COST1 122→92、CE123→74 | 支持候选级联合绑定；不证明token对应、mask或ownership |

旧EVAL32的25→28和旧EVAL128的88→99来自固定ORIGINAL7；difficult90的61→69来自另一FROZEN_C头。H593结果是五折OOF，外部和processed128使用冻结full-H593头。各面板和头不混成一条样本量学习曲线。

RPC按2026-09-18决定仅保留参考图不足诊断。附带2026-09-15旧范围文档中的RPC“外部确认”是历史表述，已被当前范围修订覆盖。完整过目不忘系统、未知拒识、持续注册后的旧身份保持、多物体ownership及新编码器训练仍是未来方向。

## 完整原文入口

以下文件按原字节保存。V1/V2/V3记录不同阶段；V3是工作假说及可证伪预测，不因收入本包就升级成普适理论定理。原文内部指向其它仓库证据的链接仍可能需要原工作区。

| 原文 | 日期版本与用途 | 包内文件 |
| --- | --- | --- |
| new HYP：定义、可证明性质与当前证据 | NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V1_20260909 | [NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V1_20260909.md](theory_sources/NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V1_20260909.md) |
| new HYP：定义、可证明性质与扩大样本证据（V2） | NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V2_20260910 | [NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V2_20260910.md](theory_sources/NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V2_20260910.md) |
| new HYP：统一的工作假说、已有证据与下一次可证伪预测 | NEW_HYP_UNIFIED_DECISION_HYPOTHESIS_V3_20260911 | [NEW_HYP_UNIFIED_DECISION_HYPOTHESIS_V3_20260911.md](theory_sources/NEW_HYP_UNIFIED_DECISION_HYPOTHESIS_V3_20260911.md) |
| new HYP：乘积对比项的数学与实验范围复核 | NEW_HYP_PRODUCT_CONTRAST_MATHEMATICAL_REVIEW_V1_20260909 | [NEW_HYP_PRODUCT_CONTRAST_MATHEMATICAL_REVIEW_V1_20260909.md](theory_sources/NEW_HYP_PRODUCT_CONTRAST_MATHEMATICAL_REVIEW_V1_20260909.md) |
| new HYP：当前校准机制的联合证据收口 | REPORT_NEW_HYP_CALIBRATION_MECHANISM_SYNTHESIS_V1_20260909 | [REPORT_NEW_HYP_CALIBRATION_MECHANISM_SYNTHESIS_V1_20260909.md](theory_sources/REPORT_NEW_HYP_CALIBRATION_MECHANISM_SYNTHESIS_V1_20260909.md) |
| new HYP：原始论文与贡献范围复核 | NEW_HYP_PRIMARY_LITERATURE_SCOPE_V1_20260909 | [NEW_HYP_PRIMARY_LITERATURE_SCOPE_V1_20260909.md](theory_sources/NEW_HYP_PRIMARY_LITERATURE_SCOPE_V1_20260909.md) |
| new HYP：当前论文收口与未来工作范围 | RC_NEW_HYP_PAPER_SCOPE_FREEZE_V1_20260915 | [RC_NEW_HYP_PAPER_SCOPE_FREEZE_V1_20260915.md](theory_sources/RC_NEW_HYP_PAPER_SCOPE_FREEZE_V1_20260915.md) |

[返回当前主总结](complete_results_zh.md) · [原文SHA索引](theory_source_index.csv)
