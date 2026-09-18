> 2026-09-15写作范围更新：当前主线已按 [new HYP论文收口文件](../plan/RC_NEW_HYP_PAPER_SCOPE_FREEZE_V1_20260915.md) 固定。GroZi/RPC外部确认与ISIC跨域探索已完成；ownership、完整过目不忘系统及新编码器训练作为未来工作。下方为2026-09-09历史工作稿，摘要和结果尚未整合上述后续实验，不作为当前总成绩或最终投稿版本。

# new HYP: Reference-conditioned Joint Evidence Calibration for Fine-grained Retrieval

工作稿。细粒度绑定检验及后续固定校准试验已陆续完成并记录；本稿仍需
按投稿要求补齐完整数据及训练协议说明，不是可直接投稿的版本。
所有实验数字以 `REPORT_RC_RETRIEVAL_ONLY_FINAL_EVIDENCE_LEDGER_20260909.md`
及其逐项来源为准。本文描述当前已经存在的检索系统，不把未实现的
superregion、额外的独立假设生成网络或ownership加入方法图。

## Abstract (working draft)

Fine-grained reference retrieval requires distinguishing visually similar
instances under occlusion, viewpoint changes, and repeated packaging text.
We study new HYP, a framework for reference-conditioned joint evidence calibration, combining
candidate-specific match quality with visibility-weighted content evidence
and a conservative decision over a fixed retrieval shortlist. In two opened
development and regression cohorts, one frozen calibration head improves
correct predictions from 25 to 27 of 32 queries and from 61 to 69 of 90 queries,
with no observed breaks of correct baseline predictions in either cohort.
Factor interventions establish that match-quality and normalized-content
statistics jointly drive the corrections in this fixed scoring graph. A
controlled experiment preserves token content, quality factors, and visibility
value distributions while disrupting visibility-content bindings. For a
separate frozen head, all three original corrections survive joint binding
disruption; the intact system retains only a one-query advantage, failing the
predeclared group-level contribution criterion. These results support
candidate-level joint calibration while limiting a stronger fine-grained
binding explanation. We also identify a lossy proposal interface and restore
the original computation exactly. The findings are confined to the evaluated
models and opened cohorts, without claiming a universal latent-hypothesis
theory or independent spatial localization.

## 研究问题

相似药品包装的准确识别需要同时处理遮挡、视角差异、重复说明文字和
真实背景物体。全图内容相似性可以找到相关包装，却不一定能把本次目标
与外观或说明高度接近的其他reference分开。我们研究：候选条件的
可见性信息，能否与已有检索内容表示共同改善固定候选集内的身份决策。

任务训练只使用检索身份与正确/错误候选，不使用本数据集的mask、box、
point或polygon作为训练监督。当前模型使用预训练的RoMa与ColNomic，
不声称这些基础模型的全部预训练都只采用检索监督。SAM及外部分区不
属于本文主线。人工像素诊断只能作为事后编码行为分析，不能成为模型
输入或被计为自动定位成绩。

## Retrieval-only 任务训练与标注成本

本任务的监督单位是检索关系：自然query、正确reference，以及由固定
候选检索器提供的自然错误references。身份/配对标签用于训练候选比较
和联合校准；任务内不使用mask、bbox、point、polygon、人工crop或
SAM/其他分割模型输出作为teacher。当前实例复用冻结的基础表示与
匹配器，检索监督用于任务校准；不声称整套基础模型由这些配对从头
训练，也不声称产生了一个额外的空间监督定位模型。

这个设计的实际价值，是在同一细粒度身份检索目标下不要求逐图绘制
精细空间标签，可省去框/分割标注这部分工作。仍需要reference资源
及query-reference身份或正负配对标签，因此术语为retrieval-only
task supervision或no task-specific spatial annotation，不能写成
annotation-free或完全无监督。

目前没有测量总标注人时，不报告节省百分比或宣称总标注时间必然
降低。本文也不以识别成绩声称替代了检测/分割任务本身的标注和输出。
预训练模型原有监督来源须独立披露；任务标注需求和基础模型预训练
成本不可混淆。最后绑定检验的负结果不改变当前训练确实未使用本任务
空间标注的事实，但也不能被用来证明尚未建立的旧Reference HYP的空间能力。

## new HYP：联合证据状态的形式化

new HYP是当前联合证据校准框架的名称。其候选证据状态统一记为
E_g=(M_g,S_g,S_g^Q,S_g^R)，核心内容相容性为L_g=S_g/max(M_g,1e-12)。
冻结action结合E_g、RAW winner的E_w以及RAW先验，计算原六维比较
特征并决定HOLD/SWITCH。后续不再用单独的字母H混称当前框架与旧理论。

这是多维联合关系的一种具体形式。它不要求二维区域，也不新增独立假设生成网络。
M与L等统计在当前计算图内共同驱动纠错的证据，与“细粒度索引绑定
是否产生额外群体贡献”的检验是两个命题。后一个检验未通过，不否定
前一个机制；前一个机制有支持，也不意味着旧空间P-only门已经通过。

## 模型

理论层使用模型无关的质量通道G和内容通道C。G提供候选条件的匹配
可靠性；C提供query与reference的细粒度内容兼容性。以下是这一类
因子化检索读出的具体实例。实验实现中，G由RoMa产生、C由ColNomic
产生；名称、权重和训练来源必须在方法与实验中披露。只在这一实例
验证的性质不表述为所有G/C组合普遍成立。

令query image tokens为q_i，candidate reference的image tokens为r_j。
对每个固定C128成员g，RoMa给出两侧完整patch网格的visibility权重
w_q^g(i)、w_r^g(j)。它们在当前scorer中为内容匹配提供权重，不强制
ColNomic只能读取RoMa预测的单个对应token。

    c_ij^g = cosine(q_i, r_j^g)
    l_i^g = max_j [w_r^g(j) c_ij^g]
    M_g = sqrt(mean_i w_q^g(i) * mean_j w_r^g(j))
    L_g = sum_i w_q^g(i) l_i^g / max(sum_i w_q^g(i), 1e-12)
    S_g = M_g L_g

完整reference内的内容搜索属于late interaction思路，不能作为本文
新发明。ColBERT将独立编码后的细粒度交互保留到检索阶段。
[Khattab and Zaharia, 2020](https://arxiv.org/abs/2004.12832)。
RoMa v2提供dense feature matching能力；本文复用其预训练模型，不
将其描述为药品规格身份分类器或本文新提出的网络。
[Edstedt et al., 2026 version](https://arxiv.org/abs/2511.15706)。

冻结action比较RAW winner与全部127 challengers。六个输入特征包括
标准化RAW gap、S差异、M差异、S/M差异以及两项Q/R辅助差异，使用一组
六权重和一个bias。零阈值决定HOLD/SWITCH。HOLD只表示保留RAW winner，
不表示reference缺席。本文不改变候选来源、阈值或完整challenger循环。

## 可检验的关系解释

当前计算可以诱导一组reference-conditioned支持关系：query内容在
candidate reference内找到匹配，并由候选条件visibility调制该内容的
贡献。该关系可以分布在图像的多个位置，不要求形成二维连通superregion。
这是对已有评分机制的解释性命题，不是另一个独立训练的假设生成网络。

核心可证伪问题是：正确的visibility-content绑定是否提供了超出相同
visibility值分布和全局质量M的识别贡献。若只改变匹配关系而保留这些
边缘信息，最终正确集合仍不变，则不能据此宣称细粒度绑定解释了识别
收益。若破坏绑定导致预先约定的群体识别损失，并有一致的margin方向，
才支持该受限机制解释。

旧Reference HYP空间命题要求连通区域生成与独立验证；本文关系解释不覆盖该命题，
也不把原空间失败改为通过。旧P-only接口的负结果保留为另一种具体
机制与资格要求下的结果，不推出所有潜在表示假说均不可能成立。

## 可独立于具体模型讨论的接口性质

设完整证据状态为E，旧评分为f(E)，压缩接口只保留T(E)。如果存在
E1、E2使T(E1)=T(E2)但f(E1)≠f(E2)，则不存在仅依赖T的读出器能在
所有输入上精确复现f。证明直接来自同一个T值无法同时映射到两个
不同输出。该性质不依赖RoMa或ColNomic，是接口设计约束；本文的具体
贡献应是指出当前压缩怎样违反它并给出可复现修复，而非将这个基本
不可恢复性事实宣称为新发现的普遍定律。

在质量因子M未被接口充分保留、且下游评分实际依赖M的条件下，只调
压缩后head的参数不能普遍恢复旧评分。该结论针对评分值的可恢复性，
不声称所有样本的类别预测必然改变，也不推出恢复M就一定提高准确率。

类似地，在相同非负权重下，完整reference搜索的局部最大值不小于
任何单点硬指派的匹配值；这一不等式同时适用于正确和错误候选，因此
不能单靠它证明检索排序一定改善。信息保留的代数性质与识别收益的
实验命题必须分别论证。

## 乘积对比提供的非线性校准项

虽然最终头对六维输入是线性的，这不意味着它对底层质量M和归一化
内容L只有加法作用。对正值、无epsilon/floor的理想实数比较，令
x为M的对称相对差、y为L的对称相对差，原S=M L的对比满足

    dS=(x+y)/(1+xy).

该有理函数不能由x、y的一般仿射组合替代。因此删除dS可能不删除
底层独立信息，却会缩小固定线性读出的函数表达范围。这个代数性质
提供了一个可检验的非线性校准解释；它不是本文独创的数学恒等式，
也不直接证明识别收益或统计交互。

原数值实现保留e=1e-12和L的保护分母。4128次原TRAIN/PAIR比较均为
正值、无M floor，含e的完整恒等式对原dS最大误差4.44e-16；模型仍
使用原列而非公式近似。具体推导与源证据见数学复核报告。S列和Q/R
响应组的固定2×2及后续固定参数干预已完成；结果分别说明训练作用与
直接输入作用，不把代数性质本身视作正确决策的充分证明。

## 已有结果

FROZEN_C在原RAW C128的current-runtime EVAL32上由25正确提升至27，
2次救回、0次破坏；在opened difficult90上由61提升至69，8次救回、
0次破坏。两者分别报告，不合并分母。另一组NATIVE7/C_PAIRED参数在
matched EVAL32上由25提升至28，3次救回、0次破坏；不能把这一28/32
与前一参数组的69/90写成一个统一模型成绩。

上述数据均已打开。当前matched EVAL32含11个supergroups，不能当成
32个独立身份或新外部确认。候选recall、候选集内识别和action应分开
报告；对不在C128内的正确reference，当前action无法补回。

在原FROZEN_C上，visibility质量与归一化内容项共同支持旧救回。固定
L/Lq/Lr而统一M=1的完整因子干预，使difficult90从69回到61。无损P/V
接口重放又对8192对输入的32768个标量和8128个action logits逐bit恢复
旧结果。这些结果支持质量与内容合作的计算解释，不单独证明空间目标
或更复杂的关系结构。

另一个独立的原RAW FULL64接口对照固定query-image-only与FP64归一化
mean-MaxSim，从R_IMAGE恢复到同源R_FULL后，EVAL32由23变为25，2救0损；
TRAIN32均为27。完整8192对R_IMAGE与旧a逐bit一致。该结果表明这一读出对reference轴敏感，不混称为原weighted scorer的
new HYP增益。后续固定D质量校准读出、只补回原上下文tokens的完整
三臂实验中，D_IMAGE为27、D_FULL为25，原C为28（同一EVAL32）；新
读出未采用。因此不能从ALL均值收益推导所有校准路径均应扩展该轴。

## 重训练后的单通道与联合读出

同一原PAIR64＋FULL TRAIN32、原优化器/损失/训练次序下，固定五个
头分别读取RAW gap、RAW+M、RAW+L、RAW+M+L和原六维特征。在已打开
RAW C128 EVAL32中，RAW2为25/32，两个三参数单通道及JOINT4均26/32，
原NATIVE7保持28/32。JOINT4的预定机制简化判定未成立。

预定的完整头次比较则为正：原NATIVE7分别胜过RAW+M和RAW+L，均为
2救0损，等权supergroup平均差分别+0.06818、+0.04545。这说明在这些
固定重训练条件下，已测单通道不能复现完整模型；比固定头删项更直接
检验了可替代性。它不证明所有单通道函数类都不可替代，也不把不同
参数数的嵌套模型声称为容量完全相同。L仍包含visibility，M臂仍使用
RAW内容先验。完整五头、所有比较与失败样本均保留。

## 细粒度绑定机制检验（已完成）

协议：`plan/RC_REFERENCE_RELATION_FINAL_V1_20260909.md`。
同一模型、同一RAW C128、同一token内容，固定原每通道M与分母，使用
预定的query/reference绑定置换。REAL、Q-only、R-only、QR四条件全部
先封存预测再join标签。唯一主比较为NATIVE7 REAL与QR，另两项只分解
机制，不根据结果改选。完整64条计算和独立literal/统计复核已通过。

| split/head | RAW | REAL | Q打乱 | R打乱 | QR打乱 |
|---|---:|---:|---:|---:|---:|
| EVAL32 / NATIVE7 C | 25 | 28 | 27 | 27 | 27 |
| EVAL32 / FROZEN_C | 25 | 27 | 28 | 27 | 26 |
| TRAIN32 / NATIVE7 C | 27 | 28 | 28 | 29 | 29 |
| TRAIN32 / FROZEN_C | 27 | 28 | 28 | 29 | 29 |

唯一主比较EVAL32/NATIVE7 REAL对QR为1救0损、净增+1（门为+4）；
严格正action-margin降幅13/32（门为21/32）；等权supergroup平均降幅
−0.00591804（要求>0）。该较强绑定贡献命题未建立。

关键集合关系是：NATIVE7原来相对RAW的三次纠错全部在QR下保留，
FROZEN_C的两次纠错也全部保留。REAL多出的一个正确为OUTCOME-0618，
它原来已被RAW识别正确，打乱绑定后被破坏。因此本次+1不能用于解释
原2/3次纠错依赖精确细粒度绑定。

## 冻定参数组干预与训练作用的区别

完成S和Q/R固定2×2重训后，EVAL32正确数为JOINT4=26、PRODUCT5=27、
RESPONSE6=26、ORIGINAL7=28。S背景上的Q/R组重训条件保护0618；补S
的重训条件恢复0212。固定原头参数直接去S，0212仍然正确，损失的是
原RAW正确的0419、0618。这说明训练路径改变共享参数后，具体受益
样本与直接输入项贡献并不相同，不能据重训差异倒推单项直接因果。

当前NATIVE7固定去Q/R由28变27，原3个rescue全保留，但0618错误切换；
另一旧FROZEN_C在opened difficult90固定去Q/R由69变70，0损失、原8个
rescue全保留。它们是不同参数与数据的固定头效应，限制了“响应统计
普遍有益”的更强解释。70是已打开数据上的诊断，不作为新外部结果。
随后同一旧head的matched32桥接仍为27，但新增0220、损失0618，
所以70/90的修改未在两组上同时保持无损。完整源、分解残差及两条
lineage见校准机制联合证据报告。

## 训练组敏感性与严格线性容量

完整TRAIN基线加12个TRAIN supergroup逐组删除条件，各4头，均按原
2000步训练并独立重训。PAIR64固定，EVAL32完整，未选择有利删除组。
S列的条件净贡献在12次删除中均非负（无Q/R时8正4平；有Q/R时11正
1平）。Q/R在S背景上的净增为5正6平1负，较依赖训练组成。原头正确
数26–28，median27；0044/0212的纠错均保留12/12，0220仅3/12，0618
为9/12。相关训练扰动不能当作12份新独立测试数据。

一个另行标识的已打开EVAL容量诊断固定原native6特征，要求严格保留
原28正确并增加任意一条剩余错误。四个系统都获得精确Farkas证书并
独立验证。这排除同一实数线性类在这一正确集合上的严格无损增一，
不排除不同正确集合、非线性读出、更多特征或精确平局解。它是标签
感知的事后容量结果，不是新性能或外部确认；更宽的
4960个29目标集合覆盖检验找到精确可行witness，而包含RAW25的35种
集合均精确不可行。这将瓶颈定位于严格保持与总体识别之间的冲突，
而非总体最多28；witness仅为已打开标签感知容量证书，不作模型结果。

## 相关工作与贡献范围

Late interaction/MaxSim已有ColBERT先例。质量与识别联合学习也有明确
基础：MagFace通过身份监督学习表征质量，AdaFace按质量调整训练margin；
PFE冻结已有embedding，再用身份对学习不确定性并联合计算匹配分数。
因此“质量加内容”“无额外质量标签”或乘积相对差的代数恒等式不能
作为本文首次发现。
[MagFace](https://arxiv.org/abs/2103.06627)、
[AdaFace](https://arxiv.org/abs/2204.00964)、
[PFE](https://arxiv.org/abs/1904.09658)。

本文区别应落在配对reference条件的匹配统计、细粒度视觉检索中的
完整C128决策、任务内免空间标注以及区分固定输入贡献和重训练贡献的
可重放证据。M是匹配质量代理，不是未经验证的正确概率；logit也不
自动成为身份后验概率。四篇文献复核并非穷尽的新颖性搜索。

## 讨论与结论

当前证据支持reference-conditioned joint evidence calibration：
候选条件的匹配质量与内容统计共同决定保守纠错。该联合状态本身
可以称为多维关系；最后试验只限制了其中更强的细粒度绑定解释。
不应将其概括为“reference失败”或“所有关系H无效”。

该机制的固定计算图作用已有重放和干预支持，但尚未证明它是唯一
解释、优于所有更简单的重新训练融合模型，或对任意表示和对象普遍
成立。本文没有给出不可提升的上界，也没有证明普遍“过目不忘”理论。
理论表述可以模型无关，具体实现和证据范围必须披露。

此前停止安排已由用户明确延期至北京时间2026-09-11 24:00。旧检验
结论保持不变，后续new HYP试验各自冻结协议，不覆盖本稿已有结果。
完整结果见`REPORT_RC_FINAL_REFERENCE_MECHANISM_AND_PAPER_CLOSURE_20260909.md`。
