# new HYP：原始论文与贡献范围复核

本轮只核对四篇原始论文，目的在于准确定位已有机制与本项目能贡献的
证据。不构成穷尽性新颖性检索，也不据此增加新模型或改变冻结试验。

**判断：质量/不确定性与身份内容联合用于识别，以及 token 级 MaxSim
检索，都已有明确先例。new HYP 的贡献应落在具体的 reference 条件证据、
检索任务监督、完整候选竞争及其可复核解释上；不能把通用思想或基本代数
写成首次发现。** 以下“本项目区别”均是与所核对方法的机制比较，不代表
其他文献从未研究过该区别。

## 1. 四篇直接相关原始论文

| 原始论文 | 论文中可直接确认的机制 | 对当前论述的作用 |
|---|---|---|
| Khattab & Zaharia, **ColBERT**, SIGIR 2020 | 分别编码 query/document；每个 query token 在 document tokens 中取最大相似度，再跨 query token 求和；可离线存 reference 表示。 | full-reference MaxSim 与可扩展索引已有基础。不能把恢复自由内容匹配或登记新 reference 本身写成新发明。 |
| Meng et al., **MagFace**, CVPR 2021 | 利用未归一化特征的模长表达识别质量，在身份识别损失下学习，不需要另给质量标签。 | “质量能从识别监督中学习”“无需额外质量标注”不是新命题。 |
| Kim, Jain & Liu, **AdaFace**, CVPR 2022 | 以特征范数作为图像质量代理，按质量调整 margin 和不同难度样本的训练权重。 | 质量、样本难度、可辨认性需要分别定义；质量与内容有关不等于二者可互相替代。 |
| Shi & Jain, **Probabilistic Face Embeddings (PFE)**, ICCV 2019 | 每幅人脸表示为均值与方差；匹配分数同时使用均值差和双方不确定性。冻结已有 embedding，另用 genuine identity pairs 学习不确定性模块。 | 与“固定内容编码器，再按身份关系学习可靠性并联合匹配”尤为接近，必须引用，不能称这套抽象思路首次出现。 |

对应可定位来源：ColBERT §3.1、§3.3、§3.4 [原文 PDF](https://arxiv.org/pdf/2004.12832)；
MagFace §1、§3 [原文 PDF](https://arxiv.org/pdf/2103.06627)；AdaFace §3.2–3.3、
Supplement B.1 [原文 PDF](https://arxiv.org/pdf/2204.00964)；PFE §4.1、§4.3，
式 (3)、(9) [原文 PDF](https://arxiv.org/pdf/1904.09658)。

## 2. 与当前系统的具体区别

ColBERT 本身是文本 passage retrieval，原 MaxSim 是 token 级相似度聚合，
并以正负 document 对训练。本项目在视觉 token 匹配上另外使用 query–reference
配对产生的可见性/匹配统计及既有 HOLD/SWITCH action；ColBERT 的结果并不能
直接验证这些新增环节在相似药盒上有效。[ColBERT §3](https://arxiv.org/pdf/2004.12832)

MagFace 的质量信号来自单幅图像的 embedding 模长；AdaFace 的质量代理用于
训练损失调节。本项目的 M 是给定候选 reference 后形成的配对匹配统计，其值
不能直接等同于图像自身质量、可辨认概率或匹配正确概率。将 M 称为“匹配质量
代理/质量统计”比未经校准的“置信概率”准确。这是按各方法定义作出的比较。
[MagFace §3](https://arxiv.org/pdf/2103.06627)，[AdaFace §3.2–3.3](https://arxiv.org/pdf/2204.00964)

PFE 的双方不确定性由各输入图像分别估计，随后共同进入概率模型的匹配分数；
本项目用配对匹配过程产生统计，并在固定 RAW C128 的全部 127 个 challenger
中作决策。PFE 的 Gaussian/互似然推导不能直接移植来宣称当前七参数 logit
就是身份后验概率。[PFE §4.1、§4.3](https://arxiv.org/pdf/1904.09658)

## 3. 本项目可以写什么

由当前实现与冻结结果支撑的叙述是：

> 我们研究细粒度 reference 检索中，候选条件的匹配质量与内容相容性如何
> 共同支持检索决策。系统保留完整 reference 内容匹配，在固定候选集合和
> action 流程中，使用身份/正负检索关系进行任务训练。通过无损重放、候选
> 绑定控制、特征删减后重训及完整候选竞争，区分质量统计、内容统计、乘积
> 对比与响应统计各自能解释的表现。

这些是当前研究对象与验证方式的定位，不能在新实验验证前补写“已经证明
哪个项必要”“已得到更高准确率”。方法独特之处与普遍新颖性仍需更广泛的
相关工作比较及新的独立数据支持。

“retrieval-only”应明确限定为**本任务训练监督只有身份/正负检索关系，
没有本任务 bbox、mask、point、人工 crop 或分割 teacher**。这不表示无身份
标签、无预训练监督，也不自动证明总标注工时减少；冻结匹配器和编码器的
预训练来源仍需披露。MagFace 已说明可不额外提供质量标签，PFE 已说明可在
原训练数据上学习不确定性，因此不应将免除某类额外标签泛化成所有识别任务
首次做到。[MagFace §1](https://arxiv.org/pdf/2103.06627)，[PFE §4.3](https://arxiv.org/pdf/1904.09658)

## 4. 数学与实验不要混淆

`dS=(dM+dL)/(1+dM*dL)` 在正数、S=M L、epsilon=0 条件下是基础代数恒等式。
它能说明线性 head 读取 dS 时具有一个非线性的 M/L 函数项，不能作为原创
数学定律，也不能自行证明泛化收益。准确的原创性候选是**该函数项在这个
任务、候选流程和训练配置中的可检验证据**，且必须等待相应冻结对照结果。
完整 epsilon/FP64 条件见
`reports/NEW_HYP_PRODUCT_CONTRAST_MATHEMATICAL_REVIEW_V1_20260909.md`。

当前共同证据框架可写得比空间区域解释更一般，但删除 RoMa/ColNomic 名称
不能产生一般性证明：方法正文应报告实际实现，抽象理论应列明条件；跨任务、
跨编码器、跨数据的有效性需分别验证。旧 Reference HYP 的空间结构或 ownership
结论，不能由这些人脸识别文献或质量/内容联合的名称间接推出。

## 5. 可直接使用的参考文献

1. Omar Khattab and Matei Zaharia. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*. SIGIR 2020. [作者论文](https://arxiv.org/abs/2004.12832)。
2. Qiang Meng, Shichao Zhao, Zhida Huang, and Feng Zhou. *MagFace: A Universal Representation for Face Recognition and Quality Assessment*. CVPR 2021, pp. 14225–14234. [CVF 官方记录](https://openaccess.thecvf.com/content/CVPR2021/html/Meng_MagFace_A_Universal_Representation_for_Face_Recognition_and_Quality_Assessment_CVPR_2021_paper.html)。
3. Minchul Kim, Anil K. Jain, and Xiaoming Liu. *AdaFace: Quality Adaptive Margin for Face Recognition*. CVPR 2022, pp. 18750–18759. [CVF 官方记录](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html)。
4. Yichun Shi and Anil K. Jain. *Probabilistic Face Embeddings*. ICCV 2019, pp. 6902–6911. [CVF 官方记录](https://openaccess.thecvf.com/content_ICCV_2019/html/Shi_Probabilistic_Face_Embeddings_ICCV_2019_paper.html)。

核对范围止于上述四篇原始论文；未宣称完成截至 2026 年的全领域文献检索。
未运行模型、读取新评价结果或查询调度器。
