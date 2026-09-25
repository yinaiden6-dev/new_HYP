# new HYP 条件表示与 score-prior 新颖性审查

日期：2026-09-25。审查范围：原文文献及现有主稿，不提出新实验，不计入 benchmark 建设贡献。本笔记未改主稿。

## 结论

**目前不能把后 LLM 路径的贡献写成新 attention、新的一般条件化算子，或首次用身份监督学习候选相关比较。** 任意条件调制视觉特征、条件化相似度/检索表示、身份监督的配对重排都已有明确先例。现有结果更支持一项具体系统与经验发现：独立冻结匹配器的候选配对支持量 M，既能用于外部分数校准，也能经过后 LLM 残差适配器进入检索表示；当前模型的大部分纠错可由跨 patch 共同表示响应保留。

这并不意味着已经证明“只是在 embedding 中复制一个加性 M 分数”。共同表示位移在归一化和与不同 reference 内容相乘后，会产生内容相关的分数变化。**“把配对证据编码进表示的条件校准/重排”是合理归类；“与 RAW+M 严格等价”不是现有事实。** 文献检索未定位到同时满足“独立 matcher 的单一配对统计量 → 冻结多向量检索器的后 LLM、前冻结投影 adapter → 不直接读 M 的匹配头”的完全同构原文；这个有限检索结果不足以声称首创。

## 已核对的原文近邻

| 原文与位置 | 原文已做什么 | 对本工作的约束与具体差异 |
|---|---|---|
| [FiLM，AAAI 2018；§2.1 Eq.1–2、§2.2、§3](https://arxiv.org/html/1709.07871#S2.SS1) | 任意条件输入产生通道缩放/偏置 γ、β，应用到中间特征；卷积特征上的调制与空间位置无关。视觉问答实现含冻结预训练特征提取器选项。§3还明确讨论条件拼接形成条件偏置。 | 条件→表示、共享偏置、冻结视觉底座都不是新概念。当前残差 adapter 不必在架构上严格等于 FiLM；应称条件化残差适配，不能凭插入位置包装成新 attention。不同的是条件来源为候选配对支持 M，而非问题文本。 |
| [Reranking Transformers，ICCV 2021；§3.2 Eq.1–4、§4.2](https://arxiv.org/html/2103.12236#S3.SS2) | 两图全局/局部描述符连同位置、尺度与图像来源编码进入 Transformer；CLS 输出配对分数；BCE 标签是同物体/不同物体；训练正样本来自同身份，负样本来自初检邻居。 | 图像级身份监督、候选相关交互、学习式二阶段相似度都已有。本工作保留独立 RoMa、用标量支持条件化 query 表示并以自由内容读取，不是 RRT 那样把双图 token 送入新配对注意力。 |
| [Conditional Similarity Networks，CVPR 2017；§3.1–3.4 Eq.3–7](https://arxiv.org/html/1603.07810#S3) | 按条件选择/重加权 embedding 维度，以条件三元组损失训练。§2已区分潜空间门控与空间注意力。 | 条件相关 embedding/相似度与可解释维度选择早有先例；本工作条件是每候选动态生成的匹配支持量，非用户指定语义相似度类别。 |
| [GeneCIS，CVPR 2023；§5.1–5.2、Appendix D Eq.2](https://arxiv.org/html/2306.07969#S5) | 图像与文本条件经 Combiner 形成查询表示，再与目标图像表示点积；Combiner 包括动态加权项及联合 MLP 残差；通过匹配三元组/对比监督学习。 | 查询表示随条件改变且图库表示不读取该条件已有先例。其条件表达用户希望关注/改变的语义，当前 M 表达被评估 query–candidate 对的支持；当前 query 表示因此每候选不同。 |
| [CLAY，CVPR 2026；§3.1 Eq.1–2、§3.2 Eq.3–5、Appendix A](https://arxiv.org/html/2604.11539v1#S3) | 区分对称/非对称条件相似度；对冻结 VLM 视觉表示，按用户文本条件建立切空间与 SVD 投影，两侧投影后比较余弦，不另训练。 | 直接覆盖“冻结 VLM 表示的条件相似度调制”大叙事。不是外部 matcher 的 pair scalar adapter 先例：其条件来自用户文本概念，同一条件用于全图库；本工作用候选配对 M，只改 query 内容路径并训练 adapter/头。 |

CLAY 作者项目页明确标注 CVPR 2026：[官方项目](https://sohwi-lim.github.io/CLAY/)。用 arXiv 原文链接是因为作者 PDF/CVF PDF 的直接抓取在本次环境出现取回失败；方法细节已由 arXiv HTML 正文核验。搜索中的“CN-CNN”不能当成 CVPR2022 检索文献的准确名称；根审查单独核对的是 CVNet / *Correlation Verification for Image Retrieval*（2204.01458），不要将缩写混用。

## 现有干预究竟证明到哪

现有本地来源：[主稿](../../RC_NEW_HYP_PAPER_CLOSURE_DRAFT_20260924.md) §3–4，[内部全量](../../REPORT_POSTLLM_H593_V1_20260924.md)，[通路分解](../../REPORT_POSTLLM_H593_DECOMPOSITION_INTERPRETATION_20260925.md)。本笔记只读取主稿，数值按主稿与任务给定口径列示，未重新运行验证器。

统一口径：冻结 ColNomic LLM 与 projection，原 H593 身份/来源组件五折，自然 C128，RAW=426/593，POST adapter 与 INTERNAL3 联合训练；后者不直接读取 M；阈值与 HOLD/SWITCH 沿现有束定义。H593 是已多轮开发使用的 OOF 面板。

| 证据 | 可支持 | 不能推出 |
|---|---|---|
| POST_REAL 478；同模型恒定 M 426；错绑 M 369 | 内部路径实际利用候选配对条件，不是仅从“加了可训参数”获益 | M 是身份概率、训练学出了 RoMa 的空间支持图、内部不再依赖 RoMa |
| 共同响应 477，保留原 54 救回中的 52；去均值剩余响应 430 | 本模型的大部分已观察纠错由共同表示响应保留 | 全部 query 共用同一个只依赖 M 的向量；局部响应毫无作用；证明新的区域发现 |
| 固定恒定条件的内容命中后 469 | 重新选 reference token 命中对部分选择有作用；大部分收益不要求每次重选命中 | 完全不使用图像内容；完全等价于只给分数加 M |

根审查的源码核对确认 QualityResidualAdapter 采用 concat(normalize(h_i), z(M)) → down → GELU → up。相同 z 被重复拼接到所有 patch，在线性 down 层自然形成共同的条件偏置方向；经 GELU 后仍会受各 patch 内容调制。因此，空间共享调制本身既符合该架构结构，也已有 FiLM 等先例；“共同响应占主导”的新增经验内容是它在真实纠错回放中保留 52/54 次救回，而不是首次发现 scalar 条件可以产生共享偏置。

设共同响应经冻结投影为 t_q(M)，旧投影 token 为 p_i，则单位 token 为 (p_i+t_q(M))/||p_i+t_q(M)||。与 reference token r_j 的点积同时取决于 p_i、t_q(M)、r_j，且自由内容匹配还含 max_j。因此，**头不直读 M 不等于头在功能上不受 M 控制**；M 可以经过 embedding 进入最终分数。反过来，上式也说明不能仅凭共同响应就宣称它是一个与内容无关的加性分数项。这个代数关系属于标准表示运算解释，不宜列作新理论。

## 建议的新颖性表述

可写：研究冻结匹配支持与冻结多向量内容之间的一种候选配对接口；在当前精确实例检索系统中，定位支持统计量的外部与内部使用位置，并用候选错绑、恒定条件、共同/剩余响应及固定内容命中回放界定其作用机制。主要新增价值由该接口的具体经验发现及适用范围承担。

应降格：新 attention、新区域识别原理、首次身份监督条件检索、首次外部信号调制 embedding、仅用 M 即学得新的空间 ownership。benchmark 由学长构建，因此不能把 benchmark 设计或建立作为当前个人方法的新颖性来源。

可直接使用的英文句子：

“Building on conditional feature modulation and learned pairwise reranking, we study how independently computed pair support enters a frozen multi-vector retriever. Our contribution is an empirical characterization of this interface: support can affect candidate decisions through both external calibration and a post-LLM adapter, with most observed internal corrections retained by a shared token-level response. This finding does not establish a new attention mechanism or learned spatial ownership.”

文献判断为截至本次检索所核原文的定性对照，不是穷尽检索后的首创证明，也不是这些官方方法的同协议性能复现。
