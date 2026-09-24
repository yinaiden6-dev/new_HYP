# 三项收口：已有工作的原文核对与可用证据底稿

日期：2026-09-24。本文是资料底稿，不宣告第1项已经完成；按照用户指定的2→3→1顺序，最终论述应在固定简化头外部迁移和强Qwen联合校准完成后整合。本文没有运行新实验，没有填入任务2/3尚未产生的结果。

## 原始文献核对

**ELViS: Efficient Visual Similarity from Local Descriptors that Generalizes Across Domains**，Pavel Suma、Giorgos Kordopatis-Zilos、Yannis Kalantidis、Giorgos Tolias；arXiv:2603.28603v1，2026-03-30，作者标注ICLR 2026。

- 已有思想：冻结局部描述子、在两图相似度矩阵上学习可解释相似度；以描述子相关dustbin gains的OT抑制无信息对应，再对强对应进行可学习投票聚合。原文§3.2–3.3、图2、式2–5。[官方全文](https://arxiv.org/html/2603.28603v1)
- 监督：正负图像对和改造BCE；不是靠人工局部对应标注才成立。原文§3.4。[训练与推理原文](https://arxiv.org/html/2603.28603v1)
- 不能把“冻结表示＋小模块”“局部匹配聚合为图像分数”“仅图像对检索监督”单独称为本项目首次提出。[作者摘要](https://arxiv.org/abs/2603.28603)
- 可定位的设计差别：本项目保留两个独立模块产生的RoMa质量M和ColNomic自由内容L，并研究压缩到整体质量的接口；ELViS上述主路径是在描述子相似矩阵内部筛选、变换、计票。这是对具体方法的比较，不证明“分离质量与内容”的广义思想没有任何其他先例。

**To Match or Not to Match: Revisiting Image Matching for Reliable Visual Place Recognition**，Davide Sferrazza、Gabriele Berton、Gabriele Trivigno、Carlo Masone；arXiv:2504.06116v1，2025-04-08。

- 已有思想：匹配重排可能损坏高质量原检索；利用匹配inlier数量估计原预测的不确定性，并分析何时值得重排。原文摘要、§1、§4.2。[官方全文](https://arxiv.org/html/2504.06116v1)
- 更接近当前设计的先例：图6将MASt3R不确定性分数交给在MSLS Val训练的Logistic Regression，再在Pitts30k与SF-XL Occlusion分析错误定位概率。因此“匹配置信度＋小校准头＋跨数据集分析”也不能独占。[§4.2与图6原文](https://arxiv.org/html/2504.06116v1)
- 当前可比较的差别：本项目对同一query的全部自然C128候选，将质量和自由内容差值联合输入RAW winner–challenger动作；研究的是候选条件证据怎样组合及哪些局部操作可删。任务与设计不同不自动等于性能更好，也不能把这里的简易门控称为该文官方复现。[作者摘要](https://arxiv.org/abs/2504.06116)

以上为原文事实及明确标出的设计比较；没有据此宣布新颖性审查已经穷尽全部文献。无需在本轮复制两篇论文的表格或长段文字。ELViS官方代码入口为[作者仓库](https://github.com/pavelsuma/ELViS/)。

## 本地直接证据能支持哪些具体说法

下表所有药盒实验均为已经多轮使用的H593开发总体，原五折按身份/来源component隔离。不同编码器使用各自自然C128，不能把跨编码器数字当作相同候选池的直接排序。当前已有结果的validation→result SHA均在本轮重新核对。

|可写的具体事实|直接证据|与既有思想的关系及限制|
|---|---|---|
|在本协议下，局部query/reference加权不是保留总体纠错收益的必要条件；重新校准很重要|原ColNomic C128：RAW426，完整COST1 481（58救/3损）；整体M＋自由L的M1Q0R0同协议重训481（57/2），冻结旧头直接替换则441。目标召回570/593。[算子报告](REPORT_H593_QUALITY_OPERATOR_V1_20260922.md)|这是当前机制的受控简化发现，不是“所有数据都只需一个标量”的定理；相同481不意味着逐图等价。|
|原收益不能全解释为让编码器重新关注局部区域|上述M1Q0R0保留整体收益，但编码器tokens始终冻结；局部加权被关闭。[算子报告](REPORT_H593_QUALITY_OPERATOR_V1_20260922.md)|说明输出后的候选质量校准有作用，不能推断内部注意力已经改变。|
|乘积读出不是已证明的核心必要项|COST1_ADDITIVE4与PRODUCT5均481（57/2），593个最终候选完全一致。[简单解释报告](REPORT_H593_SIMPLE_EXPLANATIONS_V1_20260924.md)|应突出可分离质量与内容的接口，不宣称M×L本身的非线性形式不可缺少。|
|整体质量含有当前内容小头未兑现的增量，而且必须正确绑定候选|ColPali自有C128：RAW283、CONTENT7_COST1 286、MASS5_COST1 348（65/0），M错绑252；召回467/593。[ColPali报告](REPORT_COLPALI_NATIVE_MASS_H593_FINAL_20260924.md)|支持这个接口经源域重训可用于另一检索器；不证明所有纯内容读出都失败，更不证明RoMa永远不可替代。|
|简化接口在另一表示设定下也有效|ColQwen-base自有C128：RAW227、CONTENT7_COST1 240、MASS5_COST1 322（96/1），M错绑182；召回438/593。[ColQwen-base报告](REPORT_COLQWEN_BASE_NATIVE_FINAL_20260924.md)|该base是预训练Qwen2.5-VL-3B-Instruct骨干＋固定未检索训练的投影，不是整个模型未经训练；各折头重新训练，不能称原参数零样本迁移。|
|本地简单置信度门没有复现联合校准的全部收益|同H593/C128，所实现GATE按TRAIN内预算或净增选门为434–440，COST1_ADD4@zero481。[简单解释报告](REPORT_H593_SIMPLE_EXPLANATIONS_V1_20260924.md)|仅排除这些实现及训练协议，不能概括成超过所有匹配验证法；不是ELViS/To Match的官方复现。|
|原系统与强Qwen有互补错误，但尚不能把互补称为可部署增益|同H593/同C128：Qwen直接logit522、QwenCE523；原COST1及ADD4相对Qwen直接各独有23张，但独有集合仅共享22张。[同轴互补审计](rc_h593_qwen3_simple_complementarity_audit_20260924.json)|必须报告Qwen更强的主事实；oracle并集不是实际系统。任务3将用TRAIN标签学习，固定held动作后核算。|

## 仍不能声称的内容

1. **不是重排、匹配验证或检索身份监督的首次提出。** 两篇原文已覆盖上述部分，当前差别应落在具体接口和直接消融，而非换名称。
2. **没有证明RoMa表示必不可少或ColNomic缺失必要信息。** 未充分拟合或泛化失败的学生/小读出，不是信息不可能性证明；也不能排除其他匹配器生成有效质量。
3. **没有证明空间ownership、精确对应或六项统计各自不可缺少。** 现有简化结果反而要求收窄这些表述。
4. **尚无完整的同条件速度优势。** 必须计入RoMa质量生成和内容处理；小头CPU时间不能与完整Qwen推理时间直接相比。
5. **旧完整头的GroZi/ISIC结果不替代简化头固定迁移。** 任务2尚待正式汇总；且这些外部数据早已使用过，新增复核不是未接触确认。
6. **强重排之外的联合增益待验证。** 任务3主臂已预声明CE_QWEN_ML5，对比CE_QWEN3及CE_QWEN_L4；目前只有输入/旧头回放preflight，不是联合训练结果。
7. **不存在“no-regret”逐图误伤保证。** COST1是损失偏好；观察到少量误伤或某面板0误伤，不构成任何未来样本的零误伤定理。

## 任务2/3完成后应填入的证据位置

- 任务2：单一H593全源头的参数与代码hash、GroZi480和ISIC537固定迁移逐候选封存、主臂MASS5及对照ADD4/原COST1、救回/误伤和分组不确定性。不得按外部结果挑头。正向结果仅支撑这次固定方案的迁移；负结果需要保留其失败域。
- 任务3：原Qwen基线逐折重训复现、四臂两损失完整OOF结果、主臂相对QwenCE和内容控制的配对净增与区间。主比较无增量就如实写该联合方案未增加强重排收益，不能改报oracle或事后最好次臂。
- 第1项最终文件：以以上事实决定贡献力度；本文仅为来源和论证边界，不预先写“值得主会”的结论。

## 已核对的本地结果来源

|结果文件|validation状态|result SHA256|
|---|---|---|
|[quality operator](../results/rc_h593_quality_operator_eval_v1/result.json)|QUALITY_OPERATOR_EVAL_ALL_COUNTS_PASS|aa6a2919419c4d1a6c95d31e03169f795fa67d718271f0267ad06574fc78c935|
|[simple explanations](../results/rc_h593_simple_explanations_v1/result.json)|ALL593_SEALS_ACTIONS_COUNTS_PASS|b68789604c1f81bf2de84ec2f87f595b3eb6d7b80354e83d1d5c3c0bc9cade54|
|[ColPali](../results/rc_colpali_native_mass_head_v1/result.json)|COLPALI_NATIVE_TRANSFER_FIVE_FOLDS_593_PASS|d3234afe1b093f760c9e07063ac0e0fedfbaba0f85e242c7b3b2419bf1654dfb|
|[ColQwen-base](../results/rc_colqwen_base_native_v2/head/result.json)|COLQWEN_BASE_NATIVE_TRANSFER_FIVE_FOLDS_593_PASS|3bb9bd759f809bdaa803b61f3fd2f586adfb02b45b24b71c45ff811f6fa37359|
|[Qwen rerank](../results/rc_h593_qwen3_rerank_v2_layout/result.json)|QWEN3_FULL593_AXES_BASELINES_FOLDS_COUNTS_PASS|0d0a8760fce0d3314062d1c7ffb30398435336d6c93d6d761455710747756dcc|
