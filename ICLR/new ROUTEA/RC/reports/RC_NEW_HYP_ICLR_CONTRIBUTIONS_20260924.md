# new HYP：ICLR 投稿贡献与证据定位

日期：2026-09-24。本文整理已完成结果，不新增实验、不替换原封存主结果。新增 POST 及交换读出头诊断作为探索证据；既有外部面板不改称未接触确认。

**建议主张：密集匹配可以通过候选绑定的整体支持质量，补充冻结内容检索器的决策；在已测协议中，这种收益不要求保留逐 patch 的局部重加权，且能够通过一个简化的质量—内容接口迁移。**

这是一篇以受控机制发现和可迁移接口为中心的论文定位。现有证据不支持把“首次使用匹配置信度”“七个参数”“乘法融合”或“提出通用内部注意力改造”作为核心新颖性，也不能据此保证会议接收。

本轮已进一步分解“质量来自什么、内部如何使用”：见[新增机制深析](REPORT_NEW_HYP_MECHANISM_DEPTH_20260924.md)。H593 的 reference 主效应控制与 POST 的共同方向/固定命中控制是两套范围不同的实验，以下分别使用，不合并成一条未经验证的因果链。

## 1. 贡献一：区分有用的匹配证据与其原有使用方式

我们分开检验整体配对质量、局部加权、自由内容匹配和最终校准，发现：**某个已训练头依赖局部操作，不等于该操作对获得纠错收益不可替代。**

统一协议：H593、ColNomic 自然 C128、原分组五折、COST1、固定零切换阈值；全部 593 张进入分母，23 张目标不在 C128。

| 路径 | 正确/593 | 相对 RAW 救回/损失 | 支持的判断 |
|---|---:|---:|---|
| RAW | 426 | 0/0 | 内容检索基线 |
| 原完整 NATIVE7 | 481 | 58/3 | 原系统确有纠错收益 |
| 去局部权重、保留 M 与自由内容，沿用旧头 | 441 | 15/0 | 旧参数依赖原特征算子 |
| 同一简化算子，按原协议重训头 | 481 | 57/2 | 无局部权重也可恢复总准确率 |
| RAW + 自由内容 L，三参数 | 426 | 0/0 | 当前额外内容标量未产生纠错 |
| RAW + M，三参数 | 478 | 55/3 | 更简单的质量校准已经保留大部分净增 |
| RAW + M + L，加性四参数 | 481 | 57/2 | 简化质量—内容接口有效 |
| 再加入显式 M×L，五参数 | 481 | 57/2 | 本轮全部 593 个决策与加性四参数相同 |

481 的恢复是总数恢复，不是与完整 NATIVE7 逐图等价，也不是统计等价证明。RAW+M 的 478 必须放进主文，不能回避这个强简单解释；没有依据宣称自由 L 或显式乘积在所有场景都不可缺少。

候选绑定进一步限定了信息来源：在原第一折 119 张、简化 RoMa 质量头中，保持 RAW、L 和头不变，仅循环错绑 M，正确数从 104 降至 80，原 8 次救回只保留 1 次。跨检索器也观察到同类依赖：ColPali 的 MASS5-COST1 从 348 降至 252；ColQwen-base 从 322 降至 182。这支持使用与候选一致的质量证据，而不能被解释为给所有候选统一抬分。

**最值得写出的新认识是“保留什么信息、哪些原有操作可替换”，而不是笼统的“匹配能帮助检索”。** 这里没有证明精确坐标、匹配器内部细化或单图表征普遍无用：即使下游去掉局部权重，生成 M 的过程仍使用 RoMa。

进一步的五折无标签主效应分解固定该简化头，仅在各折 TRAIN 拟合 `log M=mu+a_query+b_reference+residual`：仅保留 reference 主效应为 427/593，未保留原 57 次救回中的任何一次；去除此主效应仍为 481/593，保留其中 52 次（同时另有 5 救/5 损，正确集合不完全相同）。相同 query 的共同尺度在相对质量中抵消。这个结果限定了“只是偏好本来容易匹配的 reference”这一具体解释；不能把剩余量称为纯两图交互或已校准的身份概率。

来源：[算子分解](REPORT_H593_QUALITY_OPERATOR_V1_20260922.md)、[简单解释对照](REPORT_H593_SIMPLE_EXPLANATIONS_V1_20260924.md)、[第一折机制与绑定](REPORT_PAIR_QUALITY_FOLD0_MECHANISM_20260923.md)。

## 2. 贡献二：给出只需检索身份监督的简化接口，并检验其迁移范围

令独立内容编码器产生 patch 相似度 c_ij，匹配器产生候选条件权重 u_i、v_j。使用：

\[
M(q,g)=\sqrt{\operatorname{mean}_i u_i\;\operatorname{mean}_j v_j},\qquad
L(q,g)=\frac{1}{N_q}\sum_i\max_j c_{ij}.
\]

M 是匹配器产生的整体支持质量统计，不是经过概率校准的“身份正确率”；L 保留完整 reference 的自由内容搜索。二者并不要求共享某个硬指定的 patch 对应。

相对 RAW 原答案 w，定义 d_T(g)=(T_g-T_w)/(|T_g|+|T_w|+epsilon)。最简已验证读出为：

\[
z_g=a\,\delta_{\rm RAW}(g)+b\,d_M(g)+c\,d_L(g)+d.
\]

delta_RAW 使用原自然 C128 内的标准差归一化。比较所有 127 个挑战者；最大 z 大于零才 SWITCH，否则 HOLD 原答案。MASS5 另加入 d_(M×L)，即先对候选计算乘积，再做相对差；不是 d_M×d_L。HOLD 也不是开放集拒识。

任务适配仅使用 query–reference 身份监督，不新增框、mask、像素对应或 ownership 标签。冻结基础模型已有预训练不能被表述为完全无监督。小头参数少也不等于整套含 RoMa 的推理成本低。

### 源域拟合后冻结的外部复核

MASS5 在 H593 的 570 张候选内 TRAIN 样本拟合，外部预测前固定参数；以下各自使用 ColNomic 自然 C128，不在外部重训或选择阈值。MASS5 是预定主臂，加性四参数是对照，不能按结果互换主次。

| 面板 | RAW | 原完整头 | 冻结 MASS5 | 对 RAW 救回/损失 | 等组净增 95% 区间 |
|---|---:|---:|---:|---:|---|
| GroZi480 | 321/480 | 355/480 | 349/480 | 28/0 | +2.403～+10.400 个百分点 |
| ISIC537 实例匹配探索面板 | 466/537 | 500/537 | 506/537 | 40/0 | +3.969～+8.695 个百分点 |

组 bootstrap 分别以 27 个视频和 346 位患者为单位；组等权区间与按 query 计数的净增不是同一个加权量。加性四参数同样取得 349、506。简化头对原完整头一降一升，不能合并成“普遍无损替代”。这些面板此前已使用，是固定方案的外部复核，不是全新未接触确认。ISIC 不是临床诊断实验；RPC 依照项目决定只作诊断材料。

### 各检索器自己的候选池、分别重训的迁移

以下保持 H593 原分组五折；每个检索器自行检索 5413 张图库、产生其自然 C128，并重新拟合头。因此与上表的冻结跨数据集迁移是两类证据。

| 检索器 | C128 目标召回 | RAW | 独立纯内容 CONTENT7-COST1 | MASS5-COST1 | 对 RAW 救回/损失 |
|---|---:|---:|---:|---:|---:|
| ColPali | 467/593 | 283/593 | 286/593 | 348/593 | 65/0 |
| ColQwen-base | 438/593 | 227/593 | 240/593 | 322/593 | 96/1 |

ColQwen-base 是预训练 Qwen2.5-VL-3B-Instruct 骨干加固定、未经过检索训练的投影，不加载检索 LoRA；不是整个模型从未训练。

这些结果支持这个质量接口的适用范围，并排除“只是增加一个当前纯内容小头”对全部收益的解释。它们不证明 ColNomic/ColPali tokens 在信息论上缺少质量，也不证明任何更强的内容读出都无法取代 RoMa。

来源：[外部迁移](REPORT_SIMPLE_EXTERNAL_TRANSFER_V1_20260924.md)、[ColPali](REPORT_COLPALI_NATIVE_MASS_H593_FINAL_20260924.md)、[ColQwen-base](REPORT_COLQWEN_BASE_NATIVE_FINAL_20260924.md)。

## 3. 贡献三：强 VLM 重排之后仍可存在可利用的匹配质量增量

使用同一 H593、ColNomic 自然 C128、冻结 Qwen3-VL-Reranker-2B logits 和原五折。比较同参数量的额外 M 与额外自由内容 L，避免把一般增加模型容量当成机制证据。

| 路径 | CE 正确/593 | COST1 正确/593 |
|---|---:|---:|
| Qwen 直接排序 | 522 | 522（同一个直接排序基线） |
| Qwen 三参数校准 | 523 | 477 |
| Qwen + L，四参数 | 523 | 476 |
| Qwen + M，四参数 | 536 | 504 |
| Qwen + M + L，加性五参数 | 536 | 504 |

CE 是预定主分析：加性五参数相对三参数 24 救/11 损，净增 13；64 组件等权净增 95% 区间为 −1.820～+4.731 个百分点，尚未排除零。不能转而把更好看的次级分析改称主检验。

COST1 是预定次级保守目标：477→504，30 救/3 损，五折净增均为正；等组区间 +2.161～+8.652 个百分点。相对 RAW，原 Qwen-COST1 与质量版均观察到零损失，救回从 51 增至 78；这是本样本结果，不是未来无误伤保证。504 仍低于不加保守校准的直接 Qwen 522。

可以主张：**在已测的保守纠错目标下，配对质量提供了当前强重排分数和附加自由内容标量尚未兑现的增量。** 不能主张全面胜过 Qwen、所有损失下跨组显著，或已证明完整流水线更便宜。后来补充显式乘积的 CE 为 537、COST1 为 504；相对加性 CE 仅净增 1 且组区间跨零，不承担主结论。

来源：[强重排联合校准](../results/rc_h593_qwen_quality_joint_v1/report_zh.md)、[乘积补充](../results/rc_h593_qwen_quality_product_v1/report_zh.md)。

## 4. 新的内部实验：作为有控制的可行性结果

同一 TRAIN16、已打开 PROBE8、ColNomic 自然 C128、128 次 COST1 更新；冻结骨干，在 LLM 后、检索投影前加入由真实 M 条件化的内容适配器。末端头只读取 RAW 与适配后内容证据，不直接读取 M。

- RAW 和 PRE_REAL 均为 4/8；POST_REAL 为 5/8，一次救回、零损失。
- 恒定/错绑 M 的训练与推理控制均为 4/8；真实 POST 的 TRAIN-only 固定适配器重拟合仍为 5/8。
- 外部加性/乘积头为 6/8。内部有效不要求必须超过外部；它回答的是 M 是否能够通过内容表示路径产生实际纠错。

新补的封存内容分数×封存读出头交换，不训练、不调阈值：

| 内容分数来源 | PRE 头 | POST 头 |
|---|---:|---:|
| PRE | 4/8 | 4/8 |
| POST | 5/8 | 5/8 |

唯一新增救回的 cefdinir-fig2，PRE 的 target 相对 RAW winner 内容差为 0.00558，POST 为 0.03225；POST 内容接回 PRE 头后仍正确切换。因此这次成功不能只用读出头参数变化解释。

新增全 24 张的 CPU 机制回放进一步发现：仅保留跨 patch 共同的 M 响应，再固定恒定 M 时各 patch 原命中的 reference token，仍复现真实 POST 全部 24 张身份决策；只保留 patch 零均值剩余响应时为 TRAIN8/16、PROBE4/8，原三次救回都消失。因此在这一封存端点中，共同表示位移及其经过归一化后的余弦变化已足以解释这些纠错，不需要以重新选择匹配位置作为解释。

对 cefdinir，恒定条件的 target 切换分数 −0.61187，真实 M 加固定原命中已经达到 +0.30391，允许重新匹配后为 +0.36021。这些为同一 CPU 路径的差分；与旧 GPU 分数存在小幅数值差，但决策一致。共同 hidden 位移经每个 token 不同的归一化产生不同的余弦变化，不等于统一 scalar 加分，也不等于 LLM attention 改造。具体数据和定义见[机制深析](REPORT_NEW_HYP_MECHANISM_DEPTH_20260924.md)。

进一步只用TRAIN16构建跨query共享的M响应函数d_T(M)，不利用probe拟合该函数：probe8仍5/8，全部8张身份选择与原POST一致；再固定原匹配位置也保持一致。这把本端点所需的M响应进一步简化为TRAIN确定的共享函数。当前query内容底座、投影、归一化与自由内容比较仍保留，不能说模型只依赖scalar M。该结果受本适配器的16维瓶颈和固定条件增益约59.87影响，不能外推所有模型。

这是“质量可通过内容路径产生纠错”的初步受控证据。它不证明更好的语义定位、LLM 内部注意力变化、M 不再只是通过表示承载，或 POST 的一般位置优势；也不证明训练时头更新不必要。8 张反复开发过的 probe 不足以作为主会规模的架构验证。最稳妥的位置是机制探索小节或附录。

来源：[POST 正式报告](REPORT_POSTLLM_M_V1_20260924.md)、[结果解释](REPORT_POSTLLM_M_V1_RESULT_INTERPRETATION_20260924.md)、[交换回放原始结果](../results/rc_postllm_m_v1/head_swap_diagnostic.json)、[交换回放程序](../programs/analyze_rc_postllm_head_swap_v1.py)。

## 5. 与已有工作的区别怎么写

| 已有工作 | 已有思想，必须承认 | 本文具体研究焦点 |
|---|---|---|
| [ELViS](https://arxiv.org/html/2603.28603v1) | 由局部描述子相似度，经 OT 与投票形成可迁移图像相似度 | 对独立匹配器的整体质量与冻结语义检索器的自由内容进行分离干预，检验哪些接口字段及下游操作需要保留 |
| [To Match or Not to Match](https://arxiv.org/html/2504.06116v1) | 匹配可用于置信度验证，重排会伤害原正确结果 | 固定候选竞争下，对质量/内容/接受进行同协议控制，并比较固定头删除与重训恢复、跨检索器和强 VLM 校准的结果 |
| [Reranking Transformers](https://arxiv.org/abs/2103.12236) | 学习局部与全局特征的重排及联合特征优化 | 主体采用冻结内容和匹配骨干、检索身份监督的低维接口；不声称首次学习两图重排 |
| [FiLM](https://arxiv.org/abs/1709.07871) | 条件信息可调制网络特征 | POST 仅探索候选绑定的匹配质量能否通过检索内容路径产生纠错；不声称通用条件化结构首次提出 |

上述是贡献定位，不是已经完成这些论文官方模型的同协议比较。不可把本地加性/乘积/置信度规则控制标成 ELViS 或 To Match 的正式复现。

## 6. 可直接用于 Introduction 的贡献草稿

1. **机制发现。** 我们通过候选绑定干预、算子分解及固定头/重训配对消融，区分密集匹配提供的整体支持质量与局部重加权的作用。在所测协议中，保留候选绑定的质量并重新校准，可在去除局部权重后恢复原系统的总体纠错收益；显式质量—内容乘积也并非必要条件。
2. **简化接口与迁移。** 我们构建仅用检索身份监督训练的质量—自由内容联合校准接口，并分别检验源域参数冻结的外部迁移，以及多个冻结检索器在各自自然候选池上的重拟合。结果支持接口的可迁移用途，而无需对任务新增空间标注。
3. **强重排互补与边界。** 我们在同候选、同折、同损失和同参数量对照下，检验匹配质量对强 VLM 重排的增量；结果显示保守校准中的稳定净增，并同时披露 CE 主分析的组间不确定性、直接重排的更高绝对准确率及质量错误绑定的失败。

English working version:

> We investigate which information dense matching contributes to frozen visual retrieval. Through candidate-binding interventions and paired fixed-head/refitted operator ablations, we find that candidate-conditioned aggregate quality can retain the correction gains of local reweighting after recalibration, without requiring explicit quality–content multiplication in the tested protocol. We operationalize this finding through a compact, retrieval-supervised interface between matching quality, unrestricted content matching, and challenger acceptance, and evaluate it through source-frozen cross-dataset transfer and retrained cross-retriever tests. Finally, controlled comparisons with a strong vision-language reranker identify additional value under a conservative correction objective, while exposing the statistical and operating-point limits of that complementarity.

建议标题沿用：**What Does Dense Matching Add to Frozen Visual Retrieval?**

副标题可用：**Candidate-Conditioned Quality Beyond Local Reweighting**。

主文应围绕“哪些匹配信息值得保留，以及它们怎样改变冻结检索器的决策”展开。当前最强定位是机制研究加简化方法；内部改造还不宜抢占主线。开发集反复使用、缺少最近方法官方同协议比较，以及未完成含 RoMa 的端到端成本对照，仍是实际投稿风险。充分披露这些边界不会消除风险，但可以防止把已有证据写成它尚未支持的结论。
