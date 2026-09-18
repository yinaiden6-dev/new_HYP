> 2026-09-11历史核查更正：下述两方向在旧RCDE/C6d及V7/V8中都有明确设计、实现或训练前例，撤回“此前未探索的新方向”的判断。本稿仅保留为旧机制在当前输入上的复验候选，不作为新科学机制或立即启动实验的依据。详见[历史核查更正](../reports/NOTE_NEW_HYP_MATCH_DIRECTIONS_HISTORY_CORRECTION_20260911.md)。

# new HYP：匹配证据的信息保留与重复计票

本稿是新方向筛选，不是已验证理论或新性能结果。当前原固定头为28/32、99/128；593五折的GROUP_BASE447来自不同训练协议。已有证据支持原reference条件质量—内容机制；组均衡、旧候选绑定和旧27→28不能自动归为new HYP的新机制贡献。本轮未提交新训练任务。

## 两个尚未被近期标量头实验覆盖的问题

1. **query轴：哪些局部在支持、哪些局部在反对同一个候选。**
   现有J先在相同query支持上池化a_h，再选最强完整reference h，头只接收候选级标量。保留逐query-token的有符号竞争证据、可靠性及它们的联合分布，再由共享检索监督读出，与重复加入J、端点符号或正token attention不同。正证据和反证都需保留；不能把每个token分别取不同wrong reference拼成一个虚构的完整对手。
2. **reference轴：多个query局部是否反复读取同一块reference内容。**
   MaxSim每行独立选列，不限制列被多少行使用。软容量或依赖惩罚可以保留完整reference搜索，同时限制重复计票。它应允许未匹配和尺度差异，不能回到RoMa单点硬对应，也不能强迫所有reference背景都被匹配。

推荐先检验第二项的可定位机制，同时把第一项作为区分“完整矩阵缺失”与“仅query分布缺失”的对照。两项对应不同信息：逐token最大值相同，不意味着reference使用关系相同。

## 可严格证明的表示缺口（仅构造例）

取同一组两个单位query向量q1=e1,q2=e2。两组reference可产生下列合法余弦矩阵：

    C_A = [[0.7, 0.1],     C_B = [[0.7, 0.1],
           [0.1, 0.7]]            [0.7, 0.1]]

A的列可取(.7,.1,sqrt(.5))、(.1,.7,sqrt(.5))；B的列可取(.7,.7,sqrt(.02))、(.1,.1,sqrt(.98))，均为三维单位向量，因此不是任意不可实现的余弦矩阵。

两者每行MaxSim都是.7，平均MaxSim都是.7，矩阵均值都是.4。在共同均匀visibility下，相应的原内容/质量摘要不能区分两者。但A的两个高匹配使用不同reference局部，B的两个高匹配重复使用同一局部。

设每行质量a=(.5,.5)。普通加权MaxSim等价于

    max_{P>=0, P1=a} <P,C>.

加入两列各.5的容量上界并保留总行质量后，A最优得分.7，B最优得分.4。证明：B第一列最多贡献.5*.7，其余.5质量只能匹配第二列的.1；A能各用一列获得.7。

由此只能得出：逐行最大值映射不是单射，一些匹配互补性信息在池化后不可恢复。它没有证明自然数据存在同样碰撞、B一定wrong、A一定target，或容量约束必然提高准确率。真实六维精确碰撞此前未找到，不能用构造例替代真实瓶颈证据。

## 已有实验排除了什么

- SPECIFIC8、J相对端点和ENDPOINT2已经做过，不能把再加J或正负号称为新方向。
- 121个在场target都有正独占token，但wrong也有14362/16263；只找一个正局部不够。
- 支持容量pilot里502/508个wrong也有正容量；不能把“更多有利支持”当身份充分条件。
- 普通MaxSim本来就对复制完全相同的reference token不变；新问题是query重复计票/匹配集中度，不是声称首创reference复制不变性。
- 历史RGH已有soft assignment/reliability/dustbin。新方案不能只换这个名字；与其区别应是完整内容矩阵上的信息保留和重复证据约束。

本地来源：
- programs/cache_rc_h593_endpoint_competition_v1.py（B/F/J先池化路径）
- reports/REPORT_OPENED_EVIDENCE_LOSS_AND_SUMMARY256_V1_20260911.md（仅诊断，不作为训练输入）
- reports/REPORT_RC_REFERENCE_SUPPORT_MAXMIN_TRAIN_PILOT_RESULT_REVIEW_V1_20260910.md
- reports/REPORT_H593_ENDPOINT_SIGN_OOF5_RESULT_V1_20260911.md

## 最小可证伪实验应回答的问题

**预测A（自然前提）**：在TRAIN完整C128中，原头的强wrong相对target，是否更依赖少量reference token的反复使用？以query token权重、有效token数、已有内容分数为匹配条件，不能只看裸argmax次数；真实尺度/重复文字也可能让target集中。若此现象不成立，停止用它解释主要错误。

**预测B（模型增量）**：冻结原头与全C128动作，给同样参数预算、同训练身份/目标的附加读出分别提供：逐query分布、reference使用/软容量信息、普通标量补偿。新方法必须相对对应强基头产生观察净增；只超过弱基线或仅抬margin不称突破。旧32/128用原PAIR+FULL协议；593如开展应保持其固定OOF协议及强对照，分别报告。

**预测C（机制归因）**：用保留旧rowwise摘要而破坏reference跨行使用关系的计算干预，检验收益是否依赖新增关系。此为匹配矩阵层干预，不伪称自然图像或pixel/ownership因果；要同时检查旧score确实保持不变。

所有正常路径保留完整reference可选位置、原自然C128、不插入target。学习只用既有query-reference检索标签；不添加框、mask、OCR或对应点监督。额外成本在未池化匹配统计/容量求解，旧token与visibility缓存可部分复用，但原PAIR特定候选缓存覆盖需在具体执行前核实。

## 文献与原创性边界

- [Generalized Max Pooling, CVPR2014](https://arxiv.org/abs/1406.0312)：讨论频繁局部描述子对聚合的影响；不等于已经证明我们的MaxSim发生同一错误。
- [DeepEMD, CVPR2020及后续版本](https://arxiv.org/abs/2003.06777)：用局部匹配流和cross-reference权重学习图像相似性，已有图像标签监督路线。EMD和cross-reference weighting不是我们的原创。
- [AMES, ECCV2024](https://arxiv.org/abs/2408.03282)：实例检索中学习图内/图间局部描述子交互。仅替换为transformer matcher属于已有架构路线，不能自动形成新理论。

可能的贡献是：定位相似实例中哪些关系信息被当前聚合丢失，提出保留该信息的检索监督机制，并验证其特有增量与失败边界。数学构造、已有方法名称和自然准确率突破必须分开报告；不保证只剩一个理论补丁即可完成论文。
