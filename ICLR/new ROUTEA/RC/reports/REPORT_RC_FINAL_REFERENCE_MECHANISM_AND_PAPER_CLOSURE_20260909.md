# Reference 主线的最终机制解释与论文收口

> 命名与状态更新：当前框架统一称 **new HYP**，候选证据状态记E_g；
> 旧 **Reference HYP** 保留其空间命题含义。历史结果不变。此前停止
> 安排已被用户延长至北京时间2026-09-11 24:00。

当前结果支持参考条件化的联合证据校准。最后一次细粒度绑定检验未
达到预设门；该结论不否定reference检索机制，不否定已有识别增益，
也不否定“质量—内容联合关系”这种候选层面的解释。

## 必须分开的两个命题

**候选级联合证据：** 对同一个query，针对每个reference生成匹配
质量M与内容相容性L等统计，由冻结决策函数与RAW先验一起决定纠错。
旧成绩、逐项logit重放和M/L因子干预支持这一计算机制。可以用H表示
该联合证据状态，但这是对当前计算图的形式化，不是新增独立H网络。

**细粒度绑定贡献：** 在M、visibility值分布、内容tokens、RAW/C128
与head均相同时，正确的visibility—内容索引绑定是否提供足够群体
识别优势。最后一次试验只检验这个较强、较窄的命题，结果未建立。

两个命题不是同一个门。不能把后者未成立写成“reference完全失败”
或“所有关系H都不存在”；也不能将前者已有支持写成旧空间P-only门
通过。当前主线没有superregion，原空间H结论保持原状态。

## 最后一次检验的完整结果

固定原RAW FULL64、自然C128、全部127 challengers、两组原冻结头、
原image-reference weighted-MaxSim与零阈值。各原REAL/Q/R通道的M
和分母分别保持不变，只置换visibility与内容的绑定。所有预测先
封存，再join标签。64条、98304次literal标量检查及512个最终决策
的独立算术复核全部通过。

| 数据/冻结头 | RAW | REAL | Q绑定打乱 | R绑定打乱 | QR同时打乱 |
|---|---:|---:|---:|---:|---:|
| EVAL32 / NATIVE7 C | 25 | 28 | 27 | 27 | 27 |
| EVAL32 / FROZEN_C | 25 | 27 | 28 | 27 | 26 |
| TRAIN32 / NATIVE7 C | 27 | 28 | 28 | 29 | 29 |
| TRAIN32 / FROZEN_C | 27 | 28 | 28 | 29 | 29 |

EVAL32已打开，含11个supergroups；TRAIN32含12组且已用于训练，仅
描述。两头分开。任何控制更高的数字不作为新部署模型或择优结果。

事先固定的主比较为EVAL32/NATIVE7 REAL对QR：1救、0损，净增+1，
低于+4；action margin严格正降幅13/32，低于21/32；等权supergroup
平均降幅−0.005918040870658541，未满足正方向。三个主效应门未通过。
最终状态为RELATIONAL_BINDING_CONTRIBUTION_NOT_ESTABLISHED_INTERNAL。

NATIVE7原本对RAW的三次纠错（DIFFICULT-0044、OUTCOME-0220、
OUTCOME-0212）在QR打乱后3/3保留。FROZEN_C原来的两次纠错
（DIFFICULT-0044、OUTCOME-0212）也2/2保留。REAL相对QR多出的一个
正确均为OUTCOME-0618：它原本已被RAW识别正确，打乱绑定造成破坏。
因此该+1是这个样本的防破坏作用，不是原2/3次纠错依赖细粒度绑定。

## 已有增益及其计算解释

原FROZEN_C、RAW C128、current-runtime EVAL32为25→27（2救0损）；
同一头在opened difficult90为61→69（8救0损）。另一头NATIVE7在
matched EVAL32为25→28（3救0损），不能把28/32和前一头69/90合并。

在原FROZEN_C的difficult90上，保持L/Lq/Lr而统一M=1，会从69回到61，
八次原纠错全部消失；中和归一化内容特征也丢失原八次纠错。完整
逐项重放显示质量与归一化内容统计共同提供原纠错所需的正贡献。
这支持固定计算图内的联合校准机制，不单独证明语义内容的独立必要性、
跨模型规律、超加性的融合收益或唯一根因。

结合最后绑定检验，当前更符合证据的落点是：候选层面的总体匹配
质量与内容统计共同校准RAW决策。精确细粒度绑定有局部作用，但没有
解释这组EVAL的主要纠错收益。无须用新的区域或关系网络才能表述该
已有reference机制；也不应借重新命名宣称未实施的新模型创新。

## 论文主张与停止边界

论文主线：Reference-conditioned Joint Evidence Calibration for
Fine-grained Retrieval。理论采用模型无关的质量通道G、内容通道C
和决策函数A，实验披露实际RoMa/ColNomic实现。区分可证明的接口信息
不可恢复性质、当前组合的实验发现与尚未验证的跨模型推广。

没有证明进一步提升不可能，也没有证明普遍的“过目不忘”理论。
停止新增试验是用户约定的本轮收口，不是不可提升的上界证明。
此后只整理论文，不追加模型、seed、阈值、区域方案、V9或ownership。
原成绩与失败均保留；不改写冻结协议或旧P-only结论。

## 证据

- results/rc_reference_relation_final_v1/result.json：
  6f812de052c4644626cfa33e2e1ff91b7f89962005fdc04e4cd13762240445b3
- independent_arithmetic_validation.json：
  04d884c66127a127bd56d32e4d1d0abff74a18e198c1ad901b8a280f47832418
- source_arithmetic_validation.json：
  1e9b5fcfa14374fe2113144fa31d5d824bac282ffc8b48408db373d672a3078f
- prejoin_seal.json：
  3529260315a1e4d75302bbdc3f530accf4becd28ad6aab1f907c6fd9d0789898
- reports/REPORT_RC_ORIGINAL_ROMA_SUCCESS_AND_P_INFORMATION_LOSS_20260909.md
- reports/RC_RETRIEVAL_ONLY_PAPER_WORKING_DRAFT_20260909.md
