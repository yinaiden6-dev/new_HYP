# 当前检索系统的关系绑定贡献：最后一次 H 验证

2026-09-09。用户明确当前主线未使用 superregion，提出将 reference
HYP 的定义从空间区域改为关系；同时要求最后一次验证后停止方法迭代，
若不成立则总结已有结果、准备论文。本方案固定这一最后检验。

关系只是用户提出的候选解释，不是预设答案。已有质量与内容共同校准
的解释保留；本对照检验细粒度绑定是否在此之外贡献识别。按证据选择
解释，不为保留HYP名称改变结果。理论写作可采用模型无关符号，但
实验必须披露RoMa/ColNomic实际实现，不将单一组合外推为普遍定律。

## 命题和证据层级

旧空间命题：reference 产生连通多 patch 区域，并由独立 V 验证。
旧 P-only、superregion、P0 的结果和门保持原状态，本试验不声称通过它们。

新命题：在当前 RAW/RoMa/ColNomic 冻结评分系统中，candidate-conditioned
visibility 与内容证据的正确绑定，提供了超出相同全局质量和 visibility
边缘值分布的识别贡献。

这里的 H 是现有评分图诱导的 query-reference 支持关系，包含内容匹配
和 visibility 对该内容的绑定。它不是新增 H 网络，也不是已独立训练的
P→V 模型。ColNomic MaxSim 本身产生的匹配不能再冒充独立验证其自身的
新证据。新结论的最高层级为已打开 EVAL32 上的关系绑定机制证据。
不将任意 embedding/scorer 改名 HYP 当作额外理论或模型创新。
原 action 的 HOLD 表示保留 RAW winner，不表示某个 reference 缺席；
本试验不新增或声称验证了 candidate-absence H0。

## 固定模型和数据

原 RAW FULL64：TRAIN32、EVAL32，原自然 C128、全部127 challengers、
原 query/reference image tokens、完整原 wq/wr、原 RAW prior。主头为
已冻结 NATIVE7/C_PAIRED；旧 FROZEN_C 单列为旧27/32系统参照。两头不
合并指标，不择优宣称主结果。原 difficult90 的69/90只引用已验证历史
结果，不声称本次对90条进行了新关系干预。

本次评分使用原成功系统的 image-reference、visibility-weighted MaxSim
计算图；新 a_RFULL 含 template 的无权重基线属于独立接口恢复结果，
不暗中接入当前 w_r 计算图或混称同一 scorer。

零训练、零新 encoder/RoMa/SAM forward、无新 backbone、无外部区域或
人工空间监督、无 superregion/max-tree 选择器、无 V9、无 ownership。
不接触 D1-MI、GroZi 或未授权正式392结果。

## 一个固定的关系破坏算子

内容 tokens 和身份候选始终不变。只改变 visibility 值与其原内容索引
的对应，保留每候选 wq/wr 的完整值多重集。四条件一次性冻结：

1. REAL：原 wq、原 wr。
2. Q_BIND_DERANGED：置换 wq 的内容绑定，wr 原样。
3. R_BIND_DERANGED：wq 原样，置换 wr 的内容绑定。
4. QR_BIND_DERANGED：同时执行上述两个置换；这是唯一主破坏对照。

置换 target-free、无固定点、非仿射，seed 固定17，不扫描或重挑。
query 置换由 query source/token SHA 与 grid 确定，在该 query 全部候选
中相同；reference 置换由 reference source/token SHA 与 grid 确定，
在所有引用同一资源的 query 中相同。全部 destination→source 索引和
算法/code SHA 在标签读取前封存。候选重排不得改变资源的置换或分数。

这是输入绑定干预，不 shuffle 最终分数，不替换 candidate reference，
不只中和 head 的某个坐标项。每条件重新计算 weighted MaxSim、原四
标量、六维 feature 和完整127个 action logits。Q-only/R-only 只用于
机制分解，不能在 QR 主比较不通过时择优替代它。

操作次序固定：先得到条件 wq'=perm_q(wq)、wr'=perm_r(wr)，再执行
旧辅助 Q/R 通道的 roll(wq') 或 roll(wr')。必须是 roll(perm(w))，
不能改成一般不与之相等的 perm(roll(w))。REAL 使用 identity perm。

## 精确保留原质量与计算边界

原 verify_visibility 会分别计算 REAL/Q/R 辅助通道的质量；浮点重排
可能使各自 mean/sum 出现末位差。故先从原输入分别冻结各通道的
M_REAL、M_Q、M_R 及相应 query denominator D_REAL、D_Q、D_R。
在四条件中，各通道均使用自己相同的原质量因子和分母；对外 head 的
visibility_mass 始终为原 M_REAL。只改变相应 numerator 的内容绑定。
这是明确的因子控制定义，不把三个略有数值差异的原质量强行混为一个。

原 REAL 必须逐 bit 回归原四标量、原全部 action logits 和最终预测。
query/reference 权重多重集、所有固定 M/D、RAW/C128、内容 tokens
和无权重 ALL MaxSim 全部不变。派生 weighted MaxSim、features/logits
可以变化，不能声称这些派生量也保持不变。

## 冻结与判定

先冻结代码、plan、输入和参数 SHA、置换算法和主比较。全部 FULL64
四条件评分与预测封存后，再读取原 role/gallery identity。目标使用
所有同 exact-identity 的 C128 members；wrong 使用其余完整候选。
分别报告最终 action correctness、严格 target-minus-strongest-wrong
margin（精确 tie 失败）、paired rescue/break/net，以及原 supergroup
等权方向。TRAIN32仅为已用训练群体的描述，EVAL32为内部主检验。

margin 唯一采用 action 证据轴：原 RAW winner 的 z=0，其余127候选的
z 为对应旧头完整 challenger logit；m=max_target_equivalent(z)−
max_wrong(z)。每条件使用自己的 strongest wrong。不得改用 local S
margin 择优判门。最终 action 在最大 logit=0 时 HOLD，可能最终预测
正确而 strict margin=0；两者分别计数，net4按实际最终决策，margin
正降幅按上述唯一 m。精确 tie 不计 strict win 或正降幅。

本新命题预设的内部支持条件为全部满足：

- 工程原路径、保值和来源检查全部通过；
- NATIVE7/REAL 的原 EVAL32 识别成绩逐条保持，且相对原 RAW 的既有
  增益保留；
- 在 EVAL32，REAL 相对 QR_BIND_DERANGED 的最终识别 paired net>=4，
  rescue>break；
- REAL 对 QR 的严格 target-minus-strongest-wrong margin 正降幅
  至少21/32，且等权 supergroup 的 REAL-control margin 差>0。

这些是本关系干预的预设内部效应门，不能与旧 P-only 对 ALL/query-only
的资格门混为一谈。旧门未通过的结果不变。完整数字、失败项和各组方向
全部报告，不用 mean margin 单独替代识别贡献，不因个例失败否决整体。
32条包含重复身份视图，报告11个 EVAL supergroups；不当成32个独立身份
或新的外部确认，也不以二次参数/seed/对照选择制造支持。

若 REAL 与破坏对照保留相同正确集合和救回，或上述主门不满足，则返回
RELATIONAL_BINDING_CONTRIBUTION_NOT_ESTABLISHED_INTERNAL，不推出 H
普遍不存在。若满足则仅返回 RELATIONAL_BINDING_CONTRIBUTION_SUPPORTED_INTERNAL，
明确这是既有 weighted-MaxSim/action 的机制证据，不是新 H 网络或旧
空间 HYP 的 GO。完成独立算术/输入复核后停止追加模型，整理论文结论。
