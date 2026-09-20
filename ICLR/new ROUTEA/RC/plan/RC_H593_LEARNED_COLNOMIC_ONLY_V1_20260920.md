# H593 learned ColNomic-only correction

2026-09-20 用户授权“补”。本轮新增实验独立于已提交的六项消融，不改旧 authority、结果或部署头。

## 问题和比较

检验原完整 COST1/CE 的收益是否超过同样经过检索监督训练的纯内容纠错头。RAW2_CE 只有 RAW gap+bias，不能代替丰富内容对照；历史 FREE/dF 仍含 RoMa query 权重，本轮禁止使用。

人口、划分和训练沿用 H593 原 593 图、68 identities、64 components、grouped OOF5、自然 RAW C128 和全部127 challenger。23张 target-absent 仍在评测分母；仅在训练中按原规则排除无定义正项。监督只有 query-reference identity，无框、mask、空间标签或新数据。

## 在观察新结果前固定的内容表示

只读取已封存的原 ColNomic image tokens 和 RAW 分数/候选轴。去掉全零 padding（token L1范数>1e-6），不做新的归一化；FP16缓存提升至FP64后，矩阵 A=Q R^T。全部 query/reference tokens 参与，无坐标、RoMa 权重、置信度、可见性或其派生量。

每个候选五项内容统计固定为：

1. F：query→reference，每行最大值的均值，普通 full-reference MaxSim。
2. B：reference→query，每列最大值的均值，反向内容覆盖。
3. Gq：每行最大与第二大值之差的均值，内容匹配峰值间隙。
4. Gr：每列最大与第二大值之差的均值，反向峰值间隙。
5. U：完整 A 的均值，整体内容相似度背景。

Gq/Gr 不称为身份置信度，也不使用正确 reference 来定义。重复 token 的并列最大值给出零间隙。每侧须至少两个 active tokens，否则停止而不静默换公式。

六输入 CONTENT7=[RAW gap/std_C128, sym(Fc,Fw), sym(Bc,Bw), sym(Gqc,Gqw), sym(Grc,Grw), sym(Uc,Uw)]，sym(a,b)=(a-b)/(|a|+|b|+1e-12)。六权重+bias，与原头七参数相同。MAXSIM3只保留前两输入，作为简单伴随对照；不按结果选择两者中较弱者。另重训 RAW2_CE 作为原 RAW 特征和训练实现逐位一致性检查。

## 学习、读出与验证

CONTENT7 和 MAXSIM3 各使用 COST1、CE；COST1_CONTENT7 为主要内容基线。直接复用原训练函数：FP64，零初始化，seed17，AdamW lr=.03、weight_decay=.001，2000完整batch updates，8 CPU threads。每折训练顺序必须和旧封存头相同；不搜索特征、阈值、checkpoint或超参。

全部127分数最大值>0才SWITCH，否则HOLD=0；同分选canonical physical轴首项。20个内容头+5个RAW2复现头，各在新进程重训重放一次，共100000优化更新。原完整头的封存预测仅在新五折封存完成后读取，用于同查询配对比较。

提取按75个原token来源分片，每片最多8图；先单片资格检查，成功后继续余下74片。实际GPU仅用于FP64 token相似度，不重新执行编码器或RoMa。每一query×candidate的五项统计均以NumPy CPU独立计算复核；提取进程禁止读取任何RoMa payload、query身份和历史结果。合成预检覆盖负分数、并列最大、padding、变长token、token置换、HOLD和tie。每折重训参数和全部heldout logits精确重放，NumPy独立核算读出；RAW2_CE参数与全部127分数须复现旧封存。

## 报告和结论边界

报告RAW、普通FP64 PATCH_MAXSIM、RAW2_CE、原完整COST1/CE、四个新内容头的正确数、MRR、对RAW和对应完整头的rescue/break、动作变化、每折值；报告64 components等权差和10000次bootstrap95%（seed20260920）。主比较为 FULL_COST1−COST1_CONTENT7；CE、MAXSIM3为预定次比较。原RAW/COST1/CE计数须分别复现426/481/486；C128 recall须为570/593。

这是已开放H593上的新增开发对照，不作为新外部确认。若完整头领先，只支持相对于这些固定内容基线的增量价值，不声称所有ColNomic-only模型都不可能替代。接近、落后和分组不稳定均如实报告。无自动部署更换，无ownership结论。

输出逐候选分数、参数、逐图决策、原始五统计、验证、耗时和中文报告。任务使用accelerated、10分钟上限；内容分片最多16并发，训练五折并行。最终与六项消融合并成新的汇总文档和ZIP，保留旧版；同步GitHub时排除原图、tokens大缓存和日志。
