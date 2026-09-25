# 无 RoMa 联合关系读出：首轮执行协议

2026-09-24 用户“做”授权。承接 `RC_COLNOMIC_ROMA_FREE_RELATION_READOUT_PROPOSAL_20260924.md`，本轮只实施首轮机制试验，不启动全593或新的编码器采集。

## 数据与监督

- 原H593 fold0，474 TRAIN、119 held的既有身份/来源分组不变。
- 在读结果前，按 `SHA256(relation-v1/17/role/query_id)` 分别选64 TRAIN和32 held；不按原正确/错误、救回或目标是否在池选择。
- 原 ColNomic 自然 C128，完整128候选、自由内容 L0、RAW winner保持原始来源。所有query保留；TRAIN目标缺失时只抑制错误切换，统计时不能把HOLD算正确。
- 训练仅访问64 TRAIN的身份标签，封存所有臂预测后才由独立汇总打开held标签。该fold已有历史查看，只称开发组留出。
- 只读既有 ColNomic 原始patch tokens。输入来自早期融合缓存中的纯 `inputs/*.pt`，不是其RoMa粗细特征包。原生patch网格逐项校验，不新增池化或删token；几何邻域在该原生网格定义。
- intake从混合历史包复制候选轴和自由内容，剔除全部RoMa字段。训练worker文件访问限制只允许新分支和纯ColNomic输入，不能读取教师或held身份结果。

## 四臂与共同末端

BASE：原 RAW/L0三参数校准头。

COMPRESSED：复用历史520→16→1逐token算子结构、8353参数；保留双向soft-context、坐标期望、熵与回返质量，取 `log(sqrt(mean(u)mean(v))) + log(2)` 为残差。训练精度和末端依本协议，不冒充复现旧2000步PRODUCT5结果。

SET / SPATIAL：同一4→8、两层32→8残差、8→1结构，各577参数。输入完整矩阵上的cosine、双向log相对softmax概率、双向概率几何均值。每层读取当前对应状态、query侧邻域均值、reference侧邻域均值及双侧均值；SPATIAL使用含中心3×3格点邻域，SET使用完整集合均值。局部邻域只是软结构先验，不设几何通过门。所有软对应保留到可学习层后，再双向加权聚合成残差R。

共同logit：

`z(g) = w_raw * standardized_RAW_gap(g) + w_L * sym(Lg,Lwinner) + bias + Rg - Rwinner`。

BASE的R为零。其他臂的末层零初始化，初始行为与BASE一致。所有127challenger取最高正logit才SWITCH，否则HOLD。末端没有M、M×L或任何RoMa输入。基线头FP64、关系网络FP32；不同于旧读出的全FP64，不宣称bit-exact旧训练。

SPATIAL/SET是主要等容量隔离。COMPRESSED参数更多，仅为历史类型的同协议对照，不称四臂等参数。三参数共同头先用全部64 TRAIN做1000步AdamW拟合，再所有臂按同序8遍/512 query updates训练；COST1、seed17、reader lr=.001、head lr=.003、weight decay=.001、梯度范数clip=1。完整C128先前向，只有COST1非零VJP候选再次反传，独立检查与普通autograd等价。

## 预定读出和干预

最终固定512步端点，不按held挑选checkpoint。全部64 TRAIN诊断及32 held native预测保存128个R和L、127logit及HOLD。

SET/SPATIAL额外对全部32 held执行坐标绑定打乱、双图各自180度一致旋转对照。排列操作保持相似度和两方向softmax数值集合，最终聚合还原轴。SET原则上置换不变；SPATIAL对一致旋转保持、对错误绑定可改变。干预影响不能单独排除分布外退化。

固定held列表前2个query保存RAW winner与候选0的逐层关系状态、完整soft矩阵和边证据；所有其他中间量可由已绑定tokens、模型和程序恢复，不按结果挑图。

主要比较SPATIAL−SET，其次相对BASE与COMPRESSED，记录候选证据方向、训练拟合、rescue/break和按component的探索性区间。原RoMa+COST1同面板结果只作系统参照，训练规模和架构不同。首轮无可靠增益只能限制本配置，不能证明ColNomic缺信息。

## 执行及继续条件

先合成检验：独立NumPy邻域均值、坐标等变、非零结构响应、全图/分候选梯度、真实优化器断点恢复。随后对一个固定TRAIN query执行四臂完整C128无真实标签容量测试，核对原L0；临时模型丢弃。

容量上限每update90秒、进程RSS24GiB。通过后BASE完成，再自动提交COMPRESSED、SET、SPATIAL三项10分钟CPU任务，各8CPU/32GB，dev_cpuonly与cpuonly共同排队。逐query保存断点，400秒停止领新工作，同Job最多64次requeue；不改旧实验任务。

所有臂预测封存后自动CPU汇总，用独立NumPy重算logit和动作；汇总不重新训练。若实际容量不通过，停止扩展并保留测量，不隐式减候选、删token或换loss。
