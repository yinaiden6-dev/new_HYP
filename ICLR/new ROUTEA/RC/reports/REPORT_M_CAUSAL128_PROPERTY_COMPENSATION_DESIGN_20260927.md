# 具体性质删除后允许补偿：缓存内固定配对诊断

日期：2026-09-27。状态：设计已冻结，执行与验收另存。本实验补充四步归因中的“具体性质受损后，允许下游同族读出重新拟合能否补偿”，不重复已经完成的整条 J/P 入口删除，不新增 QRR、编码器、RoMa 前向或 GPU 计算。

冻结设计：`results/rc_m_causal128_attribution_chain_v1/property_compensation_fixed_pairs_v1/protocol.json`，SHA256 `1dc4feee484411b9e468cca1f906827b62fc4af95e525465946e251b69d36df1`。在读取新9组相位干预的 M 值之前冻结；代码与本文件另以 `execution_seal.json` 绑定。

## 问题与可回答范围

固定 RoMa 预测器和既有 A/J 输入，把 P 的相位相对结构或幅度与 patch 的绑定受控改变后，原校准是否受损？只允许同一种标量证据读出重新拟合，能补回多少？这个设计检验具体计算干预后的**有界读出补偿**，不证明信息绝对不可替代，不重训 RoMa 预测器。

现有缓存无法在新9组构造所有干预的完整 C128。故本轮严格使用原先冻结的 target 与自由内容最强 wrong 两候选，报告排序分差和损失，**不报告完整 C128 准确率、HOLD/SWITCH、target-free 部署或外部独立确认**。

## 数据、身份与隔离

- TRAIN：旧恢复实验中 target 存在的7图／7个身份 component，索引 `0,1,2,4,5,6,8`。第7号 query 的 target 不在候选内，不进入两候选训练。
- TEST：原 F71 机制开发之外的9组，每组最小序号一图，索引 `78,82,93,96,100,101,104,108,118`。
- 训练与测试 component 不相交；这16组均属于历史已开发的 H593，不能称全新数据。
- 每query沿用已冻结的两个物理图库行和身份。旧 `strongest_position` 名称可能造成歧义，但历史代码和全部7图逐项核对都确认它是**原自由内容最强 wrong**；与新组规则一致。
- RAW、自由内容 L、自然 C128、query和reference身份均逐项和主128缓存对应。自由内容没有原生 u/v 权重；干预M不会旁路读入未干预M。

## 四个性质单元

| 单元 | 幅度在 patch 上的分配 | 相位变化 |
|---|---|---|
| AMP_NATIVE_GLOBAL | 原分配 | 所有patch一致平移 |
| AMP_NATIVE_LOCAL | 原分配 | patch间正负平移 |
| AMP_PERMUTED_GLOBAL | 固定置换 | 所有patch一致平移 |
| AMP_PERMUTED_LOCAL | 固定置换 | patch间正负平移 |

每格都使用 X+/X−/Y+/Y− 四个预定重复。四个重复平均构成一个query／组的损失，不能作为四张独立图片。GLOBAL保留P之间的相对内积结构，但仍为旋转后的编码，**不是原生 NATIVE**。原始无平移NATIVE与单独AMP_PERMUTE不加入本轮拟合，以保持等扰动性质比较，不根据结果选择额外臂。

## 同族读出与训练

对固定候选 a、b，令 `sym(x,y)=(x-y)/(|x|+|y|+1e-12)`。输入为：

1. `(RAW_a−RAW_b)/std(RAW_C128)`，std为该query128个原始RAW分数的总体标准差；此量与缓存原RAW标准化差的差值核对一致。
2. `sym(M_a Lfree_a, M_b Lfree_b)`。
3. `sym(M_a, M_b)`。
4. `sym(Lfree_a, Lfree_b)`。

模型为四维线性反对称分差，**无偏置、无HOLD、无candidate序号或queryID输入**。交换候选必须使输出精确反号。监督只在排序损失中指定哪一方为正确身份；并不把“target角色”作为特征。

尺度为 TRAIN 的 AMP_NATIVE_GLOBAL 格中7组×4重复的每列RMS，不中心化；同一尺度用于全部性质格。RMS为0时按预定规则设1。这样所有格使用相同正则单位，不因单元分别标准化改变比较。

五个模型：四格各自拟合一个四维读出，再拟合只读取第1、4列的 RAW/L_ONLY 两维对照。损失为各组等权的 `softplus(−正确减错误分差)` 加 `0.001/2 × ||theta||²`。这是为补偿诊断定义的严格凸排序目标，**不是原COST1训练协议**。

采用FP64确定性阻尼Newton，零初始化，最多100次迭代，梯度无穷范数阈值1e-10；解析Hessian，固定Armijo 1e-4、回溯系数0.5、最多60次。收敛后独立标量循环核算梯度、Hessian最小特征值和目标值。L2确保Hessian最小特征值至少lambda，并给出最优目标差上界 `||grad||²/(2lambda)`。这排除本目标未充分数值优化的借口，**不排除数据不足或读出模型族不足**。

不选择lambda、不调阈值、不选轴／符号、不选有利样本、不按TEST早停。全部五个TRAIN拟合、参数、RMS与最优性证书先封存，随后才允许读取新9组干预M。

## 比较与统计

AMP_NATIVE_GLOBAL模型固定应用于四格，得到损伤；四格各自重拟合模型应用于各自TEST格，得到允许补偿后的结果。RAW/L_ONLY在全部格必须完全不变，否则判为实现错误。

主端点是新9个组分别先平均四重复的逻辑排序损失；报告三个预定对比：

- 冻结损伤：该格冻结读出的损失减GLOBAL格冻结读出损失。
- 重拟合补偿：该格冻结读出损失减该格重拟合损失。
- 剩余差：该格重拟合损失减GLOBAL格冻结读出损失。

另报各组均值／最小分差、均值分差正负组数、每组四重复正分差比例、TRAIN拟合程度及与RAW/L_ONLY的损失差。组bootstrap固定seed20260927，10000次，保留全部组值。正／负结论依据这些连续端点与不确定性，不伪装成大面板准确率。

“重训补回”仅表示这一对照下当前证据与当前读出可以补偿；损失差区间含0不能证明等价。若TRAIN优化已收敛但TEST仍差，需在模型族与跨组泛化限制下表述，不能提升为信息不可替代证书。

## 缓存、运行与输出

读取旧7组封存恢复结果、主128 RAW／自由内容清单及新9组完成后的验收结果；不加载token、模型或图像。新前向次数为0。五个四维／二维Newton拟合与验收预期远低于120秒，硬预算300秒；实际时间由调度任务另记，不作为预设结果。

程序：`programs/run_rc_m_causal128_property_compensation_v1.py`。

```
python programs/run_rc_m_causal128_property_compensation_v1.py prepare
python programs/run_rc_m_causal128_property_compensation_v1.py fit
python programs/run_rc_m_causal128_property_compensation_v1.py evaluate
```

`run`依次执行fit与evaluate；新9组验收未完成时evaluate返回75，不修改封存拟合。输出根为`results/rc_m_causal128_attribution_chain_v1/property_compensation_fixed_pairs_v1`。工程自检包括反对称性、独立特征计算、梯度／Hessian差分、严格凸目标证书、候选交换等变性、确定性重复拟合与RAW/L干预不变性。

原128主链和最终报告保持不变；本结果作为明确范围的补充归因，不能把16组两候选补偿写成全128自然候选性质必要性验证。
