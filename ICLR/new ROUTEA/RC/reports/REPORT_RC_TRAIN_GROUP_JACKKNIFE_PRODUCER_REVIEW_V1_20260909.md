# new HYP：TRAIN group jackknife producer 算术复核

**当前证据级别：producer 固定快照的独立算术复核已完成，未发现不一致；
仍须正式新进程完整重训验证。本报告未读取或宣称该轮独立重训验证完成，
不替代其结果，也不构成最终科学 PASS/GO。**

受审 producer：`results/rc_train_group_jackknife_stability_v1/result.json`，
SHA256 `ce7eaeb9a3407e4626bb2bf00d8f8c2c0c328ce20ae28215f17069db1ec21536`。
下文均是这一固定快照的条件性读数；正式对外结论须同时具备完整重训验证。

## 1. 实际独立检查了什么

复核程序没有调用训练器、head forward 或调度器。它验证 authority/source
hash、参数及预测 seal，从原完整 C128 payload 读取候选轴与 RAW 顺序，从
已独立验证的 native64 结果读取既有 query/group/target 元数据，并从冻结
identity repair 规则重建 reference 身份关系。

对 13 条件×2 split×2 mode×4 head×32 query，共 6656 个 action、845312
个封存 logit，独立执行以下计算：

- 全 127 challenger 最大值、physical-row 平局、零阈值 HOLD/SWITCH；
- 原与最终 reference 选择、身份正确性、排名及全部 action 字段；
- 完整 TRAIN/EVAL 指标和各删除条件实际保留 TRAIN 子集指标；
- 四条 factorial 边、各头相对 FULL_TRAIN 的 rescue/break/net 和组均值；
- 全部 12 删除条件的分布、全 32 query 的正确频率与三类动作/选择一致率。

整数、动作及四头原基线的完整参数/预测对象精确匹配；独立 group/MRR
求和以 1e−15 容差比较，容许不同加法顺序的舍入。未发现任何不一致。
受审条件还独立从全部 12 个 TRAIN group 重建，完整 PAIR64、原条件顺序
及删除数量与 authority 一致。

本复核**没有重新从训练数据产生删除条件参数**，也没有用参数重新计算
全部 head logits。上述两点仍由正式完整重训 validator 负责。因此这里
确认的是 producer 输出内部和原输入轴/封存预测的算术一致性。

可复现程序：`programs/review_rc_train_group_jackknife_producer_arithmetic_v1.py`。
机器可读复核：`results/rc_train_group_jackknife_producer_arithmetic_review_v1/result.json`。
复核结果 SHA256：`dabeaa886a50f956242d2b31e900980b5145fbd905c0143100d06c8cf6c60a5f`。
状态为 `PRODUCER_ARITHMETIC_REVIEW_COMPLETE_PENDING_INDEPENDENT_RETRAIN`。

## 2. 全部条件的 producer 读数

同一原 RAW C128、已打开 matched EVAL32、完整 127 challenger、REAL action；
四列分别为同协议训练的 JOINT4 / PRODUCT5 / RESPONSE6 / ORIGINAL7。
FULL_TRAIN 是单独基线，下面 12 条才构成敏感性分布。

| 条件 | 保留 FULL TRAIN 数 | JOINT4 | PRODUCT5 | RESPONSE6 | ORIGINAL7 |
|---|---:|---:|---:|---:|---:|
| FULL_TRAIN | 32 | 26 | 27 | 26 | 28 |
| DROP_GROUP_01 | 30 | 26 | 26 | 26 | 27 |
| DROP_GROUP_02 | 31 | 26 | 27 | 26 | 28 |
| DROP_GROUP_03 | 29 | 25 | 27 | 26 | 27 |
| DROP_GROUP_04 | 31 | 26 | 27 | 26 | 27 |
| DROP_GROUP_05 | 28 | 26 | 27 | 26 | 26 |
| DROP_GROUP_06 | 31 | 26 | 27 | 26 | 27 |
| DROP_GROUP_07 | 29 | 26 | 26 | 26 | 27 |
| DROP_GROUP_08 | 26 | 26 | 27 | 26 | 27 |
| DROP_GROUP_09 | 31 | 26 | 27 | 26 | 27 |
| DROP_GROUP_10 | 29 | 26 | 27 | 26 | 27 |
| DROP_GROUP_11 | 30 | 26 | 26 | 26 | 27 |
| DROP_GROUP_12 | 27 | 26 | 26 | 26 | 27 |

ORIGINAL7 删除条件范围为 26–28/32，中位数 27；11 个删除条件低于原 28，
1 个持平。没有删除条件超过原 28，也不从中选择部署模型。

## 3. 四条预定边的方向

下表严格使用全部 12 个删除条件。组指标为等权 EVAL supergroup 准确率差，
它与按 query 数计的 net 并非同一个指标。

| 预定比较 | query net 正/零/负 | query net 最小/中位/最大 | 组均值差正/零/负 |
|---|---|---|---|
| PRODUCT5 − JOINT4：无 Q/R 时加入 S | 8 / 4 / 0 | 0 / 1 / 2 | 8 / 4 / 0 |
| ORIGINAL7 − RESPONSE6：有 Q/R 时加入 S | 11 / 1 / 0 | 0 / 1 / 2 | 11 / 1 / 0 |
| RESPONSE6 − JOINT4：无 S 时加入 Q/R | 1 / 11 / 0 | 0 / 0 / 1 | 1 / 11 / 0 |
| ORIGINAL7 − PRODUCT5：有 S 时加入 Q/R | 5 / 6 / 1 | −1 / 0 / 1 | 7 / 4 / 1 |

若正式重训复现，以上可支持这一有限解释：乘积分数对比的净方向在这些
FULL TRAIN 扰动中没有变负；Q/R 响应组的增量较不稳定，不能宣称完整
七参数优势对任意训练组删除都保持。正/零/负次数是描述统计，不能用作
12 个独立样本的显著性证据。

## 4. 完整 query 频率中的关键变化

已独立检查全部 32 条频率，下面概括原成功动作及全部原错误的表现：

| ORIGINAL7 的 query | FULL_TRAIN 是否正确 | 12 删除条件正确次数 |
|---|---|---:|
| DIFFICULT-0044，原 rescue | 是 | 12 |
| OUTCOME-0212，原 rescue | 是 | 12 |
| OUTCOME-0220，原 rescue | 是 | 3 |
| OUTCOME-0618，原保留 RAW 正确 | 是 | 9 |
| DIFFICULT-0050 | 否 | 0 |
| OUTCOME-0213 | 否 | 0 |
| OUTCOME-0373 | 否 | 0 |
| OUTCOME-0676 | 否 | 0 |

因此总体波动不能简单写成“所有增益都不稳定”：0044/0212 在全部删除条件
保持正确；主要正确性敏感点是 0220 和 0618。四个原错误始终未被这些
ORIGINAL7 删除模型救回。错误 query 的 challenger 可能变化而仍错误；
正确频率、HOLD/SWITCH 一致率和最终 reference 一致率已分开检查。

本表不取代全 32 条输出，也不将单例转成新的全体资格门。

## 5. 发布边界

本报告不更新 HYP 科学状态，不采用任何删除模型，不改变阈值或训练集。
完整独立重训验证完成且匹配该 producer hash 后，才可将这些数值作为
已验证的内部敏感性结果发布。即便重训通过，它仍只属于同一已打开 EVAL32
上的训练扰动复核；PAIR 组未被扰动，也没有独立人群或跨数据确认。

旧 FROZEN_C difficult90 是另一头、另一评价谱系，不与此表合并。研究
截止时间、retrieval-only 任务监督及不监控调度器的要求保持。
