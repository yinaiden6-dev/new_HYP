# new HYP：TRAIN group jackknife 已验证结果

**作业 5138859 已完成，完整新进程重训验证通过。原乘积分数对比 S 的加入在
全部 12 个 TRAIN 删组条件下均未产生负的 query net：无 Q/R 时 8 正、4 平；
有 Q/R 时 11 正、1 平。Q/R 加在 S 之后则为 5 正、6 平、1 负，增量较不稳定。**
这是同一已打开 EVAL32 上的内部训练敏感性证据，不是 12 个独立测试集。

完整 ORIGINAL7 在删组后的正确数为 26–28/32，中位数 27；仅 1/12 条件
保留原 28/32 及全部原正确样本，另 11 个条件减少 1 或 2 个正确。
因此可保留“乘积项在本流程中贡献方向稳定”的有限解释，不能声称完整
七参数表现对任意训练组删除都稳定，也没有产生新采用的更高分模型。

## 1. 执行与验证收口

按协调者的一次 accounting 核对，5138859 为 COMPLETED、ExitCode 0:0，
Berlin 时间 2026-09-09 21:37:23–21:57:31，耗时 20 分 08 秒，8 CPU，
MaxRSS 522868K。本报告生成过程未查询调度器。

正式 producer 完成 13 条件×4 头的 52 次原协议拟合；同作业新进程再次
重训全部 52 次，逐项复现参数、封存 logits、动作和结果。FULL_TRAIN
四头参数、FULL64 REAL/C_BIND 预测及 TRAIN/EVAL 动作均精确回归父试验。

另外，独立 producer 算术复核从原候选轴和封存 logits 重新构建 6656 个
完整 action、845312 个 logit 的选择与排名，核对全部条件的指标、paired
差、等权 group 差及全 query 频率。该复核与正式重训验证绑定同一 producer
SHA，均未发现不一致。此前 producer 报告保留其“待重训”快照，本报告
通过现已存在的正式验证收口，不修改历史快照。

- Producer：`results/rc_train_group_jackknife_stability_v1/result.json`，
  SHA256 `ce7eaeb9a3407e4626bb2bf00d8f8c2c0c328ce20ae28215f17069db1ec21536`。
- 正式验证：`results/rc_train_group_jackknife_stability_v1/independent_validation.json`，
  SHA256 `9d191cb11e24b4b3f98e500f104555e680361a9ade64d7300c39390410d5db88`，
  状态 `TRAIN_GROUP_JACKKNIFE_INDEPENDENT_REEXECUTION_PASS`。
- 独立算术复核：`results/rc_train_group_jackknife_producer_arithmetic_review_v1/result.json`，
  SHA256 `dabeaa886a50f956242d2b31e900980b5145fbd905c0143100d06c8cf6c60a5f`。

## 2. 固定比较范围与全部条件

模型为 JOINT4、PRODUCT5、RESPONSE6、ORIGINAL7（NATIVE7 系列）。训练
只使用原身份/正负检索关系，PAIR64 始终完整；按排序穷举 FULL TRAIN32
的全部 12 个 supergroup，每次只删除一个组，剩余行仍保持原 execution
顺序。无种子、阈值、checkpoint 或有利训练子集选择。

评价固定为原 RAW full-gallery C128、全部 127 challenger、零阈值
HOLD/SWITCH、已打开 matched EVAL32 的 REAL 分支。原 RAW baseline 为
25/32；EVAL 的 32 query 属于既有 11 个 supergroup。下表各头分母均为 32。

| 条件 | 剩余 FULL TRAIN | JOINT4 | PRODUCT5 | RESPONSE6 | ORIGINAL7 |
|---|---:|---:|---:|---:|---:|
| FULL_TRAIN 基线 | 32 | 26 | 27 | 26 | 28 |
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

FULL_TRAIN 单独列示。全部敏感性频率以 12 个删除条件为分母；较好的删除
条件没有被提升为主模型。C_BIND、完整 TRAIN32 诊断和实际剩余 TRAIN 子集
指标也全部保存在结果中，上表不把它们混入 EVAL REAL。

## 3. 四条预定边：乘积项与响应组的稳定程度不同

| 预定比较 | query net 正/零/负 | query net 最小/中位/最大 | 等权 group 差正/零/负 |
|---|---|---|---|
| PRODUCT5 − JOINT4：无 Q/R 时加入 S | 8 / 4 / 0 | 0 / 1 / 2 | 8 / 4 / 0 |
| ORIGINAL7 − RESPONSE6：有 Q/R 时加入 S | 11 / 1 / 0 | 0 / 1 / 2 | 11 / 1 / 0 |
| RESPONSE6 − JOINT4：无 S 时加入 Q/R | 1 / 11 / 0 | 0 / 0 / 1 | 1 / 11 / 0 |
| ORIGINAL7 − PRODUCT5：有 S 时加入 Q/R | 5 / 6 / 1 | −1 / 0 / 1 | 7 / 4 / 1 |

这里的“没有负方向”指每个条件的净正确差不为负，不等于每个 query 都不
受损；query net 和等权 group 差也不应互换。完整 paired rescue/break
列表与所有方向都已报告，没有只选有利对照。

该结果与既有代数解释相容：原 dS 是 M/L 对比量的非线性函数项，可为
线性校准头补充表达形式。它支持这个固定系统中的经验机制解释，不把
基本代数恒等式升级为原创定律或普遍性能保证。

## 4. 原成功与失败分别发生什么

下表汇总原 ORIGINAL7 的三个 rescue、一个敏感的 RAW 正确保留样本，以及
全部四个原错误。其余全量 query 正确频率和动作/reference 一致率也已核对。

| Query | FULL_TRAIN 状态 | 12 删除条件正确次数 |
|---|---|---:|
| DIFFICULT-0044 | rescue 成功 | 12/12 |
| OUTCOME-0212 | rescue 成功 | 12/12 |
| OUTCOME-0220 | rescue 成功 | 3/12 |
| OUTCOME-0618 | 保留 RAW 正确 | 9/12 |
| DIFFICULT-0050 | 错误 | 0/12 |
| OUTCOME-0213 | 错误 | 0/12 |
| OUTCOME-0373 | 错误 | 0/12 |
| OUTCOME-0676 | 错误 | 0/12 |

已确认的增益存在稳定部分：0044/0212 在全部删除条件保持正确。主要
正确性敏感点是 0220 和 0618；原四个错误没有被这些 ORIGINAL7 删除模型
救回。错误 query 即使改变 challenger 仍可能错误，因此 HOLD/SWITCH、
physical candidate 与 corrected reference identity 一致率分别保存。

## 5. 可发布结论与限制

可发布为：**在原 RAW C128、已打开 EVAL32 和固定训练协议下，乘积分数
对比的增量方向对穷举 FULL TRAIN group 删除保持非负；Q/R 响应组的额外
收益与完整 28/32 表现存在训练敏感性。**

12 个条件共享训练数据和同一评价集，不报告独立样本 p 值或把它们当成
12 次外部确认。PAIR 的 20 个组没有被扰动；删除 FULL 组大小不同，剩余
FULL mean loss 的重新归一化也是扰动的一部分。结果不能推广为所有训练
分布、所有编码器或新 reference 上的可靠性保证。

本轮不采用新模型，不修改候选集/阈值，不进入 ownership、旧空间
Reference HYP 或其他受限数据。旧 FROZEN_C difficult90 的 69/90 属于
另一 head/评价谱系，不能与本表合并。

完整收口凭据：`registry/rc_train_group_jackknife_stability_completion_receipt_v1_20260909.json`。
