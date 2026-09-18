# TRAIN128 protected projection：训练保护成立，留出组 HOLD 保护失败

这轮实际结果是 **TRAIN128 按32个身份/组划分的四折 OOF：BASE7 108/128 → PROTECTED7 110/128，新增6张、损失4张**；REPAIR_ONLY7 108/128，新增6张、损失6张。这不是原模型 EVAL128 的99变成110。本次仅复核 TRAIN 产物，没有读取 EVAL32/128 的预测、标签或任何 `rc_opened_*` 容量系数。

| 同一 TRAIN128 OOF 账本 | 正确 | 相对 RAW 救回／break | 相对 BASE 新增／损失 |
| --- | ---: | ---: | ---: |
| RAW 自然 C128 winner | 86/128 | — | — |
| 原四折 BASE7 | 108/128 | 23／1 | — |
| PROTECTED7 | 110/128 | 29／5 | 6／4 |
| REPAIR_ONLY7 | 108/128 | 29／7 | 6／6 |

候选轴、六个原特征、七参数形式、FP64 运算和全部127个 challenger 的 `max logit > 0` SWITCH 规则不变。该分支复用已验证的四个折内 BASE7，没有重新拟合原头；每个折只使用其余24组的检索身份标签产生和选择 LP 候选头。它是反复使用 TRAIN 的开发实验，不是未接触外部集上的确认。

## 每折的 TRAIN 保护确实成立

| Fold | 折内训练图数 | BASE 正确集合 P | PROTECTED 正确 | 训练新增／损失 | 所选头 L1 位移 | δ 搜索值 | 生成所选头的错误图 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 93 | 79 | 85 | 6／0 | 18.893862 | 0.141340972 | OUTCOME-0538 |
| 1 | 101 | 89 | 94 | 5／0 | 29.313709 | 0.025147644 | OUTCOME-0533 |
| 2 | 96 | 80 | 89 | 9／0 | 5.808879 | 0.023192479 | OUTCOME-0742 |
| 3 | 94 | 81 | 86 | 5／0 | 4.824875 | 0.026219544 | OUTCOME-0742 |

四个所选 LP 都是 solver success，所需全部127候选动作约束的精确有理数 margin 为正，实际 FP64 动作保全各折 P。独立只读重算一致；不能把 OOF 失败归因于“求解没有收敛”或“训练保护未实现”。δ 是搜索目标，所选端点最低 margin 比 δ 低约 10⁻¹⁵ 的浮点偏差已显式记录，仍严格为正；这里没有宣称精确 δ 下限。

L1 是每个单错误 LP 的目标；跨候选选头优先级是训练组平均正确率、正确图数，最后才是 L1。故不能把所选头称为“所有能增加任意一张正确的头中改动最小者”。Fold0/1 最终位移分别18.89和29.31；Fold2 位移5.81仍产生损失，单加“小步长”是否足够尚无证据。

## 四个损失都来自正确 HOLD 被改成错误 SWITCH

| 留出图 | Fold | BASE 最大 logit | PROTECTED 最大 logit | 动作变化 | strongest challenger 是否变化 |
| --- | ---: | ---: | ---: | --- | --- |
| DIFFICULT-0101 | 0 | -1.001557 | +3.141564 | HOLD → 错误 SWITCH | 否 |
| OUTCOME-0494 | 2 | -0.685554 | +0.173282 | HOLD → 错误 SWITCH | 是 |
| DIFFICULT-0107 | 0 | -0.628195 | +4.736535 | HOLD → 错误 SWITCH | 否 |
| OUTCOME-0728 | 2 | -2.773687 | +0.216186 | HOLD → 错误 SWITCH | 否 |

4个新增损失原本都是 RAW 正确；原 BASE 的23次正确 SWITCH 全部保留。损失落在3个留出组，其中0101和0107同组。0494既跨过 SWITCH 边界，也更换了最强错误 challenger；其最终选中错误候选在 BASE 下的 logit 是 −1.227002，不能用原最大值 −0.685554 代替做参数分解。

对每个 PROTECTED 最终选中的错误候选，固定其特征后分解参数变化，得到：

- 0101：Δz=+4.143120，主要为 visibility mass 项 +2.657640、bias +2.108618；RAW 项抵消 −0.530304。
- 0107：Δz=+5.364730，主要为 visibility mass +4.190875、bias +2.108618、query control response +1.183539；RAW 项抵消 −2.133730。
- 0494：该候选 Δz=+1.400283，RAW 惩罚减弱 +1.301769、query control response +1.357837、bias +1.122435；S/L 项合计抵消 −2.381758。
- 0728：Δz=+2.989873，主要为 bias +1.122435、query control response +1.016683、RAW 惩罚减弱 +0.414112。

这说明本轮的共同失败位置是 **留出组的原正确 HOLD 边界**。具体推动项不相同：Fold0 主要是 mass 和 bias，Fold2 则是 query response、bias 与 RAW 惩罚减弱。只把原因概括为“geometry 加分过大”不覆盖四例。

训练约束只保护实际参与训练的24组，随后共享系数作用于8个留出组；新组的 HOLD 没有被这些有限训练约束自动保护。这是实测到的泛化缺口。它不证明原六维特征完全无信息、不证明全部七参数头不可能更好，也不构成对像素/空间因果来源的诊断。

## 决策与证据范围

严格冻结门中，“正确数增加”和“组平均增益为正”成立，“保全每张 BASE 正确图”失败，故本方案止于 TRAIN，不进入 EVAL。组平均差为 +7/480（约+1.46个百分点），组 bootstrap 95%区间约[−2.40,+5.36]个百分点，双侧组 sign-flip p=0.5234375；这只是弱开发信号，不能宣称可靠提升已成立。此处零损失是该方案事先冻结的晋级门，不是所有 new HYP 理论都必须满足的普遍定义。

继续研究应针对“新增 SWITCH 何时能够安全越过原 HOLD”提出可验证的共享规则，同时保留候选竞争。不能仅因为这轮训练保全成立就称为 no-regret 泛化保证，也不能把新增6张的真实开发信号抹去。

## 只读复核

复核重算了97536个原六维特征标量、48768个完整留出 logit，并复算114项保存候选的训练分数/选中头分数；重新检查训练保护集合、δ、精确端点 margin、L1与候选选择顺序。没有新增 LP、梯度训练、图像读取或 EVAL 读取。原 producer 的独立 fresh LP replay 已完成，此处不重复优化。

- 原结果：[result.json](../results/rc_train128_protected_projection_oof4_v1/result.json)，SHA `5f5cc6e1a0aba87c22b3c0fea7040d9b54304d38e7db27fce42a9147ae398042`。
- 原独立验证：[validation.json](../results/rc_train128_protected_projection_oof4_v1/validation.json)，SHA `f852f1dc4f4fd93c2688331b989259353713cb834dc38583f87e7dfb8faefc46`。
- 本次只读复核：[review.json](../results/rc_train128_protected_projection_result_review_v1/review.json)，SHA `d1050ad734f15e2cce71493a9541b0dc01261c3e9674f18d0eecdd62fbc54830`。
- 复核绑定：[validation.json](../results/rc_train128_protected_projection_result_review_v1/validation.json)，SHA `a00dee5cc230c7d0c09f267f3e980454cd6038d46d125f4aa4140c8cf134f714`。
- 可重放程序：[review_rc_train128_protected_projection_v1.py](../programs/review_rc_train128_protected_projection_v1.py)。
