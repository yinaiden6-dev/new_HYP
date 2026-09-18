# GLOBAL7统一微调：TRAIN收益未迁移到原固定面板

5142580在dev_accelerated完成0:0，用时65秒。新参数fresh-process重放、121920 logits的独立NumPy核算、动作/完整gallery排名及训练评价隔离检查通过；最大logit核算误差7.11e-15。原模型参数、预测与逐图结果均保持一致。

新模型GLOBAL7_T128冻结原ec7，使用全TRAIN128、32身份/32来源组训练七维全局残差，复用此前GLOBAL7的FP64/AdamW .03/.001/2000步/零残差/完整C128交叉熵配方。原六特征、RAW full-gallery C128及完整127 challenger HOLD/SWITCH不变。

| 同一固定面板 | RAW | 原ec7 | LISTWISE_UNIT1 | GLOBAL7_T128 | 对原头救/损/净 | 对LISTWISE救/损/净 |
|---|---:|---:|---:|---:|---|---|
| EVAL32 | 25 | 28 | 26 | 27 | 0/1/-1 | 1/0/+1 |
| EVAL128 | 88 | 99 | 101 | 98 | 3/4/-1 | 2/5/-3 |

**结果：对原强基线，两面板均净减1；不通过本轮观察净增要求。** 对LISTWISE虽在32图恢复1例，但在128图净减3，不能只报前者。没有自动替换原模型或选择两模型的逐图最优答案。

前轮108→114是TRAIN128的分组留出方法结果：每折原基头排除留出组，再用本折TRAIN学残差。本轮是原全训练ec7加全TRAIN128残差，在不同身份/组的原EVAL面板测试。两者不是同一批图或同一组参数。此结果确认该固定配方的TRAIN开发收益没有在这两个面板重现；它不单独证明是数据分布、特征缺失或某个训练环节造成。

## 逐图救损

| 比较 | 面板 | 救回 | 损失 |
|---|---|---|---|
| LISTWISE_UNIT1__to__GLOBAL7_T128 | EVAL32 | OUTCOME-0669 | 无 |
| ORIGINAL7__to__GLOBAL7_T128 | EVAL32 | 无 | OUTCOME-0618 |
| LISTWISE_UNIT1__to__GLOBAL7_T128 | EVAL128 | OUTCOME-0795, NDV2-005-P03 | OUTCOME-0106, OUTCOME-0403, OUTCOME-0278, OUTCOME-0821, OUTCOME-0818 |
| ORIGINAL7__to__GLOBAL7_T128 | EVAL128 | OUTCOME-0809, DIFFICULT-0028, OUTCOME-0769 | OUTCOME-0130, OUTCOME-0140, OUTCOME-0403, OUTCOME-0280 |

新模型相对LISTWISE恢复了0669，以及128面板中的0795和P03；同时失去0106、0278、0821、0818等成功并新增0403错误。这是具体的决策取舍，不能把“更保守”或“更积极切换”单独当成统一解释。

训练与每个评价面板在canonical query ID、原query ID、图片SHA、identity、group、component六项检查交集全0。评价标签在新预测封存及独立logit核算后才打开。两套评价均是历史已打开的开发面板，不包装成外部确认。

CBIND：新模型EVAL32为10/32，EVAL128为35/128；这是本次绑定对照，不作为模型准确率或理论突破。REAL未超过原头，不能通过重复绑定结论替代净增。

本轮已完成并收口，无待监控任务、无后续自动试验。原ec7保持28/32、99/128；LISTWISE保持26/32、101/128；GLOBAL7_T128单列27/32、98/128；TRAIN四折114仍保留原证据等级。没有新HYP GO或部署变更。

[机器结果](../results/rc_global7_train128_fixed_panels_v1/result.json) · [独立核算](../results/rc_global7_train128_fixed_panels_v1/result_validation.json) · [固定计划](../plan/RC_GLOBAL7_TRAIN128_FIXED_PANELS_V1_20260912.md)

结果SHA：162c1286a943c96c10538efcd6ec4fb33cc8094feca538e4c29d20b3dd069360。
