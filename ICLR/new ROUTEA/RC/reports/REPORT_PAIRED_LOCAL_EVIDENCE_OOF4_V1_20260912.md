# 同位置候选内容差：有观察净增，但未超过匹配对照

原 TRAIN128、32身份、32来源组的固定四折，完整 RAW C128 / 127 challenger / 阈值0 HOLD-SWITCH。原折 BASE7 直接复用；四个新残差臂均用相同 FULL-C128 CE、FP64、零初始化、AdamW .03/.001、2000步末步。该128不是旧 EVAL128。

| 模型 | 正确/128 | MRR | 相对 BASE7 救/损/净 |
|---|---:|---:|---|
| RAW | 86 | 0.780755323 | — |
| BASE7 | 108 | 0.892334884 | — |
| BIAS1 | 109 | 0.899877988 | 1/0/+1 |
| MEAN2 | 109 | 0.901328881 | 6/5/+1 |
| CURVE3 | 111 | 0.909141381 | 7/4/+3 |
| JOINT3（主臂） | 110 | 0.905235131 | 6/4/+2 |
| GLOBAL7（既有强对照） | 114 | 0.930298565 | 7/1/+6（原记录） |

JOINT3 对 BASE7 有观察净增2，不能写成完全没有积极变化。但对同参数 CURVE3 为0救1损，唯一损失 OUTCOME-0150；对 GLOBAL7 为1救5损、净-4。相对 MEAN2 的唯一新增 DIFFICULT-0107，也已由 CURVE3 救回。因此本轮没有出现仅由新局部统计带来的、超过同容量标量非线性对照的额外正确决策。

相对BASE，JOINT3为4正组/3负组，等组差bootstrap95%区间[-0.0265625,+0.0755208]，精确双侧组符号置换p=.421875。这是已多次开发过的TRAIN分组结果，不是未触碰外部检验。组区间跨0，不能据净+2宣称可靠群体提升。

## 本轮分清的内容

数学信息缺口仍成立：逐候选边际统计不一定恢复同query位置的联合比较。本轮主臂把联合差压成m2=sum(rho*d*abs(d))；m1=sum(rho*d)，d为candidate与RAW winner在同一query token上的完整reference自由MaxSim差。CURVE3仅使用m1及m1*abs(m1)，两臂均3个新参数。

无标签冗余检查发现，各折TRAIN内m1与m2的相关系数为.9840—.9887；由[1,m1,m1*abs(m1)]线性拟合m2后，剩余方差占比为1.37%—2.43%。这些比例是自然特征变化的方差，不能解释成“只剩1%身份信息”，也不能单独证明失败原因。它们结合0救1损的匹配结果，支持本统计没有展示独立检索优势；不支持继续把同类全局标量变换当成已找到的瓶颈修复。

这还没有排除完整局部分布、细粒度token语义、其他条件信息或训练分布问题。尤其本轮仍没有检验token能否读出具体药名、剂量差异；不能把失败解释成真实图片无可识别线索。

## 验证与工程修复

5142610/5142611的16缓存分片均完成，所有128×128 candidate profiles和权重经fresh NumPy重算，旧F内容分数被重放。5142612四折全部完成，参数/预测fresh位级重放、81,280 logits独立NumPy核算通过，最大logit误差2.13e-14。四折预测完整封存后才join留出标签，query/image/identity/group无交集。

原5142613汇总失败于我把5412去重身份排序误写为5413物理图的断言。上游资格程序早已明确5412；追加join-only程序及修复authority，不改原程序、训练、预测、prelabel seal或科学配方。5142654在dev_accelerated完成0:0/41秒，重建并检查完整5412身份排名。全部RAW/BASE7/GLOBAL7逐图correct/rank/selected row与上一轮已验证结果完全一致（1152项核对）。没有重训或重做预测。

## 当前处置

JOINT3未通过预定“超过全部固定对照”的screen，不晋级旧EVAL，不追加本轮温度、曲率或残差参数搜索。原各折BASE及GLOBAL预测保留；完整局部分布缓存保留供后续独立问题使用。本轮已完成，无待运行任务。

旧EVAL账本保持分开：原ec7为28/32、99/128；LISTWISE_UNIT1为26/32、101/128；GLOBAL7_T128为27/32、98/128。本轮110/128属于TRAIN四折，不能拼入上述成绩。没有new HYP GO或部署替换。

产物：results/rc_paired_local_evidence_oof4_v1/result.json、result_validation.json、join_repair_receipt.json、completion_readout.json、feature_redundancy_diagnostic.json。结果SHA256：8993ca3b81f4baee871964eca2e061c9fb6abc6b653dcd4c9607d78fc157adf1。
