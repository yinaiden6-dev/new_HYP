# 593图结果：保留正向增量，区分整体收益与新增机制

全部结果为同一593张、68身份、64来源组、RAW自然C128、五折留组读出。原28/32与99/128另属旧评价，不能直接相减。本轮包含历史开发来源，不是未触碰外部确认。

| 模型 | 正确/593 | 对同折BASE7_ALL救回/损失 |
| --- | ---: | ---: |
| RAW | 426 | — |
| BASE7_SMALL128 | 444 | 4/0 |
| BASE7_ALL | 440 | 0/0 |
| CONSTANT1自由内容统一补偿 | 444 | 5/1 |
| CONDITIONAL4自由内容条件补偿 | 445 | 5/0 |
| J端点ENDPOINT2 | 439 | 4/5 |
| J相对差＋全局偏置RELATIVE2 | 445 | 7/2 |

CONSTANT1对BASE的正确总数增加4，但其正确集合并非简单“BASE＋4”：CONDITIONAL4对CONSTANT1的唯一增量是恢复0333，故统一补偿实际5救1损。所有集合由机器账本完整保留。

## 已经成立的正向结果

CONDITIONAL4对同折全量基头5救0损，涉及5个来源组，等权组差+1.249个百分点；bootstrap95%约[+0.260,+2.563]个百分点，精确双侧组sign-flip p=0.0625。两种统计都报告，不只选有利指标。五个救回：DIFFICULT-0083、0043、0027、0015、NDV2-011-P04；全部原本target已是最强challenger，只是没越过0。

完整CONDITIONAL4对RAW为19救0损，分布15组，等权组差约+3.706个百分点，bootstrap95%约[+1.858,+5.897]个百分点，精确双侧组sign-flip p=0.0000610。BASE7_ALL对RAW为14救0损，分布11组，p=0.000977。这是更大内部群体上的正向识别证据，不应被局部新分支的失败掩盖。

## 不能据此夸大的部分

数据扩大本身没有提高原结构：SMALL128的444降到ALL的440，4个原正确损失都在fold4。CONDITIONAL4对SMALL128为3救2损，只净增1；组区间约[-0.361,+1.953]个百分点，p=0.375。CONDITIONAL4对CONSTANT1只多救1图，p=1。因此尚未证明条件化比更简单、更强对照有可靠优势。

J端点方案失败：ENDPOINT2输给BASE和同参数RELATIVE2。RELATIVE2虽有正净增，但J_BIND后正确数449，比REAL445还高，不能把该正增直接当作正确J绑定的收益；449是固定模型干预，不是采用的新模型成绩。ENDPOINT2本轮不继续作为性能候选。

## 当前继续的唯一跟进

5141332五折仅训练一个全局BIAS1，用与内容補偿同样的TRAIN标签/损失/步数；同时冻结既有CONSTANT1/CONDITIONAL4，只打乱其增量输入、保留BASE证据，区分内容特异性和一般HOLD校准。后续5141333汇总、5141334独立复核。accelerated占1GPU、CPU计算、每折10分钟。该跟进是在看到OOF后冻结，须如实披露为探索性归因，不是原主检验。

原结果与统计：reports/REPORT_NEW_HYP593_GROUPED_OOF5_RESULT_V1_20260911.md、REPORT_H593_ENDPOINT_SIGN_OOF5_RESULT_V1_20260911.md；新增精确组统计及5个救回的logit变化：results/rc_h593_completed_result_interpretation_v1/result.json。
