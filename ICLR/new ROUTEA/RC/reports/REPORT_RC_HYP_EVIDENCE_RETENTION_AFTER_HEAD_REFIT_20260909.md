# Head适配失败后的证据保留结论

2026-09-09。本轮没有得到HYP GO；已关闭共同query prior及其原七参数
head refit路径，保留所有负结果，未替换部署、修改标签或更换统计分母。

## 完整结果

同一matched RAW C128、已打开EVAL32：原NATIVE7/C为28/32；冻结head加
共同query prior为26/32；随后冻结prior重训原head为25/32。新head搭配
UNIFORM也是25/32，与新head REAL正确集合完全相同，但有4例选择了不同
错误candidate，不能说prior在数值上完全没有影响。

新head训练集29/32，EVAL下降；原prior独有新增成功0050消失，原3个
损失0212、0220、0618未修复。因此本次“只适配七参数head即可修复”
方案已被否定，不能由此推断所有head或HYP均不可能。

refit结果SHA：2aa5d46c034b130a940bfc88e99323a0bbb441416edaa28a2d390345b0458b65。
完整复算见`results/rc_shared_query_prior_head_refit_closure_v1/result.json`。
分支收口：`registry/rc_shared_query_prior_branch_closure_decision_v1_20260909.json`。

## 现有证据在哪里受到抑制

定义固定A集合：在原full-reference unweighted MaxSim下，该candidate
在同一query token胜过全部127 rivals的位置。集合由已经打开的标签和
已选wrong用于诊断，不能当部署selector。以下全用同一query轴和同一
固定A集合，追踪原reference weighting、原query贡献和新query贡献。

| target case | A独占位置 | 原wr后仍独占 | 原C贡献仍独占 | 新C贡献仍独占 |
|---|---:|---:|---:|---:|
| 0050 | 21 | 19 | 11 | 9 |
| 0212 | 12 | 10 | 9 | 9 |
| 0220 | 13 | 12 | 9 | 6 |
| 0618 | 24 | 0 | 0 | 0 |

位置存活也不代表获得了足够权重：0212 target固定集合的visibility占比
从10.99%降到0.771%；0220从2.446%降到0.0451%。0618的全部24个原
独占位置则在wr步骤就失去全127优势，发生于新prior之前。

另一方面，0212 wrong1631的5个原A独占位置在wr后全部不再独占；其
最终胜出不能归因于这些原wrong独占位置一直保留。已独立确认wrong的
Q/R特征在旧负权下增加logit，是另一个具体作用路径。

原最好C28的另外三个失败也已补齐：0213 target11→1→1→1；0373
target1→0→0→0；0676 target43→17→6→6。加0050，已覆盖原C28所有
四个失败，没有只检查新prior自己制造的回归。

来源：`results/rc_shared_query_prior_local_evidence_survival_v1/`及
`results/rc_native_c28_original_failure_local_evidence_survival_v1/`，均有
全query轴、全127 rival margin和NumPy独立复算。这里的独占性是冻结
token表征的数学性质，不自动等于可见像素里的语义身份字段。

## 原图对解释范围的约束

0050存在真实背景药品竞争，先验曾把选择改回手持目标，是有限正例。
0213只有一只药盒；四条有效成分可见，与wrong单一Loratadine成分矛盾。
0676主说明面共用，但绿色侧缘和GTIN末段51520仍提供小面积区别。
0373只露出各reference均有的演示码面，GTIN/SN等为X占位，药名剂量
不可见；尚未证明所有可见图形细节严格等价，不能直接删除或改标签。
见`reports/REPORT_RC_NATIVE_C28_THREE_REMAINING_FAILURE_VISUAL_FACTS_V1_20260909.md`。

## 当前尚需确认的像素来源

0212现成GT polygon与当前RAW token frame已闭合：target12个独占token
中心有6个在GT内、6个在外；wrong1631的5个全部在GT外。不能把GT外
token直接叫背景身份线索，因为ColNomic表征具有全图上下文。

已准备唯一四输入检验：同冻结ColNomic编码原图、仅保留target、仅擦除
target、全灰图；尺寸、processor、权重及完整C128 references固定。
原图token SHA和a矩阵必须先回归成功，之后才能解释三个干预。使用
既有target12/wrong5位置，不在干预后再挑位置，也不拟合新head。

输入mask已独立完成全像素核验：target576389像素，背景7718011，
灰RGB=(127,127,127)，KEEP与ERASE互补。资格receipt为
`results/rc_outcome0212_exact_pixel_mask_qualification_v1/receipt.json`。
执行定义见`plan/RC_OUTCOME0212_PIXEL_SOURCE_DIAGNOSTIC_V1_20260909.md`。
此处尚无encoder干预结果；它只检验信息来源，不是HYP、ownership或
新识别准确率证明。

## 实际提交

像素检验job **5138459** 已提交，唯一启动核验为PENDING（Priority），
accelerated、1GPU、8CPU、申请96G、30分钟、无依赖、无requeue。
没有持续监控。authority SHA为
e3abbd8bd4446359fe9713a42d4f70a8aecfd54661ffda51919ac81f512e3c9b，
见`registry/rc_outcome0212_pixel_source_diagnostic_authority_v1_20260909.json`。
runner和launcher独立review通过，E0四项通过；模型权重及相关源码版本
已经完整封存。唯一下一次读出应检查
`results/rc_outcome0212_pixel_source_diagnostic_v1/result.json`与
`independent_validation.json`。若原图回归STOP，不能解释后续因果效应。
