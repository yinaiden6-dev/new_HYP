# H593 M headroom与新方案审查

本次核算：原COST1五折头481/593、原ColNomic自然C128；对现有缓存做描述性分析，不训练新模型，不新增GPU工作。独立核算通过。

## 核心发现

原头112个错误中，23个target未进C128，89个target在场。后者中68个target的M高于最终错误候选；40个target的M在全部128候选中严格第一，分布在20个组件及全部五折。

这40个正确选项已经存在于原M的候选排序中，原小头没有采用它们。它支持进一步研究现有M的利用空间，尚未证明必须改造M的生成方式。

直接把全593改为选M最大者，只有379个正确：相对COST1救40、损142。用真实答案指导选择原头或M最大者，可覆盖521个正确；这是两种规则的oracle联合覆盖，不是新模型成绩，也不是所有使用M的机制的上限。

| 案例 | target的M排名（全C128） | 原头target挑战者排名 | 解释 |
|---|---:|---:|---|
| DIFFICULT-0011 |19|6|胜过原错误答案，但多个错误候选的M更高|
| OUTCOME-0477 |1|2|原M已把target排第一，小头改变了这个顺序|

原40例中，历史GAP_BIAS2已经救回12个，仍错28个；历史CONTENT_BOX_CE_POLISH18救回16个，仍错24个。不能重复宣称这些历史修复为新贡献；二者相对481的总体损失分别为5和7，仍须完整报告。

## 对附件方案的判断

可行，但应在P2前增加固定M0的利用空间检查。必须分开：读出未利用已有证据；matcher保留的信息在聚合时丢失；原图可见信息未进入冻结表示；原图未提供可辨线索。人眼可见和M相近不能直接把失败归因于标量压缩。

P2应加入原输入同预算重训、同容量M0-only、两侧均值、均值加分布统计的对照。先保留M0添加小残差支路，冻结内容定义，通过完整C128和HOLD/SWITCH检验增量；不要再次要求M单独完成全部身份分类。u/v统计的负结果不能证明空间无效或M0是充分统计量。

P3/P4保留为后续。reference差异需要双向坐标对齐、区分内容差异与未匹配，并以固定RAW winner对全部自然challengers验证；不能用已知target人工配对后当作端到端成绩。

完整修订：[方案审查与执行顺序](../plan/RC_M_HEADROOM_AND_LEARNED_SUPPORT_REVIEW_20260927.md)。

## 可复核文件

- [全593逐例表](../results/rc_h593_m_scalar_headroom_audit_20260927_v1/per_query.csv)
- [完整结果与来源哈希](../results/rc_h593_m_scalar_headroom_audit_20260927_v1/result.json)
- [独立核算](../results/rc_h593_m_scalar_headroom_audit_20260927_v1/independent_validation.json)
- [与历史方法交叉](../results/rc_h593_m_scalar_headroom_audit_20260927_v1/historical_overlap.json)
- [计算程序](../programs/audit_rc_h593_m_scalar_headroom_v1.py)
- [独立验证程序](../programs/verify_rc_h593_m_scalar_headroom_v1.py)

验证直接读取593份原operator记录，检查75,904个M与缓存一致，验证原折动作和身份，重算所有排名与计数。全部为已打开H593的事后机制分析，不能代替新的泛化确认。本次未改变原COST1、历史实验参数或预测。


## 后续全量481／492对照（2026-09-27）

[全593对照与逐例清单](../reports/REPORT_H593_M_481_492_FULL_COMPARISON_20260927.md)已补齐。492当前共有29个M第一但错误：原40例剩28，加上新误伤OUTCOME-0721。后续方案以本全量对照为起点，保留历史496强对照；不重复旧校准与乘积项实验。
