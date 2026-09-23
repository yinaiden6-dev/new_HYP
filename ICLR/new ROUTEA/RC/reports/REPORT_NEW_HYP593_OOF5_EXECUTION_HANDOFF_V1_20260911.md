# new HYP 593图五折交叉验证：执行交接

593张不同图片、68身份、64来源组。身份/组连通分量五折119/118/119/119/118；每折训练474或475张。256张复用合格输入，337张补算为43个分片。元数据独立进程复核与训练损失/梯度（含并列分数）、HOLD/物理序tie及缺席target检查已通过。

## 已提交的依赖链

| Job | 阶段 | 依赖 | 实际申请时限/并行上限 |
| --- | --- | --- | --- |
| 5140736 | 缺失337张RAW | 无 | 用户指定10分钟/片，46；本轮43片 |
| 5140737 | 缺失337张RoMa | 全部5140736成功 | 用户指定15分钟/片，46 |
| 5140738 | 复用256张的特征与独立索引复算 | 无 | 30分钟/片，46；32片 |
| 5140739 | 新337张的特征与独立索引复算 | 全部5140737成功 | 30分钟/片，46；43片 |
| 5140747 | 五折拟合及新进程重训/预测重放 | 全部5140738和5140739成功 | 1小时59分钟/折，5折并行 |
| 5140748 | 五折预测封存后join标签/统计 | 全部5140747成功 | 20分钟 |
| 5140750 | 独立动作/排名/分组计数复核并生成报告 | 5140748成功 | 20分钟 |

GPU为accelerated，CPU为cpuonly。加入dev_accelerated的尝试被Slurm账户/QOS规则拒绝，没有迁移成功。所有任务有2026-09-11 18:00 Berlin截止及运行时UTC16:00超时保护。并行46是上限，不是已获得46个GPU。

原launcher冻结值仍是59分钟/16上限的历史初始配置；当前调度值由用户override记录明确覆盖。以后若修复重提，必须显式保留RAW --time=00:10:00、RoMa --time=00:15:00及--array=0-42%46，不能仅凭旧launcher恢复较长时限。

已接完自动依赖链，不需要手动从输入再触发拟合。用户要求没事干就退出，因此本轮不继续监控队列。以上为提交和配置确认，不宣称任务已完成或模型已有收益。

## 模型与输出

BASE7_SMALL128、BASE7_ALL、CONSTANT1、CONDITIONAL4；主比较CONDITIONAL4对BASE7_ALL。折内重训，完整自然C128/127challenger，retrieval-only，无空间标注。缺席target不伪造训练正确项，评价完整保留593分母。旧EVAL来源已纳入这个新的开发交叉验证，结果不能与旧99/128直接比较。

预期结果：results/rc_new_hyp593_oof5_v1/result.json；独立复核result_validation.json；自动报告reports/REPORT_NEW_HYP593_GROUPED_OOF5_RESULT_V1_20260911.md。当前这些结果尚未在本轮读取。

程序与计划：plan/RC_NEW_HYP593_GROUPED_OOF5_V1_20260911.md。
输入/特征/训练authorities分别为registry/rc_new_hyp593_oof5_authority_v1_20260911.json、rc_new_hyp593_feature_authority_v1_20260911.json、rc_new_hyp593_training_authority_v1_20260911.json。
最终用户覆盖：registry/rc_new_hyp593_concurrency_update_v1_20260911.json和registry/rc_new_hyp593_user_time_limits_v2_20260911.json。
