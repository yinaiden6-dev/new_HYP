# 原28/99原因隔离：执行恢复记录

用户9月12日明确继续标注的“分清原因”并go；仅恢复这项有界工作，不无限续开其它研究。原模型不重训、不替换。当前无新科学结果。

已完成：
- 核心 programs/rc_convex_loss_cause_core_v1.py；SHA 9d7ef65f3c36ff2816a72f302eb11a91d7bd93279c77c2a8c1455925d80da69f；12组合成数值界资格测试PASS。
- TRAIN安全数据 results/rc_convex_train_loss_cause_v1/train_problem.json（101项、4128向量，原mixed96、原cost4）；input_seal.json在读EVAL之前封存。
- 六项已有EVAL数学条件位于 results/rc_opened_convex_loss_rescue_cones_v1/inputs/；五项保原28+99增一，一项允许取舍达到旧32至少29；全部旧witness精确重验。
- 主worker programs/run_rc_convex_cause_isolation_v1.py、freezer programs/freeze_rc_convex_cause_isolation_v1.py、plan/RC_CONVEX_CAUSE_ISOLATION_V1_20260912.md、三个Slurm入口rc_convex_cause_{train,cones,collect}_v1.sbatch已编写。

新增的第二TRAIN条件有明确对照理由：用户接受净增，但原损失仍将错误项乘4；历史PAIR/SIGN/RANK对照没有改这个成本。unit-cost96只改4→1，保留原PAIR64+FULL32、AdamW .03/.001、seed17、2000步；不同于手调评价阈值。

执行前还需收齐并冻结：
- programs/run_rc_unit_cost96_cause_v1.py（review_convex_cause_core代理负责）
- programs/collect_rc_convex_cause_isolation_v1.py（三模型版，cause_isolation_design代理负责）

随后：freeze authority registry/rc_convex_cause_isolation_authority_v1_20260912.json；运行两个无自然训练的preflight；提交TRAIN数组0-1（凸搜索+unitcost）、隔离数组0-5、依赖两数组的collector。每项accelerated/1GPU分配、实际CPU8/16G，10分钟。八项都有固定预算，不扫参数、不根据EVAL选模型。数值界均仅覆盖[-64,64]^7；AdamW的decoupled decay不冒称L2目标。隔离系数不得用于模型，只有TRAIN-only两头进入160图评价。

最终输出：results/rc_opened_convex_cause_readout_v1/result.json、independent_validation.json；reports/REPORT_CONVEX_CAUSE_ISOLATION_V1_20260912.md。


## 已提交并核对spool

- 5142472_[0-1]：原cost4凸搜索与同配方cost1对照。
- 5142473_[0-5]：六个隔离数学条件。
- 5142474：afterok两数组，自动独立汇总三头及风险界。
- 三份批处理与提交spool逐字节一致，记录results/rc_convex_train_loss_cause_v1/submission_verified.json。
- 原ec7保存参数上的真实记录损失已重算为0.9343481006454986，与原标量公式一致（不是最后一步前日志值）。
- 两份preflight通过；authority已冻结。源码和计划不得原地修改。


## 开发分区实际迁移

已对5142472训练数组、5142473_0/_1/_2诊断项增加dev_accelerated并保留accelerated备选。原Job ID、依赖、10分钟时限和代码数据不变。开发QoS每用户1运行、4提交；六项诊断整数组迁移被该限额拒绝，其余_3/_4/_5及collector暂留accelerated。dev_cpuonly/cpuonly test-only均通过，但当前估计等待更久，cpuonly按独占152CPU分配，未转入。训练第0项已在dev_accelerated完成56秒且证书/预测封存通过。详细快照registry/rc_convex_cause_partition_amendment_v1_20260912.json。


## 继续迁移与部分界结果

前三项诊断已完成且界验证PASS。继续时_3已启动，保持其运行不变；_4、_5和5142474汇总已加入dev_accelerated,accelerated。记录results/rc_convex_train_loss_cause_v1/partition_remaining_updates_20260912.json。原损失优化在固定盒中已达到[0.9311129960982376,0.9311227458347325]，原头损失0.9343481006454986；完整模型评价仍待独立汇总。


## 已全部完成

5142472两项、5142473六项、5142474汇总均COMPLETED0:0。独立三头121920logit及风险界比较PASS。原头28/99；TRAIN_CONVEX7仍28/99（两面板均0救0损）；TRAIN_UNIT_COST7为27/99（旧32为0救1损，128为5救5损）。cost1在128救回五个原HOLD错误，但新增四个RAW正确被错切、另一个原正确SWITCH转为错误候选；不是纯阈值平移。

原保存头损失0.9343481，固定盒内最优界[0.9311129961,0.9311227458]。六个既有可行改进区域的最小TRAIN损失均有严格高于原头的下界，包括允许取舍的旧32≥29条件。它证明这些具体目标与当前经验风险有冲突，不能外推全部净增方案、无界参数域或信息论不可识别。修改4倍成本不是单独有效解法；更充分优化原目标也没增加正确数。

结果results/rc_opened_convex_cause_readout_v1/{result.json,independent_validation.json,completion_readout.json}；报告reports/REPORT_CONVEX_CAUSE_ISOLATION_V1_20260912.md。原模型不替换，无新任务待监控。
