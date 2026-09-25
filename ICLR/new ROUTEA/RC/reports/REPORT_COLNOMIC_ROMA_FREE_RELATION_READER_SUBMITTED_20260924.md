# 无 RoMa 的联合关系读出：首轮已启动

本轮检验：冻结 ColNomic tokens 后，让可学习模块在对应结构被压成逐patch摘要之前联合读出，是否能获得可泛化的候选区分。用户2026-09-24“做”授权。

## 固定范围

- 原H593 fold0的64 TRAIN、32 held，按预定SHA选取，不按结果筛图。64 TRAIN包含32个原身份/来源component；目标在池内64/64，这不是筛选条件。held标签在预测封存前不读取。
- 原 ColNomic 自然C128完整保留；无新增候选、编码器或RoMa前向。3529份既有纯ColNomic图片token文件供各臂共用。
- BASE三参数末端、COMPRESSED历史逐patch算子、SET无坐标联合读出、SPATIAL邻域联合读出。SET/SPATIAL各577参数、同初始化、同训练序列和预算；COMPRESSED8353参数，参数量另列。
- 身份标签COST1训练，每臂固定8遍/512次query更新。无M教师损失、无M输入。原RoMa+COST1只在结果封存后作为同面板系统参照。
- 全部held另做SET/SPATIAL位置绑定打乱及一致180度旋转；逐候选L/R/logit保留，固定诊断图保留完整关系层状态。

## 已完成的工程验证

`synthetic_validation.json`：独立NumPy邻域计算误差0；逐候选VJP与完整autograd梯度误差不超过5.56e-17；模型优化器续跑一致。SET打乱不变，SPATIAL响应打乱，两者对一致旋转不变。

`launcher_validation.json`：六类实际shell分支用模拟调度器执行通过，覆盖正常继续、训练续跑、BASE续跑、后续提交失败重试、科学检查失败停止、重试次数耗尽。

`first_pair_validation.json`：固定首个TRAIN query、固定候选0，原自由内容L0重算误差0；真实768×594 patch关系的空间读出前反向成功，末层梯度非零。临时模型丢弃、未读真实标签。这只证明单对工程通路，不代表完整容量或科学成功。

## 作业与自动链路

首任务 **5162808**，2026-09-24 20:23:55（集群时间）提交。核查时已在 **dev_cpuonly/hkn0002 RUNNING**，8CPU、32GB、10分钟。

1. 四臂完整C128容量验证，每臂2个临时更新；真实标签不读取，临时模型丢弃。最大单update90秒、RSS24GiB为扩展门。
2. 验证通过后，完成BASE，再自动提交COMPRESSED/SET/SPATIAL三项训练；dev_cpuonly、cpuonly共同排队，逐query保存断点、同Job续跑。
3. 所有臂固定端点和全部预测封存后，自动CPU独立核算、打开held标签、生成结果报告。

20:28前的后续核查：完整C128容量门已通过。SET两次4.35/4.37秒，SPATIAL17.80/16.76秒；该验证进程峰值RSS约1.35GiB。BASE已完成512更新并封存64 TRAIN+32 held共96份预测，尚未打开held标签。

|Job|内容|核查状态|
|---|---|---|
|5162808|容量验证及BASE|产物已完成，后续提交成功|
|5162810_1|COMPRESSED旧压缩算子的同末端对照|dev_cpuonly RUNNING，日志至少20/512步|
|5162811_2|SET无坐标联合读出|dev_cpuonly,cpuonly PENDING|
|5162812_3|SPATIAL位置邻域联合读出|dev_cpuonly,cpuonly PENDING|

已取回5162810实际Slurm脚本，其SHA与通过六类shell测试的本地脚本完全一致。上述三项均为8CPU、32GB、10分钟，可断点续跑；全部封存后自动汇总。后续实际状态以 `dispatch/`、容量验证和各臂断点为准。这不是结果报告，不宣称已有检索收益。

## 文件

- 执行协议：`plan/RC_COLNOMIC_ROMA_FREE_RELATION_READOUT_EXECUTION_V1_20260924.md`
- 冻结authority：`registry/rc_colnomic_relation_reader_authority_v1_20260924.json`
- 产物目录：`results/rc_colnomic_relation_reader_v1/`
- 算子：`programs/rc_colnomic_relation_reader_v1.py`
- 运行/独立核算：`programs/run_rc_colnomic_relation_reader_v1.py`
- 调度：`programs/dispatch_rc_colnomic_relation_reader_v1.py`
- 最终报告预定：`reports/REPORT_COLNOMIC_ROMA_FREE_RELATION_READER_V1_20260924.md`

本轮是已打开开发数据的机制探索，不能称全593结果或独立外部确认。新颖性不由新增模块本身保证；首要分界是相同容量下，正确结构是否提供额外可泛化信息。
