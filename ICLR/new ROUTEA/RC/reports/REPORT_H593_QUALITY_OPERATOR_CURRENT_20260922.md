# H593 质量×内容算子实验进展

续批已实际提交：`5157222_[47–74%46]`，28片/224图；`5157223` 为afterok续接任务，全部75片合格后自动提交原五折评测、诊断及独立汇总。`5156928` 已完成续批。剩余28片逐项核实为5分钟、8 CPU、32G、无GPU；保持cpuonly正常队列，同时运行仅覆盖本数组47–74的dev_cpuonly补位程序（PID4138063，每30秒检查共享dev最多4个提交名额，不移动已运行CPU任务）。程序边界和运行跳过测试通过。回执与实时状态：`results/rc_h593_quality_operator_dispatch_v1/dev_migration_5157222/`。

2026-09-22 14:49 CEST 更新：`5157112_32–46` 全部 COMPLETED 0:0，覆盖113图。联合原完成分片，现已逐项核验47/75片、369/593图：各query验证文件、payload与中间数据SHA一致，原生评分逐位一致，独立NumPy最大误差4.440892098500626e-16。还剩shard47–74共28片、224图，尚未进入全量五折评测。完成回执：`registry/rc_h593_operator_cpu_return_20260922/5157112_completion_verified.json`。

`5156928` 的afterok已满足，原先在cpuonly因Priority排队。沿用用户CPU/dev CPU安排，已原地迁为dev_cpuonly；仍为5分钟、1 CPU、8G、无GPU，原冻结续批程序不变。核验回执：`registry/rc_h593_operator_cpu_return_20260922/5156928_dev_continuation.json`。

最新调度状态（2026-09-22 14:08 CEST）：原 `5156927` 的 shard1–31 已完成并核验；剩余 shard32–46 改由纯 CPU 替代数组 `5157112` 执行。GPU→CPU 原地迁移保留了 GPU ReqTRES，因此先核验同一冻结脚本、参数与纯 CPU 请求，再取消旧的 15 个待运行副本。后续 `5156928` 已核验改为 `afterok:5157112`。

按用户最新要求，剩余分片同时使用 `cpuonly` 正常队列及 `dev_cpuonly` 补位队列：普通 CPU 队列不再暂挂，若先获资源即可执行；每 30 秒检查共享 dev 名额，只将尚未运行的指定分片补入 dev，不移动已运行任务、不重复执行。当前共享 dev 提交上限为 4；本数组全部剩余分片均为 5 分钟、8 CPU、32G、无 GPU。`5157112_32` 已 COMPLETED 0:0，用时 2:30；完成后自动补入 `_36`，实证补位已生效。本安排仅覆盖当前数组，不自动修改未来新数组。

逐项核验回执：`registry/rc_h593_operator_cpu_return_20260922/dual_partition_release_verified.json`。补位进程 PID 3924794 已在独立会话确认存活；记录：`results/rc_h593_quality_operator_dispatch_v1/dev_migration_5157112/`。早先 `dual_partition_release.json` 的状态检查错误地比较了带括号的 squeue 原因，现已标记失效并由上述独立 scontrol 逐项核验回执替代。调度变化不构成新的科学结果，完整 593 图评测仍待完成。

以下为历史进展记录。

2026-09-22 12:41 CEST。

用户授权继续实验，并要求保留中间数据。本轮在原 H593、原五折、自然 RAW C128 上，分离整体质量、query 汇聚加权、reference MaxSim 加权三个评分操作，增加固定自由命中位置的第九臂。固定原 COST1/CE 小头与同协议重训均执行。它与旧的六输入列置零消融不同。

- `5156888_0`：CPU 资格试运行，已按用户询问迁至 `dev_cpuonly`；15 分钟、8 CPU、32G，没有 GPU 请求。当前等待共享 dev QoS 的运行名额，尚无自然样本资格结论。
- `5156904`：CPU 自动后续，`afterok:5156888`、前置失败自动取消。通过资格后以最多 46 片一波完成剩余 74 片，再自动提交五折评测、逐候选诊断及独立汇总。后续分片和评测任务目前尚未创建。
- 算子与小头合成检验已通过；五种启动阶段的真实 shell 参数/环境传递已动态验证。已提交的两个脚本均保存，并与冻结版本逐字节比较。

预设读法：固定头的下降可能包含输入分布变化，必须与重训结果合看。重选位置提高单个候选的分数，不等于提高身份区分；分析 target 与错误候选的相对收益，以及最终救回和损失。三因子分解表示固定模型分数的函数作用，不作为图像空间 ownership 的证明。

尚无新的 593 图准确率结果，亦尚未启动纯 ColNomic token 适配器或粗/细特征融合训练。下一结构选择应依据本轮作用路径及并行坐标实验的结果。

入口：

- 计划：`plan/RC_H593_QUALITY_OPERATOR_V1_20260922.md`
- 中间数据索引：`results/rc_h593_quality_operator_v1/INTERMEDIATES_README.md`
- 自动分批记录：`results/rc_h593_quality_operator_dispatch_v1/`
- 最终评测结果：`results/rc_h593_quality_operator_eval_v1/`
- 最终报告：`reports/REPORT_H593_QUALITY_OPERATOR_V1_20260922.md`（完成汇总后生成）

2026-09-22 12:55 CEST补充：5156888已COMPLETED 0:0，2分31秒，8图全128候选、36,864项独立NumPy核验通过，最大绝对误差3.3306690738754696e-16，原生逐位一致，err为空。5156904已实际提交首波CPU任务5156927（shard1–46，%46）及下一波调度5156928（afterok）。全量评测尚未完成。后续尚缺实验及已有对照复用范围见plan/RC_NEW_HYP_REMAINING_EXPERIMENTS_V1_20260922.md。

2026-09-22 13:05 CEST：用户要求将5156927迁至dev_cpuonly。整数组迁移被Slurm的dev最多4个提交任务限制拒绝，原46片保持可恢复；先逐片迁入1–4，已逐项核对15分钟、8 CPU、32G、无GPU。启动仅针对5156927_1–46的自动补位程序，每30秒检查一次共享dev名额，完成迁移或24小时上限后退出，不重提/取消任务、不改实验程序或后续5156928的afterok依赖。独立会话已核实进程持续存活，且程序已实际迁移并核验第5片；第2片已在dev_cpuonly运行。其余仍在cpuonly排队的片会按空位迁移。此安排仅限用户指定数组，不自动迁移未来新Job ID。

迁移登记：registry/rc_h593_quality_operator_dev_migration_5156927_v1_20260922.json。实时补位记录：results/rc_h593_quality_operator_dispatch_v1/dev_migration_5156927/status.json及events.jsonl。后台PID记录于同目录spawn.json；阶段完成状态应由Slurm和各shard validation另行核对，不将补位程序退出当作实验完成。

2026-09-22 13:36 CEST：按用户新指令，5156927_20–46共27片已全部迁至accelerated，每项新增1 GPU分配，原15分钟、8 CPU、32G及依赖逐项核验保留。原程序继续执行CPU FP64，不改变实验算术。最新核查其中3片RUNNING、24片PENDING。原dev补位进程3782527已停止，新V2进程3869364仅负责5156927_1–19，明确排除20–46；独立会话核验进程和心跳。V2启动记录在同目录spawn_v2.json。27片迁移完整回执：registry/rc_h593_quality_operator_accelerated_20_46_migration_20260922.json。
