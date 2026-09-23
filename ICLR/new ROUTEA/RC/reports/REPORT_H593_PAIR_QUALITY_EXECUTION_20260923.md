# 双图质量区分实验：执行记录

2026-09-23T08:07:27.775819+00:00

历史及重复性核查完成，见 REPORT_H593_PAIR_QUALITY_HISTORY_AND_OVERLAP_20260923.md。原融合、RoMa统一采集、坐标与质量通路读出均保持原协议；仅补冻结内容后的来源×交互质量学习。

- 合成NumPy独立前向、全token重排、单图/双图依赖及两遍VJP通过；最大梯度差约1.1e-17。
- 实际shell参数测试通过；Slurm接收脚本与本地逐字节一致。
- 容量任务 5159145 已提交，10分钟、1GPU、8CPU、32GiB；accelerated/dev_accelerated共同排队。首次核查PENDING，QOSMaxJobsPerUserLimit。不是运行完成或科学结果。
- 容量检查使用自然query000的完整C128、固定合成target位置、不读真实身份；新增编码器前向0。
- 后台自动链已确认独立进程存活。仅当容量验证与0退出码均通过，最多4项并行推进6臂×5折。每项2000固定query更新、FP64、原候选，保存恢复状态；异常停止。
- 全预测封存后再进行标签归约；当前尚无新准确率或替代RoMa结论。

来源：results/rc_h593_pair_quality_v1/ 下 preflight.json、launcher_validation.json、submission_verified.json、dispatch.json、dispatch_status.json、queue_before_submission.txt。

## 容量任务完成与续提（2026-09-23）

- 用户报告的 `5195145` 未在 accounting 查到；本实验任务为 `5159145`。已核实 `COMPLETED / 0:0`，运行 47 秒；`benchmark_validation.json` 为 `PAIR_QUALITY_FULL128_CAPACITY_PASS`，authority 和 payload SHA 均匹配。
- 四种来源/交互组合均完成完整 C128 更新；ColNomic 自由内容分数与原缓存误差最大约 `1.11e-16`；优化器断点恢复一致。新增编码器前向为 0，容量检查读取真实身份标签为 0。
- 这是容量及实现检查，不是精度实验结果。重复同一缓存 query 的热运行时间不能直接推算整折训练耗时。
- 优先运行容量任务时临时暂停的 46 项待运行任务已全部恢复；恢复记录位于 `results/rc_h593_pair_quality_priority_5159145/restored/`。
- 正式训练自动链已接续。`5159207` 对应 fold0 / COL_ONLY_SINGLE，`5159208` 对应 fold0 / COL_ONLY_PAIR，`5159212` 对应 fold0 / COARSE_SINGLE。
- 08:34 UTC 复核：第四项 `5159215`（fold0 / COARSE_PAIR）已自动提交，自动链恢复 `BENCHMARK_OR_TRAIN`。前三项在普通 accelerated 等待 Priority，第四项在 accelerated/dev_accelerated 等待 QOSMaxJobsPerUserLimit；均尚未产生训练结果。
- 续提遇到 dev 分区 QOS 限额：每用户最多运行 1 项、提交 4 项。上述三项转入普通 `accelerated`，逐项核实仍为 10 分钟、相同资源、相同科学程序和依赖；详情见 `results/rc_h593_pair_quality_v1/scheduler_adjustment_20260923.json`。自动链保持最多 4 项在途，原协议和源文件未修改。
- 后续判断仍须等待相同五折、相同 C128 的来源×交互与 NONE/ROMA 对照完成。目前没有新增准确率结论。

## 用户指定改为 CPU 执行（2026-09-23 08:43 UTC）

用户明确要求“不用测，直接改程序，然后去cpuonly分区”。未新跑 CPU 容量测试，已生成 CPU v2 执行入口，保留原 v1 源码、容量结果与历史 authority。新 authority 为 `registry/rc_h593_pair_quality_cpu_authority_v2_20260923.json`，输出为 `results/rc_h593_pair_quality_cpu_v2/`。

| 对照（fold0） | 旧 GPU job（未运行，已取消） | CPU job |
|---|---|---|
| COL_ONLY_SINGLE | 5159207 | 5159234 |
| COL_ONLY_PAIR | 5159208 | 5159235 |
| COARSE_SINGLE | 5159212 | 5159236 |
| COARSE_PAIR | 5159215 | 5159237 |

四项均已逐项核实 `cpuonly`、8 CPU、32GiB、10分钟，`ReqTRES` 无 GPU；Slurm 实际接收脚本与本地逐字节一致。新自动链保持最多4项在途，后续训练和汇总均为 CPU。原 GPU 自动链在收到取消状态后停止；该主动迁移不能解释为训练失败。

核心数值、内容/质量计算、训练顺序、损失与聚合函数通过静态 AST 差异核对；仅执行设备、设备加载、authority/输出路径及启动/容量入口变更。未做 CPU/GPU 数值对照，不宣称两后端逐位相同或已经验证 CPU 耗时。新旧 checkpoint 分开，原数值验证保留来源，最终原有独立 NumPy 决策核验继续执行。

迁移证据：`results/rc_h593_pair_quality_cpu_migration_v2/` 中 `old_jobs.json`、`runner.diff`、`static_validation.json`、`replacements.json`；新自动提交记录位于 `results/rc_h593_pair_quality_cpu_v2/submissions/`。

## CPU执行去重（后续核查）

发现另有CPU v1链在运行相同六臂×五折；关键输入绑定与训练/评分代码已逐项核实等价，两链均未开始训练。已停止本会话的v2调度器并取消5159234–5159237，保留v1的5159240–5159243。后续唯一CPU结果入口为 `results/rc_h593_pair_quality_cpu_v1/`；v2的 `superseded.json` 已指向该入口。此为重复执行去重，不是科学版本或模型选择。

完整收口范围见 `REPORT_H593_MECHANISM_CLOSURE_SCOPE_20260923.md`。
