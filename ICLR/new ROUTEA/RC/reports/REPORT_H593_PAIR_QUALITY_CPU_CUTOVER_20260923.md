# H593 双图质量分支：CPU 执行迁移

按用户 2026-09-23 指令“不用测，直接改程序，然后去cpuonly分区”执行。原 GPU 容量任务 5159145 此前已完成；没有追加容量测试，也不再把容量测试设为 CPU 训练前置。

首批训练任务：5159240、5159241、5159242、5159243。核查时全部 PENDING / Resources，分区 cpuonly；每项 8 CPU、32G 内存、10 分钟，无 GPU 请求。实际 Slurm batch script 已逐字节核对。最多 4 项并行，450 秒边界保存，后续分片、六臂五折其余配置及汇总均只提交 cpuonly。

旧 GPU 调度器已停止；5159207、5159208、5159212、5159215 均在启动前取消。旧 GPU 程序、authority、容量结果保留，CPU 采用独立 authority 和结果目录。

模型定义、缓存输入、完整 C128、FP64、COST1、seed17、训练顺序、2000 步初始化和 2000 次 query 更新不变。CPU 与 GPU 不宣称位级等价。中间量和检查点继续保存。当前仅确认代码迁移、提交与资源正确；尚无本 CPU 分支训练结果或速度结论。

- 程序：`programs/run_rc_h593_pair_quality_cpu_v1.py`
- 调度器：`programs/dispatch_rc_h593_pair_quality_cpu_v1.py`
- 启动脚本：`slurm/rc_h593_pair_quality_cpu_v1.sbatch`
- 计划：`plan/RC_H593_PAIR_QUALITY_CPU_V1_20260923.md`
- 封存：`registry/rc_h593_pair_quality_cpu_authority_v1_20260923.json`
- 结果：`results/rc_h593_pair_quality_cpu_v1/`
- 迁移凭据：`results/rc_h593_pair_quality_cpu_v1/cpu_submission_receipt.json`
