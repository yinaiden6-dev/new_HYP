# ColPali 原生C128评分迁至CPU

当前评分复用已有ColPali tokens和RoMa M，只进行FP64内容矩阵、MaxSim及统计量计算，无编码器/RoMa前向。原程序写死CUDA和GPU资源，直接换分区会失败；已用单独CPU执行包装替换3处设备表达式，原科学程序和authority未改。

CPU资格5160935在cpuonly完成（39秒，exit0:0）：query ordinal0、50共256个候选对全部回放，评分循环约0.99秒，资格总程序时间2.82秒。对GPU结果的最大raw分数差1.14e-13，统计量最大差2.22e-16，trace值及匹配索引一致，RAW winner一致。数值资格通过，不宣称CPU/GPU逐bit一致。

剩余49片替代数组：5160936_[1-49%50]，cpuonly，每片申请8CPU、24GB、5分钟，无GPU请求。保留50的并行上限，逐query存储，180秒主动断点，超时续跑至多16次。CPUonly分区可能整节点分配，申请CPU数与实际分配CPU数应分别报告。

仅在确认原5160876的49片全部仍PENDING后执行切换：先hold旧数组，提交hold的新CPU数组，保存并逐字节核对实际提交脚本，核对新请求无GPU；更新5160877的afterok为5160936并回读确认，再取消旧待运行数组、释放CPU替代。5160878依旧等待5160877，已完成GPU首片5160875_0结果保留。

原候选始终是ColPali自己的自然C128，模型、训练损失和身份隔离不变。当前是执行方式迁移，不是新的检索结果。

- CPU执行授权：registry/rc_colpali_native_mass_cpu_execution_v1_20260924.json
- CPU全候选核验：results/rc_colpali_native_mass_cpu_execution_v1/qualification.json
- 切换及依赖回执：results/rc_colpali_native_mass_cpu_execution_v1/cpu_replacement.json
- 逐任务执行记录：results/rc_colpali_native_mass_cpu_execution_v1/runs/
