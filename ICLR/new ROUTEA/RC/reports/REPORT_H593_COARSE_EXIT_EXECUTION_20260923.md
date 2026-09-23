# 粗阶段提前退出验证已提交

- GPU资格与计时：5158971，10分钟，1GPU/8CPU/64GB，accelerated与dev_accelerated共同排队。
- CPU自动续提：5158972，10分钟，afterok:5158971；仅在资格验证产物通过后启动首图，再以最多50并行覆盖70张并自动汇总。
- dev提交数量上限导致最初CPU后续提交被拒；已用独立调度器明确覆盖CPU分区为cpuonly，原科学程序和authority未改写。原失败提交没有生成CPU任务，也没有重复GPU任务。

新增GPU模型前向仅为固定8对的原完整路径/实际提前退出一致性与重复计时。55张粗阶段权重直接读缓存，15张只用GPU解码已缓存confidence，再由CPU重算全部C128评分。

固定原五折COST1、候选和零阈值。单列原41张与新增29张，不将70张当全593结果。全593需要后续共享阶段缓存补齐；本次不启动593个重复RoMa前向。

方案：plan/RC_H593_COARSE_EXIT_V1_20260923.md。科学结果将写入results/rc_h593_coarse_exit_v1/result.json及reports/REPORT_H593_COARSE_EXIT_V1_20260923.md，当前提交状态不等于资格或效果通过。
