# M相位贯通：线程配置故障定位与修复

5166025的DependencyNeverSatisfied来自前置5166024在原H593锚回放校验处FAILED/1，并非排队本身异常。首批00–03的512对候选已完整保存，只有03能通过旧2e-10数值门；00–02决策一致但分数不完全一致。

## 根因

旧H593分解固定4线程，本轮V1改成2线程。独立诊断确认M、适配器权重、hidden和输入轴均相同；适配器输出SHA相同。query00候选64的冻结BF16投影仅1元素出现0.0625差值，BF16范数一致；下游一个patch差4.730041e-5，除以720后L差6.569501e-8。其他差异也只涉及极少patch，没有改变命中位置和已算4张图的最终动作。

恢复4线程重算00–02全部384候选，adapter输出SHA、逐patch相似度和L均与原记录完全一致。结论由可重放脚本验证，而非依据“误差很小”推断。

## 修复与复用边界

- 新5166232：2worker×4线程、8CPU/16GB/10分钟，已在dev_cpuonly启动。
- 新5166233：4CPU/8GB/10分钟，cpuonly/dev_cpuonly，afterok:5166232，独立核算与汇总。
- 已取消永久依赖失败的旧5166025；旧失败5166024及其512对候选缓存保留。
- V2使用隔离目录重算全部8图/36世界，避免混用2线程和4线程数值结果；不重新运行RoMa、编码器或训练。
- 原2e-10门限、模型、精度、候选轴和科学干预完全不变。worlds、external、load_context、compute_candidate四个函数AST与V1一致。
- 调度按分配剩余时间派发下一查询；保留候选级断点和原Job续跑。
- 新Slurm实际提交脚本与本地文件逐字节一致。

## 产物

- [独立可重放诊断脚本](../programs/validate_rc_m_phase_bridge_thread_repair_v2.py)
- 独立证据：`results/rc_m_phase_bridge_thread_repair_v2/validation.json`
- 新计算目录：`results/rc_m_phase_external_internal_bridge_v2/`
- [修复协议](../plan/RC_M_PHASE_EXTERNAL_INTERNAL_BRIDGE_V2_THREAD_REPAIR_20260927.md)
- 提交回执：`registry/rc_m_phase_bridge_repair_v2_20260927.json`

这里只关闭工程故障；8图科学回放结果需等5166233独立验收后报告。QR/QRR已完成的其他任务与结果未改动。
