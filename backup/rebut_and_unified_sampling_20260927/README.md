# QR/QRR、M归因与统一采集快照（2026-09-27）

本批保留原工作区相对路径、源码和实验结果字节，不重新训练或修改预测。

|分支|归档状态|入口|
|---|---|---|
|H593的M与481/492逐图比较|完成的分析结果|[比较报告](../../ICLR/new%20ROUTEA/RC/reports/REPORT_H593_M_481_492_FULL_COMPARISON_20260927.md)|
|M的标量可改进空间|完成的分析及历史复核|[报告](../../ICLR/new%20ROUTEA/RC/reports/REPORT_H593_M_HEADROOM_AND_PROPOSAL_REVIEW_20260927.md)|
|QR/QRR v1|71张、60项拟合及独立回放完成；未证明新增关系分支稳定优于对照|[结果报告](../../ICLR/new%20ROUTEA/RC/results/rc_rebut_qr_qrr_v1/report.md) · [表格](../../ICLR/new%20ROUTEA/RC/results/rc_rebut_qr_qrr_v1/main_results.csv) · [独立验收](../../ICLR/new%20ROUTEA/RC/results/rc_rebut_qr_qrr_v1/validation.json)|
|QR/QRR冻结基线残差v2|协议、源码、工程检查和小规模运行检查；正式训练未提交|[计划](../../ICLR/new%20ROUTEA/RC/plan/RC_REBUT_QR_QRR_FROZEN_BASELINE_V2_20260927.md) · [工程检查](../../ICLR/new%20ROUTEA/RC/results/rc_rebut_qr_qrr_frozenbase_v2/engineering_checks.json)|
|相位干预贯通外部与内部|v1/v2、线程修复、逐候选及逐patch证据|[v2报告](../../ICLR/new%20ROUTEA/RC/reports/REPORT_M_PHASE_EXTERNAL_INTERNAL_BRIDGE_V2_20260927.md) · [结果目录](../../ICLR/new%20ROUTEA/RC/results/rc_m_phase_external_internal_bridge_v2)|
|属性恢复对照|准备、CPU探针和提交状态，不能当作已完成因果结论|[执行报告](../../ICLR/new%20ROUTEA/RC/reports/REPORT_M_PROPERTY_RESTORATION_SUBMITTED_20260927.md)|
|统一采集128张|恢复、断点和替补47分片的执行快照|[记录](../../ICLR/new%20ROUTEA/RC/reports/REPORT_UNIFIED128_RESTART_V2_20260927.md)|
|剩余465张统一采集|4个48小时GPU任务已提交；完成状态以逐query验收为准|[提交报告](../../ICLR/new%20ROUTEA/RC/reports/REPORT_UNIFIED593_LONG4_SUBMITTED_20260927.md) · [提交记录](../../ICLR/new%20ROUTEA/RC/results/rc_h593_unified_long4_v1/submission.json)|

统一采集新增索引128–592，四个GPU任务为`5166343_[0-3%4]`，对应CPU导出`5166345_[0-3%4]`，最终593张验收`5166349`。前128张继续由原链负责。运行中的进度JSON只是复制时的快照。

归档包含报告、计划、协议、执行脚本、逐候选结果和轻量归因中间量。两版相位贯通实验的小NPZ只含`worlds、M、L、maxsim、argmax`，保留逐patch证据用于后续归因；没有图像、embedding或模型权重。原图、模型/优化器二进制checkpoint、token与GPU采集缓存、ERR/OUT/LOG均排除，并在清单中记录。

缺少被排除的原图、权重和缓存时，单独克隆此仓库不能直接重跑全部实验。原绝对路径和SHA按实验封存原样保留。

[文件及排除清单](files.json) · [复制一致性核验](validation.json)
