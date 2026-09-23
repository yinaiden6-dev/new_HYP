# ColPali 跨检索器验证：完成结果（2026-09-24）

本页更新 9 月 23 日快照中的 ColPali 未完成状态；历史失败记录保留。

使用 ColPali 自有全图库检索 C128，H593 分组五折。目标命中 467/593，126 张候选缺失仍在分母中。

| 方法 | 正确/593 | 相对 RAW 救回/损失 |
|---|---:|---:|
| ColPali RAW | 283 | 0/0 |
| CONTENT7 COST1 | 286 | 3/0 |
| CONTENT7 CE | 294 | 16/5 |
| MASS5 COST1 | **348** | **65/0** |
| MASS5 CE | **365** | **93/11** |

这是整体 RoMa 质量 M 与自由内容 L 的简化 MASS5，经各折重训后的跨检索器证据。不是原七参数零样本迁移，也不是去 RoMa 学生模型。候选绑定错配使 COST1 降至 252、CE 降至 120。

- [完整报告与边界](<ICLR/new ROUTEA/RC/reports/REPORT_COLPALI_NATIVE_MASS_H593_FINAL_20260924.md>)
- [593 张最终预测与统计](<ICLR/new ROUTEA/RC/results/rc_colpali_native_mass_head_v1/result.json>)
- [五折参数及全部 challenger 决策分数](<ICLR/new ROUTEA/RC/results/rc_colpali_native_mass_head_v1>)：fold0–fold4/payload.json 含各臂每图 127 个 logits_hex（IEEE 浮点十六进制表示）；RAW winner 对应 HOLD=0。queries/queryNNN/payload.json 给出候选轴、winner、challenger_positions、RAW 分数、内容统计及 M，可重建完整 128 候选评分。
- [ColPali 自有 C128 与全图库分数](<ICLR/new ROUTEA/RC/results/rc_colpali_native_c128_v1>)
- [原生候选 RoMa 质量数据](<ICLR/new ROUTEA/RC/results/rc_colpali_native_quality_v1>)
- [CPU 数值一致性与迁移记录](<ICLR/new ROUTEA/RC/results/rc_colpali_native_mass_cpu_execution_v1>)
- [独立复核](<ICLR/new ROUTEA/RC/reports/rc_colpali_native_mass_final_independent_check_20260924.json>)
- [本次文件 SHA256 清单](backup/colpali_native_final_20260924/files.json)

源文件按字节保存。未上传原图、二进制特征缓存、训练 checkpoint 和运行日志；历史绝对路径和 SHA 指向原 workspace，不代表 clone 后全部依赖均可用。
