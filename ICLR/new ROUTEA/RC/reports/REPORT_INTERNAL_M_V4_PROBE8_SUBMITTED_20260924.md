# 内部 M V4 跨组 PROBE8：已提交

2026-09-24。固定已完成的V4第128步，不训练、不重算RoMa或视觉编码器。

- 5162474：REAL，共用真实训练终点头，推理比较真实／恒定／错绑M。
- 5162475：训练错绑及训练常量的终点对照。
- 5162476：依赖两GPU任务成功，CPU独立动作重算、标签后接入与报告。

GPU任务已核对dev_accelerated,accelerated共同排队，单GPU、8CPU、64GB、10分钟；提交核对时均Priority。CPU汇总为cpuonly、8CPU、32GB、10分钟，无GPU，afterok依赖正确。三份调度器保存的脚本与源文件逐字节一致。

候选C128、8张probe及头均预先冻结。8张是原fold0未参与V4训练的组，但已在历史开发中查看，不是全新独立确认。真实M须通过内部路径影响内容，末端三参数头不直接读取M。

保留所有128个内容分数、127个动作logit及HOLD=0。次分析复用TRAIN16冻结重训头，和联合训练终点分别报告；不按probe择优。原TRAIN16结果10/16与读出诊断11/16均不是本次probe结果。

检查：4项评分/来源/无直接M/HOLD测试通过；原模型、缓存、终点与训练头哈希封存；8张既有编码缓存就绪。结果完成后自动写REPORT_INTERNAL_M_V4_PROBE8_20260924.md，不自动扩展训练。

[执行方案](../plan/RC_INTERNAL_M_V4_PROBE8_20260924.md) · [提交与脚本核对](../results/rc_internal_m_v4_probe_v1/submission_probe.json)
