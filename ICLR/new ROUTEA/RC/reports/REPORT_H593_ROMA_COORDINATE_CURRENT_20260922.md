# RoMa坐标精度实验当前状态

2026-09-22，用户明确u/v是匹配坐标精度，不是可见性权重粗化。

- 5156826：原资格试运行完成，5分22秒，query000全部128候选×四档精度；原生可见性/四评分逐位复现，2048标量独立回放通过。
- 用户追加必须保留中间数据，故建立V2留存扩展；机制和原评分保持不变。5156849：dev_accelerated，15分钟，已COMPLETED/0:0，用时7分19秒；query000全部128候选及8个part完整，独立2048标量与MaxSim中间量回放通过。
- 5156851：afterok:5156849（前置已通过）；负责每波46个query，accelerated，每项15分钟，直到593全量。任一波失败则后续不放行；成功部分保留。一次提交592被AssocGrpSubmitJobsLimit拒绝，未生成全量V1任务。
- 全部593完整后，自动执行原COST1/CE固定头及同协议重训五折、扰动诊断、独立回放、汇总报告。尚无新H593准确率结果。

数据索引：results/rc_h593_roma_coordinate_precision_v2/INTERMEDIATES_README.md。保存两侧token可见性、粗匹配器坐标/置信度、token对齐最终坐标及空间离散度、64×64坐标抽样、free/weighted MaxSim位置和token贡献、四评分、全部127×6特征；评测保存参数及全部127logits。完整1280稠密坐标和全部隐藏激活未归档，不称全分辨率重放。

执行计划：plan/RC_H593_ROMA_COORDINATE_PRECISION_V2_20260922.md。提交登记：registry/rc_h593_roma_coordinate_submission_v2_20260922.json。自动分批登记：results/rc_h593_roma_coordinate_dispatch_v2/afterNNN.json。

用户同时提出粗/细特征直接融合ColNomic、LLM前融合和跳过RoMa匹配器。实现核对已写入plan/RC_ROMA_FEATURE_COLNOMIC_FUSION_FEASIBILITY_20260922.md；这条融合实验尚未启动，不能称已实现或已改善。优先建议无匹配器的128维后融合，先分清粗/细特征增益，再比较LLM前注入。

用户进一步明确：希望解释联合增益后改善ColNomic自身。已核对旧learned_colnomic_only仅拟合候选级内容统计，没有学习token级质量或改造编码器；它是必须保留的旧对照，不能作为排除纯ColNomic改造的证据。新方案记录于上述融合可行性文档末节，尚未启动该训练。

2026-09-22 12:03 CEST：分批调度5156851已完成，实际创建5156860（query1-46，accelerated，%46）和5156861（afterok下一批调度）。回执results/rc_h593_roma_coordinate_dispatch_v2/after001.json。

2026-09-22 12:41 CEST：按用户指定，仅将5156860的1–4移至dev_accelerated，15分钟、8 CPU、64G、1 GPU均保留并逐项核对。任务1/2分别COMPLETED 0:0，7分26秒/7分21秒，query001/002已通过全部128候选、2048标量独立核算和原生逐位一致检查。任务3运行中，4受dev QoS名额限制，5在accelerated完成清理中；其余等待Priority。另启动CPU质量×内容算子分支，见REPORT_H593_QUALITY_OPERATOR_CURRENT_20260922.md；不改本坐标实验的任何冻结源或评分。

2026-09-22 13:36 CEST：5156860的46片全部COMPLETED、0:0，约7分19秒至8分04秒。连同pilot，query000–046共47/593通过资格，累计96,256项独立四评分标量核验，原生图/评分逐位复现；本轮重新核对各validation及payload SHA。5156861已COMPLETED、0:0、38秒，自动提交下一波5157033（query47–92）和5157034（afterok后续调度）。用户本次文字中的1516860未查到记录；这里核对的是此前同分支5156860，不对未知Job ID采取操作。全量检索评测尚未完成。
