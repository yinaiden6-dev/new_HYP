# 内部 M：最终结果、失败定位与 V4 阶段快照

采集窗口：2026-09-24T10:12:34.219290+00:00 至 2026-09-24T10:12:37.384022+00:00。

[下载 ZIP](new_HYP_internal_m_diagnostics_v4_20260924.zip) · [逐文件 SHA256 清单](files.json)

本包承接 Git 提交 `0f1eead88613d5d875f81bd45ef0dea0f173ed3b` 已发布的 V3 最终结果，补齐失败定位与 V4 的源码、协议、参数配置、已完成阶段的逐候选结果及验收。

- **V3 已完成**：同一 TRAIN16 / ColNomic 自然 C128，RAW 8/16；内部真实 M 与恒定 M 均 8/16；外部加性和乘积头均 12/16。优化预算不同，不能作为同预算优劣结论。
- **诊断已完成**：固定内容后统一重拟合，真实 M 9/16、恒定训练臂 10/16。读出优化有影响，但未建立真实 M 的特有收益。单张已保存 trace 显示 M 条件支路幅度偏弱，这是待检验解释。
- **V4 尚未完成**：只放大标准化 M 输入尺度，保留真实/错绑/恒定对照；首步工程验收已通过。快照记录 PRE_REAL 至第 73/128 次更新，PRE_SHUFFLED 至第 59/128 次更新。已完成的第16步终点已收录；没有最终 V4 或新留出结论。

主要入口（仓库浏览）：

- [V3 最终结果](../../ICLR/new%20ROUTEA/RC/reports/REPORT_COLNOMIC_INTERNAL_M_LEARNED_USE_V3_FINAL_20260924.md)
- [失败定位与修复依据](../../ICLR/new%20ROUTEA/RC/reports/REPORT_COLNOMIC_INTERNAL_M_V3_CAUSE_AND_V4_REPAIR_20260924.md)
- [V4 冻结协议](../../ICLR/new%20ROUTEA/RC/plan/RC_COLNOMIC_INTERNAL_M_CONDITION_SCALE_V4_EXECUTION_20260924.md)
- [V4 阶段数据](../../ICLR/new%20ROUTEA/RC/results/rc_internal_m_condition_scale_v4/)

ZIP 内按 workspace 相对路径保留结构；上述相对链接供 GitHub 浏览，解压后请从根目录的 ICLR/new ROUTEA/RC 进入。原报告保留其记录时的状态；本 README 与 files.json 给出本次快照边界。

保存小头数值参数、完整候选分数和文本收据；排除模型/适配器二进制权重、token 缓存、原图、err/out/log、锁及 partial 文件。二进制依赖和原始封存 SHA 仍由源码/收据引用，因此本包并非可直接恢复训练的完整运行环境。历史 Qwen 乘积项和其他实验保留于仓库已有归档，本包专门收录内部 M 分支。
