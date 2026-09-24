# 论文收口稿与内部M V3完整结果

2026-09-24。内部真实M和恒定M均完成128次更新及TRAIN16终点评估；本次补齐此前88步快照之后的全部文本结果。文件SHA及排除清单见[files.json](files.json)。

- [论文收口稿：摘要、贡献、近邻工作及主表](../../ICLR/new%20ROUTEA/RC/reports/RC_NEW_HYP_PAPER_CLOSURE_DRAFT_20260924.md)
- [内部M V3最终解释及独立复核](../../ICLR/new%20ROUTEA/RC/reports/REPORT_COLNOMIC_INTERNAL_M_LEARNED_USE_V3_FINAL_20260924.md)
- [完整结果目录](../../ICLR/new%20ROUTEA/RC/results/rc_internal_m_learned_use_v3/)

| 同一TRAIN16，自然ColNomic C128 | 正确 | 对RAW救回/误伤 |
|---|---:|---:|
| RAW／无M共同读出 | 8/16 | 0/0 |
| 外部ADDITIVE4／PRODUCT5 | 12/16 | 4/0 |
| 内部PRE_CONSTANT／PRE_REAL，128更新 | 8/16 | 0/0 |
| PRE_REAL评估时换常量或错绑M | 8/16 | 0/0 |

参数、tokens和分数会变化，但内部没有产生纠错。内部128次逐query更新与外部2000次全批量更新的优化预算不同；这是TRAIN拟合诊断，没有probe/held结果，不证明内部注入原则上无效。外部模型的H593或外部面板成绩保持各自原口径。

保存完整C128内容分数、127挑战者logits、小头参数、逐步记录、配置、源码、报告及验收；二进制适配器/优化器checkpoint、token traces、原图和err/out日志未上传。绝对路径和原始封存SHA保留，恢复运行仍需要本地模型及缓存。
