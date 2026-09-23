# 当前实验与结果快照（2026-09-23）

本次追加9月20日以后已形成的实验代码、协议、逐折结果、验证记录、报告与执行状态。旧结果原样保留；运行中的任务不记作科学结论。原图、二进制特征缓存、训练checkpoint、err/out/log和重复压缩包未上传，已有模型Release保持不变。小头JSON参数和结果中的逐候选分数保留。

## 已完成、可以直接阅读的结果

### H593六项消融及纯内容学习对照

原H593分组五折、自然RAW C128，全部593张计入分母，包含23张候选缺失。相同检索监督下：

| 模型 | 正确/593 | 对RAW救回/误伤 |
|---|---:|---:|
| RAW | 426 | 0/0 |
| COST1_CONTENT7（不读取RoMa） | 427 | 1/0 |
| CE_CONTENT7（不读取RoMa） | 431 | 5/0 |
| 完整COST1 | 481 | 58/3 |
| 完整CE | 486 | 71/11 |

[纯内容完整报告](<ICLR/new ROUTEA/RC/results/rc_h593_learned_colnomic_only_v1/report_zh.md>) · [逐折参数与分数](<ICLR/new ROUTEA/RC/results/rc_h593_learned_colnomic_only_v1>) · [六项消融及边界](<ICLR/new ROUTEA/RC/results/rc_h593_six_feature_ablation_v1/report_zh.md>)

这些对照支持当前联合证据头的增量价值，不证明所有纯ColNomic读出都不能替代它，也不证明六项统计各自不可缺少。后续S置零、偏置、门控、候选排序等优化探索分别保留在结果索引中，不能用开发后挑选的较高值替换预定COST1主结果。

### 六臂第一折：表示与读出机制

固定原第一折，held119张、TRAIN474张（训练实际使用457张候选命中项），自然RAW C128；下表6张候选缺失仍计入119分母。第一折优先完成，其他折未宣称完成。

| 方法 | 正确/119 | 对RAW救回/误伤 |
|---|---:|---:|
| RAW | 98 | 0/0 |
| 原七参数COST1 | 105 | 9/2 |
| COL_ONLY_SINGLE | 92 | 0/6 |
| COL_ONLY_PAIR | 96 | 1/3 |
| COARSE_SINGLE | 95 | 0/3 |
| COARSE_PAIR | 98 | 1/1 |
| NONE | 98 | 0/0 |
| ROMA（原M＋重训五参数读出） | 104 | 8/2 |

六臂使用简化五参数读出，不能描述为原七参数头的直接替换。ROMA的M候选绑定打乱后为80/119。该第一折结果尚不能彻底区分“ColNomic表示缺信息”和“当前读出训练泛化不足”；不能外推为全593或全部纯内容模型的结论。

[机制分析](<ICLR/new ROUTEA/RC/reports/REPORT_PAIR_QUALITY_FOLD0_MECHANISM_20260923.md>) · [完整结果与独立核算](<ICLR/new ROUTEA/RC/results/rc_pair_quality_fold0_analysis_v1>)

### 已有质量路径与视觉干预分析

[41张冻结头路径回放](<ICLR/new ROUTEA/RC/results/rc_h593_subset41_frozen_paths_v1>) · [55张视觉机制分析](<ICLR/new ROUTEA/RC/results/rc_h593_visual55_mechanism_v1>) · [H593整体质量读出](<ICLR/new ROUTEA/RC/results/rc_h593_quality_operator_eval_v1>)

各面板、冻结头/重训头、干预与计算精度按各自报告解释；相同正确数不代表相同逐图决策，也不证明细匹配在全部任务上无用。

## 进行中：不能记作完成结果

| 分支 | 本次快照记录 | 解释边界 |
|---|---|---|
| H71粗/细/粗细融合 | 首批5159608正常结束；10/50配置验收，均为无融合对照；40项续跑5160117，回调5160118 | 真正融合收益尚待验收 |
| 第一折M教师诊断 | 复用固定ColNomic tokens，直接学习RoMa的M；CPU断点训练进行中 | 教师监督诊断，不是原retrieval-only主训练，也还不是新准确率结果 |
| ColPali跨编码器 | 补593张query tokens；5160112为50片、accelerated、%50；5160114等待汇总 | 尚未完成跨编码器评分/训练，不宣称迁移成功 |

[H71续跑说明](<ICLR/new ROUTEA/RC/reports/REPORT_H71_FUSION_WAVE0_CONTINUATION_20260923.md>) · [M教师方案](<ICLR/new ROUTEA/RC/plan/RC_FOLD0_ROMA_M_TEACHER_DIAGNOSTIC_V1_20260923.md>) · [ColPali采集说明](<ICLR/new ROUTEA/RC/reports/REPORT_COLPALI_H593_QUERY_CACHE_START_20260923.md>)

作业状态是上传时的快照，后续调度变化以workspace新记录为准。本仓库中的旧已封存状态文件用于保留历史，不能只凭文件名判断最新进度。

## 查找全部新增材料

- [实验目录索引](backup/experiment_sync_20260923/experiment_index.json)
- [文件SHA256清单](backup/experiment_sync_20260923/files.json)
- [上传范围与统计](backup/experiment_sync_20260923/summary.json)
- [2026-09-20完整汇总更新包](<ICLR/new ROUTEA/RC/reports/new_hyp_complete_results_20260920_controls_v1>)

这是研究归档，不声称仅靠Git仓库即可重跑全部实验。历史绝对路径、冻结来源SHA和未上传缓存的引用按原样保留。
