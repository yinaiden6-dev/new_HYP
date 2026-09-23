# 当前实验与结果快照（2026-09-23）

> 后续更新：ColPali 已完成修复及自有 C128 的 H593 五折验证，见 [2026-09-24 最终结果](COLPALI_RESULTS_20260924.md)。本页下方未完成状态保留为历史快照。

增量更新：包含2026-09-23 19:12 UTC完成的H71最终汇总；本次补入M教师完整结果、H71全部50项配置及汇总、ColPali两版失败/修复记录。已完成与未完成分开列示。

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

### 第一折M教师诊断：已完成，未复现纠错

固定原第一折457张有效TRAIN、119张held、自然RAW C128。8353参数ColNomic关系读出直接学习TRAIN的RoMa M，训练2000步；held不参与拟合。此为教师监督诊断，不是原retrieval-only身份监督主模型的新成绩。

| 路径 | 正确/119 | 对RAW救回/损失 |
|---|---:|---:|
| RAW | 98 | 0/0 |
| 原生M＋原七参数COST1 | 105 | 9/2 |
| 原生M＋冻结五参数自由内容头 | 104 | 8/2 |
| 学生M替换、冻结原七参数M_ONLY | 98 | 0/0 |
| 学生M替换、冻结五参数自由内容头 | 98 | 0/0 |

学生两条路径均逐图退回RAW。学生M几乎为常数，TRAIN候选差异也未拟合好：中心化log MSE为0.215053，常数对照0.215038；held为0.229731，常数0.229711。这不能证明ColNomic tokens缺信息，也不能只归因于跨组泛化。原七参数M_ONLY仍保留原生局部权重。

[结果分析](<ICLR/new ROUTEA/RC/reports/REPORT_FOLD0_M_TEACHER_RESULT_20260923.md>) · [完整逐候选结果](<ICLR/new ROUTEA/RC/results/rc_fold0_mass_teacher_v1/result.json>) · [独立核算](<ICLR/new ROUTEA/RC/reports/fold0_mass_teacher_recount_20260923.json>)

训练及汇总作业均COMPLETED；576份预测SHA与952个动作复核通过。二进制预测/模型文件不在Git文件树，逐候选M、logits与决策保留在结果JSON中。

### H71特征融合先导：50/50配置及最终汇总完成

71张、37组、原五折、COST1 seed17；使用同折原全量TRAIN得到的头初始化，再在71张子集微调240步。此为已打开内部先导，不能与H593或外部确认混计。

| 路径 | 正确/71 |
|---|---:|
| RAW | 51 |
| 原COST1 | 57 |
| COL_ONLY、COARSE、FINE、COARSE_FINE、NO_ADAPTER，各自FREE | 各51 |
| 同上五种来源，各自ROMA_WEIGHTED | 各57 |

当前融合配方未新增正确数，MRR略有差异；不能仅凭相同正确数认定逐图决策完全相同，也不能外推为所有额外表示都无用。最终汇总状态为`H71_JOIN_COUNTS_AND_GROUPS_PASS`。

[完整简报](<ICLR/new ROUTEA/RC/results/rc_h71_feature_fusion_pilot_v1/summary_zh.md>) · [最终配对分析与逐query结果](<ICLR/new ROUTEA/RC/results/rc_h71_feature_fusion_pilot_v1/result.json>) · [验证](<ICLR/new ROUTEA/RC/results/rc_h71_feature_fusion_pilot_v1/validation.json>)

## 进行中：不能记作完成结果

| 分支 | 本次快照记录 | 解释边界 |
|---|---|---|
| ColPali跨编码器 | v1的22片失败；v2首片5160327_0在加载完整性检查失败；query缓存0/593；5160328/5160329为DependencyNeverSatisfied | 语言模型权重加载已无missing/unexpected，但手工检查漏算437项视觉模块自动改名；尚未通过旧reference数值锚点，不宣称修复或迁移成功 |

[H71续跑说明](<ICLR/new ROUTEA/RC/reports/REPORT_H71_FUSION_WAVE0_CONTINUATION_20260923.md>) · [M教师方案](<ICLR/new ROUTEA/RC/plan/RC_FOLD0_ROMA_M_TEACHER_DIAGNOSTIC_V1_20260923.md>) · [ColPali采集说明](<ICLR/new ROUTEA/RC/reports/REPORT_COLPALI_H593_QUERY_CACHE_START_20260923.md>)

[H71全部配置](<ICLR/new ROUTEA/RC/results/rc_h71_feature_fusion_pilot_v1>) · [ColPali修复与后续失败定位](<ICLR/new ROUTEA/RC/reports/REPORT_COLPALI_QUERY_CACHE_REPAIR_20260923.md>) · [v2实际加载检查记录](<ICLR/new ROUTEA/RC/results/rc_colpali_h593_query_tokens_v2_compat/shards/00/model_loading.json>)

此外，本次包含已在本地提交的[ColNomic历史重排实验](experiments/rerank_colnomic/README.md)，保留其原协议和结果归属，不与COST1主线混计。

作业状态是上传时的快照，后续调度变化以workspace新记录为准。本仓库中的旧已封存状态文件用于保留历史，不能只凭文件名判断最新进度。

## 查找全部新增材料

- [最新实验目录索引](backup/experiment_sync_20260923_evening/experiment_index.json)
- [最新文件SHA256清单](backup/experiment_sync_20260923_evening/files.json)
- [本次上传范围与统计](backup/experiment_sync_20260923_evening/summary.json)
- [上一轮快照清单](backup/experiment_sync_20260923/files.json)
- [2026-09-20完整汇总更新包](<ICLR/new ROUTEA/RC/reports/new_hyp_complete_results_20260920_controls_v1>)

这是研究归档，不声称仅靠Git仓库即可重跑全部实验。历史绝对路径、冻结来源SHA和未上传缓存的引用按原样保留。
