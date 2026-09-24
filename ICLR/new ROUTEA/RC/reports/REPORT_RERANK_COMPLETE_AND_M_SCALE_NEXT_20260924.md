# Rerank 完成核查及 M 拟合修复

## H593 同候选重排结果

同一H593开发面板、同一ColNomic RAW自然C128，593条均纳入分母。目标在C128中570条，缺席23条。模型为冻结的本地Qwen3-VL-Reranker-2B，逐query/reference图像对前向，未调用额外OCR；直接重排不在H593上训练。校准臂只在原五折各自TRAIN上训练三参数HOLD/SWITCH头。

| 方法 | 正确/593 | 相对RAW救回 | 相对RAW误伤 | MRR@C128 |
|---|---:|---:|---:|---:|
| ColNomic RAW | 426 | 0 | 0 | 0.790135 |
| 原 RoMa COST1 | 481 | 58 | 3 | 0.866883 |
| 原 RoMa CE | 486 | 71 | 11 | 0.868011 |
| 冻结 Qwen3 直接重排 logit | 522 | 107 | 11 | 0.908901 |
| 冻结 Qwen3 官方 sigmoid 重排 | 522 | 107 | 11 | 0.908893 |
| Qwen3＋三参数 COST1 校准 | 477 | 51 | 0 | 0.867597 |
| Qwen3＋三参数 CE 校准 | 523 | 102 | 5 | 0.910091 |

直接重排相对原COST1为64救回、23损失、净增41；CE校准相对原COST1为61救回、19损失、净增42。不能将这些提升记作RoMa机制或M学生的提升。

Qwen3是图像对模型，直接读取图像；它不是固定ColNomic tokens的替代读出。因此此结果确立了更强的常规重排baseline，却不能回答“ColNomic tokens有没有足够信息”。原COST1仍有较少RAW误伤这一观察，但这不能抹去更强baseline的准确率优势。准确率最优表述应更新；部署成本必须另做同硬件同工作负载的计时，不能从模型名称推断。

官方sigmoid与logit总正确数相同，但有2条query的选择及正确性不同、彼此抵消；有限精度sigmoid排序不能视为逐条严格相同。原结果同时保存两臂，未混用。

## 核验

- 5160754全部50个推理分片、5160755全部5折及5160756汇总均COMPLETED、exit0:0。
- `QWEN3_FULL593_AXES_BASELINES_FOLDS_COUNTS_PASS`；result、authority、50片验证和5折payload的SHA全部核对。
- 从593条逐query行独立重算正确数、救回/误伤、MRR及component等权净增，全部一致。
- 模型权重未改；Conv3d channels-last存储执行版本独立重算全部75904对，未混入旧布局分数。
- 原结果：`results/rc_h593_qwen3_rerank_v2_layout/result.json`。
- 本次审阅回执：`reports/rc_rerank_review_and_colpali_migration_20260923.json`。

## ColPali 指定任务迁移

按用户指令仅将5160875_0从accelerated迁到dev_accelerated，保留1GPU、8CPU、48GB、15分钟和原依赖。迁移后已COMPLETED，运行46秒、exit0:0。下游按原依赖推进，未改变ColPali自身C128候选来源。

## M 修复的下一轮

任务5160905在dev_cpuonly继续四张TRAIN的两臂比较，均使用完整C128和原网络，固定2000步。保留绝对log偏差约束，避免纯中心化损失无法识别M标度。四项门全部通过后才在原457条有效TRAIN从头训练，并对held119封存后评估。

方案：`plan/RC_COLNOMIC_M_SCALE_FIT_V2_20260923.md`；条件式跨组复查：`plan/RC_FOLD0_M_SCALE_TEACHER_V2_20260924.md`。本轮不新增GPU/编码器/RoMa前向。
