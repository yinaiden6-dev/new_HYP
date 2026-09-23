# H593 learned ColNomic-only correction

原 H593 grouped OOF5，自然 RAW C128，检索身份监督；原完整头封存预测配对比较。23个候选缺失仍计入593分母。
CONTENT7使用RAW差、双向无权重MaxSim、双向top1−top2间隙和整体相似度均值；六输入七参数。MAXSIM3仅RAW差与前向MaxSim。全部不读取RoMa权重或坐标。

| 模型 | 正确/593 | MRR | 对RAW救/损 |
|---|---:|---:|---:|
| RAW | 426 | 0.790204 | 0/0 |
| PATCH_MAXSIM | 424 | 0.788517 | 1/3 |
| FULL_COST1 | 481 | 0.854276 | 58/3 |
| FULL_CE | 486 | 0.860845 | 71/11 |
| COST1_CONTENT7 | 427 | 0.791047 | 1/0 |
| CE_CONTENT7 | 431 | 0.794476 | 5/0 |
| COST1_MAXSIM3 | 426 | 0.790204 | 0/0 |
| CE_MAXSIM3 | 426 | 0.790204 | 0/0 |
| RAW2_CE | 426 | 0.790204 | 0/0 |

主比较：COST1_CONTENT7 → FULL_COST1；下表正差表示完整头较好。

| 比较 | 完整头救/损 | 等组差pp | bootstrap95% pp |
|---|---:|---:|---|
| COST1_CONTENT7__to__FULL_COST1 | 57/3 | 10.636 | [6.490, 15.302] |
| COST1_MAXSIM3__to__FULL_COST1 | 58/3 | 10.723 | [6.589, 15.340] |
| CE_CONTENT7__to__FULL_CE | 67/12 | 10.880 | [5.446, 16.563] |
| CE_MAXSIM3__to__FULL_CE | 71/11 | 11.872 | [6.383, 17.640] |

全部75片内容统计经过每候选NumPy独立重算；5折新进程训练重放及NumPy动作检查通过，RAW2_CE参数和全部分数复现旧值。
这是已开放H593开发对照；不宣称所有纯ColNomic模型都不能替代，不自动修改部署模型。峰值间隙不是已校准身份置信度。
本表PATCH_MAXSIM是FP64读出，旧CRISP对照的PATCH_MAXSIM为FP32，须按实际结果分别记录。
原始统计：features/shard*/payload.json；每折参数及127分数：fold*/payload.json；逐查询结果：result.json。
