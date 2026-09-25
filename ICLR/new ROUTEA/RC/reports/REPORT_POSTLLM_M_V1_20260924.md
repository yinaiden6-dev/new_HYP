# LLM后、检索投影前的M注入：固定位置对照

Fixed128 then opened PROBE8, no checkpoint or hyperparameter selection; post-hoc development only
同一TRAIN16、自然ColNomic C128、初始化、128次逐query完整C128 COST1；仅移动适配器位置。末端不直接读取M。
全部LLM隐藏特征复用已有24图缓存，投影保留原冻结LoRA和BF16计算。

| 路径 | TRAIN16 | PROBE8 | probe救回/误伤 |
|---|---:|---:|---:|
| RAW | 8/16 | 4/8 | 0/0 |
| POST_REAL | 10/16 | 5/8 | 1/0 |
| POST_REAL_REFIT | 12/16 | 5/8 | 1/0 |
| POST_REAL_CONSTANT | 8/16 | 4/8 | 0/0 |
| POST_REAL_CONSTANT_REFIT | 9/16 | 4/8 | 0/0 |
| POST_REAL_SHUFFLED | 7/16 | 4/8 | 0/0 |
| POST_REAL_SHUFFLED_REFIT | 8/16 | 4/8 | 0/0 |
| POST_CONSTANT | 9/16 | 4/8 | 0/0 |
| POST_CONSTANT_REFIT | 10/16 | 4/8 | 0/0 |
| POST_SHUFFLED | 8/16 | 4/8 | 0/0 |
| POST_SHUFFLED_REFIT | 9/16 | 4/8 | 0/0 |
| EXTERNAL_ADDITIVE4 | 12/16 | 6/8 | 2/0 |
| EXTERNAL_PRODUCT5 | 12/16 | 6/8 | 2/0 |

## 与封存PRE的同协议位置比较

| 配置 | PRE TRAIN | POST TRAIN | PRE probe | POST probe |
|---|---:|---:|---:|---:|
| POST_REAL | 10/16 | 10/16 | 4/8 | 5/8 |
| POST_CONSTANT | 8/16 | 9/16 | 4/8 | 4/8 |
| POST_SHUFFLED | 8/16 | 8/16 | 4/8 | 4/8 |
| POST_REAL_CONSTANT | 8/16 | 8/16 | 4/8 | 4/8 |
| POST_REAL_SHUFFLED | 8/16 | 7/16 | 4/8 | 4/8 |
| POST_REAL_REFIT | 11/16 | 12/16 | 4/8 | 5/8 |
| POST_CONSTANT_REFIT | 10/16 | 10/16 | 4/8 | 4/8 |
| POST_SHUFFLED_REFIT | 10/16 | 9/16 | 4/8 | 4/8 |

PRE V4完整对照引用封存结果；不得把早期V2的外部M通路6/8当作内部POST成功。
固定终点refit仅使用TRAIN标签，是预定次级诊断；未按probe选择头或阈值。
本结果只回答这套适配器、训练预算和已打开开发probe上的位置差异，不证明普遍最优注入点。
