# ColNomic internal M v3：TRAIN16 拟合能力诊断

RoMa 保留；PRE_REAL 使用真实配对 M，PRE_CONSTANT 使用恒定标准化条件。
内部末端仅读取 RAW、自由内容 Lθ 与 bias；适配器和三参数头共同训练。
原自然 C128、16 张 TRAIN、固定128次更新。没有评估或选择 probe。

| 路径 | 正确/16 | 全 TRAIN 平均 COST1 | 对 RAW 救回/损失 |
|---|---:|---:|---:|
| INTERNAL3 CPU | 8/16 | 0.922078803 | 0/0 |
| ADDITIVE4 CPU | 12/16 | 0.748420691 | 4/0 |
| PRODUCT5 CPU | 12/16 | 0.746196445 | 4/0 |
| PRE_CONSTANT step16 native | 8/16 | 0.994557860 | 0/0 |
| PRE_CONSTANT step128 native | 8/16 | 0.841984040 | 0/0 |
| PRE_REAL step16 native | 8/16 | 0.994473780 | 0/0 |
| PRE_REAL step128 native | 8/16 | 0.857835567 | 0/0 |
| PRE_REAL step128 constant | 8/16 | 0.856712761 | 0/0 |
| PRE_REAL step128 shuffled | 8/16 | 0.856416091 | 0/0 |

CPU 外部头使用同16图2000次全批量更新；内部使用128次逐 query 更新。标签预算相同，优化成本和步数不同。
这里回答是否学动、能否在 TRAIN 使用 M，不证明跨组泛化、超过完整 COST1 或必须改造编码器。

完整标量、梯度、checkpoint、每张图C128内容与127动作分数均保留在本目录。
