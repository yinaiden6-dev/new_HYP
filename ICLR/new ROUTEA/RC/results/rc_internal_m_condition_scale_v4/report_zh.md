# ColNomic internal M V4：单因素条件输入尺度检验

只将M条件输入乘sqrt(3584)，PRE_REAL使用原绑定，PRE_SHUFFLED训练和评估都固定错绑M。
V3原gain1真实M与恒定M作为封存对照复用；没有自动读取probe。
内部末端仅读取 RAW、自由内容 Lθ 与 bias；适配器和三参数头共同训练。
原自然 C128、16 张 TRAIN、固定128次更新。没有评估或选择 probe。

| 路径 | 正确/16 | 全 TRAIN 平均 COST1 | 对 RAW 救回/损失 |
|---|---:|---:|---:|
| INTERNAL3 CPU | 8/16 | 0.922078803 | 0/0 |
| ADDITIVE4 CPU | 12/16 | 0.748420691 | 4/0 |
| PRODUCT5 CPU | 12/16 | 0.746196445 | 4/0 |
| PRE_REAL step16 native | 8/16 | 0.994719824 | 0/0 |
| PRE_REAL step128 native | 10/16 | 0.790052141 | 2/0 |
| PRE_REAL step128 constant | 8/16 | 0.903199154 | 0/0 |
| PRE_REAL step128 shuffled | 8/16 | 0.960027766 | 0/0 |
| PRE_SHUFFLED step16 native | 8/16 | 0.992581062 | 0/0 |
| PRE_SHUFFLED step128 native | 8/16 | 0.832554275 | 0/0 |
| V3 reused PRE_CONSTANT | 8/16 | 0.841984040 | 0/0 |
| V3 reused PRE_REAL | 8/16 | 0.857835567 | 0/0 |

## 固定终点后的统一CPU读出诊断

仅用各自冻结L、同一warm初值和2000次全TRAIN16 COST1更新；不更新adapter，不是新的留出成绩。

| 路径 | 正确/16 | COST1 | 对RAW救回/误伤 |
|---|---:|---:|---:|
| GAIN_REAL | 11/16 | 0.714463018 | 3/0 |
| GAIN_REAL_CONSTANT | 8/16 | 0.896944332 | 0/0 |
| GAIN_REAL_SHUFFLED | 8/16 | 0.942458280 | 0/0 |
| GAIN_TRAIN_SHUFFLED | 10/16 | 0.792086469 | 2/0 |
| V3_REAL | 9/16 | 0.831175760 | 1/0 |
| V3_REAL_CONSTANT | 9/16 | 0.828283047 | 1/0 |
| V3_REAL_SHUFFLED | 9/16 | 0.827166113 | 1/0 |
| V3_CONSTANT | 10/16 | 0.804650334 | 2/0 |

CPU 外部头使用同16图2000次全批量更新；内部使用128次逐 query 更新。标签预算相同，优化成本和步数不同。
目标是内部M有效，不要求超过外部头。须比较真实、恒定和错绑条件，不能把所有臂共有的拟合提升归给M。
只改变条件尺度，尚不能穷尽优化、损失和表示问题；不证明跨组泛化或必须改造编码器。

完整标量、梯度、checkpoint、每张图C128内容与127动作分数均保留在本目录。
