# 内部 M V4：冻结终点的跨组 PROBE8 检验

ColNomic 原自然 C128；全部 adapter、小头、M标准化和尺度固定于 TRAIN16。无新训练、无probe选参。
这8张已在早期开发实验中打开；未参与V4训练，不属于全新独立确认。

| 路径 | 正确/8 | 救回/误伤 |
|---|---:|---:|
| RAW | 4/8 | 0/0 |
| REAL | 4/8 | 0/0 |
| REAL_REFIT | 4/8 | 0/0 |
| REAL_CONSTANT | 4/8 | 0/0 |
| REAL_CONSTANT_REFIT | 4/8 | 0/0 |
| REAL_SHUFFLED | 4/8 | 0/0 |
| REAL_SHUFFLED_REFIT | 4/8 | 0/0 |
| TRAIN_SHUFFLED | 4/8 | 0/0 |
| TRAIN_SHUFFLED_REFIT | 4/8 | 0/0 |
| TRAIN_CONSTANT | 4/8 | 0/0 |
| TRAIN_CONSTANT_REFIT | 4/8 | 0/0 |
| warmstart_INTERNAL3 | 4/8 | 0/0 |
| ADDITIVE4 | 6/8 | 2/0 |
| PRODUCT5 | 6/8 | 2/0 |
| REAL_CONSTANT_REAL_REFIT_HEAD | 4/8 | 0/0 |
| REAL_SHUFFLED_REAL_REFIT_HEAD | 4/8 | 0/0 |

实际component数：6；目标在C128：7/8。
REAL是128步联合训练终点；REFIT为仅在原TRAIN16完成的固定读出重训，二者分别报告，不能按probe择优。
REAL_CONSTANT/SHUFFLED共用REAL原终点头；带REAL_REFIT_HEAD的干预共用真实M重训头。
TRAIN_SHUFFLED与TRAIN_CONSTANT为训练条件对照。末端内部头只有RAW差、条件化内容和bias。
八图只能检验方向和失败方式；不自动追加实验，不宣称普遍有效。
