# V4冻结表示：原始内容与内部变化分开读出

Motivated by opened PROBE8 failure audit; exploratory development, not untouched validation
只训练小头，固定TRAIN16/C128/2000步COST1；不直接读M，不改变适配器。PROBE8已打开，不能称独立确认。

| 路径 | TRAIN16 | PROBE8 | probe救回/误伤 |
|---|---:|---:|---:|
| RAW | 8/16 | 4/8 | 0/0 |
| ORIGINAL3_CONTINUE | 8/16 | 4/8 | 0/0 |
| PRESERVE_REAL | 11/16 | 4/8 | 0/0 |
| PRESERVE_TRAIN_CONSTANT | 10/16 | 4/8 | 0/0 |
| PRESERVE_TRAIN_SHUFFLED | 9/16 | 4/8 | 0/0 |
| PRESERVE_REAL_INFER_REAL_CONSTANT | 8/16 | 4/8 | 0/0 |
| PRESERVE_REAL_INFER_REAL_SHUFFLED | 8/16 | 4/8 | 0/0 |
| ADAPTED3_FROZEN_REFIT | 11/16 | 4/8 | 0/0 |
| EXTERNAL_ADDITIVE4 | 12/16 | 6/8 | 2/0 |

新增变化列只是保留原始和调制内容的独立自由度，不增加信息源；原内容列的存在不保证原排序或正确样本保留。
ORIGINAL3_CONTINUE排除继续训练本身；两个TRAIN条件对照均为同四参数头；推理干预共用PRESERVE_REAL头。
本轮源于旧probe失败分析；不按probe选择参数、阈值、checkpoint或宣布独立成功。
