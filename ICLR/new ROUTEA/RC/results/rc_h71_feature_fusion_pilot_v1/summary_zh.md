# 71张融合收益先导

71 queries / 37 groups; COST1 seed17; 原五折; 同折原全量TRAIN初始化后，在71张子集微调240步。

| 方法 | 正确/71 | MRR |
|---|---:|---:|
| RAW | 51 | — |
| ORIGINAL_COST1 | 57 | — |
| COL_ONLY/FREE | 51 | 0.774905222032724 |
| COL_ONLY/ROMA_WEIGHTED | 57 | 0.8525186204948992 |
| COARSE/FREE | 51 | 0.7761292066770618 |
| COARSE/ROMA_WEIGHTED | 57 | 0.8524283351933464 |
| FINE/FREE | 51 | 0.7759798893980566 |
| FINE/ROMA_WEIGHTED | 57 | 0.8525186204948992 |
| COARSE_FINE/FREE | 51 | 0.7736392899913136 |
| COARSE_FINE/ROMA_WEIGHTED | 57 | 0.8524283351933464 |
| NO_ADAPTER/FREE | 51 | 0.7711901747113016 |
| NO_ADAPTER/ROMA_WEIGHTED | 57 | 0.8525598032640286 |

全部配对比较、救回/损失、每折净增、分组区间、逐query分差见result.json；所有预测保存完整C128中间量。区间为描述性边际区间，未据此选择最佳臂，不能将多臂中最好的一个当成独立确认。

Opened71-query grouped pilot with original full-TRAIN warm heads; not external GO
