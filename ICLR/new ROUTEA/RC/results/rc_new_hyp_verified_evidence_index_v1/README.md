# new HYP 已验证证据索引

九个明确指定的已验证试验，共118条记录。每条CSV/JSON记录都绑定原结果、独立验证、参数来源与算术路径；全部正确数和救/损从原动作重算。没有新增预测或训练。

| 试验 | 数据与头范围 | 已打开 REAL 正确数 | RAW基线 |
|---|---|---|---:|
| 绝对证据尺度扩展 | native系列 matched EVAL32 | NATIVE7=28/32；ABS12=26/32；REL12=26/32 | 25 |
| 固定训练目标对照 | native系列 matched EVAL32 | PAIR_SIGN=28/32；PAIR_RANK=27/32；FULL_SIGN=27/32；FULL_RANK=27/32 | 25 |
| PAIR控制尺度统一 | native系列 matched EVAL32 | ORIGINAL7=28/32；HARMONIZED7=26/32 | 25 |
| 完整reference内容桥接 | native系列 matched EVAL32 | ORIGINAL_C=28/32；D_IMAGE=27/32；D_FULL=25/32 | 25 |
| 重新训练证据子集 | native系列 matched EVAL32 | RAW2=25/32；RAW_PLUS_M3=26/32；RAW_PLUS_L3=26/32；JOINT4=26/32；ORIGINAL7=28/32 | 25 |
| S与Q/R固定2×2对照 | native系列 matched EVAL32 | JOINT4=26/32；PRODUCT5=27/32；RESPONSE6=26/32；ORIGINAL7=28/32 | 25 |
| 固定NATIVE7组效应 | native系列 matched EVAL32 | ORIGINAL7=28/32；DROP_S=26/32；DROP_QR=27/32；DROP_S_QR=26/32；REFIT_RESPONSE6=26/32；REFIT_PRODUCT5=27/32；REFIT_JOINT4=26/32 | 25 |
| 旧FROZEN_C difficult90组效应 | 旧FROZEN_C difficult90 | ORIGINAL=69/90；DROP_S=68/90；DROP_QR=70/90；DROP_S_QR=65/90 | 61 |
| 旧FROZEN_C matched32同头桥接 | 旧FROZEN_C matched EVAL32 | ORIGINAL=27/32；DROP_QR=27/32 | 25 |

旧FROZEN_C的DROP_QR在difficult90为70/90、9救0损；同一旧头在matched32为27/32、3救1损。后者与原27/32同分，但正确集合有变化，因此不能称两个bundle都无损。NATIVE7的28/32属于另一个头，不能与旧69/70合并。

反复出现的原模型和已训练子集回放标为 `baseline_replay_duplicate=true`，不算新增独立实验。固定特征置零都标为计算反事实；索引没有采用任何新模型。TRAIN、EVAL、REAL与C_BIND分别保存，分母不合并。

完整数据：[CSV](evidence.csv)、[JSON](evidence.json)、[来源清单](sources_manifest.json)、[独立重建验证](independent_validation.json)。
