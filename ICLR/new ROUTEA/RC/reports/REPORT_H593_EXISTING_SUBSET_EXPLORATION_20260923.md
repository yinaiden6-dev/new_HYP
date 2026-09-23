# 已完成共同子集：无需新增 GPU 的机制探索

共 41 张、30 个组；三类数据完整后纳入。不是旧 EVAL32/EVAL128，也不是全 H593 结论。

原生 H593 缓存：593 张，75 个分片，75904 个 query–candidate 配对；全部原生分片 receipt/validation 已核对。

下表只比较候选直接排序，不经过 COST1/CE 小头和 HOLD/SWITCH。M×内容中的内容固定为完整 reference 的自由 MaxSim。没有训练或新增 GPU 前向。

|干预（HR1）|M 排第一正确数|M×固定自由内容正确数|相对原生 M×内容：救回/损失|
|---|---:|---:|---:|
|visual/NATIVE|26/41|28/41|0/0|
|visual/Q_GRAY|24/41|26/41|0/2|
|visual/R_GRAY|23/41|25/41|0/3|
|visual/Q_LOWPASS|10/41|12/41|0/16|
|visual/R_LOWPASS|7/41|7/41|1/22|
|visual/Q_SHUFFLE|14/41|16/41|1/13|
|visual/R_SHUFFLE|18/41|22/41|2/8|
|inside/A0J0P0|3/41|19/41|5/14|
|inside/A0J0P1|0/41|2/41|1/27|
|inside/A0J1P0|3/41|8/41|1/21|
|inside/A0J1P1|19/41|25/41|1/4|
|inside/A1J0P0|0/41|18/41|3/13|
|inside/A1J0P1|1/41|5/41|1/24|
|inside/A1J1P0|14/41|14/41|0/14|
|inside/A1J1P1|26/41|28/41|0/0|
|inside/NO_COARSE_LOGIT|1/41|26/41|6/8|
|inside/NO_LR_DELTA|25/41|28/41|0/0|
|inside/NO_HR_DELTA|25/41|27/41|0/1|
|inside/NO_REFINER_DELTA|24/41|25/41|0/3|

全部七阶段逐 query 数值见 `result.json`。此轮未运行坐标干预下的小头评测，不能把坐标数据已存在说成最终性能评测已完成。

当前子集历史 RAW/COST1/CE 决策正确数（仅做样本背景）：{'RAW': 32, 'COST1_FROZEN_NATIVE': 35, 'CE_FROZEN_NATIVE': 35}。这些数与表内直接排序是不同读出机制。

样本量减少可加速机制定位；应按身份/组分布检查代表性。任何从本次探索选择的方法，需在其余未用于选择的数据上固定验证；不能据此缩改既有593评测分母。
