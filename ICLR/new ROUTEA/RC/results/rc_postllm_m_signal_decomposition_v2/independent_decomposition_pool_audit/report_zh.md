# POST残差与TRAIN共享方向：独立逐patch输出/决策核算

核对 4224 个完整候选证据文件、重算 29337 个logit；最大L误差 2.22e-16，最大logit误差 7.11e-15。

|分支|臂|正确/n|救回|损失|
|---|---|---:|---:|---:|
|拆分/train|CONSTANT|8/16|0|0|
|拆分/train|NATIVE|10/16|2|0|
|拆分/train|COMMON_ONLY|10/16|2|0|
|拆分/train|SPATIAL_ONLY|8/16|0|0|
|拆分/train|RECOMPOSED|10/16|2|0|
|拆分/train|NATIVE_FIXED_ARGMAX|10/16|2|0|
|拆分/train|COMMON_FIXED_ARGMAX|10/16|2|0|
|拆分/probe|CONSTANT|4/8|0|0|
|拆分/probe|NATIVE|5/8|1|0|
|拆分/probe|COMMON_ONLY|5/8|1|0|
|拆分/probe|SPATIAL_ONLY|4/8|0|0|
|拆分/probe|RECOMPOSED|5/8|1|0|
|拆分/probe|NATIVE_FIXED_ARGMAX|5/8|1|0|
|拆分/probe|COMMON_FIXED_ARGMAX|5/8|1|0|
|共享TRAIN方向/probe|TRAIN_POOL_COMMON|5/8|1|0|
|共享TRAIN方向/probe|IDEAL_SELF_COMMON|5/8|1|0|
|共享TRAIN方向/probe|TRAIN_POOL_COMMON_FIXED_ARGMAX|5/8|1|0|
|共享TRAIN方向/probe|IDEAL_SELF_COMMON_FIXED_ARGMAX|5/8|1|0|
|共享TRAIN方向/probe|ACTUAL_COMMON_BRIDGE|5/8|1|0|
|共享TRAIN方向/probe|CPU_NATIVE_REFERENCE|5/8|1|0|
|共享TRAIN方向/probe|CPU_CONSTANT_REFERENCE|4/8|0|0|
|共享TRAIN方向/train_engineering|TRAIN_POOL_COMMON|1/1|0|0|
|共享TRAIN方向/train_engineering|IDEAL_SELF_COMMON|1/1|0|0|
|共享TRAIN方向/train_engineering|TRAIN_POOL_COMMON_FIXED_ARGMAX|1/1|0|0|
|共享TRAIN方向/train_engineering|IDEAL_SELF_COMMON_FIXED_ARGMAX|1/1|0|0|
|共享TRAIN方向/train_engineering|ACTUAL_COMMON_BRIDGE|1/1|0|0|
|共享TRAIN方向/train_engineering|CPU_NATIVE_REFERENCE|1/1|0|0|
|共享TRAIN方向/train_engineering|CPU_CONSTANT_REFERENCE|1/1|0|0|

## cefdinir-fig2

|分支|臂|target−RAWwinner L|target logit|正确|
|---|---|---:|---:|---|
|decomposition|CONSTANT|-0.004624817|-0.611872901|False|
|decomposition|NATIVE|0.032241924|0.360213625|True|
|decomposition|COMMON_ONLY|0.032290543|0.361511712|True|
|decomposition|SPATIAL_ONLY|-0.004671705|-0.613206393|False|
|decomposition|RECOMPOSED|0.032241924|0.360213625|True|
|decomposition|NATIVE_FIXED_ARGMAX|0.029979284|0.303910077|True|
|decomposition|COMMON_FIXED_ARGMAX|0.030027891|0.305206908|True|
|pool|TRAIN_POOL_COMMON|0.032289031|0.361481777|True|
|pool|IDEAL_SELF_COMMON|0.032277620|0.361179072|True|
|pool|TRAIN_POOL_COMMON_FIXED_ARGMAX|0.030022029|0.305062597|True|
|pool|IDEAL_SELF_COMMON_FIXED_ARGMAX|0.030011085|0.304772299|True|
|pool|ACTUAL_COMMON_BRIDGE|0.032290543|0.361511712|True|
|pool|CPU_NATIVE_REFERENCE|0.032241924|0.360213625|True|
|pool|CPU_CONSTANT_REFERENCE|-0.004624817|-0.611872901|False|

## 核算范围

- Independent replay starts at saved per-patch MaxSim/fixed-assignment outputs; it does not rerun projection or MaxSim matrices.
- RECOMPOSED saved patch outputs, candidate scores and decisions are independently compared; hidden/token error bounds are checked producer measurements, not fresh independent full tensor recomputation.
- POOL source membership and preserved nonlinear latent/candidate outputs are checked; the original 16-query adapter-hook calculation is a producer validation, not rerun here.
- TRAIN16 and repeatedly opened PROBE8; descriptive interventions, not new independent accuracy or optimal model selection.
- A shared hidden direction can yield patch-dependent rotations after normalization; shared direction is not a uniform final score or proof of better semantic attention.

## 架构与两套证据的边界

本次冻结POST_REAL128适配器的瓶颈为16，标准化M的固定condition_gain为59.86651818838306（约59.87）。共同响应占比和TRAIN共享方向复现纠错只描述这一具体参数化及TRAIN16/已打开PROBE8；固定输入标度与瓶颈结构可能影响该现象，不能推为所有模型普遍定律。

H593外部质量头的主效应分解，与这里POST内部TRAIN16/PROBE8的残差/共享方向干预，是两套范围不同的证据；不能将它们拼接成已经证明的同一中介链。POOL的四个新臂均在8/8张上选择与NATIVE完全相同的候选，不仅是正确数相同。
