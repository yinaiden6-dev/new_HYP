# POST残差：独立逐patch输出/决策核算（仅24图分解）

核对 3072 个完整候选证据文件、重算 21336 个logit；最大L误差 2.22e-16，最大logit误差 7.11e-15。

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

## 核算范围

- Independent replay starts at saved per-patch MaxSim/fixed-assignment outputs; it does not rerun projection or MaxSim matrices.
- RECOMPOSED saved patch outputs, candidate scores and decisions are independently compared; hidden/token error bounds are checked producer measurements, not fresh independent full tensor recomputation.
- Decomposition-only mode does not validate the POOL intervention outputs or decisions; these await the combined independent report. No POOL efficacy conclusion is established by this report.
- TRAIN16 and repeatedly opened PROBE8; descriptive interventions, not new independent accuracy or optimal model selection.
- A shared hidden direction can yield patch-dependent rotations after normalization; shared direction is not a uniform final score or proof of better semantic attention.
