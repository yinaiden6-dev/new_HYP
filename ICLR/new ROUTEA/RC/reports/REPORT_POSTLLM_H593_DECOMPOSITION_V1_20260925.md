# H593 后 LLM：共同响应、patch 剩余与匹配位置归因

原五折封存 POST_REAL、自然 ColNomic C128、固定 INTERNAL3。全593张；不训练、不运行编码器或RoMa。

|回放|正确/593|救回RAW|误伤RAW|保留原54次救回|与CPU原生决策不同|
|---|---:|---:|---:|---:|---:|
|CONSTANT|426|5|5|4|67|
|NATIVE|478|54|2|54|0|
|COMMON_ONLY|477|53|2|52|3|
|SPATIAL_ONLY|430|7|3|6|63|
|RECOMPOSED|478|54|2|54|0|
|NATIVE_FIXED_ARGMAX|469|44|1|44|14|
|COMMON_FIXED_ARGMAX|468|43|1|43|14|

CPU与封存GPU的决策差异：{'NATIVE': 0, 'CONSTANT': 0}。存在差异时，CPU原生是干预基线，不能宣称精确解释原GPU478。
逐patch独立重算头分数最大误差：2.13e-14。
COMMON_ONLY保留跨patch均值；SPATIAL_ONLY仅指去均值后的patch差异，不代表空间ownership。固定命中由同一适配器的恒定M路径提供。
本报告为已打开OOF上的固定参数机制诊断，正确数保留不能替代逐图决策与纠错集合比较。
[完整结果及逐图分数](../results/rc_postllm_h593_decomposition_v1/result.json)
