# H593 质量与内容算子实验：数据索引

本目录复用已有 ColNomic tokens 和 RoMa 可见性缓存，不重新编码图像，也不重新运行 RoMa。坐标精度实验的数据另存于 `results/rc_h593_roma_coordinate_precision_v2/`。

## 算子定义

M：整体质量缩放 `sqrt(mean(u) * mean(v))`；Q：query token 加权汇聚；R：reference 权重参与 MaxSim 的最大值选择。`NATIVE` 为三个开关全开，另有七种组合。关闭 Q 或 R 时，该侧局部权重设为 1；M 单独控制是否保留原整体质量。

第九臂 `FIXED_FREE_ARGMAX` 固定自由 ColNomic MaxSim 的命中位置，在该位置读取 reference 权重。它与原生路径的区别是是否允许 reference 权重改变匹配位置。这里不是把小头的某一输入列置零。

## 中间文件

- `workers.json`：593 图按原缓存分成 75 片，含原始来源及 SHA。没有 query 身份标签。
- `queryNNN/intermediates.pt`：完整 128 候选轴；每对的 u/v、自由/加权 MaxSim 命中 token、局部相似度、原生逐 token 贡献，以及 reference 在自由命中位置上的数值变化和重新选择位置的增量。原始 token 缓存通过路径与 SHA 绑定，不重复存储。
- `queryNNN/payload.json`：九臂的四项评分及各自完整 127×6 小头输入；RAW 原始排序、原答案和 challenger 轴。
- `queryNNN/validation.json`：原生评分逐位一致、独立 NumPy 数值复核和中间量重放的封存结果。
- `shardNN/validation.json`：该片全部 query 的复核清单。

reference 数值变化加上重选增量，等于原生分数减去 `M1Q1R0` 分数，逐 token 求和核对。重选增量对单个候选非负，不意味着正确身份受益更多；最终必须比较 target 与错误候选。

## 小头与分析结果

位于 `results/rc_h593_quality_operator_eval_v1/`：

- `foldN/payload.json`：固定原 COST1/CE 参数与同协议重训参数，以及所有臂的完整 127 logits、最终选择；HOLD 的分数为 0。
- `foldN/validation.json`：新进程重训与 NumPy logits/动作复核；原生参数、分数和动作逐位复现。
- `operator_diagnostics.json`：每张图、每个候选的 reference 数值改变、重选增量与匹配位置改变比例。
- `result.json`、`validation.json`：五折预测全部封存后才合并标签；完整九臂固定头/重训结果、救回/损失清单和三个算子的固定头 margin 分解。

文件只有在相应阶段执行成功后才会出现。当前文档是索引，不代表结果已经完成。原图、完整图像隐藏激活不在本目录；原始图像/token 缓存仍是来源文件。

自然 RAW C128，593 图分母保留全部 23 个召回缺失；原 COST1 481/593、CE 486/593作为本轮对应基线。已经反复开发的 H593 结果属于机制分析及开发证据，不作为新的外部确认。
