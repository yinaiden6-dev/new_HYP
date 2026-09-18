# CRISP 与不训练参数头：固定缓存对照

用户授权：试验 CRISP baseline，并检验参数头不训练、手动填写参数的结果。

## 问题与边界

在 H593（完整 593 张开发面板）和 RPC600（已打开的外部面板）上，复用冻结的 query/reference tokens、RAW 自然 C128 和 RoMa 缓存。不训练任何参数、不修改既有模型、不调整 candidate、不用 target 插入、不新增 OCR/SAM、不触碰 D1-MI/formal392。RPC 是后续已打开面板对照，不能改称新的 untouched 外部确认。

这是 **CRISP 的 image-to-image、RAW-C128 reranking 适配**。原论文是文字检索流程图，故不声称原任务完全复现，也不把本轮结果当成 CRISP 全图库召回效果。

## 看结果前固定的实验臂

- RAW：复用各面板已核验的完整图/完整图库 RAW 决策。
- PATCH_MAXSIM：在相同自然 C128 上，用相同 image-only 有效 tokens 做正向 MaxSim 均值。这个匹配对照用于分开“去掉模板 tokens”与“CRISP 评分改动”。
- CRISP：作者公开 `evaluation/baselines/crisp.py`，Git blob `fa2c71e533c477cf805f2bf119a9d3c9160b61c9`，tree `313822fe979d0ab2d3eaedebfb4722292e833191`。函数不改动，调用输入为既有 image tokens 转 FP32；不额外 L2 normalize，不包含模板/padding。沿用作者 epsilon 与权重/反向项。非有限评分报错，不丢弃该 query。
- VISIBILITY_DIRECT：直接按既有 `real_score` 选第一，不训练参数、不使用标签选阈值。
- ZERO_HEAD：6 个权重与 bias 全为 0，正 logit 才 SWITCH，因此应严格复现 RAW。这是实现检查，不当作有用新模型。
- MANUAL_EQUAL：6 个权重均为 1，bias=0。
- MANUAL_BALANCED：RAW-gap 权重为 1，另外 5 个证据项各为 0.2，bias=0。它把 RAW 与五项证据的平均值等权组合。
- COST1、CE：H593 仅引用既有分组 OOF 逐图预测；RPC 仅引用既有 full-H593 冻结头逐图预测。禁止在 H593 上用 full-H593 拟合头冒充 OOF。

手工系数按上述公式一次固定，不扫描、不根据成绩换参数。它们是两种明确的手工规则，不代表手工规则族的最优值。把已训练的 theta 抄到代码里是同一训练模型的部署，不能称为无训练 baseline。

## 核验与报告

worker 只读已列明的 tokens、RoMa payload、receipt、validation 和公开源码，禁止读取 curator、以往逐图结果或当前汇总。所有预测密封、索引与 NumPy action 独立重放通过后，再读取标签和旧预测做配对。

主报告每面板的 top1 正确数、相对 RAW/匹配 MaxSim 的救回、损失、净增、C128 遗漏；H593 按 source component 分组 bootstrap，RPC 按 SKU 重采样（3 张相机图一起），均 100000 次、seed=20260918。比较无训练规则与训练头时用同一 query 的配对结果。无新科学 GO 门，报告负结果，不挑最高分臂宣布胜利。

独立检查：作者 CRISP 与简洁 NumPy 实现在随机/负相似度/非等长序列下吻合；相同 tokens 的 MaxSim；固定参数的正 logit SWITCH 和 tie 顺序；ZERO_HEAD 完全等于 RAW；已验证缓存的 source SHA、query/candidate/reference 绑定；脚本真实参数展开；汇总恒等式和旧 RAW 数量重放。

不按结果改变分片：每面板原始 worker 顺序每 8 个 query 一片，H593 75 片（最后 1 张），RPC 75 片（每片 8 张）。先各跑首片，再放行其余片；每片 GPU1 / CPU8 / 96G / 10分钟，最大并发46。完成后自动 CPU 汇总。运行开销另记为缓存评分时间，不与原系统全部编码/RoMa耗时混用。
