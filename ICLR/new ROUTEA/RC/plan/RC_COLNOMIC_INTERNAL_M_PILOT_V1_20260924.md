# ColNomic 内部使用真实 M：位置与条件的最小对照

用户授权：2026-09-24 开始 LLM 前改造，并增加其他注入位置作对照。保留 RoMa，不做质量蒸馏；下游新增训练只使用检索身份标签。

## 范围与比较

冻结真实 ColNomic-7B（包括已有检索 LoRA）、原 reference tokens、RoMa M、原 RAW C128、原 fold0 分组和 COST1 的 M-only 重训头。只训练 query 分支的新残差模块。四臂使用相同 3584 维输入、rank16 残差结构、初始化、训练次序及更新数：

1. PRE_REAL：视觉 merger 后、LLM 前，读取 patch 内容与真实 M。
2. PRE_CONSTANT：同位置同容量，M 固定为训练面板 log-M 均值。
3. POST_REAL：LLM 末层后、128 维检索投影前，读取 patch 内容与真实 M。
4. POST_CONSTANT：同位置同容量，M 固定为训练面板 log-M 均值。

恒定 M 臂仍训练内容适配器，控制新增参数与训练的影响。所有臂的最终决策头都继续读取同一份真实 M；这里隔离的是额外的内部条件化，而非剥夺外部基线的已有信息。真实 M 的固定候选循环错绑仅作推理干预，不另增训练臂。未直接修改 attention 分数。

## 数据与训练

首轮为工程与小规模开发验证，不扩跑 H593：最初按 hash 选出的4个不同component TRAIN均为RAW正确，保留为工程面板及独立快照。训练面板固定16张：在fold0 TRAIN中分层选择8张RAW正确、8张RAW错误、target自然进入C128的query，同层按hash取不同component优先；这是仅依TRAIN监督的分层，不读取probe结果。8 个 probe 从原 outer-held119 按 query ID 的 hash 选择，选择时不读身份标签。probe 标签只在四臂预测封存后用于 join；不用于步数、超参数或快照选择。原头来自完整 fold0 TRAIN，因此报告必须同时说明新模块只有 16 图训练、原头已经在原 TRAIN 训练，不宣称整个系统仅使用 16 个标签。

保持全部 128 候选，原 RAW winner 的 HOLD=0。五参数头复用 `COST1_REFIT_M1Q0R0` 的 [0,1,2,3,6]，特征为 RAW 标准化差、sym(M*L)、sym(M)、sym(L)，其中 L 是归一化图像 tokens 的 full-reference mean MaxSim。head 和 M 均冻结。

小模块 FP32、主干 BF16、评分与 COST1 损失 FP64。rank16、残差末层零初始化，从原编码开始；AdamW lr=3e-4、weight_decay=1e-3、seed17、1 epoch 共16次逐query更新，clip norm=1。M 使用 log(max(M,1e-8))，仅用16 TRAIN图的全部候选计算均值和标准差。该设置先冻结，不依 probe 改动。

对完整 C128 先不保留计算图地计算分数，精确求出 COST1 对128个 L 的导数，再只重算非零导数对应的候选反传，避免同时保存128份语言主干激活。此标量VJP必须通过直接autograd对照；不能改成抽样负例、改 seed/family 或裁剪候选。

## 缓存与运行

原M和reference tokens直接复用。每张query仅补一次 merger输出、冻结LLM末层状态、原检索tokens、processor输入和几何信息。POST臂复用末层状态，训练时不再执行LLM；PRE臂复用merger，但必须重新执行受条件化的LLM。这种计算差异必须报告，不能宣称同等wall time。所有臂保持相同优化更新数；记录前向、反向次数和GPU用时。

新文件和作业单独放置，不修改旧实验或 site-packages。GPU短任务10分钟，可原Job requeue，逐候选评分与完整优化步原子保存；每阶段最多16次requeue。首个GPU任务先做源一致性、零适配器等价与梯度检查，通过后才允许训练。若仍排队，报告排队；若一致性失败，保存失败证据并停止下游，不能用新编码结果覆盖旧基线。

## 验收与证据边界

- 真实模型、LoRA与processor加载记录；图像frame、token数量和旧缓存对齐。新鲜基线与零适配器必须等价，先缓存输入的路径也必须等价；旧FP16 token缓存允许精度差，但不得悄悄改变原基线决策。
- 两位置同参数量；仅图像位置允许新增模块，文本输入、位置编号与attention mask不改。主干与原头没有可训练参数；新增模块收到有限非零梯度。
- 保存全部128个候选L、M、最终127个挑战者分数、模型/输入hash、训练损失、梯度、峰值内存和时间。首个样本保存patch残差与检索token变化；这不是物体mask或ownership证据。
- 训练终点固定，四臂预测封存后统一join。报告RAW、旧外部M头、新鲜零适配器基线、四臂的救回/损失/改变决策，并单列内部M错绑干预。8图只能检验运行与开发方向，不能验证可推广性，也不能据负结果判定表示缺信息。
- 成功标准首先是原流程可回放、梯度和真实M条件有效；性能收益须另看留出结果，工程PASS不是HYP GO。

原工程4图清单及其生成程序另存 engineering_manifest4.json、engineering_manifest_validation4.json、engineering_prepare4.py；扩面在任何GPU训练之前冻结。
