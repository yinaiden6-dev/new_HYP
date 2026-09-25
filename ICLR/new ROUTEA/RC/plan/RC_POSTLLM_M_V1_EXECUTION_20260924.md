# POST：LLM后、检索投影前的M注入

授权：用户明确要求“现在做后注入”。目标是检验内部使用M能否纠错，不以超过外部M头作为必要条件。

## 只改变位置

复用V4固定TRAIN16、已打开PROBE8、原ColNomic自然C128、冻结ColNomic主干与原reference tokens。POST把原来的同构质量残差适配器移到LLM最后隐藏状态之后、custom_text_proj之前，仅改变图像patch位置的hidden。保留原检索投影的base权重、bias和检索LoRA，以及原BF16加法、投影、归一化。PRE封存结果不改动。

所有24图的hidden、image_mask、native_tokens、fresh_L0已在V2缓存，可复用；不重跑视觉编码器、LLM或RoMa。GPU仅执行小适配器、原投影和full-reference MaxSim。实际运行时按缓存native tokens和fresh_L0验证，不将仅找到文件等同于前向一致。

## 三条训练臂

- POST_REAL：真实M条件训练和推理。
- POST_CONSTANT：标准化M固定0，同参数量、同初始化和训练，必须新训，不能复用PRE_CONSTANT。
- POST_SHUFFLED：M按C128固定循环左移一位；候选、RAW、标签、reference顺序不动。

对POST_REAL另做相同冻结头、相同适配器的constant/shuffled推理，分别在TRAIN终点和probe保存。训练错绑与推理错绑分开报告。

固定V4配方：hidden3584、瓶颈16、118304个适配器参数、相同seed17和V3零残差初始state、M增益sqrt3584、TRAIN冻结logM标准化、residual_scale .1；原warm INTERNAL3头；128次逐query完整C128 COST1；adapter lr3e-4，head lr.03，AdamW wd.001，分别clip1。适配器FP32、head/损失FP64；末端只读RAW与Ltheta及bias，不读M。

checkpoint16/64/128，TRAIN endpoint16/128；probe只评价固定128终点。固定128后，预定CPU小头refit从同一warm出发，用TRAIN16训练2000次完整批量更新，作为与PRE一致的次级诊断。所有臂完整报告，不按probe选阈值、checkpoint、超参或优胜模型。

## 工程与执行

新代码隔离于rc_postllm_m_v1目录。训练、endpoint、重放VJP和恢复函数从封存V4复制，仅替换PRE/POST名称和记录hidden来源；单元测试核对源级差异。实际pilot必须通过全部24图零残差token一致、首图C128分数一致、非零有限梯度、适配器及头均更新、token变化可见、参数/优化器断点恢复一致。

先提交单GPU10分钟pilot到dev_accelerated/accelerated，合格后自动提交三臂及依赖CPU refit、join。每个任务8CPU，GPU任务32GB，CPU任务16GB，10分钟；候选中途保存，超时同Job requeue至多48次。保留已有源缓存、中间分数、参数、梯度、输出tokens、127挑战者和HOLD=0。其它任务不改动。

## 结论边界

PRE/POST比较锁定算法和训练预算，不声称两位置计算成本相等。原始预训练内容表示、匹配器均冻结；本轮不提供学习新编码器或空间ownership证据。probe8是6个component的历史开发面板，虽与TRAIN组/身份分离，已被反复分析，因此只能形成机制开发结果，不能冒充新独立确认。若本轮负结果，不等于所有LLM后注入无效。
