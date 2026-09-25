# RoMa粗细特征与ColNomic融合：代码核对及最小对照

2026-09-22用户提出三个问题：为什么用RoMa；其粗/细embedding能否在ColNomic输出端或LLM前融合；能否跳过RoMa匹配器。本文是新方向的实现依据和待验证方案，不是已完成结果，不替代正在运行的坐标精度消融。

## 当前真实接口

本地RoMaV2默认：Descriptor为DINOv3 ViT-L/16，读取索引11、17两个中间层，各1024维；800×800输入对应50×50网格。FineFeatures为VGG19-BN，尺度1/2/4处通道64/128/256。FineFeatures初始化时pretrained=False，但完整romav2.0.1.pt随后加载覆盖其参数；抽取时必须使用该封存checkpoint，不能把初始化CNN当成细特征模型。

两类特征本身首先是单图表示。matcher联合两图粗特征产生对应与置信度，refiners使用两侧细特征、已有坐标、局部相关性继续更新warp和confidence。当前new HYP评分只读取overlap_AB/BA形成两侧token可见性，再与ColNomic full-reference MaxSim联合；未直接把DINO/VGG embedding相乘到ColNomic。

因此RoMa的实际作用是提供candidate-conditioned匹配证据。粗细多尺度是其实现来源，不足以解释为什么必须用完整RoMa；“预训练特征是否已足够，匹配器是否有额外价值”仍需要同协议对照。

ColNomic本地模型：Qwen2.5-VL视觉编码器和merger -> 3584维图像嵌入散射到序列 -> 28层语言主干 -> custom_text_proj -> 128维 -> token级L2归一化。LoRA适配器是冻结原模型的一部分。所谓“后半段”须明确是3584维隐藏状态还是最终128维输出；建议第一轮采用最终128维，能复用封存tokens，干预边界更清楚。

## 三个可行位置

1. 后融合：将单图粗/细特征按原EXIF/canonical geometry汇聚到同图ColNomic token网格，经过可学习投影后，调制最终128维token，再沿用MaxSim。query/reference使用共享适配器。无需每个candidate重跑7B主干。
2. LLM前融合：在视觉merger输出、masked_scatter进inputs_embeds之前调制3584维图像token，保留原位置编号、文本提示和mask。不能直接将DINO1024或VGG64/128/256维向量与3584维向量逐元素相乘。需要空间对齐和通道投影。即便主干参数冻结，要学入口适配器仍要反向传播经过后续语言主干；原ColNomic输出缓存不再适用。
3. 跳过匹配器：独立抽取DINOv3/VGG特征，与每张图的ColNomic表示融合，省去matcher/refiners。若读取RoMa checkpoint里的CNN，仍依赖RoMa训练过的特征，准确名称是“无RoMa匹配器”，不是完全无RoMa预训练。若换成独立通用预训练编码器，属于另一个来源对照。

不建议直接裸乘两个embedding：即使维度偶然相同，通道语义也未对齐；数值尺度、正负号会改变原有内容相似性。正标量先乘token再L2归一化会被抵消。现有可见性权重作用于已计算的相似度/加权聚合，不能等同于重新归一化后的token缩放。

可学习通道门控的一种待测实现：z_t为原ColNomic向量；f_t为对齐后的粗/细表示；g_t=tanh(A LN(f_t)); z'_t=normalize(z_t*(1+alpha*g_t))。A使用低秩投影，最后投影零初始化、alpha固定非零，从原功能起步；不能把alpha和A都置零造成双重零梯度。对输入3584维版本同理，但不额外强加输出端归一化规则。也可用残差相加作为不同融合形式；先固定一种再做形式比较。

## 优先顺序与对照

先完成“后融合×跳过匹配器”，比较内容来源：ColNomic-only等容量适配器、粗特征、细特征、粗+细；原COST1和CE保持已冻结基线。第一轮固定原RAW C128、五折和HOLD/SWITCH，以原retrieval identity标签训练共享小型适配器及相同形式决策头，不用box/mask/point标签、不靠heldout选超参。保持超参/种子/训练预算一致，报告参数量；ColNomic-only加头控制用于分清额外训练容量和额外特征来源。

“有/无匹配证据”的关键对照，应在相同融合token上分别采用全参考MaxSim与RoMa weighted MaxSim，再使用相同训练协议；不能只比较一个新适配器与完全没重训的旧头。全部593计分，23个不入原C128继续保留；另报RAW原正确损失、救回、跨组件区间。若随后换候选生成器，另开全库检索指标，不能混记到当前C128结果。

只有后融合确定有效后，再比较同一特征来源在LLM前和最终128维的两个插入位置。前融合若按candidate reference条件化，则每个query-candidate需重新计算受调制的语言主干，失去单图索引复用；第一轮采用单图融合，把条件交互留在MaxSim/匹配阶段。若图像编码各自独立仍可缓存reference；缓存必须按适配器checkpoint版本重建。

上述方案保持retrieval-only下游训练，但预训练RoMa、DINO、ColNomic的既有监督来源应按原说明披露。改善与否未知；直接特征融合成功也不能被提前写成已经解释现有RoMa增益。

## 中间数据要求

按图像SHA、编码器checkpoint SHA、EXIF/frame/网格、fold、训练step版本索引，保存：对齐后的粗/细token；原始ColNomic128维token及其原缓存引用；学习投影与门控；融合后的token；每候选free/weighted MaxSim命中reference token及贡献；全127挑战者分数、头参数和HOLD/SWITCH。LLM前实验另存受调制的merger输出和输入门控。避免只保留top1及正确数。未汇聚的高分辨率CNN特征不默认全量落盘，如需分析原像素细节另明确存储范围；不会将汇聚缓存描述为完整隐藏激活。

## 核对来源

- 本地 third_party/RoMaV2/src/romav2/features.py、matcher.py、refiner.py、romav2.py；封存romav2.0.1.pt。
- 本地 .venv-colpali/lib/python3.13/site-packages/colpali_engine/models/qwen2_5/colqwen2_5/modeling_colqwen2_5.py：custom_text_proj与L2归一化。
- 本地 .venv-colpali/lib/python3.13/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py：get_image_features、masked_scatter、language_model。
- 本地 models/downloaded_models/colqwen2.5-7B-base/config.json、colnomic-embed-multimodal-7b/adapter_config.json。
- RoMaV2官方代码 https://github.com/Parskatt/RoMaV2 ，论文 https://arxiv.org/abs/2511.15706 。
- ColNomic官方模型卡 https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b 。

## 用户进一步明确目标：解释增益后，单独改造ColNomic

“加入DINO/VGG但去掉RoMa matcher”和“推理只用ColNomic”是两个不同模型目标。后者不再引入额外视觉编码器。2026-09-22当前问题应区分：额外图像信息、候选条件化的证据使用方式、最终决策校准。三者可以同时存在，不能把组合提升归因于某一项后直接宣称单模型必然可复制。

代码层面已知的三个作用位置：query加权汇聚；reference权重改变MaxSim最大值及命中位置；可见性总量进入评分并由决策头联合校准。这说明算法如何改变证据，但尚未量化每条路径对全部跨组净增的贡献。此次保存free/weighted命中索引、逐token贡献、坐标和权重，正是为了回放这些作用。坐标精度干预也不能自动替代粗/细embedding来源对照。

纯ColNomic的最小改造建议：共享编码器维持原图输入与原tokens，以两图的token相似度、位置及局部匹配分布为输入，学习candidate-conditioned的逐token质量/可靠性，再进行full-reference内容比较；先训练小型token级质量模块，必要时再训练ColNomic输出适配器。质量应有query-reference条件，单图“清晰度/显著性”不是同一概念。保持主干冻结的成功只能证明改进读取方式；需要重新编码/改变adapter或主干才可称表示也改善。

检索标签仍是唯一新增人工监督；第一阶段不增加RoMa权重拟合/蒸馏辅助损失。若以后训练student拟合RoMa质量图，须单列teacher-assisted分支，不能与严格仅检索损失的版本混为一谈。作为诊断读取RoMa中间文件不等于给训练加入这些目标。

已有learned_colnomic_only不能跳过：programs/run_rc_h593_learned_colnomic_only_v1.py中的FEATURES为RAW,F,B,Gq,Gr,U，statistics()先将token相似度压成五个候选级内容统计，再训练3/7参数头，new_encoder_forwards=0。这一对照应复用；它没有训练token级质量图，也没有改变ColNomic表示，因此无法排除上述纯ColNomic改造。

验证路径：先用完整中间数据隔离主要收益路径；然后比较原ColNomic、原内容统计小头、纯ColNomic token级质量模块、原RoMa联合系统。保持原C128和分组五折，训练预算及容量对照明确。逐query保留全部分数与中间量，报告净增、损失、候选缺失及跨组稳定性。任何新模型提升都必须由留出结果确认，解释成立本身不构成提升证明。此节是待实施方案，尚未提交融合/纯ColNomic新训练。
