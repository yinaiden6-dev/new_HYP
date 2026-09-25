# H593 同协议图像重排对照

用户要求补齐本地 reranker 对照，回答常规图像重排能否复现 COST1 收益。

## 固定设计

- H593 原五折、原图库、原自然 RAW C128、原 query 字节与图像帧。全部593张分母，目标缺席不注入、不删除。
- 冻结本地 Qwen3-VL-Reranker-2B 预训练模型。图像query＋图像reference，不读取OCR、文件名文本、身份文字或标签。提示词在看结果前固定于authority。
- 对全部75,904个图像对评分。主排序读出官方 yes−no 线性层的 sigmoid 前 logit，避免 BF16 sigmoid 饱和造成伪并列；同时完整报告官方 sigmoid 排序。同分先按 RAW rank，再按物理行。
- 保持官方图片像素预算（每张1280视觉token上限）、模型BF16、SDPA。每次两张图片；记录 token/grid，遇到官方NULL回退、图片缺失、截断、非有限分数立即失败。
- 原 RAW426、COST1481、CE486 必须逐query复核并复现汇总。不得与旧Top10、OCR或不同药盒集合直接比较。

## 已有结果的适用范围

`rerank/colnomc_rerank` 已有 outcome1132/difficult224 的Top10结果；Qwen3 hybrid混入RAW排名并要求0.15切换优势。该结果不是本同C128对照，不能宣称该预训练模型无用。MonoQwen现有实现为OCR文本query→图像reference，不能代替本图像对图像主对照。

## 同任务训练对照

一次GPU评分供两个CPU校准使用：原COST1和原CE；仅输入RAW标准化分差与Qwen logit标准化分差，两个权重加一个偏置。逐query在128候选内部标准化，分母最低1e-12。三个参数全零初始化，FP64 AdamW lr0.03、weight_decay0.001、2000步；保持原有效TRAIN记录与次序。HOLD logit0，只有最高挑战者分数严格大于0才SWITCH。

直接预训练排序回答即用效果；训练校准回答同样得到本任务身份监督后常规重排能做到多少。后者不是端到端微调，也不宣称与七参数头容量完全相同。

所有图像评分及校准held预测封存之后才读取全局身份标签。汇总R@1、候选内MRR（缺席为0）、救回/误伤/决策变化、分组bootstrap区间、全部候选分数和运行时间。每折拟合仅可读取自身TRAIN身份与图库映射。

## 执行与缓存

首张完整C128进行官方输出一致性、模型加载、图片输入检查，合格后释放50片，最大并行50；每次GPU申请15分钟，按候选保存断点、同Job requeue（最多16次）。CPU五折训练/汇总10分钟，可续跑。GPU无需为校准重复评分。

保存各候选logit、官方sigmoid、candidate physical row、图片SHA、模型/代码SHA、图像grid、token数量、计算时间、head参数与held logits。中间量保存在专属result根，不改变旧结果。原图与模型不复制入GitHub。

## 结论边界

这是已打开H593开发面板上的五折比较，不是外部确认。reranker基线回答相对效果，不单独证明机制创新、精确空间ownership或ColNomic表示缺信息；绑定干预和质量/内容机制对照仍承担解释工作。不得预设哪个模型胜出。
