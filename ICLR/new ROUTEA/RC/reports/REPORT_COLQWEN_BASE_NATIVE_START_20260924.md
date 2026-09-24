# ColQwen2.5 base 3B 对照已提交（2026-09-24）

用户指定 /hkfs/work/workspace/scratch/ap7811-benchmark/colqwen，实际历史image-only程序默认模型为 models/downloaded_models/colqwen2.5-base。config确认Qwen2.5-VL-3B-Instruct骨干，hidden_size=2048，36层。checkpoint包含固定custom_text_proj权重；本对照不加载检索LoRA，明确为未做检索微调的base，而非未经预训练模型或训练版ColQwen。

H593原分组五折，完整5413图库自行检索并去重得到C128；RAW、纯内容CONTENT7、整体质量MASS5分别报告，两个头各训COST1/CE，并保留M候选错配对照。未命中候选的query继续计入分母。

已确认旧difficult图库缓存5413项，indices=0..5412、与当前图库文件stem.strip()顺序一致；另一个outcome旧缓存5412项未采用。真实数值兼容性仍需首片重编码3个reference锚点，不因名称一致就宣称兼容。

新首片5161030：accelerated/dev_accelerated共同排队，1GPU/8CPU，申请64G内存与15分钟时限。提交核查状态PENDING，QOSMaxJobsPerUserLimit；实际spooled脚本逐字节一致。数值小检验已通过：ragged FP64 MaxSim与独立NumPy误差1.42e-14。完整模型loading与真实锚点尚未验收。

首片成功后自动提交剩余49片（最大并行50）；随后只补ColNomic/ColPali已有原生候选质量缓存中缺少的配对，再接5折CPU训练与汇总。GPU软预算660秒，允许断点requeue，最多16次；任一失败不释放依赖。现有M蒸馏链未改动。

保存query tokens、image masks、全图库分数、完整C128统计、匹配索引、新计算质量的u/v，以及各折参数、全部127个challenger logits。结果目录results/rc_colqwen_base_native_v1；协议plan/RC_COLQWEN_BASE_NATIVE_V1_20260924.md；authority registry/rc_colqwen_base_native_authority_v1_20260924.json。

本实验回答质量机制是否在未检索微调的base上仍有增益，不能单独回答ColNomic应改LLM前还是后，不能将3B base与7B ColNomic差异全部归因于检索训练。
