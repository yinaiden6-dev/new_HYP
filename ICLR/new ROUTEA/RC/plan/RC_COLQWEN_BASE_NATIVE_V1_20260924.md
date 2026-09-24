# ColQwen2.5 base 3B：自有 C128 与整体质量机制

2026-09-24 用户明确指定 workspace/colqwen。采用该历史路线模型 models/downloaded_models/colqwen2.5-base（Qwen2.5-VL-3B-Instruct骨干＋固定未训练检索投影），不加载vidore v0.2或ColNomic检索LoRA，不称未预训练模型或训练版检索器。base不含检索微调，但下游小头依然训练。

H593已打开分组五折开发协议。RAW为完整5413图库FP64 MaxSim，自有自然C128，图库身份去重；不插入target，全部593计分。CONTENT7 COST1/CE与MASS5 COST1/CE按原五折重训，原预算2000步；MASS5包含M×L、M、L与RAW差距＋偏置。推理时M循环错位64项作候选绑定控制。不是LLM前后信息丢失定位实验，不是外部确认。

复用 colqwen/difficult/raw_gallery/cache/colqwen_gallery_emb_difficult.pt 的5413项缓存，按索引与原文件名（strip后）核对顺序；不使用只有5412项的早期缓存。首片重新编码图库物理行0/2706/5412，检查序列长度、每token cosine>=.999与relativeL2<=.03；不匹配即停止，不混用。完整模型权重加载不允许missing/mismatched键。当前transformers映射旧model.*→language_model.*，保存映射和加载记录。保留checkpoint固定投影，不允许本次随机新建检索投影。

Query按模型原process_images编码，PIL RGB，无新增EXIF处理；BF16模型、FP16存储，RAW全部有效序列tokens，内容L仅image tokens归一化MaxSim。旧图库image token范围从同processor固定prefix/suffix确定，三个重新编码锚点验证；token长度不匹配停止。保存新query tokens、token mask、全图库分数、C128全部统计与匹配索引。

RoMa M采用原质量定义与原网格；优先复用相同query/reference物理图的ColNomic、ColPali原生池已有M，只对缺少的配对重算，并保留u/v中间量。质量未使用候选正确身份。所有候选预测封存后才进行held标签汇总。

GPU50片最大并行50，先首片通过，再自动放行其余；15分钟时限、软预算660秒，保存并续跑，上限16次。后续缺失质量首片与49片依赖执行，CPU五折与汇总自动接续。失败不释放下游。申请accelerated首片，可用dev_accelerated；不改变现有蒸馏实验。
