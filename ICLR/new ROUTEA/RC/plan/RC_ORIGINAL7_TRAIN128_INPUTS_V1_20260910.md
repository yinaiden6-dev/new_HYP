# 原七参数机制：TRAIN128输入扩充

本阶段承接用户持续推进retrieval-only研究的授权，截止北京时间2026-09-11 24:00。
已有EVAL128固定原模型的88→99、12救/1损结果独立保留。本阶段只准备训练输入，
不修改旧数据角色、旧结果或参数；不将新EVAL128的身份、图像或标签加入训练。

## 原因与待检验问题

OUTCOME-0337的两reference分解已定位：保留query权重的自由内容比较仍支持target，
reference权重改变argmax位置后发生反转，再由权重幅度放大。FREE8追加自由内容已试过，
旧EVAL28→27，不能换名重复。原完整模型也仍有19个新EVAL target不过阈值的错误。
更丰富训练视图及完整候选负样本是否改善七参数校准，是下一项待检验假设，
不是已经证明的全部失败根因。本阶段不据0337拟合阈值、方向、归一化或新增特征。

## 冻结样本与角色

仅原FULL_TRAIN32的12身份/组和PAIR64的20身份/组，共32身份/32组。
严格排除旧EVAL32及新EVAL128身份、来源组、图片，并排除formal392负名单query IDs/image SHA；
不读取其P/RoMa结果。可用269不同图片：保留原FULL32、PAIR64的96张，再选32张。
不使用组内另一个新增身份，不按RAW名次、是否进入C128或训练成绩挑选。

固定seed为RC_ORIGINAL7_TRAIN128_20260910；补图时优先当前已选图片最少且仍有
图片的组，tie按SHA(seed+'|group|'+group)；组内按SHA(seed+'|image|'+image_sha)。
图片以SHA去重；与必保留原图重复的记录不作为新增。执行顺序为原FULL32原序、
原PAIR64原序、新增32选择序。worker只含匿名T128、图片路径/哈希及处理元数据；
身份/组/原queryID在独立TRAIN curator中。生产和不同实现的metadata重放顺序一致。

冻结清单目录results/rc_original7_train128_manifest_v1，128图、32身份、32组；
18 difficult、9 new_difficult_train、101 outcome。它是训练输入扩展，不是新EVAL结论。

## 输入计算与复用

复用已经资格化的模型、图库、processor、gallery身份修复、当前runtime和原FP64算子。
原TRAIN56 TOKEN桥保留V1/95 authority；RoMa桥保留EVAL V2/bc671 authority；明确复用，
不重跑、不改写旧桥。新增authority逐项绑定与parent不变的数值/模型源码。

当前runtime重新提取128张query的image/template tokens并算完整5413物理reference
RAW分数，沿用correctedidentity去重后的5412排名及自然C128。旧redacted tokens只可作
元数据来源，不能替代fresh tokens。原FULL32位于前四分片，fresh token、C128及RAW
须逐bit匹配旧原训练输入；差异属于接口阻断，不能混入数据扩容结论。

RoMa沿自然C128全部reference物化原wq/wr、S、M、SQ、SR；匹配、自由full-reference
MaxSim、控制位移、FP64归约和训练特征公式完全沿用已验证实现。CPU独立重算全部
65536个C4 scalar和REAL/C_BIND原六列特征。此阶段不算新head、不读TRAIN target roles。
不插入target、不因后续target缺席C128而替换或丢弃预选图片。

资源：16片、每片8图，GPU数组最多16并发；每片1GPU/8CPU/96G/30分钟；
RAW汇总8CPU/16G/15分钟，全部C4与特征资格8CPU/16G/30分钟。所有作业带研究截止时间。
仅允许token_raw_shard→raw_aggregate→roma_shard→qualify_train四阶段；提交后继续独立工作，
不持续轮询调度器、不改运行中源码，完整分片在后续失败时保留。

## 后续拟合必须另行冻结的规则

后续只检验同一六列七参数结构的训练覆盖，不自动新增自由内容列或替换模型。
应保留原PAIR64辅助损失、seed17、零初始化、AdamW(lr=.03,weight_decay=.001)、2000步
及原FULL query loss。原32数据配方必须复现当前baseline参数；扩充后新参数先封存，
再分别重放旧EVAL32和已打开EVAL128，报告rescue/break、group和绑定，不能混合分母。
若新TRAIN有target不在自然C128，不能静默插入、替换或错误赋予winner身份；须在正式
fit协议中预先说明原定义域之外的处理并报告原选图数与有效FULL loss数。
当前输入authority不授权fit/eval阶段执行，不把输入ready、metadata PASS或提交状态称为科学GO。

不使用SAM、空间teacher、任务框/掩膜/点、D1-MI、GroZi或受保护推理结果。任务监督
仍仅检索身份/正负配对；冻结基础模型既有预训练来源照实披露。原时间截止保持不变。
