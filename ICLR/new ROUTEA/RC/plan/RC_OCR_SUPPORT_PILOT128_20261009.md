# OCR空间支持首轮试验

用户于2026-10-09授权“试一下”。仅新增独立试验，不修改593上游归因任务。

面板：H593 execution ordinal 0–127，选择不看OCR覆盖、target、原成功失败。自然ColNomic C128不变，原五折身份component划分不变。该128是H593前缀，不等同于旧EVAL128，更不是全新外部数据。

全部query及其候选图库图像统一使用本地PP-OCRv5_server_det＋en_PP-OCRv5_mobile_rec。冻结OCR，CPU执行一次每张图片，保留文本、多边形、识别分数、检测输出、原图SHA与EXIF方向。已有DeepSeek全图字符串及Paddle＋dots混合输出不能提供同一识别器和完整位置，故只用于后续独立对照，不混入本次臂比较。无人工文字、框标注或字段标签参与训练。

空间支持直接复用原RoMa query/reference_visibility；没有新RoMa或ColNomic前向。OCR框与token cell统一在完整EXIF方向坐标系，按交叠面积汇聚u/v，各自除以原整图均值以分离整体M和局部形状。保留原M=sqrt(mean(u)mean(v))。不把检测/识别分数当作身份概率。

文字内容使用逐词自由匹配到整个reference OCR，保留单字符数字、十进制和单位。每个query词按图库IDF及识别分数加权，reference取所有含同一词的文字框中最大支持；不作RoMa坐标硬绑定。数值精确匹配，不把10和20当成近义。重复通用文字仍可能造成混淆，本轮不预设解决。

五臂：

1. CALIBRATION_ONLY：原COST1折头＋同训练的标量/偏置再校准。
2. TEXT：加无空间加权OCR文字匹配。
3. TEXT_M：加相同文字分数与M×文字项。
4. TEXT_UV：相同结构，文字用正确u/v框支持；保留M×文字项。
5. TEXT_UV_SHUFFLED：每对图分别确定性置换文字框的支持值，保持支持分布与M，破坏文字-支持绑定；单独同预算重训。

所有新增读出从原对应外折COST1起步，四个残差参数（base logit、text差、M×text差、偏置）；无对应输入的列固定为0。原模型始终作为冻结baseline报告。每外折的TRAIN和held都用该外折头，不混用其他折OOF头防止TRAIN分数混入held标签。新增头只读此128面板TRAIN身份，target不在C128的TRAIN行不参与原COST1损失，held缺失目标仍计失败。保留整图库候选身份轴，不用文件名推断文本。

原COST1 unit loss，FP64 AdamW，2000步，lr=.03、weight_decay=.001、seed17，固定0阈值，无held阈值搜索。特征RMS仅在有效TRAIN拟合。输出五折逐候选logits、训练损失、参数、baseline逐query复现、标量评分独立核算、rescue/break、component等权探索性bootstrap区间；不将未校正多重比较称为确认性显著性。

主要比较TEXT_UV−TEXT_M（局部空间使用）与TEXT_UV−TEXT_UV_SHUFFLED（正确文字绑定）。若只有TEXT胜出，只支持OCR增量；若真实与打乱相同，不能宣称空间支持指导文字的机制成立。全头仍保留原视觉M，不能解释为去掉RoMa。

首图CPU真实OCR通过后，32片OCR→完整性索引→16片支持回放→5折训练→独立汇总。10分钟预算，逐图/逐query/50步断点；不吞工程异常。缺失输入或非有限结果阻断下游。
