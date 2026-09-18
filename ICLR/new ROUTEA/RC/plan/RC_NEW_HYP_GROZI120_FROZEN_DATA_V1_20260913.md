# GroZi-120：获准只读的外部身份识别，冻结数据V1

用户已明确回复“允许，按这套只读外部确认方案使用 GroZi-120”。该授权仅覆盖GroZi只读外部确认；不解封D1-MI或formal392。原封存目录及DO_NOT_CONSUME保留不动，派生产物写入`results/rc_new_hyp_grozi120_external_v1`。用户另明确不再提供新药盒实拍；本次为公开商品外部评测。

## 已核对的实际来源

原下载日期2026-09-05；source_receipt记录仅下载/完整性检查，model_read_count、image_decode_count、reference/query_selection_count及target_outcome_read_count均0。本轮重新哈希与下载记录一致：

- inVitro.zip：00017cecf617dac20bb376dd4a8c10eeeca4699650888b052bbade06dfe25498；120商品、676个JPEG reference视图，另有PNG重复格式及mask，后两者不用。
- inSitu.zip：c59b26238f1dcda10f2fe607d477de9e19e87b69842899f478b1bbb9f40c2875；120商品、11194个既有商品裁图PNG。

公开作者论文：https://www.michelemerler.com/papers/grozi_cvprw07.pdf ；官方项目：https://grozi.ucsd.edu/ 。作者将数据用于web商品图到店内商品识别与定位研究。当前官方页未给出明确再分发许可证，旧receipt也未确认；本次限定内部研究分析，不重新分发图片、不声称取得额外再分发授权，论文引用原作者。已下载不等于预训练模型从未见过。

## 元数据修正：仅压缩包外派生，不改原件

商品82的coordinates.txt有122行，但图片仅61张；前61行与后61行逐行完全相同，info.txt也为重复两遍的相同61张说明。精确断言这个重复后只取前61行。其余119商品不进行此修正。修正后必须11194个坐标映射与PNG文件序号完整一一对应。

只使用coordinates.txt的video编号与frame编号用于防止连续帧伪独立；bbox四列、mask和点均不进入模型或选择评分。相同视频帧同SKU出现多个实例时，按预定hash选择一个已有裁图；不生成新真值裁图。

## 先定来源，再选帧：480 query、120 reference

每SKU从其原JPEG reference中按`GROZI_EXTERNAL_V1_20260913|reference|<archive_member>`的SHA256升序取第一张，不看图、不选最有利视角。

每SKU仅选择一个原视频，避免同一SKU跨多个视频把绝大多数数据连成一个依赖分量。可选视频要求至少4个不同frame；按`GROZI_EXTERNAL_V1_20260913|video|<identity>|<video>`的SHA256升序取第一项。这个规则只看元数据，未看图像、token或模型结果。120个SKU全部符合，形成27个选中视频组（原始29视频）。

在选中视频内，按不同frame排序；每个frame若有多个裁图，按固定`frame_instance` hash取一张。对n个frame取索引`floor((k+.5)*n/4)`，k=0,1,2,3。每SKU四张，共480张。最小可用frame数5，取样索引必须互异。Query按另一固定hash排序后匿名编号；target与视频分组只在curator文件。参考gallery标签供身份注册使用，query目标不交给scorer。

下载包内容在本协议前只检查了目录和文本元数据，没有图片目视、解码或模型forward。协议及清单封存后才提取600张被选图片，核对原始byte/EXIF-RGB像素指纹和精确重复。出现样本或reference/query精确重复、跨标签冲突、损坏或遗漏时，在推理前记录并停止本版，不凭模型表现替换样本。

## 模型、竞争空间与统计

使用已经训练并独立验证的固定模型包`results/rc_new_hyp_external_head_freeze_v1/bundle.json`，主COST1、次CE；对照COST4、GROUP_COST4、RAW2_CE和RAW，另做冻结COST1的CBIND。GroZi零训练、零校准、无checkpoint选择。

120个新reference追加到原5413物理行gallery，合计5533物理行；原714/715等价修正保持，名义5532身份（若数据资格发现语义/精确重复需在推理前处置，不能假装为不同身份）。RAW全gallery排序产生自然C128，全部127 challenger；不插target、不补重复候选。RAW保持原FP32 MaxSim/求和后转FP64，C4与head保持FP64；reference采用原未做EXIF转置的编码帧，query采用明确EXIF转正帧，RoMa映射沿原canonical geometry处理器契约。

主联合门沿固定头计划：COST1对GROUP_COST4、COST4、RAW及RAW2_CE的query净增>0、等视频组平均差>0且95%视频组bootstrap下界>0；正确绑定相对CBIND也满足相同条件。视频组作为27个重采样单位，100000次、seed20260913。报告等SKU/微平均、MRR、naturalC128召回、救回/损失、分视频及动作情况。不能把480张当成480个独立来源，更不能把11194连续帧当成同等数量独立样本。该单商店数据不能证明跨商店泛化。CE为次分析，不替换主模型。

本轮若通过，结论限于“固定药盒开发模型，经仅reference注册，能够迁移到本GroZi商品裁图身份检索，并以绑定内容证据取得可靠净增”。不能宣称整图无框定位、ownership、开放集或完整过目不忘系统。它与RPC协议分别报告，不能挑成功的数据集隐去失败。原28/32、99/128及H593 OOF保持各自账本。

图像派生与模型运行的工程PASS不等于外部new HYP GO。外部评分仍须新gallery数值适配完成、原输入重放通过、全部预测先封存，再join标签。
