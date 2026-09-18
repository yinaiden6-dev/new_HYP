# new HYP：RPC 单商品跨相机外部迁移，V1

状态：在任何RPC图像目视、编码或模型推理之前固定数据设计。用户已明确选择“先用公开商品数据验证跨数据集效果，再补药盒实拍”。本任务是外部受控商品迁移，不是新拍药盒自然场景确认。

## 数据适用性与来源

官方项目：https://rpc-dataset.github.io/ ，官方分发：https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset 。单商品exemplar分割有53739张整图、200个细粒度商品类别。只采用该分割；不使用多物体checkout图或真值框裁剪。原作者称其为training set，但本项目不在RPC上训练，它对本项目是新的外部迁移数据来源。官方许可CC BY-NC-SA 4.0，保留作者及来源。

本地已读取官方目录和instances_train2019.json元数据，尚未读取任何RPC图片。已有RC reports/plan/registry关键词检索未找到过去RPC实验记录；这只是本项目可见记录的暴露核对，不能证明预训练模型未见过该公开数据。

SOP已经取得文件名元数据，但其身份按eBay item/listing组织，不能未经核对等同准确SKU；不将它自动替换为药盒身份确认。Products-10K具有SKU标签，但官方OneDrive匿名目录探测404，未以不可达接口宣称数据已到位。选择RPC依据任务接口与可达性，不依据模型结果。

## 固定600 query与200 reference

全部200类别均进入固定gallery，每类一张reference，共200张。以源码中固定hash盐排序，在camera0且文件名产品前缀不含'-back'的照片中取首张；若某类没有此类照片，先在读取任何图像前报告元数据不适配，不偷偷换选择规则。

每类camera1、camera2、camera3各取一张query：在该类别/相机可用文件中按另一固定hash排序取首张，共600张。没有挑选角度、困难样本或模型错误；query可包含back。全部抽样只依赖已存在的公开类别、相机文件名及SHA256。gallery/query图像ID严格分离；query ID按所有选中文件的第三个hash排序后匿名编号。

清单冻结后，按官方单文件下载端点接收800张原始整图，记录原始字节SHA和EXIF转正RGB像素SHA。不得用框、mask、segmentation、point_xy或人工空间标签；源JSON只有image_id→category_id投影可进入curator标签。worker只收到匿名图片路径和query ID，类别只在gallery注册与评测join使用。

完整下载后、任何推理之前检查精确重复或跨身份冲突。出现query与gallery同图、选中图像跨身份冲突、缺图、损坏或非单商品元数据等问题时冻结该版并报告；不看评分后补图。所有600条应进入最终分母，target缺自然C128记失败。

## 冻结模型与自然竞争

使用RC_NEW_HYP_EXTERNAL_HEAD_FREEZE_V1_20260913.md产出的全H593开发数据固定头：主COST1，次CE，同协议COST4、GROUP_COST4、RAW2_CE以及无头RAW。全部head/bundle与编码器、处理器、RoMa、soft-visibility/full-reference MaxSim、六统计实现SHA必须在任何RPC推理前封存。RPC零训练、零校准、零选checkpoint；仅用新reference注册身份。

RAW在固定200-reference gallery完整排名产生自然C128，不插target、不改宽度。完整127 challenger，FP64，maxlogit>0才SWITCH，否则HOLD；同样六统计和顺序/tie规则。另做冻结COST1的CBIND，保留RAW轴，不借target标签设计错配。当前旧代码部分入口硬编码5413行gallery，新adapter必须通过旧输入原样重放和新gallery索引/全候选核算后才能推理。该adapter尚待实现，下载完成不能当作推理已完成。

## 一次完整检验

沿用固定头计划的主联合门：COST1分别优于GROUP_COST4、COST4、RAW，并优于RAW2_CE和自身CBIND；query净增>0，等SKU差>0，95%bootstrap下界>0。17个官方supercategory内按SKU重采样，每个SKU的三张query一起重采样；100000次，seed20260913。所有SKU同样三张，主等SKU与query点估计等价。额外报告按17个supercategory整组bootstrap的敏感性区间、三个相机与17类分层效果；若只在特定相机/品类成立，不宣称普适。

CE为次分析，不在主模型失败后换主臂。完整报告MRR、RAW recallC128、rescue/break、HOLD/SWITCH组成。置信区间不足即未确认；不进行基于结果的扩样、阈值/温度扫描或图像替换。抽样层选择、800图像清单和全部数值配方在评分之前固定。

通过最多支持“药盒开发集学到的reference证据校准，能够无RPC微调迁移到这批新商品的受控跨相机识别，并产生可靠净增”。不支持自然杂乱药盒场景、空间ownership、开放集拒识、所有视角可识别或完整过目不忘系统。旧28/32、99/128和H593 OOF481/486分别保留，不能与RPC分母拼账。
