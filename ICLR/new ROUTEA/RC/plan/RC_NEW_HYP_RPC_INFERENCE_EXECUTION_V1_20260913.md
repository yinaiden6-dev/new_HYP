# RPC600固定模型外部执行 V1

承接用户“开始”及已经在任何RPC推理前冻结的 RC_NEW_HYP_RPC_TRANSFER_V1_20260913.md。全部800原始单商品整图已下载并通过完整性/重复核对。本轮不训练、不改头、不改抽样，不根据GroZi结果选择模型。原先指定的主COST1、次CE与COST4/GROUP_COST4/RAW2_CE全部保留。

## 固定计算与运行

独立200-reference gallery，物理行0..199且身份互异，600 query按worker顺序处理。RAW使用原ColNomic/processor、FP32 MaxSim求和后存FP64；完整200排名取自然C128，不插入target。RoMa使用冻结原数值循环和profile；六统计/七参数、全127 challenger、maxlogit>0才SWITCH、否则HOLD不变。CBIND为C128物理轴完整证据循环移位64，RAW不动；所有头REAL/CBIND封存全部logit及动作。

参考编码单任务15分钟：原4张已开放TRAIN的tokens、5413 RAW分数及自然C128逐bit重放，然后注册固定200 reference。参考完整sequence含template供RAW，image tokens供RoMa/C4；不读RPC query target。

75片，每片8 query。首片RAW10分钟→RoMa15分钟并完成独立CPU C4重放和head动作重算；只按工程通过释放余下74片。RAW和RoMa两个数组依次执行、各最多并行46，每片分别10/15分钟。所有75片预测封存后才一次join600标签。最后汇总10分钟。RAW使用.venv-colpali，RoMa使用原profile指定.venv-romav2，TORCH_HOME固定；真实spool逐字节验证。

Worker无curator、图像框/mask/点、D1-MI、formal392私有标签或GroZi结果访问。所有消费的authority、程序、manifest、head和stage receipt逐层SHA绑定；reference编码与自然推理不修改旧实现。全部RPC target缺召回保留600分母。

## 固定统计

按原数据协议：200 SKU，每SKU三相机图一起重采样；17个supercategory内分别对SKU有放回重采样，维持每层SKU数量，100000次、seed20260913。主等SKU与query平均相同。主联合门仍为COST1对RAW、COST4、GROUP_COST4、RAW2_CE和COST1_CBIND均有正净增且95%分层SKU区间下界>0。

附加敏感性为17个supercategory等组重采样区间，单独报告，不替代主门；报告各相机、品类、SKU、MRR、C128召回与HOLD/SWITCH分解。CE为次分析，不能看结果换主模型。独立计数和全部logit动作重建验证后才能报告RPC结果。GroZi独立账本不改写。

结论最多为受控单商品跨相机迁移；不宣称多物体结账定位、ownership、开放集或任意新环境成立。
