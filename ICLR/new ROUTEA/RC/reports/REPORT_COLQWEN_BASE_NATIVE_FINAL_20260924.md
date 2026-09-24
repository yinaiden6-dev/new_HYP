# ColQwen base 自有C128：H593最终结果（2026-09-24）

全部GPU采集、五折CPU训练和汇总已完成。5161390五折均成功，5161391汇总COMPLETED/0:0（33秒）。历史5161250是成功生成质量与head输入后续提失败，已恢复，不重复GPU。

模型是colqwen2.5-base：Qwen2.5-VL-3B-Instruct骨干与固定、未经过检索训练的投影，不加载检索LoRA。不是训练版ColQwen，也不是整个骨干无预训练。

固定原H593分组五折，使用本编码器对5413张图库检索得到的自然C128；目标召回438/593，155张候选缺失仍在593分母内。图库与query同编码配置；旧不兼容缓存不混用。

|方法|正确/593|对RAW救回/误伤|
|---|---:|---:|
|COLQWEN_BASE_RAW_C128|227|0/0|
|COLQWEN_BASE_IMAGE_L|225|1/3|
|CONTENT7_COST1|240|13/0|
|CONTENT7_CE|260|46/13|
|MASS5_COST1|322|96/1|
|MASS5_COST1_CBIND|182|3/48|
|MASS5_CE|333|117/11|
|MASS5_CE_CBIND|91|7/143|

MASS5 COST1比纯内容CONTENT7 COST1多88救、6损，净增82。相对RAW净增95；64组件等权增益95%bootstrap区间[10.80,21.42]个百分点。MASS5 CE相对RAW净增106。候选绑定错配后COST1=182，CE=91。

支持：整体候选质量M与自由内容L的简化接口，在这一base检索器下经小头重训仍有增量。不能将增益外推为训练版ColQwen指标、原七参数零样本迁移、独立外部确认或RoMa不可替代性。

来源质量共75904对：复用30074，新算45830。完整候选分数、质量、小头参数和全部127 challenger logits均保留。独立NumPy逐项乘加重算451866个分数，最大差7.1054e-15，593动作和计数一致，验证身份/component不跨折。

结果：results/rc_colqwen_base_native_v2/head/result.json；验收：同目录validation.json；独立核对：reports/rc_colqwen_base_final_independent_check_20260924.json。
