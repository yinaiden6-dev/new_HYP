# 八臂融合正式训练中间数据

目录存在不表示训练完成。实际状态见 `../rc_h593_feature_fusion_dispatch_v2/status.json` 与各fit的validation。

- `benchmark.json`：真实图像完整C128、固定合成目标的耗时/显存/梯度资格。不是身份精度结果，其参数不进入正式训练。
- `fitNNN/checkpoint.pt`：可原子覆盖的训练恢复文件，保存精确step、七参数头、4096参数适配器（NO_ADAPTER除外）、AdamW状态、初始化、固定配置与来源。配置轴由authority固定。
- `fitNNN/final_model.pt`：2000步结束时不可变FP64模型；预测和分析副本绑定其SHA，而不是不断写入的恢复文件。
- `fitNNN/chunks/<array>_<index>.json`：每个成功分片的正常退出、step及完成标志。分片成功不等于全部训练/预测已完成。
- `fitNNN/encoded/<key>.pt`：预测时query/reference的融合token和门控FP32分析副本；原始token/固定投影在全量cache、准确FP64模型在final_model中，数值评分仍用FP64。分析副本不是逐位原始FP64激活。
- `fitNNN/predictions/queryNNN.pt`：完整128候选C4、自由/加权匹配位置、逐token贡献、127×6输入、全部127logits、原RAW胜者与HOLD/SWITCH决策。身份标签不在该文件中。
- `fitNNN/payload.json`、`validation.json`：预测清单及独立NumPy读出核验；检查全部127logits/动作、全128特征及贡献求和，不声称重新训练验证。
- 全部200配置验证齐全后才建立prelabel seal并进行结果join。`result.json`保留每query、两种seed、两loss及所有臂，不挑最好seed。

全量原/粗/细特征按图像/几何/token SHA去重保存在`../rc_h593_feature_fusion_cache_v1/`，合格的129个pilot文件仍引用原目录，不复制或改写。
