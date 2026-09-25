# 原H593七参数COST1：已有证据的独立算术归因

自然ColNomic C128，原五折held参数；RAW426/593，原生481/593（58救回/3误伤）。不使用简化PRODUCT5头代替原头。

原58例救回中：58例M相对RAW winner更高；58例原生质量归一化局部分数更高；57例自由ColNomic内容分数反而更低。
这是按原结果事后选出的救回组的描述统计，57/58不是因果比例。

| 原头输入项 | 对target相对HOLD logit的平均贡献 |
|---|---:|
| standardized_raw_gap | -0.395384113 |
| symmetric_local_score | -1.480011391 |
| symmetric_visibility_mass | +2.472485947 |
| symmetric_visibility_normalized_local_similarity | +1.495394394 |
| symmetric_query_spatial_robustness | +0.148973483 |
| symmetric_reference_spatial_robustness | -0.016325776 |
| bias | -1.250423684 |

这些是算术贡献。原生S/M仍含u加权和reference v加权，不能称纯内容；Q/R robustness是滚动权重控制的评分差，不是独立算子贡献。

独立重算75311个原始logit，最大误差3.553e-15；全部593个动作与原封存预测一致。58例原生特征另从原算子S/M及控制分数重建，最大误差0.000e+00。

逐例分差、7项贡献、完整参数与原始来源见rescues58.json；源文件SHA、公式和边界见summary.json；复核入口recompute.py。
