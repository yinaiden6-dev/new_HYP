# ColNomic：1117张查询的错误、Token与竞争机制审计

本报告讨论 1117 张独立查询、5405 条图库记录、5330 个SHA设计和35个历史配置。39095是重复评测次数。原标签用于复现历史成绩；视觉发现的标签歧义单独保留。

## 首页五个问题

**哪些情况下失败？** B03错误 247/1117，F03错误 145/1117。两者都对 829，图像错融合对 143，图像对融合错 41，两者都错 104。场景、面、数字规格、排版、主色、图案、清晰度等是可观察多标签，频数不能相加，也不能直接叫失败机制。

**哪些原因有证据？** 冻结特征与原始评分的确定性复放、Token贡献守恒、融合新增错误发生在评分公式加入文字之后，有直接证据。输入可辨识性、身份标签歧义、空间约束、特征Probe和像素编辑分别有范围限定的证据；不支持为全部失败指定唯一编码器原因。

**匹配、汇总和融合分别改变什么？** 下表来自全部1117×5405实际评分。OCR对齐包含文字支持，γ/λ拆分后不能把对齐可用性奖励的收益都算为布局匹配。Query自权重另保存完整候选界和确切top10/rank，未计算的低界scalar明确NaN。融合选α与门控仅用验证集，同时保存修复、损害和原143救回样本的损失。

|configuration|method|R1|R5|R10|MRR|repairs|harms|net|
|---|---|---|---|---|---|---|---|---|
|B03|baseline|0.778872|0.89615|0.920322|0.833984|0|0|0|
|B03|availability_only|0.823635|0.924799|0.944494|0.871753|54|4|50|
|B03|spatial_only|0.755595|0.885407|0.91137|0.813126|5|31|-26|
|B03|combined|0.826321|0.923903|0.942704|0.871956|58|5|53|
|B03|shuffled_spatial_only|0.743062|0.871083|0.903312|0.801059|4|44|-40|
|B03|shuffled_combined|0.823635|0.923903|0.940018|0.869947|56|6|50|
|B03|ocr_alignment_support|0.469114|0.522829|0.536258|0.493562|123|469|-346|
|B03|coordinate_only|0.330349|0.435989|0.467323|0.377422|63|564|-501|
|B03|ocr_anchor_overlap_control|0.661594|0.772605|0.78872|0.707518|169|300|-131|
|B03|bm25_text_control|0.831692|0.954342|0.975828|0.886173|157|98|59|
|F03|baseline|0.870188|0.962399|0.9812|0.913331|0|0|0|
|F03|availability_only|0.887198|0.969561|0.986571|0.925736|19|0|19|
|F03|spatial_only|0.870188|0.962399|0.9812|0.913331|0|0|0|
|F03|combined|0.885407|0.969561|0.985676|0.924872|18|1|17|
|F03|shuffled_spatial_only|0.870188|0.962399|0.9812|0.913331|0|0|0|
|F03|shuffled_combined|0.887198|0.969561|0.985676|0.92574|19|0|19|
|F03|ocr_alignment_support|0.469114|0.522829|0.536258|0.493562|52|500|-448|
|F03|coordinate_only|0.330349|0.435989|0.467323|0.377422|35|638|-603|
|F03|ocr_anchor_overlap_control|0.661594|0.772605|0.78872|0.707518|76|309|-233|
|F03|bm25_text_control|0.831692|0.954342|0.975828|0.886173|21|64|-43|


匹配规则×汇总规则的2×2对照独立列出。空间两行均为γ=0、λ=0.1；Query权重η=0.2仅从验证身份选择。最后两行额外保留γ=0.2的文字对齐可用性奖励作为对照，不能合称纯空间匹配。修复/损害列均相对于original_uniform；各自均匀汇总的损益和来源组bootstrap另见逐图过渡与交互表。

|cell|photos|eta|gamma|spatial_lambda|r1|r5|r10|mrr|package_macro_r1|repaired_vs_original_uniform|harmed_vs_original_uniform|
|---|---|---|---|---|---|---|---|---|---|---|---|
|original_uniform|1117|0.0|0.0|0.0|0.7788719785138765|0.8961504028648165|0.9203222918531782|0.833983800093587|0.790922912557528|0|0|
|original_query_weighted|1117|0.2|0.0|0.0|0.7797672336615935|0.8979409131602507|0.9212175470008953|0.8346035305724281|0.7899606269798578|5|4|
|spatial_only_uniform|1117|0.0|0.0|0.1|0.7555953446732319|0.8854073410922113|0.9113697403760072|0.8131256508381166|0.7594290965444812|5|31|
|spatial_only_query_weighted|1117|0.2|0.0|0.1|0.756490599820949|0.8845120859444942|0.9131602506714414|0.8148895358965724|0.7594873716027561|6|31|
|availability_spatial_uniform|1117|0.0|0.2|0.1|0.8263205013428827|0.9239033124440466|0.9427036705461056|0.8719561652135652|0.8331752222136838|58|5|
|availability_spatial_query_weighted|1117|0.2|0.2|0.1|0.8254252461951657|0.9230080572963295|0.9444941808415398|0.8720804651207186|0.8312521452906069|59|7|


[完整候选界、精确top10/正确候选排名、参数选择和逐图2×2交互](weighted_spatial_only_reviewed/REPORT.md)已保存。未精算的候选保留NaN及有效上界，不能把排名认证说成保存了全部修正后的标量分数。空间约束与加权没有形成通用修复；原本成功样本的损害一并保留。

**多因素怎样共同作用？** 每张卡分别列直接观察、分数差额、改评分响应、像素干预、竞争解释和未知。文字内容、布局、尺度、包装面和标签可同时不同；无法隔离时保留混杂。例如Mucus DM重点例不仅布局不同，正/负剂量是60/1200mg与30/600mg；Capstar负设计为猫+狗且计数不同，Query下部计数被鼠标遮挡。

**哪些可写进论文？** 可以报告忠实基线成绩、融合修复/损害、限定身份分组的Probe/完整图库差距、OCR可用性与空间项的对照、经过像素守恒验收的合成响应、人工修订定位的输入有效性反例。不能声称绝对位置预测证明自然实例布局可分，不能把Probe失败写成信息不存在，不能将共享文字相关或少数编辑响应升级成普遍因果机制。历史测试已被观察，不能称新盲测。

## 覆盖与分母

全部主档案 1117/1117；全部Token分析 1117/1117；去重正负对 1648；完整关键矩阵2765。详细验收：[本轮Token独立复放](delivery_verification/full_tokens/independent_validation.json)。

视觉覆盖必须按overview、场景各维度、细节放大和机制证据分别解释：

```json
{
  "schema": "colnomic.consolidated_actual_vision.v1",
  "queries": 1117,
  "unique_queries": 1117,
  "candidate_overview_reviewed": 1117,
  "candidate_overview_missing": [],
  "scene_audited_by_dimension": {
    "background": 1117,
    "placement": 1117,
    "occlusion": 1117
  },
  "scene_known_by_dimension": {
    "background": 1111,
    "placement": 1117,
    "occlusion": 1116
  },
  "scene_unreviewed_ids": {
    "background": [],
    "placement": [],
    "occlusion": []
  },
  "scene_uncertain_ids": {
    "background": [
      "photo_045575d4d47e1205",
      "photo_12b37e2d68711d62",
      "photo_3e01b05369dd6cf5",
      "photo_46b3373ecce22234",
      "photo_4920adb7ad190a5f",
      "photo_813fb072c0f3b380"
    ],
    "placement": [],
    "occlusion": [
      "photo_ca454f8a9d866abb"
    ]
  },
  "scene_counts": {
    "background": {
      "single_object": 858,
      "multi_object": 253,
      "unclear": 6
    },
    "placement": {
      "tabletop": 601,
      "handheld": 516
    },
    "occlusion": {
      "none": 1072,
      "foreground": 44,
      "unclear": 1
    }
  },
  "all_three_scene_audited": 1117,
  "all_three_scene_known": 1110,
  "explicit_fine_detail_verified": 13,
  "selected_detail_observation_records": 85,
  "original_images_sha_verified": 1117,
  "objective_outcomes": {
    "both_correct": 829,
    "image_wrong_fusion_correct": 143,
    "both_wrong": 104,
    "image_correct_fusion_wrong": 41
  },
  "primary_source_groups": {
    "photo_0007f82bf7613055.json": 1,
    "photo_002246f5f205397a.json": 1,
    "batches_001_004.json": 14,
    "batches_005_008.json": 13,
    "visual_audit": 204,
    "results": 329,
    "zoom_photo_045575d4d47e1205.json": 1,
    "full_tokens_agent": 90,
    "geometry_agent": 51,
    "tokens_resume": 260,
    "root_resume": 77,
    "scene35_agent": 76
  },
  "same_face_success_controls": {
    "missing_confirmed_same_face_success": 251,
    "actual_same_face_success_quality_not_matched": 37
  },
  "mechanism_limit": "1117 observed images does not imply 1117 established mechanisms",
  "narrow_package_count_policy": "token-agent package counts retained in raw records; actual scene rechecks supply background",
  "same_face_pair_comparisons_actually_viewed": 49
}
```

全部现有可观察困难的发生率如下，分母是1117，包括成功控制；多标签同时出现。细则未核验不填成“无差异”。错误内部比例及身份分布见统一视觉和场景报告。

|difficulty|photos|denominator|fraction|
|---|---|---|---|
|blur_low_resolution_or_small|586|1117|0.52462|
|shared_text_or_brand|380|1117|0.340197|
|graphics_or_color|372|1117|0.333035|
|different_or_hidden_face|351|1117|0.314235|
|numbers_specs_or_quantity|346|1117|0.309758|
|layout_or_linewrap|293|1117|0.26231|
|unclear_or_no_explicit_difficulty_tag|246|1117|0.220233|
|perspective_rotation_or_deformation|155|1117|0.138765|
|product_name_ingredient_or_use_difference|130|1117|0.116383|
|extra_background_entities|108|1117|0.096688|
|insufficient_identity_information|96|1117|0.085944|
|occlusion|40|1117|0.03581|
|reflection_or_low_contrast|40|1117|0.03581|
|barcode_or_code_detail|40|1117|0.03581|
|dense_or_sparse_print|19|1117|0.01701|
|multilingual_text|12|1117|0.010743|
|identity_or_gallery_ambiguity|7|1117|0.006267|
|crop_or_partial_view|4|1117|0.003581|


以上沿用人工观察的既有18类统一映射，原始细标签与审核历史一并保留。实际同面成功控制覆盖如下；missing或unconfirmed不表示该照片不存在：

|configuration|category|failure_queries|queries_with_actual_same_face_success|control_missing_or_unconfirmed|identities_with_control|
|---|---|---|---|---|---|
|B03|all|247|24|223|17|
|B03|layout_or_linewrap|68|10|58|8|
|B03|numbers_specs_or_quantity|50|5|45|5|
|B03|shared_text_or_brand|103|9|94|6|
|F03|all|145|18|127|13|
|F03|layout_or_linewrap|35|7|28|5|
|F03|numbers_specs_or_quantity|44|8|36|6|
|F03|shared_text_or_brand|54|5|49|4|


[同面固定正负候选对照](correspondence_controls/fixed_candidate_same_face_controls.csv)同时保持两种参考和候选Token数相同，保存完整矩阵及来源组区间；Query尺度、清晰度、角度仍未匹配。全量匹配集中度按候选Token数分层，跨行和多对一现象不能直接判为错误。

语义ROI：4张重点图、13个人工区域；全量独立分区仍守恒，但其余关键规格/布局/图案未定位时保留未知。[审后区域范围](reviewed_regions/localization_coverage.json)。视觉patch可跨行，并非单词。

## 35配置、三个场景维度与互补性

权威明细：[35配置报告](scene35_reviewed/REPORT.md)、[逐图配置—场景—Token关联表](delivery/config_scene_photo_token_evidence.csv)。物体数量、手持/桌面、前景遮挡独立统计；前三者不能相加。原标签、118已审核子集和审后统计并行保留，原审核子集有选择偏差。

E0–E7保留原规则与优先级，未校准。E2只是正确设计排名>10，E3只是2–10；E4/E5是通道正确性关系。E2主标签数不等于完整Recall@10未命中数。C0–C7沿用项目定义，未人工核验的不自行推断。

B03是原图ColNomic、B05是BM25、B08是OCR→图像、B12组合B03/B08、F03组合B03/B05；这些融合输入不同。B16 image_crisp改变forward/reverse Token评分，并未裁切或提高清晰度。D01人工框裁切具有额外标注监督；D03是SAM3遮背景并裁切且有fallback。裁切同时改变背景、缩放和内容完整性。

原标签中多物体177→181与单体693→791、有前景遮挡32→24均逐图复核，详见原场景过渡表及8张融合新增错的文字项分解。这些是原标签下统计，审后数字另列；不能由32张图像全对推普遍遮挡鲁棒性，不能将融合错误当成遮挡因果作用。

全量实际审核后的三个独立维度如下。每行分母为该类可用照片数；未知值单列。区间为身份相关的来源组配对bootstrap，尤其小类不宜作因果解释。

|dimension|label|available_photos|identities|baseline_correct|variant_correct|both_correct|rescues|harms|both_wrong|net|source_group_bootstrap_low|source_group_bootstrap_high|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|background|multi_object|253|45|193|195|167|28|26|32|2|-0.08118381183811837|0.08580888872469336|
|background|single_object|858|110|672|774|659|115|13|71|102|0.08813549363659902|0.15011727058025284|
|background|unclear|6|5|5|3|3|0|2|1|-2|-0.625|0.0|
|placement|handheld|516|122|378|439|361|78|17|60|61|0.0763023522662077|0.16116978687192973|
|placement|tabletop|601|130|492|533|468|65|24|44|41|0.02590371096784216|0.10859333270747878|
|occlusion|foreground|44|29|41|35|33|2|8|1|-6|-0.3333333333333333|0.05|
|occlusion|none|1072|130|828|936|795|141|33|103|108|0.0674776000377375|0.13321632163216318|
|occlusion|unclear|1|1|1|1|1|0|0|0|0|0.0|0.0|


任一35方法成功的oracle只是需真值的比较上界；共同失败/互补逐图保存。总体成绩接近不保证错误相同，跨模型成功只支持竞争比较。

实际交集如下，B03比较对象按真实“原图image”输入合同预先选择，不把文字/裁切方法混作同一编码器对照：

|scope|queries|identities|
|---|---|---|
|all35_fail|25|18|
|any_method_oracle_correct|1092|130|
|fusion_introduced|41|23|
|B03_fail_others_all_original_image_correct|0|0|
|B03_fail_at_least_one_original_image_correct|125|56|


[困难×配置的类内失败率、错误构成、逐图交集与损益](difficulty_config/REPORT.md)保留独立分母和多标签重叠；多个方法共有的错误、相对于指定方法的B03错误和融合新增错分别可追溯。它们不是机制已查明数量。

## 冻结特征Probe与受控布局

身份隔离为91/21/18包装和82/18/18来源组，783/173/161查询，所有实际交集已检查。标准化仅训练数据，参数/阈值仅验证；feature-only不输入坐标/ID。坐标与ROI辅助头单独标额外输入。阈值准确率、同Query配对排序、差额、AUC、身份宏平均并行报告。

原受限候选对的高配对成绩无法代表检索：

|head|queries|identities|recall_at_1|recall_at_5|recall_at_10|mrr|repaired|harmed|
|---|---|---|---|---|---|---|---|---|
|raw_maxsim|161|18|0.7763975155279503|0.9192546583850931|0.9254658385093167|0.8432511952493451|0|0|
|maxsim_only|161|18|0.7763975155279503|0.9192546583850931|0.9254658385093167|0.8432511952493451|0|0|
|pooled_only|161|18|0.19254658385093168|0.35403726708074534|0.42857142857142855|0.2667315340658795|0|94|
|pooled_plus_maxsim|161|18|0.3167701863354037|0.5341614906832298|0.6086956521739131|0.4236493232162381|1|75|
|kernel_set_only|161|18|0.018633540372670808|0.055900621118012424|0.09316770186335403|0.042706652856725175|0|122|


上表测试集161张/18身份，全图库5405。池化只检验均值/标准差组合，kernel mean是有限集合分布摘要，二者失败都不能证明完整Token没有布局信息。完整Query×Reference点积关系在池化前读取的38维头，以及Token数量单独评分对照的完整图库成绩如下：

|head|queries|identities|recall_at_1|recall_at_5|recall_at_10|mrr|identity_macro_recall_at_1|repaired|harmed|net_change|
|---|---|---|---|---|---|---|---|---|---|---|
|relation|161|18|0.09316770186335403|0.2795031055900621|0.37267080745341613|0.19549768369015494|0.08497474747474748|0|110|-110|
|count_only|161|18|0.0|0.0|0.0|0.0005528367383403022|0.0|0|125|-125|


[关系头全部配对、困难对、完整图库及独立验收](cross_token_relation/REPORT.md)均保留。完整图库候选的Token数量范围与训练配对明显不同；小Token参考大量成为赢家，这是支持范围差异的证据，不能独自解释全部下降。此头仍为有限关系摘要，受限配对高分不能当作检索修复。

OCR行序标签是主体OCR行的纵向排序位置，非阅读行号。仅y坐标基线显著解释它，绝对位置预测不直接检验同文字身份布局。旧targets的9个打乱头已恢复，当前审核框也重新拟合并独立重建标签/验证全部主头、9null与坐标对照：

|probe|control|mae|package_macro_mae|r2|dim0_four_bin_accuracy|
|---|---|---|---|---|---|
|absolute|feature_only|0.20919502664831544|0.20981669071924428|0.24219198495068994|0.27828998447204967|
|absolute|train_mean|0.2500215957210529|0.24990379584544106|-7.404784247100338e-06|0.25228066770186336|
|subject_relative|feature_only|0.2337085799731952|0.23380008028589794|0.0885550195022955|0.26044198895027626|
|subject_relative|train_mean|0.24933768162984182|0.24929845757740546|-1.060566293098475e-05|0.2525414364640884|
|ocr_vertical_order|feature_only|0.24129280377264475|0.2531934825152556|0.20433256191622984|0.28098360655737703|
|ocr_vertical_order|train_mean|0.28404548617694353|0.29833055995461044|-0.0008644807106326091|0.1980327868852459|
|ocr_vertical_order|y_coordinate_only|0.11075803580168518|0.11633591209430993|0.7665293381200207|0.6577049180327869|


[当前定位Probe验收](positions_reviewed/independent_validation.json)只验证冻结特征可预测这些几何标签；坐标对照故意输入y，与feature-only头分开。

文字内容单独预测同一OCR纵向序号也有可测关联：训练词表的256维词存在向量，不输入坐标/Token、索引、图像ID或OCR行数；词表/标准化仅训练，ridge仅验证选择，主头与3null闭式独立重拟合。它检查文字/位置相关性，不直接证明feature-only头利用某种捷径。

|control|seed|samples|packages|mae|package_macro_mae|r2|dim0_four_bin_accuracy|
|---|---|---|---|---|---|---|---|
|OCR_content_only||6100|18|0.27744568006597436|0.2877630357669277|0.034065128103134334|0.2124590163934426|
|label_shuffle|17|6100|18|0.28391233611756933|0.29683898343996606|0.002568339179243928|0.18918032786885247|
|label_shuffle|31|6100|18|0.2847489630321157|0.29664356066515596|-0.002780895044127707|0.20442622950819672|
|label_shuffle|53|6100|18|0.2848433868865662|0.2961679146637014|-0.0016579100570643934|0.1977049180327869|


[文字内容与位置关联控制](position_content_control/REPORT.md)的样本关联仍依靠OCR多边形/主体框外部监督。

从130身份按固定像素质量规则选出全部41合格词位置源，27/9/5身份划分，246编码，41原图与缓存逐位一致。等幅左右移动原字形，原图/sham/editor-only对照均保存；只修改单个孤立词，不能称自然换行或身份布局实验。测试5身份，解释限于合成编辑：

|probe|accuracy|auc|pair_accuracy|package_macro_pair_accuracy|packages|gap_mean|
|---|---|---|---|---|---|---|
|kernel_set_only|0.5|0.44|0.2|0.2|5|-0.029817493170458943|
|maxsim_only|0.5|0.6|0.8|0.8|5|0.0010014327781378324|
|pooled_only|0.5|0.4|0.2|0.2|5|-1.382006469394372|
|pooled_plus_maxsim|0.5|0.44|0.2|0.2|5|-0.004747251635408611|
|word_ROI_assisted|0.4|0.4|0.0|0.0|5|-0.0586703740457317|


历史唯一测试身份50%是pilot，阈值和配对必须拆开。旧38身份实验比较位移幅度，同样不是身份布局证据。全图库精确OCR词bag相同且异SHA组0是代理匹配缺失，不能证明自然同文字包装不存在。

自然共享短语局部身份配对：固定质量规则在1117查询/130身份上筛得238组、202张图、65身份，训练/验证/测试身份{'train': 47, 'val': 10, 'test': 8}。每组正负均含相同四条OCR短语，但其它内容、颜色、字体与整图上下文不同；只检验这些局部特征与身份判断的关系，不是布局单因素实验。65身份的首例四短语实际放大查验，后续同身份案例依旧含OCR核验范围限制。

|probe|samples|packages|accuracy|auc|pair_accuracy|package_macro_pair_accuracy|gap_mean|
|---|---|---|---|---|---|---|---|
|local_maxsim_only|48|8|0.5208333333333334|0.53125|0.5416666666666666|0.58125|0.007571819928048289|
|shared_phrase_pooled_only|48|8|0.5208333333333334|0.5034722222222222|0.375|0.4458333333333333|-0.07184871745829564|
|shared_phrase_pooled_plus_maxsim|48|8|0.5416666666666666|0.5052083333333334|0.375|0.4458333333333333|-0.07303361920171692|
|shared_phrase_relation_only|48|8|0.5625|0.5920138888888888|0.625|0.6|0.2988305766477985|
|ROI_token_count_control|48|8|0.5208333333333334|0.40625|0.3333333333333333|0.4041666666666667|-0.026450431051058095|


|probe|test_identities|test_pairs|identity_macro_pair_accuracy|ci95_low|ci95_high|
|---|---|---|---|---|---|
|local_maxsim_only|8|24|0.58125|0.36244791666666665|0.8000520833333327|
|shared_phrase_pooled_only|8|24|0.4458333333333333|0.20833333333333331|0.6958333333333333|
|shared_phrase_pooled_plus_maxsim|8|24|0.4458333333333333|0.20833333333333331|0.6958333333333333|
|shared_phrase_relation_only|8|24|0.6|0.38541666666666663|0.7875|
|ROI_token_count_control|8|24|0.4041666666666667|0.15625|0.671875|


测试8身份/24配对，小样本区间较宽；ROI选Token含OCR位置辅助监督，patch可越过选定短语。所有5头、15null、标准化/阈值/参数均独立重放，结果不能证明纯布局可分或特征完全无信息。[局部Probe验收](natural_shared_phrase_layout/independent/validation.json)。

跨两张不同原图的同词左右等幅编辑进一步排除了同一图片自身匹配捷径。全130身份固定规则实际筛查、视觉核实后仅3身份合格，2训练/1验证/0测试，6原图/36编码；因此只报告预定义同向/反向关系和sham、editor-only差额，不训练/宣称held-test Probe或自然身份精度。[跨视角pilot](controlled_crossview_layouts/analysis/REPORT.md)。

## 编码审计、图像干预与人工框修订

近期Query drift已定位PIL后端；TV+FP32 LoRA原图可逐位复现。旧图库5405均从已缺失的历史canonical/gallery缓存复用，缺少原processor张量与中间层日志；现多recipe测试仍无法逐位复现。不能唯一归因GPU型号、CPU卸载或某个编码层。冻结分析不受此影响。

12独立身份、4客观结果各3张进行了数字区域/等面积非关键区/组合遮挡与sham，60编码、145矩阵保存Token变化、匹配切换/未切换贡献与交互项。数字OCR行有的并非身份关键，有的非关键区在背景；编辑影响语义、对比度和纹理，报告保留这些限定。新Query对旧冻结图库的条件排名不等于新运行时全图库准确率。

另补同数值运行时参考对照：预先冻结的全部28参考及12原Query重新编码，12 Query base全部逐位等于已有编辑臂基准；包版本、模型/处理后端、数值encoder源码一致，因此复用60个兼容编辑臂。历史worker完整源码SHA不同且快照缺失，无法唯一说明其差异。新旧参考各145个完整矩阵、17对×5arms×2参考bank的170行配对和区域/交互分解均保留。28个新参考均未逐位复现旧缓存；17个base正负差额的符号均保持不变，不能由特征漂移直接断言它造成原错误。

[新参考配对及独立重放](fresh_reference_interventions/analysis/REPORT.md)明确区分新参考条件响应和原始全图库统计。同Token长度或ordinal索引轴不证明新旧物理格点相同；跨bank argmax编号变化仅为索引诊断，同bank edit-sham切换另保存。

[按身份重采样的编辑响应和新旧参考差异](fresh_reference_interventions/interpretation/REPORT.md)给出全部预定义单因素/组合效应及区间，保留改善、无变化和反向响应；它们不是编辑后身份准确率。

用户复核后5张框坐标变化，旧框/新框版本分别保存，旧统计不覆写。5张旧D01裁切重新编码全部逐位一致；完整图库比较为：

|photo_id|split|B03_rank|F03_rank|before_D01_rank|fresh_before_D01_rank|after_D01_rank|
|---|---|---|---|---|---|---|
|photo_41de3eb383934ec3|test|2|2|2|2|2|
|photo_67f24002bcb357c9|train|1|1|2|2|1|
|photo_88e7f77db7da30c3|val|1|1|1|1|1|
|photo_c81d2a7abb5af160|train|5|7|189|189|1|
|photo_d33ae6dfc0cb239b|train|1|1|1|1|1|


这是历史结果被查看后的人工作业，不能作为无标注定位算法或独立盲测成绩。B03/F03原图特征与基线分数未改变。

[8张原前景遮挡融合新增错的实际关键区域审核](foreground_harms_review/REPORT.md)区分“有遮挡”和“规格/身份信息被挡”；图像项为正而文字项反向更大是确定性评分事实，是否由遮挡改变OCR还需要反事实对照，不能仅凭这8例推因果。

## 复现、验收和限制

Token/区域守恒、评分轴、SHA排序、参数选择、损益、矩阵逐点重放均由独立程序检查。范围边界：[当前逐项验收与限制](delivery/acceptance.json)。completion_audit中的旧inventory仅记录接续时情况，当前总验收以此处为准。冻结数学验证与语义人工审核不是同一保证。

只验证和全部可复现分析入口、代码/环境/协议/日志/恢复点及归档清单位于交付根。原缓存保持不变；subject_boxes是用户授权修改，前后快照及SHA另存。

未知/未执行/不适用的范围继续显式留在验收矩阵；1117张有数据与审计，不等于1117种失败原因全部查明。

[可浏览HTML与全部照片筛选](delivery/index.html) · [一键冻结复现/只验说明](reproduction_support/README.md)

[最终完整入口说明](reproduction_support/README_FINAL.md) · [原九阶段及35配置证据映射](FINAL_PROTOCOL.json) · [缺失及尚未证明范围](MISSING_AND_LIMITATIONS.json)
