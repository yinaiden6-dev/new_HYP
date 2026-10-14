# 35 配置统计接续核验

共有 1117 张独立查询、5405 条参考记录、5330 个 SHA 去重设计。39095 条是同一批照片上的 35 次重复评测，不是独立图片。缺失预测 0 条。

当前项目汇总多了 L01, L02；本报告按历史 manifest 的预测路径恢复原35配置，未混入额外配置。
历史 summary/failure_types/failed_photos/joint_statistics/photo_labels 与三个宽表均逐字段重算；结果见 validation.json。

## 原标签下 B03 → F03

修复与损害分别报告；净变化是两者之差。以下分母是对应场景可用的逐图预测数。

|场景维度|场景|N|身份|两者都对|图像错融合对|图像对融合错|两者都错|净变化|
|---|---|---|---|---|---|---|---|---|
|overall|all|1117|130|829|143|41|104|102|
|background|multi_object|235|45|155|26|22|32|4|
|background|single_object|882|111|674|117|19|72|98|
|placement|handheld|509|122|357|75|17|60|58|
|placement|tabletop|608|130|472|68|24|44|44|
|occlusion|foreground|32|20|24|0|8|0|-8|
|occlusion|none|1085|130|805|143|33|104|110|
|joint|handheld/multi_object/foreground|9|9|6|0|3|0|-3|
|joint|handheld/multi_object/none|95|28|54|14|9|18|5|
|joint|handheld/single_object/foreground|4|4|4|0|0|0|0|
|joint|handheld/single_object/none|401|88|293|61|5|42|56|
|joint|tabletop/multi_object/foreground|0|0|0|0|0|0|0|
|joint|tabletop/multi_object/none|131|45|95|12|10|14|2|
|joint|tabletop/single_object/foreground|19|16|14|0|5|0|-5|
|joint|tabletop/single_object/none|458|111|363|56|9|30|47|

三个场景维度独立，各自分区完整；不能把维度数量相加。交叉场景、split、独立身份、来源组和配对 bootstrap 已保存。小样本 bootstrap 不稳定，不能把身份组成、可见面、尺度、清晰度差异解释为场景因果。

## 审核状态

正式元数据 reviewed=true 经实际读取为 118/1117；与历史声明匹配为 True。已读取独立场景修订 1117 张、视觉记录 1117 张。未核验字段继续保留原标签和未知状态。

original_labels/ 保存原标签全量统计，original_reviewed_subset/ 保存原已审核子集统计；verified_labels/ 在存在独立场景修订时保存核验快照。原元数据与 reviewed 标志均未改写，修订见 scene_label_revisions.json。

原已审核子集不具有随机抽样保证：前景遮挡：已审核 29/118（24.6%），全量 32/1117（2.9%）；桌面：已审核 97/118（82.2%），全量 608/1117（54.4%）；多物体：已审核 57/118（48.3%），全量 235/1117（21.0%）。这些数字是逐图标签快照的描述，不能代表全量。身份、split、场景构成见 scene_population_splits.csv。前景遮挡与身份关键区域被挡分开记录，后者需要逐图视觉证据。


## 全量视觉审后重算

审后场景标签逐维度读取实际图像观察；不确定值单独列为unclear。reviewed_scene_dimension_denominators.csv给出每维已知、不确定、未审分母，configuration_scene_uncertainty.csv分别保存原全量、原reviewed子集和审后快照。

|维度|标签|N|身份|B03正确|F03正确|修复|损害|净变化|
|---|---|---|---|---|---|---|---|---|
|overall|all|1117|130|870|972|143|41|102|
|background|multi_object|253|45|193|195|28|26|2|
|background|single_object|858|110|672|774|115|13|102|
|background|unclear|6|5|5|3|0|2|-2|
|placement|handheld|516|122|378|439|78|17|61|
|placement|tabletop|601|130|492|533|65|24|41|
|occlusion|foreground|44|29|41|35|2|8|-6|
|occlusion|none|1072|130|828|936|141|33|108|
|occlusion|unclear|1|1|1|1|0|0|0|

## 32 张原前景遮挡照片与8例融合损害实际复核

|照片|遮挡物|图像贡献|文字贡献|融合差额|一般身份区被挡|对比候选区分规格被挡|
|---|---|---|---|---|---|---|
|photo_12b37e2d68711d62|拇指|0.13918709754943848|-0.26849842071533203|-0.12931156158447266|none|none|
|photo_21bd7df9ed6795e7|黑色鼠标|0.33829593658447266|-0.9334259033203125|-0.5951299667358398|yes|yes|
|photo_28d813f7c9451126|白色手持设备背面，具体设备类别未判定|0.09324121475219727|-0.5329866409301758|-0.4397449493408203|yes|yes|
|photo_3657bb9dffa08975|拇指|0.2715339660644531|-0.4866619110107422|-0.21512794494628906|yes|none|
|photo_4920adb7ad190a5f|拇指；另有原始拍摄左边缘截出|0.38187599182128906|-1.388716220855713|-1.006840705871582|yes|unknown|
|photo_7e0b721a9c25b555|白纸|0.11455726623535156|-0.619269847869873|-0.5047125816345215|yes|none|
|photo_c0babaac8a7306cd|白纸|0.37569522857666016|-2.6627635955810547|-2.2870683670043945|yes|yes|
|photo_daac4b2576b9e9f5|黑色鼠标|0.14984822273254395|-0.1616649627685547|-0.011816978454589844|yes|none|

8例均经原图、局部与正负候选实际审核；区分规格被挡yes3、none4、unknown1。q4920数量前缀缺失不能区分手指遮挡与原始画面截出；q21负参考为蓝色60片猫狗同框，Query数量行不可读；q3657的6片数仍可读；qdaac的144可读而125mg被挡。
[8例细节与逐图图片链接](foreground_harms_reviewed.csv)及[原始人工记录](../foreground_harms_review/manual_reviews.json)保存不确定范围；旧32表与旧scene统计保留于before_final_scene_refresh/。

图像项与文字项来自 full_fusion 的正负候选确定性分解，原始 OCR 词贡献表用于定位评分来源。是否遮挡关键区域以独立视觉审核为准；32 张图像全部正确不证明普遍遮挡鲁棒性，这8例也不证明遮挡导致融合失败。

## 方法含义与竞争解释

B03 是原图 mean MaxSim；B05 是 PP-OCRv6 文本 BM25；B08 是 OCR 文字作为 ColNomic 查询匹配图像参考；B12 融合 B03+B08；F03 融合 B03+B05。两种融合的文字通道不同，虽同用逐 Query population z-score 和 α=0.5。

B16 image_crisp 复用 B03 的同一输入、embedding 和相似度，改为参考 Token distinctiveness 加权 forward MaxSim − 加权均值 + reverse MaxSim。它没有裁切或提升像素清晰度，不能把差异归因于清晰度。全部35配置的 checkpoint、revision、输入合同与实际评分代码已记入 configuration_semantics.json。

D01 使用人工 confirmed 主体框，在 EXIF 转正原图上加最大边5%边距裁切，属于额外定位监督；D03 使用 SAM3 'product package' 自动最高置信度 mask（阈值0.2，mask阈值0.5），先白底合成再加同样边距裁切。两者原分辨率存储，随后通过模型处理器缩放；背景、尺度与内容完整性同时变化。preprocessing_inputs.csv 保存2234个实际输入、SHA验证、裁切范围、主体像素和原OCR中心保留代理；代理不是人工可见性审核。

## 全35配置错误交集

|配置|B03错而该方法对|B03对而该方法错|共同失败|净变化|
|---|---|---|---|---|
|B01|17|307|230|-290|
|B02|69|120|178|-51|
|B04|65|126|182|-61|
|B05|157|98|90|59|
|B06|33|342|214|-309|
|B07|69|164|178|-95|
|B08|58|152|189|-94|
|B09|46|372|201|-326|
|B10|34|247|213|-213|
|B11|84|111|163|-27|
|B12|60|48|187|12|
|B13|70|132|177|-62|
|B14|19|449|228|-430|
|B15|36|213|211|-177|
|B16|22|140|225|-118|
|B17|33|357|214|-324|
|B18|58|200|189|-142|
|B19|59|176|188|-117|
|B20|48|386|199|-338|
|B21|7|723|240|-716|
|B22|16|535|231|-519|
|B23|73|155|174|-82|
|B24|26|238|221|-212|
|B25|46|235|201|-189|
|B26|47|169|200|-122|
|B27|21|312|226|-291|
|B28|43|259|204|-216|
|D01|62|51|185|11|
|D03|57|109|190|-52|
|F01|141|77|106|64|
|F02|154|67|93|87|
|F03|143|41|104|102|
|F04|159|63|88|96|
|F05|145|73|102|72|

任一配置正确的 oracle 为 1092/1117；它需要事后知道哪个方法正确，不是可部署系统的成绩。总体指标相近不代表错误相同。跨模型成功仅提供比较证据，不能确定 ColNomic 编码器或某个层的失败原因。

## E0–E7 与 C0–C7

保留历史 E 主标签以及 E0/E1 → E4/E5/E6 → E2/E3 → E7 优先级，全部为未人工校准规则。E2 的事实仅是正确设计排名>10；E3 仅是正确排名2–10；E4/E5 是图像、文字和融合正确性关系。literal_ranks_vs_E2.csv 分开保存 Recall@10 未命中与 E2 数量，避免优先级导致的漏计。

C0–C7 已读取沿用 dataset/tools/error_analysis.py 原定义，但本脚本不猜测逐图包装关系。数据/参考有效性、实际排名、通道正确性、可见差异和机制证据分别存储。

## 数据接口与复现

- prediction_matrix.csv/.npz：1117×35 的逐图正确性、排名、第一名、可用完整图库分数差额；缺失使用显式哨兵。
- all_config_pair_intersections.csv、controlled_method_transitions.csv、cross_config_photo_evidence.csv：全部方法两两交集与逐图互补。
- B03_F03_scene_transitions.csv：每个独立场景/交叉场景及 split 下四格、修复、损害、净变化与来源组区间。
- photo_evidence_interface.json：场景—配置—照片—独立视觉审核—Token 证据的连接。
- validation.json：实际轴、全图库稳定排序、SHA去重、历史逐表逐字段、源文件未修改的验证。
- final_scene_statistics/manifest.json：从已核验且未改变的1117×35矩阵重算最终scene定义修订，保留原35重放与新审核的独立来源。

```bash
OMP_NUM_THREADS=2 /home/jwei/miniconda3/envs/rebut/bin/python -B -m scripts.colnomic_mechanism.scene35
```

上述统计属于确定性评分关系和描述性关联；尚未有逐图图像干预或竞争证据的类别不能升级为机制结论。
